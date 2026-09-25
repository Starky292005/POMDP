import math
import numpy as np
from scipy.spatial import Voronoi

import config
import world
import models
import planners

def build_world(
    rng,
    n_seeds=config.N_SEEDS,
    obstacle_fraction=config.OBSTACLE_FRACTION,
    threat_fraction=config.THREAT_FRACTION,
    min_reachable=config.MIN_REACHABLE,
    max_tries=50,
):
    """
    Executes the environment generation sequence. Iterates until the
    reachability condition is satisfied.
    """
    for attempt in range(max_tries):
        points = world.generate_points(rng, n_seeds)
        points = world.lloyd_relax_points(
            points, base_idx=0, iterations=4,
            xmin=config.BOUNDS[0], xmax=config.BOUNDS[1], ymin=config.BOUNDS[0], ymax=config.BOUNDS[1],
        )
        reflected_points = world._create_reflected_boundaries(points, config.BOUNDS[0], config.BOUNDS[1], config.BOUNDS[0], config.BOUNDS[1])
        vor = Voronoi(reflected_points)
        regions, vertices = world.voronoi_finite_polygons_2d(vor)

        polygons = []
        for region in regions[:n_seeds]:
            poly_pts = [tuple(vertices[i]) for i in region]
            poly_pts = world.clip_polygon(poly_pts, config.BOUNDS[0], config.BOUNDS[1], config.BOUNDS[0], config.BOUNDS[1])
            polygons.append(poly_pts)

        adjacency = world.build_adjacency(vor, n_seeds)

        base_idx = 0
        non_base = np.array([i for i in range(n_seeds) if i != base_idx])
        perm = rng.permutation(non_base)

        n_obstacles = max(1, round(obstacle_fraction * n_seeds))
        n_threats = max(1, round(threat_fraction * n_seeds))

        obstacles = set(int(c) for c in perm[:n_obstacles])
        remaining = perm[n_obstacles:]
        threats = set(int(c) for c in remaining[:n_threats])

        transition_model = models.build_movement_model(points, adjacency, obstacles)
        reachable = models.movement_reachable(base_idx, transition_model)
        free_non_base = [c for c in reachable if c != base_idx]

        if len(free_non_base) >= min_reachable:
            victim_idx = int(rng.choice(free_non_base))
            return {
                "points": points,
                "adjacency": adjacency,
                "polygons": polygons,
                "obstacles": obstacles,
                "threats": threats,
                "base_idx": base_idx,
                "victim_idx": victim_idx,
                "reachable": reachable,
                "model": transition_model,
                "n_cells": n_seeds,
                "attempt": attempt,
            }

    raise RuntimeError("Matrix configuration failed. Target connectivity threshold not met.")


def sample_from_distribution(distribution, rng):
    items = list(distribution.items())
    states = [item[0] for item in items]
    probabilities = [item[1] for item in items]
    index = rng.choice(len(states), p=probabilities)
    return states[index]


def sample_observation(points, uav_cell, victim_cell, action, rng):
    if action != "SEARCH":
        return "NONE"
    distribution = models.search_observation_distribution(points, uav_cell, victim_cell)
    return sample_from_distribution(distribution, rng)


def entropy(belief):
    result = 0.0
    for p in belief.values():
        if p > 0:
            result -= p * math.log2(p)
    return result


def most_likely_victim(belief):
    return max(belief, key=belief.get)


def run_simulation(
    seed=None,
    n_seeds=config.N_SEEDS,
    obstacle_fraction=config.OBSTACLE_FRACTION,
    threat_fraction=config.THREAT_FRACTION,
    min_reachable=config.MIN_REACHABLE,
    max_search_steps=config.MAX_SEARCH_STEPS,
    max_return_steps=config.MAX_RETURN_STEPS,
):
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**32))
    rng = np.random.default_rng(seed)

    env_world = build_world(
        rng, n_seeds=n_seeds, obstacle_fraction=obstacle_fraction,
        threat_fraction=threat_fraction, min_reachable=min_reachable,
    )
    transition_model = env_world["model"]
    points = env_world["points"]
    base = env_world["base_idx"]
    victim = env_world["victim_idx"]

    victim_states = sorted(c for c in env_world["reachable"] if c != base)
    belief = {v: 1.0 / len(victim_states) for v in victim_states}

    planner = planners.SearchPhasePlanner(env_world, transition_model)

    uav_cell = base
    path = [uav_cell]
    history = []
    found = False

    try:
        with open("seeds_latest_last.txt", "r") as file:
            seeds = file.readlines()
    except FileNotFoundError:
        seeds = []
    seeds.append(f"{seed}\n")
    seeds = seeds[-50:]
    with open("seeds_latest_last.txt", "w") as file:
        file.writelines(seeds)

    print("=" * 64)
    print(" UAV DISASTER RESPONSE — VORONOI-GRID POMDP")
    print("=" * 64)
    print(f"\nRandom seed parameter : {seed}")
    print(f"Total cells           : {env_world['n_cells']} (reachable: {len(env_world['reachable'])})")
    print(f"Base cell index       : {base}")
    print(f"Target cell index     : {victim}")
    print(f"Obstacle arrays       : {sorted(env_world['obstacles'])}")
    print(f"Threat arrays         : {sorted(env_world['threats'])}")
    print(f"Search step limit     : {max_search_steps}")
    print(f"Planning horizon      : {planner.horizon}   Discount factor: {planner.gamma}\n")

    for step in range(1, max_search_steps + 1):
        action, action_values = planner.best_action(belief, uav_cell)
        old_cell = uav_cell

        if action == "SEARCH":
            observation = sample_observation(points, uav_cell, victim, action, rng)
            next_cell = uav_cell
        else:
            dist = models.transition_distribution(transition_model, uav_cell, action)
            next_cell = sample_from_distribution(dist, rng)
            observation = "NONE"

        belief = models.update_belief(points, belief, old_cell, action, observation)
        uav_cell = next_cell
        path.append(uav_cell)

        likely = most_likely_victim(belief)

        history.append({
            "phase": "SEARCH",
            "step": step,
            "position": uav_cell,
            "action": action,
            "observation": observation,
            "likely_victim": likely,
            "likely_probability": belief[likely],
            "entropy": entropy(belief),
            "expected_value": action_values[action],
            "belief": belief.copy(),
        })

        print(
            f"[SEARCH] Step {step:2d}: {old_cell:2d} -> {uav_cell:2d} | "
            f"Action = {action:6s} | Observation = {observation:9s} | "
            f"Target parameter = {likely:2d} ({belief[likely]:.3f}) | "
            f"Entropy metric = {entropy(belief):.3f} | Value = {action_values[action]:.3f}"
        )

        if action == "SEARCH" and observation == "FOUND" and uav_cell == victim:
            found = True
            print(f"\nTarget object identified at index: {victim}")
            break

    search_steps_used = len(history)
    if not found:
        print("\nSearch limit reached. Executing return sequence.")

    print("\n" + "-" * 64)
    print(" PHASE 2: RETURN SEQUENCE (MDP Value Iteration)")
    print("-" * 64)

    policy, V = planners.solve_return_policy(env_world, transition_model)

    returned = uav_cell == base
    if returned:
        print("UAV system is at base coordinates.")

    for step in range(1, max_return_steps + 1):
        if returned:
            break

        action = policy.get(uav_cell)
        if action is None:
            returned = True
            break

        old_cell = uav_cell
        dist = models.transition_distribution(transition_model, uav_cell, action)
        next_cell = sample_from_distribution(dist, rng)
        uav_cell = next_cell
        path.append(uav_cell)

        likely = most_likely_victim(belief)

        history.append({
            "phase": "RETURN",
            "step": search_steps_used + step,
            "position": uav_cell,
            "action": action,
            "observation": "NONE",
            "likely_victim": likely,
            "likely_probability": belief[likely],
            "entropy": entropy(belief),
            "expected_value": V.get(old_cell, 0.0),
            "belief": belief.copy(),
        })

        print(
            f"[RETURN] Step {step:2d}: {old_cell:2d} -> {uav_cell:2d} | "
            f"Action = {action:6s} | Value = {V.get(old_cell, 0.0):.3f}"
        )

        if uav_cell == base:
            returned = True

    print("\n" + "-" * 64)
    print(f"Target object identified : {int(found)}")
    print(f"Return sequence complete : {int(returned)}")
    print(f"Total steps executed     : {len(history)}")
    print(f"Search steps executed    : {search_steps_used}")
    print(f"Return steps executed    : {len(history) - search_steps_used}")

    return {
        "world": env_world,
        "model": transition_model,
        "path": path,
        "history": history,
        "final_belief": belief,
        "found": found,
        "returned": returned,
    }
