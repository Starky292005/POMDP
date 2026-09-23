import math
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.collections import PolyCollection
from scipy.spatial import Voronoi


# ============================================================
# UAV DISASTER RESPONSE — VORONOI-GRID POMDP
#
# Changes from the original square-grid version:
#
#   1. Round trip     : after the search phase ends (victim found,
#                        or the search step budget runs out), the
#                        UAV is REQUIRED to fly back to BASE.
#   2. Dynamic hazards : obstacles (fully blocked) and threats
#                        (passable but risky) are placed randomly
#                        each run, anywhere reachable on the map.
#   3. Voronoi grid    : the map is an irregular Voronoi diagram
#                        built from random seed points, not a
#                        square grid. UAV movement follows the
#                        cell-adjacency graph instead of fixed
#                        N/S/E/W offsets.
#
# The planning logic keeps two distinct, purpose-fit techniques:
#   - PHASE 1 (search)  : belief-space POMDP lookahead, same
#                          V_h(b) = max_a[R(b,a)+gamma*sum_o P(o|b,a)V_(h-1)] as
#                          the original, now over Voronoi cells.
#   - PHASE 2 (return)  : once the victim's hidden-state search is
#                          over, there's nothing left to be
#                          uncertain about -- getting back to BASE
#                          is a fully observable MDP, solved
#                          exactly with value iteration rather than
#                          reusing belief-space lookahead.
# ============================================================


# -----------------------------
# Configuration
# -----------------------------

BOUNDS = (0.0, 10.0)

N_SEEDS = 22
OBSTACLE_FRACTION = 0.15
THREAT_FRACTION = 0.15
MIN_REACHABLE = 10

GAMMA = 0.95
PLANNING_HORIZON = 3

SEARCH_RADIUS_DIST = 2.0
P_FOUND_NEAR = 0.90
P_FALSE_ALARM = 0.02

MOVE_COST = -1.0
BLOCKED_PENALTY = -5.0
THREAT_OCCUPANCY_PENALTY = -3.0
THREAT_ENTRY_PENALTY = -5.0
SEARCH_FOUND_REWARD = 100.0
SEARCH_NOT_FOUND_PENALTY = -2.0

MAX_SEARCH_STEPS = 30
MAX_RETURN_STEPS = 40

CANONICAL_ANGLES = {
    "RIGHT": 0.0,
    "UP": math.pi / 2,
    "LEFT": math.pi,
    "DOWN": -math.pi / 2,
}
MOVE_ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
ACTIONS_SEARCH = ("UP", "DOWN", "LEFT", "RIGHT", "SEARCH")
ACTIONS_RETURN = MOVE_ACTIONS

ROLE_COLORS = {
    "free": "#e8e8e8",
    "obstacle": "#2b2b2b",
    "threat": "#f4a259",
}


# ============================================================
# Voronoi world construction
# ============================================================

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Standard recipe to reconstruct infinite Voronoi regions into
    finite polygons by extending unbounded ridges out to `radius`
    and closing them. Returns (regions, vertices): regions is a
    list (one per input point, in point order) of vertex-index
    lists; vertices is the array of all vertex coordinates
    (original + any newly added far points).
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def _line_intersect_x(a, b, x):
    t = (x - a[0]) / (b[0] - a[0])
    return (x, a[1] + t * (b[1] - a[1]))


def _line_intersect_y(a, b, y):
    t = (y - a[1]) / (b[1] - a[1])
    return (a[0] + t * (b[0] - a[0]), y)


def clip_polygon(poly, xmin, xmax, ymin, ymax):
    """Sutherland-Hodgman clip of a polygon against an axis-aligned box."""

    def clip_edge(points, inside_fn, intersect_fn):
        if not points:
            return []
        result = []
        prev = points[-1]
        prev_in = inside_fn(prev)
        for curr in points:
            curr_in = inside_fn(curr)
            if curr_in:
                if not prev_in:
                    result.append(intersect_fn(prev, curr))
                result.append(curr)
            elif prev_in:
                result.append(intersect_fn(prev, curr))
            prev, prev_in = curr, curr_in
        return result

    poly = clip_edge(poly, lambda p: p[0] >= xmin, lambda a, b: _line_intersect_x(a, b, xmin))
    poly = clip_edge(poly, lambda p: p[0] <= xmax, lambda a, b: _line_intersect_x(a, b, xmax))
    poly = clip_edge(poly, lambda p: p[1] >= ymin, lambda a, b: _line_intersect_y(a, b, ymin))
    poly = clip_edge(poly, lambda p: p[1] <= ymax, lambda a, b: _line_intersect_y(a, b, ymax))
    return poly


def _make_mirrored_points(real_points, xmin, xmax, ymin, ymax):
    """
    Reflects every real point across all 4 boundary walls (and the 4
    corners) before computing the Voronoi diagram. Without this, a
    point near an edge/corner has no real neighbors on the outside,
    so its cell balloons inward with far-flung neighbors -- exactly
    the "UAV teleports halfway across the map in one step" artifact.
    Mirroring gives every boundary cell a same-side "ghost" neighbor
    that properly closes it off, so cell sizes stay reasonably even
    everywhere, including corners (where BASE lives).
    """
    x, y = real_points[:, 0], real_points[:, 1]
    ghosts = [
        np.column_stack([2 * xmin - x, y]),
        np.column_stack([2 * xmax - x, y]),
        np.column_stack([x, 2 * ymin - y]),
        np.column_stack([x, 2 * ymax - y]),
        np.column_stack([2 * xmin - x, 2 * ymin - y]),
        np.column_stack([2 * xmin - x, 2 * ymax - y]),
        np.column_stack([2 * xmax - x, 2 * ymin - y]),
        np.column_stack([2 * xmax - x, 2 * ymax - y]),
    ]
    return np.vstack([real_points] + ghosts)


def _polygon_centroid(poly):
    """Area-weighted centroid of a (closed, non-self-intersecting) polygon."""
    poly = np.asarray(poly)
    if len(poly) < 3:
        return poly.mean(axis=0)
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area2 = cross.sum()
    if abs(area2) < 1e-9:
        return poly.mean(axis=0)
    cx = ((x + x1) * cross).sum() / (3 * area2)
    cy = ((y + y1) * cross).sum() / (3 * area2)
    return np.array([cx, cy])


def lloyd_relax_points(points, base_idx, iterations, xmin, xmax, ymin, ymax):
    """
    A few iterations of Lloyd's algorithm (move each seed point to its
    own cell's centroid, recompute, repeat). This is the standard way
    to turn an arbitrary random point set into a much more evenly
    spaced ("centroidal") Voronoi tessellation -- it directly shrinks
    the oversized, sparse-corner cells that cause huge single-step
    jumps. BASE is kept pinned at its fixed corner location so it
    stays the entry/exit point; only the other seed points relax.
    """
    pts = points.copy()
    for _ in range(iterations):
        mirrored = _make_mirrored_points(pts, xmin, xmax, ymin, ymax)
        vor = Voronoi(mirrored)
        regions, vertices = voronoi_finite_polygons_2d(vor)

        new_pts = pts.copy()
        for i in range(len(pts)):
            if i == base_idx:
                continue
            poly = [tuple(vertices[v]) for v in regions[i]]
            poly = clip_polygon(poly, xmin, xmax, ymin, ymax)
            if len(poly) >= 3:
                new_pts[i] = _polygon_centroid(poly)
        pts = new_pts
    return pts


def build_adjacency(vor, n_points):
    """
    n_points is the number of REAL points; any ridge touching a
    mirrored ghost point (index >= n_points) is ignored -- ghosts
    exist only to bound real cells geometrically, they are never
    real UAV-reachable cells.
    """
    adjacency = {i: set() for i in range(n_points)}
    for p1, p2 in vor.ridge_points:
        if p1 < n_points and p2 < n_points:
            adjacency[p1].add(int(p2))
            adjacency[p2].add(int(p1))
    return adjacency


def generate_points(rng, n_seeds):
    """Seed 0 is always pinned near a corner -> that cell becomes BASE."""
    base_point = np.array([[0.6, 0.6]])
    margin = 0.4
    others = rng.uniform(BOUNDS[0] + margin, BOUNDS[1] - margin, size=(n_seeds - 1, 2))
    return np.vstack([base_point, others])


# -----------------------------
# Movement model
# -----------------------------
#
# Voronoi cells don't have a fixed number of neighbors like a square
# grid does, so "UP/DOWN/LEFT/RIGHT" can't be fixed offsets. Instead,
# each action is mapped to a canonical compass angle; for a given
# cell, the neighbor whose direction is closest to that angle becomes
# the intended target (p=0.8), and the neighbors closest to the two
# perpendicular angles become the stochastic slip targets (p=0.1
# each) -- the same 80/10/10 structure as a standard grid's
# UP -> {UP:0.8, LEFT:0.1, RIGHT:0.1}, just driven by real neighbor
# geometry instead of fixed grid offsets. A target that is an
# obstacle is redirected to "stay in place", exactly like the
# original boundary/collision rule.

def circular_diff(a, b):
    d = (a - b) % (2 * math.pi)
    if d > math.pi:
        d = 2 * math.pi - d
    return d


def build_movement_model(points, adjacency, obstacles):
    """Returns model[cell][action] = {target_cell: probability}."""
    model = {}
    n = len(points)

    for c in range(n):
        neighbors = list(adjacency[c])
        model[c] = {}

        if not neighbors:
            for action in CANONICAL_ANGLES:
                model[c][action] = {c: 1.0}
            continue

        angles = {}
        for nb in neighbors:
            dx = points[nb][0] - points[c][0]
            dy = points[nb][1] - points[c][1]
            angles[nb] = math.atan2(dy, dx)

        for action, canonical in CANONICAL_ANGLES.items():
            intended = min(neighbors, key=lambda x: circular_diff(angles[x], canonical))
            slip_a = min(neighbors, key=lambda x: circular_diff(angles[x], canonical + math.pi / 2))
            slip_b = min(neighbors, key=lambda x: circular_diff(angles[x], canonical - math.pi / 2))

            dist = {}

            def add(target, p, dist=dist, c=c):
                if target in obstacles:
                    target = c
                dist[target] = dist.get(target, 0.0) + p

            add(intended, 0.8)
            add(slip_a, 0.1)
            add(slip_b, 0.1)

            model[c][action] = dist

    return model


def transition_distribution(model, cell, action):
    if action == "SEARCH":
        return {cell: 1.0}
    return model[cell][action]


def movement_reachable(base_idx, model):
    """
    BFS over the *actual* movement model (not raw Voronoi adjacency).
    Because the compass-angle action mapping doesn't necessarily use
    every graph neighbor, a cell can be graph-adjacent yet never
    actually reachable by any action -- so connectivity must be
    checked against the model, not the raw graph.
    """
    visited = {base_idx}
    dq = deque([base_idx])
    while dq:
        c = dq.popleft()
        outs = set()
        for action in MOVE_ACTIONS:
            for target in model[c][action]:
                if target != c:
                    outs.add(target)
        for t in outs:
            if t not in visited:
                visited.add(t)
                dq.append(t)
    return visited


def build_world(
    rng,
    n_seeds=N_SEEDS,
    obstacle_fraction=OBSTACLE_FRACTION,
    threat_fraction=THREAT_FRACTION,
    min_reachable=MIN_REACHABLE,
    max_tries=50,
):
    """
    Generates a fresh, randomized Voronoi world each call: cell
    polygons, the cell-adjacency graph, the movement model, and a
    dynamic, randomized placement of obstacles (fully blocked),
    threats (passable but risky) and the hidden victim cell -- all
    of which can land anywhere reachable on the map. Retries with a
    fresh layout if the random obstacles disconnect the map too much.
    """
    for attempt in range(max_tries):
        points = generate_points(rng, n_seeds)
        points = lloyd_relax_points(
            points, base_idx=0, iterations=4,
            xmin=BOUNDS[0], xmax=BOUNDS[1], ymin=BOUNDS[0], ymax=BOUNDS[1],
        )
        mirrored = _make_mirrored_points(points, BOUNDS[0], BOUNDS[1], BOUNDS[0], BOUNDS[1])
        vor = Voronoi(mirrored)
        regions, vertices = voronoi_finite_polygons_2d(vor)

        polygons = []
        for region in regions[:n_seeds]:
            poly_pts = [tuple(vertices[i]) for i in region]
            poly_pts = clip_polygon(poly_pts, BOUNDS[0], BOUNDS[1], BOUNDS[0], BOUNDS[1])
            polygons.append(poly_pts)

        adjacency = build_adjacency(vor, n_seeds)

        base_idx = 0
        non_base = np.array([i for i in range(n_seeds) if i != base_idx])
        perm = rng.permutation(non_base)

        n_obstacles = max(1, round(obstacle_fraction * n_seeds))
        n_threats = max(1, round(threat_fraction * n_seeds))

        obstacles = set(int(c) for c in perm[:n_obstacles])
        remaining = perm[n_obstacles:]
        threats = set(int(c) for c in remaining[:n_threats])

        model = build_movement_model(points, adjacency, obstacles)
        reachable = movement_reachable(base_idx, model)
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
                "model": model,
                "n_cells": n_seeds,
                "attempt": attempt,
            }

    raise RuntimeError(
        "Could not generate a sufficiently connected Voronoi world "
        "after max_tries attempts; try lowering obstacle_fraction."
    )


# ============================================================
# Belief tracking & observation model
# ============================================================

def search_observation_distribution(points, uav_cell, victim_cell):
    d = float(np.linalg.norm(points[uav_cell] - points[victim_cell]))
    near = d <= SEARCH_RADIUS_DIST
    p_found = P_FOUND_NEAR if near else P_FALSE_ALARM
    return {"FOUND": p_found, "NOT_FOUND": 1.0 - p_found}


def observation_probability(points, belief, uav_cell, action, observation):
    if action != "SEARCH":
        return 1.0 if observation == "NONE" else 0.0
    p = 0.0
    for victim, b in belief.items():
        z = search_observation_distribution(points, uav_cell, victim)[observation]
        p += b * z
    return p


def update_belief(points, belief, uav_cell, action, observation):
    """Bayesian update b'(v) = eta * Z(o|v,a) * b(v)."""
    if action != "SEARCH":
        return belief.copy()
    unnorm = {}
    for victim, b in belief.items():
        z = search_observation_distribution(points, uav_cell, victim)[observation]
        unnorm[victim] = b * z
    total = sum(unnorm.values())
    if total <= 0:
        return belief.copy()
    return {v: p / total for v, p in unnorm.items()}


# ============================================================
# Reward model
# ============================================================

def move_reward(model, threats, uav_cell, action):
    """
    Shared movement reward used by BOTH the search-phase lookahead and
    the return-phase value iteration, so a threat cell costs the same
    whichever phase the UAV is in.

      MOVE_COST               : per-step cost of moving.
      BLOCKED_PENALTY         : extra expected cost if the intended/slip
                                 target was an obstacle (redirected to
                                 staying in place), weighted by that
                                 probability.
      THREAT_ENTRY_PENALTY    : extra expected cost if the transition
                                 may land the UAV in a threat cell,
                                 weighted by that probability.
      THREAT_OCCUPANCY_PENALTY: flat cost for the step if the UAV is
                                 currently standing in a threat cell
                                 (regardless of action) -- danger zones
                                 are risky to linger in, not just to
                                 enter.
    """
    occupancy = THREAT_OCCUPANCY_PENALTY if uav_cell in threats else 0.0
    dist = transition_distribution(model, uav_cell, action)
    p_blocked = dist.get(uav_cell, 0.0)
    p_threat_entry = sum(p for target, p in dist.items() if target in threats)
    return occupancy + MOVE_COST + p_blocked * BLOCKED_PENALTY + p_threat_entry * THREAT_ENTRY_PENALTY


def immediate_reward_search(points, threats, belief, uav_cell, action, model):
    if action == "SEARCH":
        occupancy = THREAT_OCCUPANCY_PENALTY if uav_cell in threats else 0.0
        p_found = observation_probability(points, belief, uav_cell, action, "FOUND")
        return occupancy + p_found * SEARCH_FOUND_REWARD + (1.0 - p_found) * SEARCH_NOT_FOUND_PENALTY
    return move_reward(model, threats, uav_cell, action)


# ============================================================
# Phase 1: search-phase POMDP planner (finite-horizon lookahead)
# ============================================================
#
#   V_h(b) = max_a [ R(b,a) + gamma * sum_o P(o|b,a) V_(h-1)(b_a,o) ]

class SearchPhasePlanner:
    def __init__(self, world, model, horizon=PLANNING_HORIZON, gamma=GAMMA):
        self.points = world["points"]
        self.threats = world["threats"]
        self.model = model
        self.horizon = horizon
        self.gamma = gamma

    def action_value(self, belief, uav_cell, action, depth):
        reward = immediate_reward_search(
            self.points, self.threats, belief, uav_cell, action, self.model
        )

        if depth == 1:
            return reward

        total_future = 0.0
        observations = ("FOUND", "NOT_FOUND") if action == "SEARCH" else ("NONE",)

        for observation in observations:
            p_obs = observation_probability(self.points, belief, uav_cell, action, observation)
            if p_obs <= 1e-12:
                continue

            next_belief = update_belief(self.points, belief, uav_cell, action, observation)

            if action == "SEARCH":
                next_positions = {uav_cell: 1.0}
            else:
                next_positions = transition_distribution(self.model, uav_cell, action)

            future_for_obs = 0.0
            for next_cell, p_trans in next_positions.items():
                future_for_obs += p_trans * self.value(next_belief, next_cell, depth - 1)

            total_future += p_obs * future_for_obs

        return reward + self.gamma * total_future

    def value(self, belief, uav_cell, depth):
        return max(self.action_value(belief, uav_cell, a, depth) for a in ACTIONS_SEARCH)

    def best_action(self, belief, uav_cell):
        values = {a: self.action_value(belief, uav_cell, a, self.horizon) for a in ACTIONS_SEARCH}
        best = max(values, key=values.get)
        return best, values


# ============================================================
# Phase 2: return-phase MDP value iteration
# ============================================================
#
# Once the search phase ends (victim found, or the search step
# budget is exhausted), there is no more hidden state to track --
# the UAV just needs to get back to BASE as cheaply as possible
# under the same stochastic movement model and threat penalties.
# That's a fully observable MDP over the (small) set of reachable
# cells, so instead of belief-space lookahead we solve it exactly
# with value iteration.

def solve_return_policy(world, model, gamma=GAMMA, max_iters=500, tol=1e-6):
    base = world["base_idx"]
    threats = world["threats"]
    reachable = world["reachable"]

    V = {c: 0.0 for c in reachable}

    for _ in range(max_iters):
        delta = 0.0
        V_new = {}
        for c in reachable:
            if c == base:
                V_new[c] = 0.0
                continue
            best_q = -float("inf")
            for a in ACTIONS_RETURN:
                dist = transition_distribution(model, c, a)
                r = move_reward(model, threats, c, a)
                future = sum(p * V.get(c2, 0.0) for c2, p in dist.items())
                best_q = max(best_q, r + gamma * future)
            V_new[c] = best_q
            delta = max(delta, abs(V_new[c] - V[c]))
        V = V_new
        if delta < tol:
            break

    policy = {}
    for c in reachable:
        if c == base:
            policy[c] = None
            continue
        best_a, best_q = None, -float("inf")
        for a in ACTIONS_RETURN:
            dist = transition_distribution(model, c, a)
            r = move_reward(model, threats, c, a)
            future = sum(p * V.get(c2, 0.0) for c2, p in dist.items())
            q = r + gamma * future
            if q > best_q:
                best_q, best_a = q, a
        policy[c] = best_a

    return policy, V


# ============================================================
# Simulation
# ============================================================

def sample_from_distribution(distribution, rng):
    items = list(distribution.items())
    states = [item[0] for item in items]
    probabilities = [item[1] for item in items]
    index = rng.choice(len(states), p=probabilities)
    return states[index]


def sample_observation(points, uav_cell, victim_cell, action, rng):
    if action != "SEARCH":
        return "NONE"
    distribution = search_observation_distribution(points, uav_cell, victim_cell)
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
    n_seeds=N_SEEDS,
    obstacle_fraction=OBSTACLE_FRACTION,
    threat_fraction=THREAT_FRACTION,
    min_reachable=MIN_REACHABLE,
    max_search_steps=MAX_SEARCH_STEPS,
    max_return_steps=MAX_RETURN_STEPS,
):
    # seed=None (the default) draws a fresh random seed from OS entropy
    # each call, so obstacles/threats/victim placement differ every run.
    # Pass an explicit integer seed to reproduce one specific layout.
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**32))
    rng = np.random.default_rng(seed)

    world = build_world(
        rng, n_seeds=n_seeds, obstacle_fraction=obstacle_fraction,
        threat_fraction=threat_fraction, min_reachable=min_reachable,
    )
    model = world["model"]
    points = world["points"]
    base = world["base_idx"]
    victim = world["victim_idx"]

    victim_states = sorted(c for c in world["reachable"] if c != base)
    belief = {v: 1.0 / len(victim_states) for v in victim_states}

    planner = SearchPhasePlanner(world, model)

    uav_cell = base
    path = [uav_cell]
    history = []
    found = False

    print("=" * 64)
    print(" UAV DISASTER RESPONSE — VORONOI-GRID POMDP")
    print("=" * 64)
    print()
    print(f"Random seed used    : {seed}   (pass this to run_simulation(seed=...) to replay this exact layout)")
    print(f"Cells               : {world['n_cells']} (reachable: {len(world['reachable'])})")
    print(f"Base cell           : {base}")
    print(f"True victim cell    : {victim}  [hidden from UAV]")
    print(f"Obstacles           : {sorted(world['obstacles'])}")
    print(f"Threat cells        : {sorted(world['threats'])}")
    print(f"Search step budget  : {max_search_steps}")
    print(f"Planning horizon    : {planner.horizon}   Discount γ: {planner.gamma}")
    print()

    # ---------------- PHASE 1: SEARCH ----------------
    for step in range(1, max_search_steps + 1):
        action, action_values = planner.best_action(belief, uav_cell)
        old_cell = uav_cell

        if action == "SEARCH":
            observation = sample_observation(points, uav_cell, victim, action, rng)
            next_cell = uav_cell
        else:
            dist = transition_distribution(model, uav_cell, action)
            next_cell = sample_from_distribution(dist, rng)
            observation = "NONE"

        belief = update_belief(points, belief, old_cell, action, observation)
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
            f"Likely victim = {likely:2d} ({belief[likely]:.3f}) | "
            f"Entropy = {entropy(belief):.3f} | V = {action_values[action]:.3f}"
        )

        if action == "SEARCH" and observation == "FOUND":
            found = True
            print()
            print(">>> VICTIM DETECTED BY UAV <<<")
            print(f">>> True victim cell: {victim}")
            break

    search_steps_used = len(history)
    if not found:
        print()
        print(">>> Search budget exhausted without a FOUND observation. Returning to base. <<<")

    # ---------------- PHASE 2: RETURN TO BASE ----------------
    print()
    print("-" * 64)
    print(" PHASE 2: RETURN TO BASE (MDP value iteration)")
    print("-" * 64)

    policy, V = solve_return_policy(world, model)

    returned = uav_cell == base
    if returned:
        print("UAV was already at base when the search phase ended.")

    for step in range(1, max_return_steps + 1):
        if returned:
            break

        action = policy.get(uav_cell)
        if action is None:
            returned = True
            break

        old_cell = uav_cell
        dist = transition_distribution(model, uav_cell, action)
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
            f"Action = {action:6s} | V(cell) = {V.get(old_cell, 0.0):.3f}"
        )

        if uav_cell == base:
            returned = True

    print()
    print("-" * 64)
    print(f"Victim found           : {'YES' if found else 'NO'}")
    print(f"Returned to base        : {'YES' if returned else 'NO (step budget exhausted)'}")
    print(f"Total steps taken       : {len(history)}")
    print(f"  search steps          : {search_steps_used}")
    print(f"  return steps          : {len(history) - search_steps_used}")

    return {
        "world": world,
        "model": model,
        "path": path,
        "history": history,
        "final_belief": belief,
        "found": found,
        "returned": returned,
    }


# ============================================================
# Visualization
# ============================================================

def cell_face_colors(world):
    colors = []
    for c in range(world["n_cells"]):
        if c in world["obstacles"]:
            colors.append(ROLE_COLORS["obstacle"])
        elif c in world["threats"]:
            colors.append(ROLE_COLORS["threat"])
        else:
            colors.append(ROLE_COLORS["free"])
    return colors


def draw_static_world(ax, world):
    pc = PolyCollection(
        world["polygons"], facecolors=cell_face_colors(world),
        edgecolors="white", linewidths=0.7,
    )
    ax.add_collection(pc)
    ax.set_xlim(*BOUNDS)
    ax.set_ylim(*BOUNDS)
    ax.set_aspect("equal")
    return pc


def role_legend_handles():
    return [
        Patch(facecolor=ROLE_COLORS["free"], edgecolor="gray", label="Free cell"),
        Patch(facecolor=ROLE_COLORS["obstacle"], edgecolor="gray", label="Obstacle (blocked)"),
        Patch(facecolor=ROLE_COLORS["threat"], edgecolor="gray", label="Threat zone"),
    ]


def plot_results(result):
    """Show the final environment (with UAV round-trip path) and the final belief map."""
    world = result["world"]
    points = world["points"]
    path = result["path"]
    final_belief = result["final_belief"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    ax, ax2 = axes

    draw_static_world(ax, world)
    ax.scatter(*points[world["base_idx"]], marker="^", s=200, color="#2ca02c", zorder=5)
    ax.scatter(*points[world["victim_idx"]], marker="*", s=260, color="#d62728", zorder=5)

    xs = [points[c][0] for c in path]
    ys = [points[c][1] for c in path]
    ax.plot(xs, ys, color="#1f77b4", linewidth=2, marker="o", markersize=3, zorder=4)

    ax.legend(
        handles=role_legend_handles() + [
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c", markersize=10, label="Base"),
            plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#d62728", markersize=13, label="True victim"),
            plt.Line2D([0], [0], color="#1f77b4", linewidth=2, label="UAV path (round trip)"),
        ],
        loc="upper left", fontsize=8,
    )
    ax.set_title("UAV Disaster Environment (Voronoi grid)")

    belief_values = np.array([final_belief.get(c, 0.0) for c in range(world["n_cells"])])
    pc2 = PolyCollection(
        world["polygons"], array=belief_values, cmap="viridis",
        edgecolors="white", linewidths=0.5,
    )
    ax2.add_collection(pc2)
    ax2.set_xlim(*BOUNDS)
    ax2.set_ylim(*BOUNDS)
    ax2.set_aspect("equal")
    fig.colorbar(pc2, ax=ax2, label="Probability")
    ax2.scatter(*points[world["victim_idx"]], marker="*", s=180, color="white", edgecolor="black", zorder=5)
    ax2.set_title("Final Belief Map")

    plt.tight_layout()
    plt.show()


def _draw_animation_frames(fig, ax, ax2, world, history, path):
    """Shared frame-drawing logic for the live animation and the saved GIF."""
    points = world["points"]

    draw_static_world(ax, world)
    ax.scatter(*points[world["base_idx"]], marker="^", s=200, color="#2ca02c", zorder=5)
    ax.scatter(*points[world["victim_idx"]], marker="*", s=260, color="#d62728", zorder=5)
    ax.legend(
        handles=role_legend_handles() + [
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c", markersize=10, label="Base"),
            plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#d62728", markersize=13, label="True victim"),
        ],
        loc="upper left", fontsize=7,
    )

    path_line, = ax.plot([], [], color="#1f77b4", linewidth=2, zorder=4)
    uav_marker, = ax.plot([], [], marker="o", markersize=10, color="#1f77b4", linestyle="None", zorder=6)

    n_cells = world["n_cells"]
    pc2 = PolyCollection(
        world["polygons"], array=np.zeros(n_cells), cmap="viridis",
        edgecolors="white", linewidths=0.5,
    )
    ax2.add_collection(pc2)
    ax2.set_xlim(*BOUNDS)
    ax2.set_ylim(*BOUNDS)
    ax2.set_aspect("equal")
    fig.colorbar(pc2, ax=ax2, label="Probability")
    ax2.scatter(*points[world["victim_idx"]], marker="*", s=160, color="white", edgecolor="black", zorder=5)

    status = fig.suptitle("", fontsize=12)

    def update(frame):
        item = history[frame]
        cell = item["position"]

        current_path = path[: frame + 2]
        xs = [points[c][0] for c in current_path]
        ys = [points[c][1] for c in current_path]
        path_line.set_data(xs, ys)
        uav_marker.set_data([points[cell][0]], [points[cell][1]])

        ax.set_title(f"Real Environment — Step {item['step']}  [{item['phase']}]")

        belief = item["belief"]
        vals = np.array([belief.get(c, 0.0) for c in range(n_cells)])
        pc2.set_array(vals)
        # Rescale per frame so the map keeps showing contrast as belief
        # concentrates, instead of saturating against a fixed vmax.
        pc2.set_clim(vmin=0, vmax=max(vals.max(), 1e-9))

        ax2.set_title(
            "Belief Map\n"
            f"Most likely = cell {item['likely_victim']} (P={item['likely_probability']:.3f})"
        )

        status.set_text(
            f"Step {item['step']} [{item['phase']}]  |  Action: {item['action']}  |  "
            f"Observation: {item['observation']}  |  Entropy: {item['entropy']:.3f}  |  "
            f"V: {item['expected_value']:.3f}"
        )

        return path_line, uav_marker, pc2, status

    return update


def animate_pomdp(result, interval=700):
    """Live animation of the search + return round trip."""
    from matplotlib.animation import FuncAnimation

    world = result["world"]
    history = result["history"]
    path = result["path"]

    if not history:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    ax, ax2 = axes

    update = _draw_animation_frames(fig, ax, ax2, world, history, path)

    animation = FuncAnimation(
        fig, update, frames=len(history), interval=interval, repeat=False, blit=False,
    )

    plt.tight_layout()
    fig._pomdp_animation = animation
    plt.show()


def save_pomdp_animation(result, filename="uav_pomdp_voronoi.gif", interval=1000):
    """Save the search + return round trip as a GIF."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    world = result["world"]
    history = result["history"]
    path = result["path"]

    if not history:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    ax, ax2 = axes

    update = _draw_animation_frames(fig, ax, ax2, world, history, path)

    animation = FuncAnimation(
        fig, update, frames=len(history), interval=interval, repeat=False, blit=False,
    )

    animation.save(filename, writer=PillowWriter(fps=1000 / interval))
    plt.close(fig)

    print()
    print("=" * 60)
    print(f"POMDP animation saved to: {filename}")
    print(f"Frames / decision steps : {len(history)}")
    print(f"Time per step           : {interval / 1000:.1f} seconds")
    print("=" * 60)


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    result = run_simulation()  # seed=None -> a fresh random world every run

    save_pomdp_animation(
        result,
        filename="uav_pomdp_voronoi.gif",
        interval=1000,
    )