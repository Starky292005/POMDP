import math
from collections import deque
import numpy as np
import config

def circular_diff(a, b):
    d = (a - b) % (2 * math.pi)
    if d > math.pi:
        d = 2 * math.pi - d
    return d

def build_movement_model(points, adjacency, obstacles):
    """
    Computes the state transition matrix for all coordinates.
    Maps cardinal actions to valid neighbor trajectories.
    """
    model = {}
    n = len(points)

    for c in range(n):
        neighbors = list(adjacency[c])
        model[c] = {}

        if not neighbors:
            for action in config.CANONICAL_ANGLES:
                model[c][action] = {c: 1.0}
            continue

        angles = {}
        for nb in neighbors:
            dx = points[nb][0] - points[c][0]
            dy = points[nb][1] - points[c][1]
            angles[nb] = math.atan2(dy, dx)

        for action, canonical in config.CANONICAL_ANGLES.items():
            intended = min(neighbors, key=lambda x: circular_diff(angles[x], canonical))
            deviation_a = min(neighbors, key=lambda x: circular_diff(angles[x], canonical + math.pi / 2))
            deviation_b = min(neighbors, key=lambda x: circular_diff(angles[x], canonical - math.pi / 2))

            dist = {}

            def add(target, p, dist=dist, c=c):
                if target in obstacles:
                    target = c
                dist[target] = dist.get(target, 0.0) + p

            add(intended, 0.8)
            add(deviation_a, 0.1)
            add(deviation_b, 0.1)

            model[c][action] = dist

    return model

def transition_distribution(model, cell, action):
    if action == "SEARCH":
        return {cell: 1.0}
    return model[cell][action]

def movement_reachable(base_idx, model):
    """
    Executes a breadth-first search on the transition matrix to identify reachable nodes.
    """
    visited = {base_idx}
    dq = deque([base_idx])
    while dq:
        c = dq.popleft()
        outs = set()
        for action in config.MOVE_ACTIONS:
            for target in model[c][action]:
                if target != c:
                    outs.add(target)
        for t in outs:
            if t not in visited:
                visited.add(t)
                dq.append(t)
    return visited

def search_observation_distribution(points, uav_cell, victim_cell):
    d = float(np.linalg.norm(points[uav_cell] - points[victim_cell]))
    near = d <= config.SEARCH_RADIUS_DIST
    p_found = config.P_FOUND_NEAR if near else config.P_FALSE_ALARM
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
    """
    Computes the Bayesian update for the target state probability distribution.
    """
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

def move_reward(model, threats, uav_cell, action):
    """
    Calculates the expected reward value for a state transition.
    """
    occupancy = config.THREAT_OCCUPANCY_PENALTY if uav_cell in threats else 0.0
    dist = transition_distribution(model, uav_cell, action)
    p_blocked = dist.get(uav_cell, 0.0)
    p_threat_entry = sum(p for target, p in dist.items() if target in threats)
    return occupancy + config.MOVE_COST + p_blocked * config.BLOCKED_PENALTY + p_threat_entry * config.THREAT_ENTRY_PENALTY

def immediate_reward_search(points, threats, belief, uav_cell, action, model):
    if action == "SEARCH":
        occupancy = config.THREAT_OCCUPANCY_PENALTY if uav_cell in threats else 0.0
        p_found = observation_probability(points, belief, uav_cell, action, "FOUND")
        return occupancy + p_found * config.SEARCH_FOUND_REWARD + (1.0 - p_found) * config.SEARCH_NOT_FOUND_PENALTY
    return move_reward(model, threats, uav_cell, action)