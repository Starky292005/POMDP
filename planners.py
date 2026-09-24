import config
import models

class SearchPhasePlanner:
    """
    Executes a finite-horizon lookahead in belief space to identify the optimal action.
    """
    def __init__(self, world, transition_model, horizon=config.PLANNING_HORIZON, gamma=config.GAMMA):
        self.points = world["points"]
        self.threats = world["threats"]
        self.transition_model = transition_model
        self.horizon = horizon
        self.gamma = gamma

    def action_value(self, belief, uav_cell, action, depth):
        reward = models.immediate_reward_search(
            self.points, self.threats, belief, uav_cell, action, self.transition_model
        )

        if depth == 1:
            return reward

        total_future = 0.0
        observations = ("FOUND", "NOT_FOUND") if action == "SEARCH" else ("NONE",)

        for observation in observations:
            p_obs = models.observation_probability(self.points, belief, uav_cell, action, observation)
            if p_obs <= 1e-12:
                continue

            next_belief = models.update_belief(self.points, belief, uav_cell, action, observation)

            if action == "SEARCH":
                next_positions = {uav_cell: 1.0}
            else:
                next_positions = models.transition_distribution(self.transition_model, uav_cell, action)

            future_for_obs = 0.0
            for next_cell, p_trans in next_positions.items():
                future_for_obs += p_trans * self.value(next_belief, next_cell, depth - 1)

            total_future += p_obs * future_for_obs

        return reward + self.gamma * total_future

    def value(self, belief, uav_cell, depth):
        return max(self.action_value(belief, uav_cell, a, depth) for a in config.ACTIONS_SEARCH)

    def best_action(self, belief, uav_cell):
        values = {a: self.action_value(belief, uav_cell, a, self.horizon) for a in config.ACTIONS_SEARCH}
        best = max(values, key=values.get)
        return best, values


def solve_return_policy(world, transition_model, gamma=config.GAMMA, max_iters=500, tol=1e-6):
    """
    Executes value iteration to compute the optimal deterministic policy for the return phase.
    """
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
            for a in config.ACTIONS_RETURN:
                dist = models.transition_distribution(transition_model, c, a)
                r = models.move_reward(transition_model, threats, c, a)
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
        for a in config.ACTIONS_RETURN:
            dist = models.transition_distribution(transition_model, c, a)
            r = models.move_reward(transition_model, threats, c, a)
            future = sum(p * V.get(c2, 0.0) for c2, p in dist.items())
            q = r + gamma * future
            if q > best_q:
                best_q, best_a = q, a
            policy[c] = best_a

    return policy, V