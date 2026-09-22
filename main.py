import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter

# --- 1. SIMULATION PARAMETERS ---
BOUNDS = 50.0
NUM_PARTICLES = 300
DT = 1.0
WIND_NOISE_STD = 0.4
GPS_NOISE_STD = 1.5
SENSOR_RADIUS = 8.0
UAV_SPEED = 2.0


# --- 2. CORE CLASSES ---
class UAV_Environment:
    def __init__(self, start_pos):
        self.true_pos = np.array(start_pos, dtype=float)

    def move(self, velocity):
        wind = np.random.normal(0, WIND_NOISE_STD, 2)
        self.true_pos += (velocity * DT) + wind
        return self.true_pos

    def get_gps_reading(self):
        return self.true_pos + np.random.normal(0, GPS_NOISE_STD, 2)

    def check_sensor_for_threats(self, hidden_threat):
        return np.linalg.norm(self.true_pos - hidden_threat) < SENSOR_RADIUS


class ParticleFilter:
    def __init__(self, start_pos, num_particles):
        self.particles = np.tile(start_pos, (num_particles, 1)) + np.random.normal(0, 0.5, (num_particles, 2))
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, velocity):
        self.particles += (velocity * DT) + np.random.normal(0, WIND_NOISE_STD, self.particles.shape)

    def update(self, gps_reading):
        distances = np.linalg.norm(self.particles - gps_reading, axis=1)
        self.weights = np.exp(- (distances ** 2) / (2 * GPS_NOISE_STD ** 2)) + 1e-300
        self.weights /= sum(self.weights)

    def resample(self):
        indices = np.random.choice(range(len(self.particles)), size=len(self.particles), p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(len(self.particles)) / len(self.particles)

    def get_estimated_position(self):
        return np.average(self.particles, axis=0)


# [NEW] Helper function to just build the graph
def build_voronoi_graph(obstacles):
    vor = Voronoi(obstacles)
    G = nx.Graph()
    for ridge in vor.ridge_vertices:
        if ridge[0] != -1 and ridge[1] != -1:
            p1, p2 = vor.vertices[ridge[0]], vor.vertices[ridge[1]]
            if (0 <= p1[0] <= BOUNDS and 0 <= p1[1] <= BOUNDS and 0 <= p2[0] <= BOUNDS and 0 <= p2[1] <= BOUNDS):
                G.add_edge(tuple(p1), tuple(p2), weight=np.linalg.norm(p1 - p2))
    return G


# [NEW] Replaces the point-to-point logic with a Greedy TSP Coverage Planner
def get_greedy_coverage_path(G, start_pos, unvisited_targets, base_pos):
    if len(G.nodes) == 0: return [start_pos, base_pos]

    nodes = list(G.nodes)
    current_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - start_pos, axis=1))])

    # Snap unvisited targets to the nearest valid graph nodes
    target_nodes = set()
    for t in unvisited_targets:
        target_nodes.add(tuple(nodes[np.argmin(np.linalg.norm(nodes - np.array(t), axis=1))]))

    full_path = [np.array(current_node)]

    # Greedy TSP: Find the closest unvisited node, go to it, cross it off, repeat.
    while target_nodes:
        best_target = None
        best_dist = float('inf')

        for t in target_nodes:
            try:
                dist = nx.shortest_path_length(G, source=current_node, target=t, weight='weight')
                if dist < best_dist:
                    best_dist = dist
                    best_target = t
            except nx.NetworkXNoPath:
                continue

        if best_target is None:
            break  # No reachable targets left

        subpath = nx.shortest_path(G, source=current_node, target=best_target, weight='weight')
        full_path.extend([np.array(p) for p in subpath[1:]])
        current_node = best_target
        target_nodes.remove(best_target)

    # [NEW] Final requirement: Return to the starting point
    base_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - base_pos, axis=1))])
    try:
        return_path = nx.shortest_path(G, source=current_node, target=base_node, weight='weight')
        full_path.extend([np.array(p) for p in return_path[1:]])
    except nx.NetworkXNoPath:
        pass 

    full_path.append(np.array(base_pos))

    return full_path


# --- 3. ANIMATION FRAME CONTROLLER ---
class SimulationContainer:
    def __init__(self):
        self.start = np.array([5.0, 5.0])
        # The original obstacles
        core_obstacles = [[15, 15], [20, 35], [35, 20], [10, 40], [40, 10], [25, 25]]

        # Add a "fence" of obstacles just outside the bounds to trap the Voronoi lines
        fence_obstacles = [
            [-10, -10], [-10, 25], [-10, 60],
            [60, -10], [60, 25], [60, 60],
            [25, -10], [25, 60]
        ]

        # Combine them
        self.known_obstacles = np.array(core_obstacles + fence_obstacles)
        self.hidden_threat = np.array([30.0, 30.0])
        self.threat_discovered = False

        self.env = UAV_Environment(self.start)
        self.pf = ParticleFilter(self.start, NUM_PARTICLES)

        # [NEW] Setup the initial grid and track unvisited nodes
        G = build_voronoi_graph(self.known_obstacles)
        self.unvisited_nodes = [np.array(n) for n in G.nodes]
        self.current_path = get_greedy_coverage_path(G, self.start, self.unvisited_nodes, self.start)

        self.current_waypoint_idx = 0
        self.mission_complete = False

    def update_frame(self, frame_idx, ax_real, ax_belief):
        if self.mission_complete:
            return

        # 1. PLAN
        estimated_pos = self.pf.get_estimated_position()
        target_waypoint = self.current_path[self.current_waypoint_idx]

        # [NEW] Check off visited nodes from our checklist memory
        for i in range(len(self.unvisited_nodes) - 1, -1, -1):
            if np.linalg.norm(estimated_pos - self.unvisited_nodes[i]) < 2.0:
                self.unvisited_nodes.pop(i)

        # Advance waypoint logic
        if np.linalg.norm(estimated_pos - target_waypoint) < 2.0:
            if self.current_waypoint_idx < len(self.current_path) - 1:
                self.current_waypoint_idx += 1
                target_waypoint = self.current_path[self.current_waypoint_idx]
            else:
                self.mission_complete = True
                print("Grid fully covered. UAV has landed back at base!")
                return

        direction = target_waypoint - estimated_pos
        # Prevent division by zero if we overshoot exactly
        dist = np.linalg.norm(direction)
        if dist > 0.1:
            velocity = (direction / dist) * UAV_SPEED
        else:
            velocity = np.array([0.0, 0.0])

        # 2. ACT & SENSE
        true_pos = self.env.move(velocity)
        gps_reading = self.env.get_gps_reading()

        # [NEW] Dynamic Replanning if threat discovered mid-mission
        if not self.threat_discovered and self.env.check_sensor_for_threats(self.hidden_threat):
            self.threat_discovered = True
            print("Threat Discovered! Replanning coverage route...")
            self.known_obstacles = np.vstack([self.known_obstacles, self.hidden_threat])

            # Rebuild graph and calculate a NEW coverage path for the remaining unvisited nodes
            G = build_voronoi_graph(self.known_obstacles)
            self.current_path = get_greedy_coverage_path(G, estimated_pos, self.unvisited_nodes, self.start)
            self.current_waypoint_idx = 0

        # 3. UPDATE POMDP BELIEF
        self.pf.predict(velocity)
        self.pf.update(gps_reading)
        self.pf.resample()

        # 4. RENDER PANELS
        ax_real.clear();
        ax_belief.clear()

        # Panel 1: Reality
        ax_real.set_title(f"Reality View | Step {frame_idx}")
        ax_real.set_xlim(0, BOUNDS);
        ax_real.set_ylim(0, BOUNDS)
        ax_real.scatter(self.known_obstacles[:, 0], self.known_obstacles[:, 1], c='black', s=80, label='Obstacles')
        ax_real.scatter(self.hidden_threat[0], self.hidden_threat[1], c='red', s=100, label='Hidden Threat')
        ax_real.scatter(true_pos[0], true_pos[1], c='blue', s=60, label='True UAV')
        ax_real.scatter(self.start[0], self.start[1], c='green', s=150, marker='s', label='Base')
        ax_real.add_patch(plt.Circle(true_pos, SENSOR_RADIUS, color='blue', fill=False, linestyle=':'))
        ax_real.legend(loc='upper left')

        # Panel 2: UAV Brain
        ax_belief.set_title("UAV Internal Belief & Coverage Plan")
        ax_belief.set_xlim(0, BOUNDS);
        ax_belief.set_ylim(0, BOUNDS)

        # Plot Voronoi Lines
        vor_visual = Voronoi(self.known_obstacles)
        for ridge in vor_visual.ridge_vertices:
            if ridge[0] != -1 and ridge[1] != -1:
                p1 = vor_visual.vertices[ridge[0]]
                p2 = vor_visual.vertices[ridge[1]]
                if (0 <= p1[0] <= BOUNDS and 0 <= p1[1] <= BOUNDS and 0 <= p2[0] <= BOUNDS and 0 <= p2[1] <= BOUNDS):
                    ax_belief.plot([p1[0], p2[0]], [p1[1], p2[1]], color='teal', linestyle='--', linewidth=1.5,
                                   alpha=0.8)

        # Plot elements
        ax_belief.scatter(self.known_obstacles[:, 0], self.known_obstacles[:, 1], c='black', s=80)
        if self.threat_discovered:
            ax_belief.scatter(self.hidden_threat[0], self.hidden_threat[1], c='red', marker='X', s=150,
                              label='Discovered!')
        ax_belief.scatter(self.pf.particles[:, 0], self.pf.particles[:, 1], color='gray', s=4, alpha=0.4,
                          label='Belief Cloud')
        ax_belief.scatter(estimated_pos[0], estimated_pos[1], color='purple', marker='+', s=100, label='Est. Pos')
        ax_belief.scatter(self.start[0], self.start[1], c='green', s=150, marker='s')

        # [NEW] Visually show which nodes still need to be visited
        if len(self.unvisited_nodes) > 0:
            unv = np.array(self.unvisited_nodes)
            ax_belief.scatter(unv[:, 0], unv[:, 1], c='orange', s=50, zorder=5, label='Unvisited')

        # Plot Path
        path_array = np.array(self.current_path)
        if len(path_array) > 0:
            ax_belief.plot(path_array[self.current_waypoint_idx:, 0], path_array[self.current_waypoint_idx:, 1], 'g-',
                           linewidth=2, label='Current Route')

        ax_belief.legend(loc='upper left')


def main():
    fig, (ax_real, ax_belief) = plt.subplots(1, 2, figsize=(14, 6))
    sim = SimulationContainer()

    print("Recording simulation frames...")
    # [NEW] Increased frames to ensure the drone has enough time to complete the longer coverage mission
    anim = FuncAnimation(
        fig, sim.update_frame, frames=500,
        fargs=(ax_real, ax_belief), repeat=False
    )

    output_filename = "uav_coverage_voronoi.gif"
    anim.save(output_filename, writer=PillowWriter(fps=5))
    print(f"Success! Saved rendering dashboard directly to '{output_filename}'")


if __name__ == "__main__":
    main()