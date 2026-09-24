import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation, PillowWriter

import config

def cell_face_colors(world):
    colors = []
    for c in range(world["n_cells"]):
        if c in world["obstacles"]:
            colors.append(config.ROLE_COLORS["obstacle"])
        elif c in world["threats"]:
            colors.append(config.ROLE_COLORS["threat"])
        else:
            colors.append(config.ROLE_COLORS["free"])
    return colors


def draw_static_world(ax, world):
    pc = PolyCollection(
        world["polygons"], facecolors=cell_face_colors(world),
        edgecolors="white", linewidths=0.7,
    )
    ax.add_collection(pc)
    ax.set_xlim(*config.BOUNDS)
    ax.set_ylim(*config.BOUNDS)
    ax.set_aspect("equal")
    return pc


def role_legend_handles():
    return [
        Patch(facecolor=config.ROLE_COLORS["free"], edgecolor="gray", label="Free cell"),
        Patch(facecolor=config.ROLE_COLORS["obstacle"], edgecolor="gray", label="Obstacle parameter"),
        Patch(facecolor=config.ROLE_COLORS["threat"], edgecolor="gray", label="Threat parameter"),
    ]


def plot_results(result):
    """
    Renders the terminal environment state and the terminal probability matrix.
    """
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
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c", markersize=10, label="Base coordinate"),
            plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#d62728", markersize=13, label="Target coordinate"),
            plt.Line2D([0], [0], color="#1f77b4", linewidth=2, label="UAV trajectory"),
        ],
        loc="upper left", fontsize=8,
    )
    ax.set_title("UAV Environment Topology")

    belief_values = np.array([final_belief.get(c, 0.0) for c in range(world["n_cells"])])
    pc2 = PolyCollection(
        world["polygons"], array=belief_values, cmap="viridis",
        edgecolors="white", linewidths=0.5,
    )
    ax2.add_collection(pc2)
    ax2.set_xlim(*config.BOUNDS)
    ax2.set_ylim(*config.BOUNDS)
    ax2.set_aspect("equal")
    fig.colorbar(pc2, ax=ax2, label="Probability Density")
    ax2.scatter(*points[world["victim_idx"]], marker="*", s=180, color="white", edgecolor="black", zorder=5)
    ax2.set_title("Terminal Probability Matrix")

    plt.tight_layout()
    plt.show()


def _draw_animation_frames(fig, ax, ax2, world, history, path):
    points = world["points"]

    draw_static_world(ax, world)
    ax.scatter(*points[world["base_idx"]], marker="^", s=200, color="#2ca02c", zorder=5)
    ax.scatter(*points[world["victim_idx"]], marker="*", s=260, color="#d62728", zorder=5)
    ax.legend(
        handles=role_legend_handles() + [
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c", markersize=10, label="Base coordinate"),
            plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#d62728", markersize=13, label="Target coordinate"),
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
    ax2.set_xlim(*config.BOUNDS)
    ax2.set_ylim(*config.BOUNDS)
    ax2.set_aspect("equal")
    fig.colorbar(pc2, ax=ax2, label="Probability Density")
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

        ax.set_title(f"Environment State — Step {item['step']}  [{item['phase']}]")

        belief = item["belief"]
        vals = np.array([belief.get(c, 0.0) for c in range(n_cells)])
        pc2.set_array(vals)
        pc2.set_clim(vmin=0, vmax=max(vals.max(), 1e-9))

        ax2.set_title(
            "Probability Matrix\n"
            f"Maximum probability parameter = {item['likely_victim']} (P={item['likely_probability']:.3f})"
        )

        status.set_text(
            f"Step {item['step']} [{item['phase']}]  |  Action: {item['action']}  |  "
            f"Observation: {item['observation']}  |  Entropy: {item['entropy']:.3f}  |  "
            f"Value: {item['expected_value']:.3f}"
        )

        return path_line, uav_marker, pc2, status

    return update


def animate_pomdp(result, interval=700):
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

    print(f"\nPOMDP output saved to     : {filename}")
    print(f"Total frame count         : {len(history)}")
    print(f"Time allocation per frame : {interval / 1000:.1f} seconds")