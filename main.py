import simulation
import visualization

if __name__ == "__main__":
    result = simulation.run_simulation()

    visualization.save_pomdp_animation(
        result,
        filename="uav_pomdp_voronoi.gif",
        interval=1000,
    )