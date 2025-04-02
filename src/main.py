import numpy as np
from environment import env
from algorithms import run_a_star

def test_warehouse_env():
    # Create warehouse environment with rendering
    warehouse_env = env(grid_size=(34, 32), n_agents=8, n_humans=1, num_shelves=2048, num_pickup_points=3,
                        num_dropoff_points=2, render_mode="human")

    run_a_star(warehouse_env, n_steps=1000)


if __name__ == "__main__":
    test_warehouse_env()