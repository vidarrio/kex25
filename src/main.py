import numpy as np
from environment import env
from algorithms import run_a_star, run_q_learning, train_DQN

CURRENT_VERSION = "v1.0.0"

def eval_DQN():
    # Create warehouse environment with rendering
    warehouse_env = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=6, n_humans=10, num_shelves=2048, num_pickup_points=3,
                        num_dropoff_points=2, render_mode="human")

    # Run DQN agent
    run_q_learning(warehouse_env, n_steps=1000, model_path="models/dqn_v1.pth")


def train_DQL():
    # Create warehouse environment with rendering
    warehouse_env = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=6, n_humans=10, num_shelves=2048, num_pickup_points=3,
                        num_dropoff_points=2, render_mode="human")

    # Train DQN agent
    train_DQN(warehouse_env, n_episodes=1000, max_steps=1000, save_every=100, model_path="models/dqn_v1.pth")

def test_a_star():
    # Create warehouse environment with rendering
    warehouse_env = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=6, n_humans=10, num_shelves=2048, num_pickup_points=3,
                        num_dropoff_points=2, render_mode="human")

    run_a_star(warehouse_env, n_steps=1000)

if __name__ == "__main__":
    # test_a_star()

    train_DQL()

    # eval_DQN()