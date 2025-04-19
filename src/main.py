import numpy as np
from environment import env
from algorithms import run_a_star, run_q_learning, train_DQN
import os

CURRENT_VERSION = "v1.2.1"

# Construct path to the model directory
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models')
model_name = CURRENT_VERSION + ".pth"
model_path = os.path.join(models_dir, model_name)


def eval_DQN():
    # Create warehouse environment with rendering
    warehouse_env = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=6, n_humans=10, num_shelves=2048, num_pickup_points=3,
                        num_dropoff_points=2, render_mode="human")

    # Run DQN agent
    run_q_learning(warehouse_env, n_steps=1000, model_path=model_path)


def train_DQL():
    # Create warehouse environments with rendering
    simplest_env = env(grid_size=(10, 10), n_agents=1, n_humans=0, num_shelves=0, num_pickup_points=1,
                        num_dropoff_points=1, render_mode="human")
    warehouse_env = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=6, n_humans=10, num_shelves=2048, num_pickup_points=3,
                        num_dropoff_points=4, render_mode="human")

    # Train DQN agent
    train_DQN(warehouse_env, n_episodes=300, max_steps=1000, save_every=100, model_path=model_path)

def test_a_star():
    # Create warehouse environment with rendering
    simplest_env = env(grid_size=(20, 20), n_agents=3, n_humans=4, num_shelves=5, num_pickup_points=4,
                        num_dropoff_points=4, render_mode="human")
    warehouse_env = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=6, n_humans=5, num_shelves=2048, num_pickup_points=4,
                        num_dropoff_points=8, render_mode="human")

    # run_a_star(simplest_env, n_steps=1000, debug_level=5)
    run_a_star(simplest_env, n_steps=1000, debug_level=5)

# Take first argument as task ("a_star" or "dqn_train" or "dqn_eval")
def main(task):
    if task == "a_star":
        test_a_star()
    elif task == "dqn_train":
        train_DQL()
    elif task == "dqn_eval":
        eval_DQN()
    else:
        raise ValueError("Invalid task. Choose 'a_star', 'dqn_train', or 'dqn_eval'.")
    

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <task>")
        print("Tasks: a_star, dqn_train, dqn_eval")
        sys.exit(1)
    
    task = sys.argv[1]
    main(task)