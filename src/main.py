import numpy as np
from environment import env
from algorithms import run_a_star, run_q_learning, train_DQN
import os

LOAD_MODEL = "v3.0.0-simple_env_multiple_agent-10x10,2agent,0human,0shelf,1pickup,1dropoff"
CURRENT_VERSION = "v4.0.0-simple_env_multiple_drop_pickup-10x10,2agent,0human,0shelf,4pickup,8dropoff"

# Construct path to the model directory
def get_path(name):
    if name is not None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, 'models')
        model_name = name + ".pth"
        model_path = os.path.join(models_dir, model_name)
        return model_path




def eval_DQN():
    # Create warehouse environment with rendering
    warehouse_env = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=6, n_humans=10, num_shelves=2048, num_pickup_points=3,
                        num_dropoff_points=2, render_mode="human")
    

    # Run DQN agent
    run_q_learning(warehouse_env, n_steps=1000, model_path=get_path(CURRENT_VERSION))


def train_DQL():
    # Create warehouse environments with rendering
    v1_simplest_env_10x10_1agent_0human_0shelves_1pickup_1dropoff = env(grid_size=(10, 10), n_agents=1, n_humans=0, num_shelves=0, num_pickup_points=1,
                        num_dropoff_points=1, render_mode="human")
    
    v2_simple_env_20x20_1agent_0human_0shelves_1pickup_1dropoff = env(grid_size=(20, 20), n_agents=1, n_humans=0, num_shelves=0, num_pickup_points=1,
                        num_dropoff_points=1, render_mode="human")
    
    v3_simple_env_multiple_agents_10x10_2agent_0human_0shelves_1pickup_1dropoff = env(grid_size=(10, 10), n_agents=2, n_humans=0, num_shelves=0, num_pickup_points=1,
                        num_dropoff_points=1, render_mode="human")
    
    v4_simple_env_multiple_drop_pickup_10x10_2agent_0human_0shelves_4_pickup_8dropoff = env(grid_size=(10, 10), n_agents=2, n_humans=0, num_shelves=0, num_pickup_points=4,
                        num_dropoff_points=8, render_mode="human")
    
    v5_simple_shelves_10x14_1agent_0human_32shelves_2pickup_4_dropoff = env(grid_size=(10, 14), n_agents=1, n_humans=0, num_shelves=32, num_pickup_points=2,
                        num_dropoff_points=4, render_mode="human")
    
    v6_warehouse_env = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=5, n_humans=10, num_shelves=2048, num_pickup_points=3,
                        num_dropoff_points=4, render_mode="human")
    
    # curriculum = [simplest_env, simple_env, simple_env_multiple_agents, simple_shelves, warehouse_env]
    curriculum = [v1_simplest_env_10x10_1agent_0human_0shelves_1pickup_1dropoff, 
                  v2_simple_env_20x20_1agent_0human_0shelves_1pickup_1dropoff, 
                  v3_simple_env_multiple_agents_10x10_2agent_0human_0shelves_1pickup_1dropoff, 
                  v4_simple_env_multiple_drop_pickup_10x10_2agent_0human_0shelves_4_pickup_8dropoff, 
                  v5_simple_shelves_10x14_1agent_0human_32shelves_2pickup_4_dropoff, 
                  v6_warehouse_env]
    curriculum_path = ["v1_simplest_env_10x10_1agent_0human_0shelves_1pickup_1dropoff", 
                       "v2_simple_env_20x20_1agent_0human_0shelves_1pickup_1dropoff", 
                       "v3_simple_env_multiple_agents_10x10_2agent_0human_0shelves_1pickup_1dropoff", 
                       "v4_simple_env_multiple_drop_pickup_10x10_2agent_0human_0shelves_4_pickup_8dropoff", 
                       "v5_simple_shelves_10x14_1agent_0human_32shelves_2pickup_4_dropoff", 
                       "v6_warehouse_env"]
    for env_instance in curriculum:
        current_v = curriculum_path[curriculum.index(env_instance)]
        previous_v = curriculum_path[curriculum.index(env_instance) - 1] if curriculum.index(env_instance) > 0 else curriculum_path[curriculum.index(env_instance)]
        train_DQN(env_instance, n_episodes=2, max_steps=1000, save_every=100, model_path=get_path(current_v), load_path=get_path(previous_v))
            
        

    # Train DQN agent
    # train_DQN(v4_simple_env_multiple_drop_pickup, n_episodes=1000, max_steps=1000, save_every=100, model_path=get_path(CURRENT_VERSION), load_path=get_path(LOAD_MODEL))

def test_a_star():
    # Create warehouse environment with rendering
    simplest_env = env(grid_size=(20, 20), n_agents=3, n_humans=4, num_shelves=5, num_pickup_points=4,
                        num_dropoff_points=4, render_mode="human")
    warehouse_env = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=6, n_humans=5, num_shelves=2048, num_pickup_points=4,
                        num_dropoff_points=8, render_mode="human")

    # run_a_star(simplest_env, n_steps=1000, debug_level=5)
    run_a_star(warehouse_env, n_steps=1000, debug_level=5)

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