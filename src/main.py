import numpy as np
from environment import env
from algorithms import run_a_star, run_q_learning, train_DQN
import os
import matplotlib.pyplot as plt
from algorithms.PPO import train_PPO, evaluate_PPO

# PPO curriculum training function
def train_PPO_curriculum(max_steps=1000):
    """
    Train PPO across a sequence of increasingly difficult environments.
    """
    stages = [
        (
            "stage1_simple",
            dict(
                grid_size=(10, 10), human_grid_size=(10, 10),
                n_agents=1, n_humans=0, num_shelves=0,
                num_pickup_points=1, num_dropoff_points=8,
                render_mode=None
            ),
            1500
        ),
        (
            "stage2_shelves8",
            dict(
                grid_size=(10, 10), human_grid_size=(10, 10),
                n_agents=1, n_humans=0, num_shelves=8,
                num_pickup_points=1, num_dropoff_points=8,
                render_mode=None
            ),
            5000
        ),
        
        (
            "stage3_shelves16",
            dict(
                grid_size=(10, 10), human_grid_size=(10, 10),
                n_agents=1, n_humans=0, num_shelves=16,
                num_pickup_points=1, num_dropoff_points=8,
                render_mode=None
            ),
            5000
        ),
        (
            "stage4_shelves32",
            dict(
                grid_size=(10, 10), human_grid_size=(10, 10),
                n_agents=1, n_humans=0, num_shelves=32,
                num_pickup_points=1, num_dropoff_points=8,
                render_mode=None
            ),
            5000
        ),
        (
            "stage5_shelves64",
            dict(
                grid_size=(10, 10), human_grid_size=(10, 10),
                n_agents=1, n_humans=0, num_shelves=64,
                num_pickup_points=1, num_dropoff_points=8,
                render_mode=None
            ),
            5000
        ),
        (
            "stage6_humans",
            dict(
                grid_size=(10, 10), human_grid_size=(10, 10),
                n_agents=1, n_humans=3, num_shelves=0,
                num_pickup_points=1, num_dropoff_points=8,
                render_mode=None
            ),
            3000
        ),
        (
            "stage7_robots",
            dict(
                grid_size=(10, 10), human_grid_size=(10, 10),
                n_agents=3, n_humans=0, num_shelves=0,
                num_pickup_points=1, num_dropoff_points=8,
                render_mode=None
            ),
            3000
        ),
        (
            "stage8_full",
            dict(
                grid_size=(10, 10), human_grid_size=(10, 10),
                n_agents=1, n_humans=3, num_shelves=64,
                num_pickup_points=1, num_dropoff_points=8,
                render_mode=None
            ),
            5000
        ),
        (
            "stage9_advanced",
            dict(
                grid_size=(34, 32), human_grid_size=(34, 32),
                n_agents=5, n_humans=5, num_shelves=2048,
                num_pickup_points=1, num_dropoff_points=8,
                render_mode=None
            ),
            5000
        ),
    ]

    for name, env_kwargs, episodes in stages:
        print(f"\n=== PPO Curriculum Stage: {name} ({episodes} episodes) ===")
        env_instance = env(**env_kwargs)
        train_PPO(env_instance, n_episodes=episodes, max_steps=max_steps, save_prefix=name)
    print("\n=== PPO Curriculum Complete ===")


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
    
    v1_simplest_env_10x10_1agent_0human_0shelves_1pickup_1dropoff = env(grid_size=(10, 10), human_grid_size=(10, 10), n_agents=1, n_humans=0, num_shelves=0, num_pickup_points=1,
                        num_dropoff_points=4, render_mode="human")
    
    run_q_learning(v1_simplest_env_10x10_1agent_0human_0shelves_1pickup_1dropoff, n_steps=1000, model_path=get_path("v1_simplest_env_10x10_1agent_0human_0shelves_1pickup_1dropoff"))

    

    # Run DQN agent


def train_DQL_curriculum():
    # Create warehouse environments with rendering
    v1_simplest_env_10x10_1agent_0human_0shelves_1pickup_1dropoff = env(grid_size=(10, 10), human_grid_size=(10, 10), n_agents=1, n_humans=0, num_shelves=0, num_pickup_points=1,
                        num_dropoff_points=1, render_mode="human")
    
    v2_simple_env_20x20_1agent_0human_0shelves_1pickup_1dropoff = env(grid_size=(20, 20), human_grid_size=(20, 20), n_agents=1, n_humans=0, num_shelves=0, num_pickup_points=1,
                        num_dropoff_points=1, render_mode="human")
    
    v3_simple_env_multiple_agents_10x10_3agent_0human_0shelves_1pickup_1dropoff = env(grid_size=(10, 10), human_grid_size=(10, 10), n_agents=3, n_humans=0, num_shelves=0, num_pickup_points=1,
                        num_dropoff_points=1, render_mode="human")
    
    v4_simple_env_multiple_drop_pickup_10x10_2agent_0human_0shelves_4_pickup_8dropoff = env(grid_size=(10, 10), human_grid_size=(10, 10), n_agents=2, n_humans=0, num_shelves=0, num_pickup_points=4,
                        num_dropoff_points=8, render_mode="human")
    
    v5_simple_shelves_10x14_1agent_0human_32shelves_2pickup_4_dropoff = env(grid_size=(10, 14), human_grid_size=(10, 14), n_agents=1, n_humans=0, num_shelves=32, num_pickup_points=2,
                        num_dropoff_points=4, render_mode="human")
    
    v6_simple_human_10x10_1agent_5human_0shelves_2pickup_4_dropoff = env(grid_size=(10, 10), human_grid_size=(10, 10), n_agents=1, n_humans=5, num_shelves=0, num_pickup_points=2,
                        num_dropoff_points=4, render_mode="human")
    
    v7_simple_human_and_shelves_10x14_1agent_5human_32shelves_2pickup_4_dropoff = env(grid_size=(10, 14), human_grid_size=(10, 14), n_agents=1, n_humans=5, num_shelves=32, num_pickup_points=2,
                        num_dropoff_points=4, render_mode="human")
    
    v8_warehouse_env = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=5, n_humans=10, num_shelves=2048, num_pickup_points=3,
                        num_dropoff_points=4, render_mode="human")
    
    # curriculum = [simplest_env, simple_env, simple_env_multiple_agents, simple_shelves, warehouse_env]
    curriculum = [v1_simplest_env_10x10_1agent_0human_0shelves_1pickup_1dropoff, 
                  v2_simple_env_20x20_1agent_0human_0shelves_1pickup_1dropoff, 
                  v3_simple_env_multiple_agents_10x10_3agent_0human_0shelves_1pickup_1dropoff, 
                  v4_simple_env_multiple_drop_pickup_10x10_2agent_0human_0shelves_4_pickup_8dropoff, 
                  v5_simple_shelves_10x14_1agent_0human_32shelves_2pickup_4_dropoff,
                  v6_simple_human_10x10_1agent_5human_0shelves_2pickup_4_dropoff,
                  v7_simple_human_and_shelves_10x14_1agent_5human_32shelves_2pickup_4_dropoff,
                  v8_warehouse_env]
    curriculum_path = ["v1_simplest_env_10x10_1agent_0human_0shelves_1pickup_1dropoff", 
                       "v2_simple_env_20x20_1agent_0human_0shelves_1pickup_1dropoff", 
                       "v3_simple_env_multiple_agents_10x10_3agent_0human_0shelves_1pickup_1dropoff", 
                       "v4_simple_env_multiple_drop_pickup_10x10_2agent_0human_0shelves_4_pickup_8dropoff", 
                       "v5_simple_shelves_10x14_1agent_0human_32shelves_2pickup_4_dropoff",
                       "v6_simple_human_10x10_1agent_5human_0shelves_2pickup_4_dropoff",
                       "v7_simple_human_and_shelves_10x14_1agent_5human_32shelves_2pickup_4_dropoff",
                       "v8_warehouse_env"]
    for env_instance in curriculum:
        current_v = curriculum_path[curriculum.index(env_instance)]
        previous_v = curriculum_path[curriculum.index(env_instance) - 1] if curriculum.index(env_instance) > 0 else curriculum_path[curriculum.index(env_instance)]
        
        if env_instance == v2_simple_env_20x20_1agent_0human_0shelves_1pickup_1dropoff:
            train_DQN(env_instance, n_episodes=500, max_steps=1000, save_every=100, model_path=get_path(current_v), load_path=get_path(previous_v))
        
        elif env_instance == v3_simple_env_multiple_agents_10x10_3agent_0human_0shelves_1pickup_1dropoff:
            train_DQN(env_instance, n_episodes=2000, max_steps=1000, save_every=100, model_path=get_path(current_v), load_path=get_path(previous_v))
        
        elif env_instance == v5_simple_shelves_10x14_1agent_0human_32shelves_2pickup_4_dropoff:
            # Train DQN agent
            train_DQN(env_instance, n_episodes=2000, max_steps=1000, save_every=100, model_path=get_path(current_v), load_path=get_path(previous_v))
        
        elif env_instance == v6_simple_human_10x10_1agent_5human_0shelves_2pickup_4_dropoff:
            train_DQN(env_instance, n_episodes=2000, max_steps=1000, save_every=100, model_path=get_path(current_v), load_path=get_path(previous_v))
        
        else:
            train_DQN(env_instance, n_episodes=1000, max_steps=1000, save_every=100, model_path=get_path(current_v), load_path=get_path(previous_v))
        
            
def train_DQL():
    v1_simplest_env_10x10_1agent_0human_0shelves_1pickup_1dropoff = env(grid_size=(10, 10), human_grid_size=(10, 10), n_agents=1, n_humans=0, num_shelves=0, num_pickup_points=1,
                        num_dropoff_points=1, render_mode="human")
    
    v2_simple_env_20x20_1agent_0human_0shelves_1pickup_1dropoff = env(grid_size=(20, 20), human_grid_size=(20, 20), n_agents=1, n_humans=0, num_shelves=0, num_pickup_points=1,
                        num_dropoff_points=1, render_mode="human")
    
    v3_simple_env_multiple_agents_10x10_3agent_0human_0shelves_1pickup_1dropoff = env(grid_size=(10, 10), human_grid_size=(10, 10), n_agents=3, n_humans=0, num_shelves=0, num_pickup_points=1,
                        num_dropoff_points=1, render_mode="human")
    
    v4_simple_env_multiple_drop_pickup_10x10_2agent_0human_0shelves_4_pickup_8dropoff = env(grid_size=(10, 10), human_grid_size=(10, 10), n_agents=2, n_humans=0, num_shelves=0, num_pickup_points=4,
                        num_dropoff_points=8, render_mode="human")
    
    v5_simple_shelves_10x14_1agent_0human_32shelves_2pickup_4_dropoff = env(grid_size=(10, 14), human_grid_size=(10, 14), n_agents=1, n_humans=0, num_shelves=32, num_pickup_points=2,
                        num_dropoff_points=4, render_mode="human")
    
    v6_simple_human_10x10_1agent_5human_0shelves_2pickup_4_dropoff = env(grid_size=(10, 10), human_grid_size=(10, 10), n_agents=1, n_humans=5, num_shelves=0, num_pickup_points=2,
                        num_dropoff_points=4, render_mode="human")
    
    v7_simple_human_and_shelves_10x14_1agent_5human_32shelves_2pickup_4_dropoff = env(grid_size=(10, 14), human_grid_size=(10, 14), n_agents=1, n_humans=5, num_shelves=32, num_pickup_points=2,
                        num_dropoff_points=4, render_mode="human")
    
    v8_warehouse_env = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=5, n_humans=10, num_shelves=2048, num_pickup_points=3,
                        num_dropoff_points=4, render_mode="human")

    curriculum_path = ["v1_simplest_env_10x10_1agent_0human_0shelves_1pickup_1dropoff", 
                    "v2_simple_env_20x20_1agent_0human_0shelves_1pickup_1dropoff", 
                    "v3_simple_env_multiple_agents_10x10_3agent_0human_0shelves_1pickup_1dropoff", 
                    "v4_simple_env_multiple_drop_pickup_10x10_2agent_0human_0shelves_4_pickup_8dropoff", 
                    "v5_simple_shelves_10x14_1agent_0human_32shelves_2pickup_4_dropoff",
                    "v6_simple_human_10x10_1agent_5human_0shelves_2pickup_4_dropoff",
                    "v7_simple_human_and_shelves_10x14_1agent_5human_32shelves_2pickup_4_dropoff",
                    "v8_warehouse_env"]
    train_DQN(v1_simplest_env_10x10_1agent_0human_0shelves_1pickup_1dropoff, n_episodes=1000, max_steps=1000, save_every=100, model_path=get_path("v1_simplest_env_10x10_1agent_0human_0shelves_1pickup_1dropoff"), load_path=get_path("v1_simplest_env_10x10_1agent_0human_0shelves_1pickup_1dropoff"))


def test_a_star():
    # Create warehouse environment with rendering
    simplest_env = env(grid_size=(50, 50), n_agents=10, n_humans=0, num_shelves=0, num_pickup_points=4,
                        num_dropoff_points=4, render_mode="human")
    
    
    warehouse_env = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=1, n_humans=1, num_shelves=0, num_pickup_points=1,
                        num_dropoff_points=8, render_mode="human")

    # run_a_star(simplest_env, n_steps=1000, debug_level=5)
    # sum = 0
    # for i in range(100):
    #     sum += run_a_star(simplest_env, n_steps=1000, debug_level=5)
    #     # Run A* algorithm with simplest environment
    # print("Average time taken for A* algorithm with simplest environment: ", sum/100)
    human_counts = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    throughputs = []

    for shelves in range(0, 321, 8):
        
        warehouse_env = env(
            grid_size=(34, 32),
            human_grid_size=(34, 32),
            n_agents=1,
            n_humans=0,
            num_shelves=shelves,
            num_pickup_points=4,
            num_dropoff_points=8,
            render_mode=None
        )

        warehouse_env.reset(seed=1)  # Always same start
        result = run_a_star(warehouse_env, n_steps=1000)

        throughput_per_agent = result / 1
        throughputs.append(throughput_per_agent)

    # Plotting
    plt.figure()
    plt.plot(list(range(0, 41)), throughputs, marker='o')
    plt.xticks(list(range(0, 41)))
    plt.xlabel('Number of Shelves')
    plt.ylabel('Throughput per Agent')
    plt.title('Throughput per Agent vs Number of Shelves')
    plt.grid(True)
    plt.show()

# Take first argument as task ("a_star" or "dqn_train" or "dqn_eval")
def main(task):
    if task == "a_star":
        test_a_star()
    elif task == "dqn_train_curriculum":
        train_DQL_curriculum()
    elif task == "dqn_train":
        train_DQL()
    elif task == "dqn_eval":
        eval_DQN()
    elif task == "ppo_train":
        # Create warehouse environment for PPO training
        warehouse_env = env(
            grid_size=(10, 10), human_grid_size=(10, 10),
            n_agents=1, n_humans=0, num_shelves=0,
            num_pickup_points=1, num_dropoff_points=4,
            render_mode=None
        )
        train_PPO(warehouse_env, n_episodes=1000, max_steps=1000)
    elif task == "ppo_eval":
        # Build the same env you trained on
        warehouse_env = env(
            grid_size=(15, 15), human_grid_size=(15, 15),
            n_agents=1, n_humans=3, num_shelves=8,
            num_pickup_points=1, num_dropoff_points=8,
            render_mode="human"
        )
        # Point this to the .pth you saved after training
        model_path = get_path("stage2_shelves8")
        evaluate_PPO(
            warehouse_env,
            model_path,
            n_episodes=5,
            max_steps=1000,
            render=True
        )
    elif task == "ppo_train_curriculum":
        train_PPO_curriculum()
    else:
        raise ValueError("Invalid task. Choose 'a_star', 'dqn_train', or 'dqn_eval'.")
    

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <task>")
        print("Tasks: a_star, dqn_train, dqn_eval, ppo_train, ppo_eval, ppo_train_curriculum")
        sys.exit(1)
    
    task = sys.argv[1]
    main(task)