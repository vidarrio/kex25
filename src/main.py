import numpy as np
from environment import env
from algorithms import run_a_star, run_q_learning, train_DQN_curriculum, train_DQN, benchmark_environment
import os

CURRENT_VERSION = "v1.2.1"

# Construct path to the model directory
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models')
model_name = CURRENT_VERSION + ".pth"
model_path = os.path.join(models_dir, model_name)



def get_env(env_name):
    # Environments
    stage1_env = env(grid_size=(5, 5), n_agents=1, n_humans=0, num_shelves=0, 
                    num_pickup_points=1, num_dropoff_points=1, render_mode="human")
    stage2_env = env(grid_size=(10, 8), human_grid_size=(10, 8), n_agents=1, n_humans=0, num_shelves=16, 
                        num_pickup_points=1, num_dropoff_points=1, render_mode="human")
    stage3_env = env(grid_size=(25, 25), n_agents=1, n_humans=0, num_shelves=50, 
                        num_pickup_points=1, num_dropoff_points=1, render_mode="human")

    warehouse_env = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=6, n_humans=10, num_shelves=2048, num_pickup_points=3,
                            num_dropoff_points=2, render_mode="human")
    
    if env_name == "stage1":
        return stage1_env
    elif env_name == "stage2":
        return stage2_env
    elif env_name == "stage3":
        return stage3_env
    elif env_name == "warehouse":
        return warehouse_env
    else:
        raise ValueError("Invalid environment name.")


def start_DQN_eval(model_path=None):
    # Run DQN agent
    run_q_learning(get_env("warehouse"), n_steps=1000, full_model_path=model_path)

def start_DQN_training():
    # Train DQN agent
    train_DQN_curriculum(get_env("warehouse"), n_episodes=1000, max_steps=1000, save_every=100, model_path=model_path)

def test_a_star():
    run_a_star(get_env("warehouse"), n_steps=1000, debug_level=0)

def start_dqn_training_no_curriculum():
    # Train DQN agent without curriculum
    train_DQN(get_env("warehouse"), n_episodes=20000, max_steps=300, save_every=100)

def benchmark_agents():
    """Compare A* and DQN performance across different environments"""
    environments = [
        ("Simple (5x5)", stage1_env),
        ("Medium (10x8 with shelves)", stage2_env),
        ("Large (25x25 with shelves)", stage3_env)
    ]
    
    results = {}
    
    for name, test_env in environments:
        print(f"\nBenchmarking {name} environment...")
        results[name] = benchmark_environment(test_env, n_steps=200)
        
    # Print summary table
    print("\n=== BENCHMARK SUMMARY ===")
    print(f"{'Environment':<25} {'A* Tasks':<10} {'RL Tasks':<10} {'Ratio':<10}")
    print("-" * 55)
    
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['astar_tasks']:<10} {metrics['rl_tasks']:<10} {metrics['performance_ratio']:.2f}")

# Update main function to include benchmark option
def main(task, argv=None):
    if task == "a_star":
        test_a_star()
    elif task == "dqn_train":
        start_DQN_training()
    elif task == "dqn_train_nc":
        start_dqn_training_no_curriculum()
    elif task == "dqn_eval":
        model = argv[0] if argv else None
        if model:
            start_DQN_eval(model_path=models_dir + "/" + model)
    elif task == "benchmark":
        benchmark_agents()
    elif task == "manual_benchmark":
        model = argv[0] if argv else None
        phase = int(argv[1]) if len(argv) > 1 else None
        if model and phase:
            benchmark_environment(env_phase=phase, model_path=models_dir + "/" + model)
    else:
        raise ValueError("Invalid task. Choose 'a_star', 'dqn_train', 'dqn_eval', or 'benchmark'.")

# Update usage message
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <task>")
        print("Tasks: a_star, dqn_train, dqn_train_nc, dqn_eval, benchmark")
        sys.exit(1)
    
    task = sys.argv[1]
    main(task, argv=sys.argv[2:])