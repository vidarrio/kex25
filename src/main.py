import numpy as np
import sys
import json
from environment import env
from algorithms import run_a_star, run_q_learning, train_DQN_curriculum, train_DQN, benchmark_environment
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tensorboard.backend.event_processing import event_accumulator

CURRENT_VERSION = "v1.2.3"

# Construct path to the model directory
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models')
model_name = CURRENT_VERSION + ".pth"
model_path = os.path.join(models_dir, model_name)
outer_dir = os.path.dirname(current_dir)
figures_dir = os.path.join(outer_dir, 'figures')
data_dir = os.path.join(outer_dir, 'data')
tensorboard_dir = os.path.join(outer_dir, 'runs')

def get_env(env_name):
    # Environments
    stage1_env = env(grid_size=(5, 5), n_agents=1, n_humans=0, num_shelves=0, 
                    num_pickup_points=1, num_dropoff_points=1, render_mode="human")
    stage2_env = env(grid_size=(10, 8), human_grid_size=(10, 8), n_agents=1, n_humans=0, num_shelves=16, 
                        num_pickup_points=1, num_dropoff_points=1, render_mode="human")
    stage3_env = env(grid_size=(25, 25), n_agents=1, n_humans=0, num_shelves=50, 
                        num_pickup_points=1, num_dropoff_points=1, render_mode="human")

    warehouse_env = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=6, n_humans=10, num_shelves=2048, num_pickup_points=10,
                            num_dropoff_points=10, render_mode="human")
    
    test_env = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=10, n_humans=10, num_shelves=2048, num_pickup_points=10,
                            num_dropoff_points=10, render_mode="human")
    
    if env_name == "stage1":
        return stage1_env
    elif env_name == "stage2":
        return stage2_env
    elif env_name == "stage3":
        return stage3_env
    elif env_name == "warehouse":
        return warehouse_env
    elif env_name == "test_env":
        return test_env
    else:
        raise ValueError("Invalid environment name.")


def start_DQN_eval(model_path=None):
    # Run DQN agent
    run_q_learning(get_env("test_env"), n_steps=1000, full_model_path=model_path)

def start_DQN_training():
    # Train DQN agent
    train_DQN_curriculum(get_env("warehouse"), n_episodes=1000, max_steps=1000, save_every=100)

def test_a_star():
    run_a_star(get_env("warehouse"), n_steps=1000, debug_level=0)

def start_dqn_training_no_curriculum():
    # Train DQN agent without curriculum
    train_DQN(get_env("warehouse"), n_episodes=20000, max_steps=300, save_every=100)

# Generate data comparing A* and QMIX throughput per agent for increasing number of dynamic obstacles
def generate_human_graph_data(model_path):

    # From 0 to 100 dynamic obstacles, in steps of 5
    num_humans = np.arange(0, 101, 5)

    # Result dict
    results = {
        "num_humans": num_humans,
        "A_star_avg": [],
        "RL_avg": [],
        "A_star_std": [],
        "RL_std": []
    }

    for idx, num_human in enumerate(num_humans):
        env_params = {
            "grid_size": (34, 32),
            "human_grid_size": (34, 32),
            "n_agents": 10,
            "n_humans": num_human,
            "num_shelves": 320,
            "num_pickup_points": 10,
            "num_dropoff_points": 8,
            "render_mode": None,
            "n_steps": 1000,
        }

        benchmark_results = benchmark_environment(env_phase=4, n_steps=1000, env_params=env_params, model_path=model_path)

        # Extract totals
        results["A_star_avg"].append(benchmark_results["avg_astar_per_agent"])
        results["RL_avg"].append(benchmark_results["avg_dqn_per_agent"])
        results["A_star_std"].append(benchmark_results["astar_std_per_agent"])
        results["RL_std"].append(benchmark_results["dqn_std_per_agent"])

    # Save results to a JSON file
    results_file = os.path.join(data_dir, "astar_vs_rl_throughput_over_humans.json")

    # Convert any numpy arrays to lists
    for key in results:
        if isinstance(results[key], np.ndarray):
            results[key] = results[key].tolist()
        elif isinstance(results[key], list):
            results[key] = [float(item) if isinstance(item, np.number) else item for item in results[key]]

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_file}")
    
    return results_file

def generate_shelf_graph_data(model_path):
    # From 0 to 320 shelves, in steps of 20
    num_shelves = np.arange(0, 321, 20)

        # Result dict
    results = {
        "num_shelves": num_shelves,
        "A_star_avg": [],
        "RL_avg": [],
        "A_star_std": [],
        "RL_std": []
    }

    for idx, num_shelf in enumerate(num_shelves):
        env_params = {
            "grid_size": (34, 32),
            "human_grid_size": (34, 32),
            "n_agents": 10,
            "n_humans": 10,
            "num_shelves": num_shelf,
            "num_pickup_points": 10,
            "num_dropoff_points": 8,
            "render_mode": None,
            "n_steps": 1000,
        }

        benchmark_results = benchmark_environment(env_phase=4, n_steps=1000, env_params=env_params, model_path=model_path)

        # Extract totals
        results["A_star_avg"].append(benchmark_results["avg_astar_per_agent"])
        results["RL_avg"].append(benchmark_results["avg_dqn_per_agent"])
        results["A_star_std"].append(benchmark_results["astar_std_per_agent"])
        results["RL_std"].append(benchmark_results["dqn_std_per_agent"])

    # Save results to a JSON file
    results_file = os.path.join(data_dir, "astar_vs_rl_throughput_over_shelves.json")

    # Convert any numpy arrays to lists
    for key in results:
        if isinstance(results[key], np.ndarray):
            results[key] = results[key].tolist()
        elif isinstance(results[key], list):
            results[key] = [float(item) if isinstance(item, np.number) else item for item in results[key]]

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_file}")
    
    return results_file

def generate_robot_graph_data(model_path):
    # From 1 to 51 robots, in steps of 5
    num_robos = np.arange(1, 52, 5)

    # Result dict
    results = {
        "num_agents": num_robos,
        "A_star_avg": [],
        "RL_avg": [],
        "A_star_std": [],
        "RL_std": []
    }

    for idx, num_robot in enumerate(num_robos):
        env_params = {
            "grid_size": (34, 32),
            "human_grid_size": (34, 32),
            "n_agents": num_robot,
            "n_humans": 10,
            "num_shelves": 320,
            "num_pickup_points": 10,
            "num_dropoff_points": 8,
            "render_mode": None,
            "n_steps": 1000,
        }

        benchmark_results = benchmark_environment(env_phase=4, n_steps=1000, env_params=env_params, model_path=model_path)

        # Extract totals
        results["A_star_avg"].append(benchmark_results["avg_astar_per_agent"])
        results["RL_avg"].append(benchmark_results["avg_dqn_per_agent"])
        results["A_star_std"].append(benchmark_results["astar_std_per_agent"])
        results["RL_std"].append(benchmark_results["dqn_std_per_agent"])

    # Save results to a JSON file
    results_file = os.path.join(data_dir, "astar_vs_rl_throughput_over_robots.json")

    # Convert any numpy arrays to lists
    for key in results:
        if isinstance(results[key], np.ndarray):
            results[key] = results[key].tolist()
        elif isinstance(results[key], list):
            results[key] = [float(item) if isinstance(item, np.number) else item for item in results[key]]

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_file}")
    
    return results_file

def generate_gridsize_graph_data(model_path):
    # Define grid sizes
    grid_sizes = [
        (10, 10),
        (15, 15),
        (20, 20),
        (25, 25),
        (30, 30),
        (34, 32),
        (40, 40),
        (50, 50),
    ]

    # Result dict
    results = {
        "grid_sizes": grid_sizes,
        "A_star_avg": [],
        "RL_avg": [],
        "A_star_std": [],
        "RL_std": []
    }

    for idx, grid_size in enumerate(grid_sizes):
        env_params = {
            "grid_size": grid_size,
            "human_grid_size": grid_size,
            "n_agents": 10,
            "n_humans": 10,
            "num_shelves": 320,
            "num_pickup_points": 10,
            "num_dropoff_points": 8,
            "render_mode": None,
            "n_steps": 1000,
        }

        benchmark_results = benchmark_environment(env_phase=4, n_steps=1000, env_params=env_params, model_path=model_path)

        # Extract totals
        results["A_star_avg"].append(benchmark_results["avg_astar_per_agent"])
        results["RL_avg"].append(benchmark_results["avg_dqn_per_agent"])
        results["A_star_std"].append(benchmark_results["astar_std_per_agent"])
        results["RL_std"].append(benchmark_results["dqn_std_per_agent"])

    # Save results to a JSON file
    results_file = os.path.join(data_dir, "astar_vs_rl_throughput_over_gridsizes.json")

    # Convert any numpy arrays to lists
    for key in results:
        if isinstance(results[key], np.ndarray):
            results[key] = results[key].tolist()
        elif isinstance(results[key], list):
            results[key] = [float(item) if isinstance(item, np.number) else item for item in results[key]]

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_file}")
    
    return results_file

def generate_human_graph_plot_diff(data_path):
    # Load the data from the JSON file
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Convert the data to numpy arrays
    rl_avg = np.array(data["RL_avg"])
    a_star_avg = np.array(data["A_star_avg"])
    rl_std = np.array(data["RL_std"])
    a_star_std = np.array(data["A_star_std"])
    num_humans = np.array(data["num_humans"])

    # Calculate the performance difference
    diff = rl_avg - a_star_avg

    # Calculate the standard deviation of the difference
    diff_std = np.sqrt(rl_std**2 + a_star_std**2)

    # Set the style for seaborn
    sns.set_theme(style="whitegrid")

    # Set the font size for all plots
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    # Set good aspect ratio for latex
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the difference in average throughput for A* and RL
    ax.plot(data["num_humans"], diff, label="QMIX - A*", marker='o', markersize=5, color='purple')
    

    # Add shaded areas for the standard deviation
    ax.fill_between(num_humans,
                    diff - diff_std,
                    diff + diff_std,
                    alpha=0.2, color='purple')
    
    # Add a horizontal line at y=0
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    # Set the title and labels
    ax.set_xlabel("Number of Dynamic Obstacles")
    ax.set_ylabel("Average Throughput Difference (QMIX - A*)")

    # Add a legend
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray', framealpha=0.7, borderpad=0.8)

    # Adjust grid for a cleaner look
    ax.grid(True, linestyle='--', alpha=0.7)

    # Subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('lightgray')

    # Thicker lines scale better
    for line in ax.get_lines():
        line.set_linewidth(1.25)

    # Use tight layout
    plt.tight_layout()

    # Save the plot as a svg file
    plt.savefig(os.path.join(figures_dir, "astar_vs_rl_throughput_diff_over_humans.svg"), format='svg', bbox_inches='tight', dpi=300)

    print(f"Graph saved to {os.path.join(figures_dir, 'astar_vs_rl_throughput_diff_over_humans.svg')}")

def generate_human_graph_plot(data_path):

    # Load the data from the JSON file
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Set the style for seaborn
    sns.set_theme(style="whitegrid")

    # Set the font size for all plots
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    # Set good aspect ratio for latex
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the average throughput for A* and RL
    ax.plot(data["num_humans"], data["A_star_avg"], label="A*", marker='o', markersize=5)
    ax.plot(data["num_humans"], data["RL_avg"], label="QMIX", marker='o', markersize=5)
    # Add shaded areas for the standard deviation
    ax.fill_between(data["num_humans"],
                    np.array(data["A_star_avg"]) - np.array(data["A_star_std"]),
                    np.array(data["A_star_avg"]) + np.array(data["A_star_std"]),
                    alpha=0.2, color='blue')
    ax.fill_between(data["num_humans"],
                    np.array(data["RL_avg"]) - np.array(data["RL_std"]),
                    np.array(data["RL_avg"]) + np.array(data["RL_std"]),
                    alpha=0.2, color='orange')
    
    # Set the title and labels
    ax.set_xlabel("Number of Dynamic Obstacles")
    ax.set_ylabel("Average Throughput per Agent")

    # Add a legend
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray', framealpha=0.7, borderpad=0.8)

    # Adjust grid for a cleaner look
    ax.grid(True, linestyle='--', alpha=0.7)

    # Subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('lightgray')

    # Thicker lines scale better
    for line in ax.get_lines():
        line.set_linewidth(1.25) 

    # Use tight layout
    plt.tight_layout()
    
    # Save the plot as a svg file
    plt.savefig(os.path.join(figures_dir, "astar_vs_rl_throughput_over_humans.svg"), format='svg', bbox_inches='tight', dpi=300)

    print(f"Graph saved to {os.path.join(figures_dir, 'astar_vs_rl_throughput_over_humans.svg')}")

def generate_shelf_graph_plot_diff(data_path):
    # Load the data from the JSON file
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Convert the data to numpy arrays
    rl_avg = np.array(data["RL_avg"])
    a_star_avg = np.array(data["A_star_avg"])
    rl_std = np.array(data["RL_std"])
    a_star_std = np.array(data["A_star_std"])
    num_shelves = np.array(data["num_shelves"])

    # Calculate the performance difference
    diff = rl_avg - a_star_avg

    # Calculate the standard deviation of the difference
    diff_std = np.sqrt(rl_std**2 + a_star_std**2)

    # Set the style for seaborn
    sns.set_theme(style="whitegrid")

    # Set the font size for all plots
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    # Set good aspect ratio for latex
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the difference in average throughput for A* and RL
    ax.plot(data["num_shelves"], diff, label="QMIX - A*", marker='o', markersize=5, color='purple')
    
    # Add shaded areas for the standard deviation
    ax.fill_between(num_shelves,
                    diff - diff_std,
                    diff + diff_std,
                    alpha=0.2, color='purple')
    
    # Add a horizontal line at y=0
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    # Set the title and labels
    ax.set_xlabel("Number of Static Obstacles")
    ax.set_ylabel("Average Throughput Difference (QMIX - A*)")

    # Add a legend
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray', framealpha=0.7, borderpad=0.8)

    # Adjust grid for a cleaner look
    ax.grid(True, linestyle='--', alpha=0.7)

    # Subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('lightgray')

    # Thicker lines scale better
    for line in ax.get_lines():
        line.set_linewidth(1.25)

    # Use tight layout
    plt.tight_layout()

    # Save the plot as a svg file
    plt.savefig(os.path.join(figures_dir, "astar_vs_rl_throughput_diff_over_shelves.svg"), format='svg', bbox_inches='tight', dpi=300)

    print(f"Graph saved to {os.path.join(figures_dir, 'astar_vs_rl_throughput_diff_over_shelves.svg')}")

def generate_shelf_graph_plot(data_path):

    # Load the data from the JSON file
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Set the style for seaborn
    sns.set_theme(style="whitegrid")

    # Set the font size for all plots
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    # Set good aspect ratio for latex
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the average throughput for A* and RL
    ax.plot(data["num_shelves"], data["A_star_avg"], label="A*", marker='o', markersize=5)
    ax.plot(data["num_shelves"], data["RL_avg"], label="QMIX", marker='o', markersize=5)
    # Add shaded areas for the standard deviation
    ax.fill_between(data["num_shelves"],
                    np.array(data["A_star_avg"]) - np.array(data["A_star_std"]),
                    np.array(data["A_star_avg"]) + np.array(data["A_star_std"]),
                    alpha=0.2, color='blue')
    ax.fill_between(data["num_shelves"],
                    np.array(data["RL_avg"]) - np.array(data["RL_std"]),
                    np.array(data["RL_avg"]) + np.array(data["RL_std"]),
                    alpha=0.2, color='orange')
    
    # Set the title and labels
    ax.set_xlabel("Number of Static Obstacles")
    ax.set_ylabel("Average Throughput per Agent")

    # Add a legend
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray', framealpha=0.7, borderpad=0.8)

    # Adjust grid for a cleaner look
    ax.grid(True, linestyle='--', alpha=0.7)

    # Subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('lightgray')

    # Thicker lines scale better
    for line in ax.get_lines():
        line.set_linewidth(1.25) 

    # Use tight layout
    plt.tight_layout()
    
    # Save the plot as a svg file
    plt.savefig(os.path.join(figures_dir, "astar_vs_rl_throughput_over_shelves.svg"), format='svg', bbox_inches='tight', dpi=300)

    print(f"Graph saved to {os.path.join(figures_dir, 'astar_vs_rl_throughput_over_shelves.svg')}")

def generate_robot_graph_plot_diff(data_path):
    # Load the data from the JSON file
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Convert the data to numpy arrays
    rl_avg = np.array(data["RL_avg"])
    a_star_avg = np.array(data["A_star_avg"])
    rl_std = np.array(data["RL_std"])
    a_star_std = np.array(data["A_star_std"])
    num_agents = np.array(data["num_agents"])

    # Calculate the performance difference
    diff = rl_avg - a_star_avg

    # Calculate the standard deviation of the difference
    diff_std = np.sqrt(rl_std**2 + a_star_std**2)

    # Set the style for seaborn
    sns.set_theme(style="whitegrid")

    # Set the font size for all plots
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    # Set good aspect ratio for latex
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the difference in average throughput for A* and RL
    ax.plot(data["num_agents"], diff, label="QMIX - A*", marker='o', markersize=5, color='purple')
    
    # Add shaded areas for the standard deviation
    ax.fill_between(num_agents,
                    diff - diff_std,
                    diff + diff_std,
                    alpha=0.2, color='purple')
    
    # Add a horizontal line at y=0
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    # Set the title and labels
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Average Throughput Difference (QMIX - A*)")

    # Add a legend
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray', framealpha=0.7, borderpad=0.8)

    # Adjust grid for a cleaner look
    ax.grid(True, linestyle='--', alpha=0.7)

    # Subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('lightgray')

    # Thicker lines scale better
    for line in ax.get_lines():
        line.set_linewidth(1.25)

    # Use tight layout
    plt.tight_layout()

    # Save the plot as a svg file
    plt.savefig(os.path.join(figures_dir, "astar_vs_rl_throughput_diff_over_robots.svg"), format='svg', bbox_inches='tight', dpi=300)

    print(f"Graph saved to {os.path.join(figures_dir, 'astar_vs_rl_throughput_diff_over_robots.svg')}")

def generate_robot_graph_plot(data_path):

    # Load the data from the JSON file
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Set the style for seaborn
    sns.set_theme(style="whitegrid")

    # Set the font size for all plots
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    # Set good aspect ratio for latex
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the average throughput for A* and RL
    ax.plot(data["num_agents"], data["A_star_avg"], label="A*", marker='o', markersize=5)
    ax.plot(data["num_agents"], data["RL_avg"], label="QMIX", marker='o', markersize=5)
    # Add shaded areas for the standard deviation
    ax.fill_between(data["num_agents"],
                    np.array(data["A_star_avg"]) - np.array(data["A_star_std"]),
                    np.array(data["A_star_avg"]) + np.array(data["A_star_std"]),
                    alpha=0.2, color='blue')
    ax.fill_between(data["num_agents"],
                    np.array(data["RL_avg"]) - np.array(data["RL_std"]),
                    np.array(data["RL_avg"]) + np.array(data["RL_std"]),
                    alpha=0.2, color='orange')
    
    # Set the title and labels
    ax.set_xlabel("Number of Robots")
    ax.set_ylabel("Average Throughput per Agent")

    # Add a legend
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray', framealpha=0.7, borderpad=0.8)

    # Adjust grid for a cleaner look
    ax.grid(True, linestyle='--', alpha=0.7)

    # Subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('lightgray')

    # Thicker lines scale better
    for line in ax.get_lines():
        line.set_linewidth(1.25) 

    # Use tight layout
    plt.tight_layout()
    
    # Save the plot as a svg file
    plt.savefig(os.path.join(figures_dir, "astar_vs_rl_throughput_over_robots.svg"), format='svg', bbox_inches='tight', dpi=300)

    print(f"Graph saved to {os.path.join(figures_dir, 'astar_vs_rl_throughput_over_robots.svg')}")

def generate_gridsize_graph_plot_diff(data_path):
    # Load the data from the JSON file
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Convert the data to numpy arrays
    rl_avg = np.array(data["RL_avg"])
    a_star_avg = np.array(data["A_star_avg"])
    rl_std = np.array(data["RL_std"])
    a_star_std = np.array(data["A_star_std"])
    grid_sizes_temp = np.array(data["grid_sizes"])
    # Extract first dimension of each tuple
    grid_sizes = np.array([grid_size[0] for grid_size in grid_sizes_temp])

    # Calculate the performance difference
    diff = rl_avg - a_star_avg

    # Calculate the standard deviation of the difference
    diff_std = np.sqrt(rl_std**2 + a_star_std**2)

    # Set the style for seaborn
    sns.set_theme(style="whitegrid")

    # Set the font size for all plots
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    # Set good aspect ratio for latex
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the difference in average throughput for A* and RL
    ax.plot(grid_sizes, diff, label="QMIX - A*", marker='o', markersize=5, color='purple')
    
    # Add shaded areas for the standard deviation
    ax.fill_between(grid_sizes,
                    diff - diff_std,
                    diff + diff_std,
                    alpha=0.2, color='purple')
    
    # Add a horizontal line at y=0
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    # Set the title and labels
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Average Throughput Difference (QMIX - A*)")

    # Add a legend
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray', framealpha=0.7, borderpad=0.8)

    # Adjust grid for a cleaner look
    ax.grid(True, linestyle='--', alpha=0.7)

    # Subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('lightgray')

    # Thicker lines scale better
    for line in ax.get_lines():
        line.set_linewidth(1.25)

    # Use tight layout
    plt.tight_layout()

    # Save the plot as a svg file
    plt.savefig(os.path.join(figures_dir, "astar_vs_rl_throughput_diff_over_gridsize.svg"), format='svg', bbox_inches='tight', dpi=300)

    print(f"Graph saved to {os.path.join(figures_dir, 'astar_vs_rl_throughput_diff_over_gridsize.svg')}")

def generate_gridsize_graph_plot(data_path):

    # Load the data from the JSON file
    with open(data_path, 'r') as f:
        data = json.load(f)

    grid_sizes_temp = np.array(data["grid_sizes"])
    # Extract first dimension of each tuple
    grid_sizes = np.array([grid_size[0] for grid_size in grid_sizes_temp])

    # Set the style for seaborn
    sns.set_theme(style="whitegrid")

    # Set the font size for all plots
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    # Set good aspect ratio for latex
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the average throughput for A* and RL
    ax.plot(grid_sizes, data["A_star_avg"], label="A*", marker='o', markersize=5)
    ax.plot(grid_sizes, data["RL_avg"], label="QMIX", marker='o', markersize=5)
    # Add shaded areas for the standard deviation
    ax.fill_between(grid_sizes,
                    np.array(data["A_star_avg"]) - np.array(data["A_star_std"]),
                    np.array(data["A_star_avg"]) + np.array(data["A_star_std"]),
                    alpha=0.2, color='blue')
    ax.fill_between(grid_sizes,
                    np.array(data["RL_avg"]) - np.array(data["RL_std"]),
                    np.array(data["RL_avg"]) + np.array(data["RL_std"]),
                    alpha=0.2, color='orange')
    
    # Set the title and labels
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Average Throughput per Agent")

    # Add a legend
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray', framealpha=0.7, borderpad=0.8)

    # Adjust grid for a cleaner look
    ax.grid(True, linestyle='--', alpha=0.7)

    # Subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('lightgray')

    # Thicker lines scale better
    for line in ax.get_lines():
        line.set_linewidth(1.25) 

    # Use tight layout
    plt.tight_layout()
    
    # Save the plot as a svg file
    plt.savefig(os.path.join(figures_dir, "astar_vs_rl_throughput_over_gridsize.svg"), format='svg', bbox_inches='tight', dpi=300)

    print(f"Graph saved to {os.path.join(figures_dir, 'astar_vs_rl_throughput_over_gridsize.svg')}")

def generate_training_tasks_plot(data_path=None):
    if data_path is None:
        # Throw exception if data_path is None
        raise ValueError("data_path cannot be None. Please provide a valid path to the data file.")

    # Trim down path to just the file name
    data_path = os.path.basename(data_path)
    # Remove "_best" if it exists in the file name
    if "_best" in data_path:
        data_path = data_path.replace("_best", "")
    # Point at tensorboard dir
    data_path = tensorboard_dir + "/" + data_path
    # Get file in the folder by scanning the directory
    file = os.listdir(data_path)
    # Check if the file exists
    if not file:
        print(f"Error: No files found in {data_path}.")
        return
    # Get the first file in the directory
    data_path = os.path.join(data_path, file[0])

    # Initialize event accumulator
    ea = event_accumulator.EventAccumulator(data_path)
    ea.Reload()

    if 'Evaluation/RL_Tasks' not in ea.scalars.Keys():
        print(f"Error: 'Evaluation/RL_Tasks' not found in {data_path}.")
        return
    
    # Get data
    rl_metrics = ea.scalars.Items('Evaluation/RL_Tasks')
    rl_steps = [event.step for event in rl_metrics]
    rl_values = [event.value for event in rl_metrics]
    rl_values = smooth(rl_values, window=7)

    # Get std data
    rl_std_metrics = ea.scalars.Items('Evaluation/RL_StdDev')
    rl_std_steps = [event.step for event in rl_std_metrics]
    rl_std_values = [event.value for event in rl_std_metrics]

    # Set the style for seaborn
    sns.set_theme(style="whitegrid")

    # Set the font size for all plots
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    # Set good aspect ratio for latex
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the average throughput for RL during training
    ax.plot(rl_steps, rl_values, label="QMIX", marker='o', markersize=5, color='orange')

    # Add shaded areas for the standard deviation
    ax.fill_between(rl_steps,
                    np.array(rl_values) - np.array(rl_std_values),
                    np.array(rl_values) + np.array(rl_std_values),
                    alpha=0.2, color='orange')
    
    # Add a horizontal line at y=0
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    # Set the title and labels
    ax.set_xlabel("Training Episodes")
    # 6 agents, 300 time steps
    ax.set_ylabel("Average System Throughput (6 agents, 300 time steps)")

    # Add a legend
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray', framealpha=0.7, borderpad=0.8)

    # Adjust grid for a cleaner look
    ax.grid(True, linestyle='--', alpha=0.7)

    # Subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('lightgray')

    # Thicker lines scale better
    for line in ax.get_lines():
        line.set_linewidth(1.25)

    # Use tight layout
    plt.tight_layout()

    # Save the plot as a svg file
    plt.savefig(os.path.join(figures_dir, "rl_training_delivered.svg"), format='svg', bbox_inches='tight', dpi=300)

    print(f"Graph saved to {os.path.join(figures_dir, 'rl_training_delivered.svg')}")
    
def smooth(y, window=5):
    box = np.ones(window)/window
    return np.convolve(y, box, mode='same')

def gen_graph(type, model_path, data_path=None): 

    if type == "human":
        if data_path is None:
            data_path = generate_human_graph_data(model_path)
        generate_human_graph_plot(data_path)
        generate_human_graph_plot_diff(data_path)
    if type == "shelf":
        if data_path is None:
            data_path = generate_shelf_graph_data(model_path)
        generate_shelf_graph_plot(data_path)
        generate_shelf_graph_plot_diff(data_path)
    if type == "robot":
        if data_path is None:
            data_path = generate_robot_graph_data(model_path)
        generate_robot_graph_plot(data_path)
        generate_robot_graph_plot_diff(data_path)
    if type == "gridsize":
        if data_path is None:
            data_path = generate_gridsize_graph_data(model_path)
        generate_gridsize_graph_plot(data_path)
        generate_gridsize_graph_plot_diff(data_path)
    if type == "training":
        generate_training_tasks_plot(data_path=model_path)

def main():
    parser = argparse.ArgumentParser(description="DQN and A* Benchmarking")

    subparsers = parser.add_subparsers(dest="command", help="commands")

    # Subparser for "gen_graph" command
    gen_graph_parser = subparsers.add_parser("gen_graph", help="generate graphs")
    gen_graph_parser.add_argument("-t", "--type", help="type of graph to generate", choices=["human", "shelf", "robot", "gridsize", "training"], required=True)
    gen_graph_parser.add_argument("-m", "--model", help="model name including .pth", required=True)
    gen_graph_parser.add_argument("-d", "--data", help="data file name including .json")

    # Subparser for "benchmark" command
    benchmark_parser = subparsers.add_parser("benchmark", help="run benchmark")
    benchmark_parser.add_argument("-m", "--model", help="model name including .pth")
    benchmark_parser.add_argument("-p", "--phase", type=int, help="phase number")

    # Subparser for "dqn_eval" command
    dqn_eval_parser = subparsers.add_parser("dqn_eval", help="run DQN evaluation")
    dqn_eval_parser.add_argument("-m", "--model", help="model name including .pth")

    # Subparser for "dqn_train" command
    dqn_train_parser = subparsers.add_parser("dqn_train", help="run DQN training")
    dqn_train_parser.add_argument("-nc", "--no_curriculum", help="train DQN without curriculum")

    # Subparser for "a_star" command
    a_star_parser = subparsers.add_parser("a_star", help="run A* algorithm")

    # Store the subparsers in a dictionary for easy access
    command_parsers = {
        'gen_graph': gen_graph_parser,
        'benchmark': benchmark_parser,
        'dqn_eval': dqn_eval_parser,
        'dqn_train': dqn_train_parser,
        'a_star': a_star_parser
    }

    # Parse the arguments
    args = parser.parse_args()

    if args.command == "a_star":
        test_a_star()

    elif args.command == "dqn_train":
        start_DQN_training()

    elif args.command == "dqn_eval":
        try:
            start_DQN_eval(model_path=models_dir + "/" + args.model)
        except Exception as e:
            print(f"Error: {e}")
            command_parsers['dqn_eval'].print_help()


    elif args.command == "benchmark":
        try:
            env_phase = args.phase if args.phase is not None else 4
            benchmark_environment(env_phase=env_phase, n_steps=1000, model_path=models_dir + "/" + args.model)
        except Exception as e:
            print(f"Error: {e}")
            command_parsers['benchmark'].print_help()

    elif args.command == "gen_graph":
        try:
            data_path = data_dir + "/" + args.data if args.data else None
            gen_graph(args.type, models_dir + "/" + args.model, data_path=data_path)
        except Exception as e:
            print(f"Error: {e}")
            command_parsers['gen_graph'].print_help()

    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()