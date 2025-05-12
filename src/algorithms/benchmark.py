# Standard library imports
import os
import random
import time

# Third party imports
import numpy as np
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt

# Local application imports
from .common import DEBUG_ALL, DEBUG_NONE, DEBUG_CRITICAL, DEBUG_INFO, DEBUG_VERBOSE, DEBUG_SPECIFIC, get_model_path, check_gpu_usage, determine_process_count
from .agent import QLAgent
from algorithms.a_star import run_a_star
from environment import env

# Run a single benchmark episode for A* and RL agents
def run_benchmark_episode(seed, phase, steps, model_path):

    # Set up environments with the same seed
    if phase == 1:
        env_a_star = env(grid_size=(5, 5), n_agents=1, n_humans=0, num_shelves=0, 
                    num_pickup_points=1, num_dropoff_points=1, render_mode=None, seed=seed)
        env_dqn = env(grid_size=(5, 5), n_agents=1, n_humans=0, num_shelves=0, 
                    num_pickup_points=1, num_dropoff_points=1, render_mode=None, seed=seed)
    elif phase == 2:
        env_a_star = env(grid_size=(10, 8), n_agents=1, n_humans=0, num_shelves=16, 
                    num_pickup_points=1, num_dropoff_points=1, render_mode=None, seed=seed)
        env_dqn = env(grid_size=(10, 8), n_agents=1, n_humans=0, num_shelves=16, 
                    num_pickup_points=1, num_dropoff_points=1, render_mode=None, seed=seed)
    elif phase == 3:
        env_a_star = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=6, n_humans=10, num_shelves=2048, 
                        num_pickup_points=3, num_dropoff_points=2, render_mode=None, seed=seed)
        env_dqn = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=6, n_humans=10, num_shelves=2048, 
                        num_pickup_points=3, num_dropoff_points=2, render_mode=None, seed=seed)
    elif phase == 4:
        env_a_star = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=10, n_humans=10, num_shelves=2048, 
                        num_pickup_points=3, num_dropoff_points=2, render_mode=None, seed=seed)
        env_dqn = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=10, n_humans=10, num_shelves=2048, 
                        num_pickup_points=3, num_dropoff_points=2, render_mode=None, seed=seed)
    
    # Reset environments
    env_a_star.reset()
    env_dqn.reset()
    
    # Run A* agent
    astar_tasks = run_a_star(env_a_star, n_steps=steps, debug_level=DEBUG_NONE)
    env_a_star.close()
    
    # Run RL agent with explicit model loading
    rl_tasks = run_q_learning(env_dqn, full_model_path=model_path, n_steps=steps, debug_level=DEBUG_NONE)
    env_dqn.close()
    
    # Clean up plots that might be open
    plt.close('all')
    
    return astar_tasks, rl_tasks

def benchmark_environment(env_phase, n_steps=200, debug_level=DEBUG_NONE, model_path=get_model_path(), n_jobs=None, eval_episodes=100):
    """
    Run both A* and RL agents on the same environment to establish performance benchmarks.
    
    Args:
        env_phase: Phase of the environment to benchmark (1, 2, or 3).
        n_steps: Maximum steps per episode (default: 200).
        debug_level: Debug level (default: DEBUG_NONE).
        model_path: Path to the model file (default: get_model_path()).
        n_jobs: Number of processes for parallel execution (default: None, auto-determined).
        eval_episodes: Number of episodes to run for evaluation (default: 100).
        
    Returns:
        dict: Results dictionary containing benchmark metrics.
    """
    n_jobs = determine_process_count(n_jobs)

    # Check if model exists before running benchmark
    model_exists = os.path.isfile(model_path)
    if not model_exists:
        print(f"WARNING: Model file not found at {model_path}")
        # Try to find best model
        best_model_path = model_path.replace('.pth', '_best.pth')
        if os.path.isfile(best_model_path):
            print(f"Using best model found at {best_model_path}")
            model_path = best_model_path
        else:
            print("No model found! Cannot run RL benchmark.")
            return None
    else:
        print(f"Using model from {model_path}")
    
    # Generate seeds for all episodes
    seeds = [random.randint(0, 10000) for _ in range(eval_episodes)]

    # Create a partial function with fixed arguments
    benchmark_fn = partial(run_benchmark_episode, phase=env_phase, steps=n_steps, model_path=model_path)
    
    # Set up multiprocessing pool and run benchmarks in parallel
    print(f"Running {eval_episodes} benchmark episodes in parallel using {n_jobs} processes...")
    start_time = time.time()

    # Set start method to 'spawn' for better compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set, ignore

    # Use context manager to ensure proper cleanup
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(benchmark_fn, seeds)

    # Aggregate results
    total_a_star_completed_tasks = sum(r[0] for r in results)
    total_dqn_completed_tasks = sum(r[1] for r in results)
    
    # Store individual episode results for additional metrics
    a_star_tasks_per_episode = [r[0] for r in results]
    dqn_tasks_per_episode = [r[1] for r in results]
    
    # Calculate ratio per episode (avoiding division by zero)
    ratios = []
    for a, d in zip(a_star_tasks_per_episode, dqn_tasks_per_episode):
        if a > 0:
            ratios.append(d / a)
        else:
            ratios.append(0 if d == 0 else float('inf'))
    
    # Calculate standard deviation for results
    a_star_std = np.std(a_star_tasks_per_episode) if a_star_tasks_per_episode else 0
    dqn_std = np.std(dqn_tasks_per_episode) if dqn_tasks_per_episode else 0
    ratio_std = np.std([r for r in ratios if r != float('inf')]) if ratios else 0

    end_time = time.time()
    print(f"Benchmark completed in {end_time - start_time:.2f} seconds.")
    
    # Calculate performance ratio (RL / A*)
    if total_a_star_completed_tasks and total_a_star_completed_tasks > 0:
        performance_ratio = total_dqn_completed_tasks / total_a_star_completed_tasks
    else:
        performance_ratio = 0
    
    # Calculate average deliveries per episode
    avg_a_star = total_a_star_completed_tasks / eval_episodes if eval_episodes > 0 else 0
    avg_dqn = total_dqn_completed_tasks / eval_episodes if eval_episodes > 0 else 0
        
    # Print benchmark results
    print("\n=== BENCHMARK RESULTS ===")
    print(f"A* completed tasks: {total_a_star_completed_tasks} (avg: {avg_a_star:.2f} ± {a_star_std:.2f})")
    print(f"RL completed tasks: {total_dqn_completed_tasks} (avg: {avg_dqn:.2f} ± {dqn_std:.2f})")
    print(f"Performance ratio (RL/A*): {performance_ratio:.2f} ± {ratio_std:.2f}")
    
    return {
        "astar_tasks": total_a_star_completed_tasks,
        "rl_tasks": total_dqn_completed_tasks,
        "performance_ratio": performance_ratio,
        "avg_astar_tasks": avg_a_star,
        "avg_rl_tasks": avg_dqn,
        "astar_std": a_star_std,
        "rl_std": dqn_std,
        "ratio_std": ratio_std,
        "astar_tasks_per_episode": a_star_tasks_per_episode,
        "dqn_tasks_per_episode": dqn_tasks_per_episode,
        "episode_ratios": ratios
    }

def run_q_learning(env, full_model_path, n_steps=1000, debug_level=DEBUG_NONE, sampling_mode='argmax'):
    """
    Run trained Q-learning agent in the warehouse environment.
    """
    
    # Initialize agent in evaluation mode
    QL_agent = QLAgent(env, debug_level=debug_level, use_tensorboard=False)
    QL_agent.epsilon = 0.05  # Small epsilon for minimal exploration during evaluation
    
    # Load the trained model if it exists
    try:
        QL_agent.load_model(full_model_path)
        QL_agent.debug(DEBUG_INFO, f"Loaded trained model from {full_model_path}")
    except FileNotFoundError:
        QL_agent.debug(DEBUG_CRITICAL, f"Model file not found at {full_model_path}. Using untrained agent.")

    # Reset environment
    observations, _ = env.reset()
    
    total_rewards = 0
    completed_tasks = 0
    
    # Track action statistics to detect bias
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

    # Run for n_steps
    for step in range(n_steps):
        # Get actions for each agent
        actions = {}
        
        for agent in env.agents:
            # Use select_action with eval_mode=True to ensure consistent action selection
            action = QL_agent.select_action(
                observations[agent], 
                agent, 
                eval_mode=True,
                sampling_mode=sampling_mode
            )
            actions[agent] = action
            action_counts[action] += 1
            
        # Take actions in the environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Track metrics
        total_rewards += sum(rewards.values())
        
        # Update task completion for tracking
        completed_tasks = sum(env.completed_tasks.values())
        
        # Render the environment
        env.render()

        # Debug info
        QL_agent.debug(DEBUG_INFO, f"Step {step}, Tasks: {completed_tasks}")
        QL_agent.debug(DEBUG_INFO, f"Actions: {actions}")
        QL_agent.debug(DEBUG_INFO, f"Rewards: {rewards}")

        # Print action distribution every 20 steps
        if step > 0 and step % 20 == 0:
            QL_agent.debug(DEBUG_VERBOSE, "\nAction distribution so far:")
            total = sum(action_counts.values())
            for action, count in action_counts.items():
                action_name = ["LEFT", "DOWN", "RIGHT", "UP", "PICKUP", "DROPOFF", "WAIT"][action]
                percentage = (count / total) * 100
                QL_agent.debug(DEBUG_VERBOSE, f"  {action_name}: {count} ({percentage:.1f}%)")

        # Check if done
        if all(terminations.values()) or all(truncations.values()):
            break
    
    # Print final results
    QL_agent.debug(DEBUG_CRITICAL, f"Final score: {total_rewards}")
    QL_agent.debug(DEBUG_CRITICAL, f"Completed tasks: {completed_tasks}")
    
    # Print final action distribution
    QL_agent.debug(DEBUG_INFO, "\nFinal action distribution:")
    total = sum(action_counts.values())
    for action, count in action_counts.items():
        action_name = ["LEFT", "DOWN", "RIGHT", "UP", "PICKUP", "DROPOFF", "WAIT"][action]
        percentage = (count / total) * 100
        QL_agent.debug(DEBUG_INFO, f"  {action_name}: {count} ({percentage:.1f}%)")
    
    # Clean up
    import matplotlib.pyplot as plt
    plt.close('all')
    env.close()
    
    return completed_tasks
    

