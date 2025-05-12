# Standard library imports
import sys
import os
import time
import tty
import termios
import threading
import select

# Third party imports
import numpy as np
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# Local application imports
from .agent import QLAgent
from .qnet import QNetwork
from .replay import ReplayBuffer, PrioritizedReplayBuffer
from environment import env
from .benchmark import benchmark_environment
from .common import DEBUG_ALL, DEBUG_CRITICAL, DEBUG_INFO, DEBUG_VERBOSE, DEBUG_SPECIFIC, DEBUG_NONE, get_model_path, device


def setup_keyboard_listener():
    """Set up keyboard listener for interactive control."""
    # Create a dictionary to hold our control flags and terminal settings
    control = {
        'next_phase': False,
        'quit_training': False,
        'evaluate_model': False,
        'original_settings': None
    }
    
    def check_input():
        # Store original terminal settings so we can restore them later
        control['original_settings'] = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while True:
                if select.select([sys.stdin], [], [], 0.1)[0]:  # Added small timeout for better responsiveness
                    try:
                        key = sys.stdin.read(1)
                        if key == 'n':
                            # Force print to show immediately with flush=True
                            print("\nNext phase requested! Saving current model and proceeding to next phase...", flush=True)
                            control['next_phase'] = True
                        elif key == 'q':
                            print("\nQuit requested! Saving current model and stopping training...", flush=True)
                            control['quit_training'] = True
                        elif key == 'e':
                            print("\nEvaluation requested! Saving current model and running benchmark...", flush=True)
                            control['evaluate_model'] = True
                    except IOError:
                        # Handle potential IOError when reading from stdin
                        pass
                time.sleep(0.05)  # Shorter sleep time for better responsiveness
        except Exception as e:
            print(f"Input thread error: {e}")
        finally:
            # Restore terminal settings in the finally block
            if control['original_settings']:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, control['original_settings'])
                except:
                    pass  # Ignore errors during cleanup
    
    # Start input checking thread
    input_thread = threading.Thread(target=check_input, daemon=True)
    input_thread.start()
    
    # Give thread a moment to initialize
    time.sleep(0.2)
    
    # Return the control dictionary that will be updated by the thread
    return control

def initialize_agent(env, agent, debug_level, phase, sampling_mode, force_new_model=False):
    """
    Initialize or update agent for training.
    
    Args:
        env: Environment to train in
        agent: Existing agent or None to create new one
        debug_level: Debug verbosity level
        phase: Current training phase
        sampling_mode: Exploration strategy
        force_new_model: Whether to force creation of new model path and writer
    """
    if agent is None:
        # Create new agent
        return QLAgent(env, debug_level=debug_level, use_tensorboard=True, phase=phase)
    
    # Use existing agent with updated settings
    old_phase = agent.phase
    agent.env = env
    agent.phase = phase

    # Only generate new model name if forced or if phase changed
    if force_new_model or agent.phase != old_phase:
        agent.model_name = f"phase_{phase}_dqn_{time.strftime('%Y%m%d-%H%M%S')}.pth"
        agent.model_path = os.path.join(get_model_path(), agent.model_name)
        
        # Reset TensorBoard writer
        if hasattr(agent, 'writer') and agent.writer:
            agent.writer.close()
        agent.writer = SummaryWriter(log_dir=f"runs/{agent.model_name}")
        
        # Log hyperparameters
        hp_dict = {
            'phase': agent.phase,
            'alpha': agent.alpha,
            'gamma': agent.gamma,
            'epsilon_start': agent.epsilon,
            'epsilon_end': agent.epsilon_min,
            'epsilon_decay': agent.epsilon_decay,
            'hidden_size': agent.q_networks[env.agents[0]].fc1.out_features,
            'buffer_size': agent.memory.buffer_size,
            'batch_size': agent.batch_size,
            'update_freq': agent.update_freq,
            'tau': agent.tau,
            'frame_history': agent.frame_history,
            'use_per': agent.use_per,
            'use_softmax': agent.use_softmax,
            'use_qmix': getattr(agent, 'use_qmix', False),
            'sampling_mode': sampling_mode
        }
        param_str = "\n".join([f"{k}: {v}" for k, v in hp_dict.items()])
        agent.writer.add_text("Hyperparameters", param_str)
        
        # Reset performance tracking
        agent.completed_tasks = []
        agent.episode_rewards = []
        agent.t_step = 0
        agent.epsilon = agent.epsilon_start
        
        # Reset memory buffer
        if agent.use_per:
            agent.memory = PrioritizedReplayBuffer(agent.memory.buffer_size, agent.batch_size)
        else:
            agent.memory = ReplayBuffer(agent.memory.buffer_size, agent.batch_size)
    
    return agent

def run_evaluation(agent, phase, episode, debug_level, max_steps, eval_episodes, eval_history):
    """Run benchmark evaluation of current agent."""
    # Save current model for evaluation without modifying agent's model_path attribute
    temp_model_path = os.path.join(get_model_path(), f"temp_eval_{phase}.pth")
    agent.save_model(temp_model_path)
    print(f"\nEvaluating model at episode {episode}...")
    
    # Run benchmark
    benchmark_results = benchmark_environment(
        env_phase=phase,
        n_steps=max_steps,
        debug_level=debug_level,
        model_path=temp_model_path,  # Use temporary path
        eval_episodes=eval_episodes
    )
    
    # Clean up temporary file after benchmarking
    if os.path.exists(temp_model_path):
        try:
            os.remove(temp_model_path)
        except Exception as e:
            print(f"Warning: Could not remove temporary evaluation file {temp_model_path}: {e}")
    
    if not benchmark_results:
        return False, 0, eval_history, False
    
    # Update evaluation history
    eval_history['episodes'].append(episode)
    eval_history['performance_ratio'].append(benchmark_results['performance_ratio'])
    eval_history['avg_rl_tasks'].append(benchmark_results['avg_rl_tasks'])
    eval_history['avg_astar_tasks'].append(benchmark_results['avg_astar_tasks'])
    
    # Log to TensorBoard - using existing writer
    if hasattr(agent, 'writer'):
        agent.writer.add_scalar("Evaluation/A*_Tasks", benchmark_results['avg_astar_tasks'], episode)
        agent.writer.add_scalar("Evaluation/RL_Tasks", benchmark_results['avg_rl_tasks'], episode)
        agent.writer.add_scalar("Evaluation/Performance_Ratio", benchmark_results['performance_ratio'], episode)
        agent.writer.add_scalar("Evaluation/A*_StdDev", benchmark_results['astar_std'], episode)
        agent.writer.add_scalar("Evaluation/RL_StdDev", benchmark_results['rl_std'], episode)
        agent.writer.add_scalar("Evaluation/Ratio_StdDev", benchmark_results['ratio_std'], episode)
        
        # Log distributions
        agent.writer.add_histogram("Evaluation/A*_Distribution", 
                                np.array(benchmark_results['astar_tasks_per_episode']), episode)
        agent.writer.add_histogram("Evaluation/RL_Distribution", 
                                np.array(benchmark_results['dqn_tasks_per_episode']), episode)
        agent.writer.add_histogram("Evaluation/Ratio_Distribution", 
                                np.array([r for r in benchmark_results['episode_ratios'] if r != float('inf')]), episode)
    
    # Handle best model saving - without changing model path
    improved = False
    current_performance = benchmark_results['avg_rl_tasks']
    
    if len(eval_history['avg_rl_tasks']) > 1:
        best_previous = max(eval_history['avg_rl_tasks'][:-1])
        if current_performance > best_previous:
            print(f"New best model! {current_performance:.2f} deliveries vs previous best {best_previous:.2f}")
            best_model_path = agent.model_path.replace('.pth', '_best.pth')
            agent.save_model(best_model_path)
            improved = True
        else:
            print(f"No improvement: {current_performance:.2f} deliveries vs best {best_previous:.2f}")
    elif len(eval_history['avg_rl_tasks']) == 1:
        # First evaluation, save as best
        best_model_path = agent.model_path.replace('.pth', '_best.pth')
        agent.save_model(best_model_path)
        improved = True
        
    return True, current_performance, eval_history, improved

def run_episode(agent, max_steps, episode, sampling_mode):
    """Run a single training episode, using QMIX if the agent supports it."""
    # Initialize timing and metrics
    t_start = time.time()
    env_time = 0
    select_time = 0
    learn_time = 0
    
    # Reset environment
    t_reset = time.time()
    observations, _ = agent.env.reset()
    env_time += time.time() - t_reset
    
    # Get initial global state if agent uses QMIX
    if hasattr(agent, 'use_qmix') and agent.use_qmix:
        global_state, _ = agent.env.get_global_state()
    
    # Track episode data
    score = 0
    episode_actions = {agent_id: [] for agent_id in agent.env.agents}
    
    # Determine sampling mode
    current_sampling_mode = sampling_mode
    if sampling_mode == 'auto':
        current_sampling_mode = 'sample' if episode < 50 else 'argmax'
    
    # Run steps
    for step in range(max_steps):
        # Select actions
        t_select = time.time()
        actions = agent.select_action_batch(observations, sampling_mode=current_sampling_mode)
        select_time += time.time() - t_select
        
        # Record actions
        for agent_id, action in actions.items():
            episode_actions[agent_id].append(action)
            
        # Take environment step
        t_env = time.time()
        next_observations, rewards, terminations, truncations, infos = agent.env.step(actions)
        env_time += time.time() - t_env
        
        # Agent learning
        t_learn = time.time()
        # Check if agent uses QMIX
        if hasattr(agent, 'use_qmix') and agent.use_qmix:
            # Get next global state
            next_global_state, _ = agent.env.get_global_state()
            # Use QMIX step with global state
            agent.step_qmix(observations, actions, rewards, next_observations, terminations,
                          global_state, next_global_state)
            # Update global state for next step
            global_state = next_global_state
        else:
            # Standard DQN step
            agent.step(observations, actions, rewards, next_observations, terminations)
            
        learn_time += time.time() - t_learn
        
        # Update state and score
        score += sum(rewards.values())
        observations = next_observations
        
        # Check for episode termination
        if all(terminations.values()) or all(truncations.values()):
            break
    
    # Get episode metrics
    episode_duration = time.time() - t_start
    episode_deliveries = sum(agent.env.completed_tasks.values())
    
    return score, episode_actions, episode_deliveries, {
        'env_time': env_time,
        'select_time': select_time,
        'learn_time': learn_time,
        'total_time': episode_duration
    }

def check_early_stopping(agent, env, episode, metrics_history, no_improvement):
    """Check if training should stop early based on learning metrics."""
    if episode <= 10:
        return no_improvement, False
    
    # Get latest loss and TD error values
    recent_losses = []
    recent_td_errors = []
    
    for agent_id in env.agents:
        if hasattr(agent, 'last_loss') and agent_id in agent.last_loss:
            recent_losses.append(agent.last_loss[agent_id])
        if hasattr(agent, 'last_td_error') and agent_id in agent.last_td_error:
            recent_td_errors.append(agent.last_td_error[agent_id])
    
    if not (recent_losses and recent_td_errors):
        return no_improvement, False
    
    # Calculate average across agents
    avg_loss = sum(recent_losses) / len(recent_losses)
    avg_td_error = sum(recent_td_errors) / len(recent_td_errors)
    
    # Update history
    metrics_history['loss'].append(avg_loss)
    metrics_history['td_error'].append(avg_td_error)
    
    # Calculate exponential moving average
    if 'ema_loss' not in metrics_history:
        metrics_history['ema_loss'] = avg_loss
        metrics_history['ema_td_error'] = avg_td_error
    else:
        metrics_history['ema_loss'] = 0.95 * metrics_history['ema_loss'] + 0.05 * avg_loss
        metrics_history['ema_td_error'] = 0.95 * metrics_history['ema_td_error'] + 0.05 * avg_td_error
    
    # Calculate learning score
    learning_score = (0.6 * metrics_history['ema_loss'] + 0.4 * metrics_history['ema_td_error'])
    
    # Log the score
    agent.writer.add_scalar("Training/LearningScore", learning_score, episode)
    
    # Check for stabilization after significant training
    if episode > 1000:
        if 'recent_scores' not in metrics_history:
            metrics_history['recent_scores'] = deque(maxlen=300)
        
        metrics_history['recent_scores'].append(learning_score)
        
        if len(metrics_history['recent_scores']) >= 200:
            # Calculate slope using moving average
            window_size = 50
            smoothed_scores = []
            recent = list(metrics_history['recent_scores'])
            
            for i in range(len(recent) - window_size + 1):
                smoothed_scores.append(sum(recent[i:i+window_size]) / window_size)
            
            x = np.array(range(len(smoothed_scores)))
            y = np.array(smoothed_scores)
            slope = np.polyfit(x, y, 1)[0]
            
            agent.writer.add_scalar("Training/LearningScoreSlope", slope, episode)
            
            # Check if learning has stabilized
            slope_threshold = 0.00002
            if abs(slope) < slope_threshold:
                no_improvement += 1
                return no_improvement, False
            
            if episode > 1500:
                # Check for absolute stability
                recent_100 = list(metrics_history['recent_scores'])[-100:]
                max_val = max(recent_100)
                min_val = min(recent_100)
                relative_range = (max_val - min_val) / np.mean(recent_100)
                
                if relative_range < 0.03:
                    no_improvement += 3
    
    return no_improvement, False

def train_DQN(env, agent=None, n_episodes=1000, max_steps=1000, debug_level=DEBUG_CRITICAL, 
             save_every=100, phase=1, sampling_mode='auto', use_early_stopping=False,
             eval_every=200, eval_episodes=100, use_qmix=False):
    """
    Train Q-learning agent with interactive control and periodic evaluation.
    
    Args:
        env: Training environment
        agent: Optional pre-initialized QLAgent 
        n_episodes: Maximum number of training episodes
        max_steps: Maximum steps per episode
        debug_level: Verbosity level for logging
        save_every: Episodes between model saves
        phase: Curriculum training phase
        sampling_mode: Action selection mode ('auto', 'argmax', or 'sample')
        use_early_stopping: Whether to use early stopping based on performance
        eval_every: Episodes between evaluations
        eval_episodes: Number of episodes for each evaluation
        use_qmix: Whether to use QMIX for training (default: False) - only used if creating a new agent
        
    Returns:
        agent: Trained QLAgent
        exit_reason: Reason for training termination
        eval_history: Dictionary of evaluation metrics
    """
    # Verify hardware
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Set up CPU threads
    if hasattr(torch, 'set_num_threads'):
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        print(f"Using {num_threads} CPU threads for data loading.")
    
    # Initialize agent and keyboard controls
    if agent is None:
        ql_agent = QLAgent(
            env, debug_level=debug_level, phase=phase, 
            use_tensorboard=True, use_qmix=use_qmix
        )
    else:
        ql_agent = agent
        ql_agent.env = env
        
    # Log whether QMIX is being used
    if hasattr(ql_agent, 'use_qmix') and ql_agent.use_qmix:
        print("Using QMIX for centralized training with decentralized execution")
    
    control = setup_keyboard_listener()
    print("\nTraining control: Press 'n' to move to next phase, 'e' to evaluate current model, 'q' to quit training")
    
    # Initialize tracking variables
    eval_history = {'episodes': [], 'performance_ratio': [], 'avg_rl_tasks': [], 'avg_astar_tasks': []}
    scores = []
    timing = {'env': 0, 'select': 0, 'learn': 0, 'total': 0}
    deliveries_per_episode = []
    last_10_deliveries = deque(maxlen=10)
    metrics_history = {'loss': [], 'td_error': []}
    best_learning_score = float('inf')
    no_improvement = 0
    best_episode = 0
    avg_deliveries_history = []
    
    # Training loop
    exit_reason = "completed"
    episode = 1
    
    while episode <= n_episodes:
        # Check for control requests - FIRST PRIORITY
        if control['next_phase']:
            control['next_phase'] = False  # Reset flag immediately
            ql_agent.save_model(ql_agent.model_path)
            print(f"\nMoving to next phase at episode {episode}")
            exit_reason = "nextphase"
            break
            
        if control['quit_training']:
            control['quit_training'] = False  # Reset flag immediately
            ql_agent.save_model(ql_agent.model_path)
            print(f"\nQuitting training at episode {episode}")
            exit_reason = "quit"
            break
        
        # Evaluation checkpoint
        if control['evaluate_model'] or (episode % eval_every == 0 and episode > 0):
            # Reset flag immediately
            control['evaluate_model'] = False
            
            # Save current state of stdin
            old_stdin_settings = None
            try:
                # Save current terminal settings
                old_stdin_settings = termios.tcgetattr(sys.stdin)
                # Reset terminal to sane state temporarily
                os.system('stty sane')
            except Exception as e:
                # Could be running in an environment without proper terminal
                print(f"Note: Could not save terminal settings: {e}")
            
            # Run evaluation
            success, performance, eval_history, improved = run_evaluation(
                ql_agent, phase, episode, debug_level, 
                max_steps, eval_episodes, eval_history
            )
            
            # Get user input for manual evaluation
            if episode % eval_every != 0:
                # Handle the input with a completely isolated approach
                try:
                    # Make absolutely sure terminal is in a sane state
                    os.system('stty sane')
                    
                    # Flush any pending input
                    termios.tcflush(sys.stdin, termios.TCIOFLUSH)
                    
                    valid_input = False
                    while not valid_input:
                        print("\nEvaluation complete. Press 'c' to continue training, 'n' for next phase, 'q' to quit: ", end='', flush=True)
                        
                        # Use simpler direct read approach
                        ch = sys.stdin.read(1)
                        print(ch)  # Echo the character
                        
                        if ch.lower() == 'c':
                            valid_input = True
                            print("\nResuming training...")
                        elif ch.lower() == 'n':
                            valid_input = True
                            exit_reason = "nextphase"
                        elif ch.lower() == 'q':
                            valid_input = True
                            exit_reason = "quit"
                        else:
                            print(f"Unrecognized command: '{ch}'. Please try again.")
                    
                except Exception as e:
                    print(f"Error handling input: {e}")
                    print("Resuming training...")
                finally:
                    # Restore terminal settings if we have them
                    if old_stdin_settings is not None:
                        try:
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_stdin_settings)
                        except Exception as e:
                            print(f"Could not restore terminal settings: {e}")
                            # Force reset terminal as a fallback
                            os.system('stty sane')
                    else:
                        # Force reset terminal as a fallback
                        os.system('stty sane')
                    
                # If we got here after breaking from the while loop for 'n' or 'q', 
                # we need to also break from the episode loop
                if exit_reason in ["nextphase", "quit"]:
                    break
            else:
                # This is a scheduled evaluation
                try:
                    # Make sure terminal is back to normal
                    os.system('stty sane')
                    
                    print(f"\nScheduled evaluation at episode {episode} complete.")
                    print("\nResuming training...")
                    
                    # Update improvement counter
                    if not improved:
                        no_improvement += eval_every
                    else:
                        no_improvement = 0
                        best_episode = episode
                    
                    # Early stopping check
                    if use_early_stopping and no_improvement >= 500 and episode > eval_every:
                        print(f"\nEarly stopping at episode {episode}. No improvement for {no_improvement} episodes.")
                        print(f"Restoring best model from episode {best_episode}")
                        
                        best_model_path = ql_agent.model_path.replace('.pth', '_best.pth')
                        if os.path.exists(best_model_path):
                            ql_agent.load_model(best_model_path)
                        exit_reason = "earlystopping"
                        break
                finally:
                    # Restore terminal settings if we have them
                    if old_stdin_settings is not None:
                        try:
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_stdin_settings)
                        except:
                            # Force reset terminal as a fallback
                            os.system('stty sane')
                    else:
                        # Force reset terminal as a fallback
                        os.system('stty sane')
            
            # Move to next episode after evaluation
            episode += 1
            continue
        
        # Run single episode
        score, episode_actions, episode_deliveries, episode_timing = run_episode(
            ql_agent, max_steps, episode, sampling_mode
        )
        
        # Check control flags AGAIN after episode - critical for responsiveness
        if control['next_phase']:
            # Don't increment episode - continue to the top of the loop
            # where the flag will be detected and properly handled
            continue
        
        if control['quit_training']:
            # Don't increment episode - continue to the top of the loop
            continue
            
        if control['evaluate_model']:
            # Don't increment episode - continue to the top of the loop
            continue
        
        # Update exploration parameters
        ql_agent.update_epsilon()
        if hasattr(ql_agent, 'update_temperature'):
            ql_agent.update_temperature()
        
        # Update tracking metrics
        timing['env'] += episode_timing['env_time']
        timing['select'] += episode_timing['select_time']
        timing['learn'] += episode_timing['learn_time']
        timing['total'] += episode_timing['total_time']
        
        scores.append(score)
        ql_agent.episode_rewards.append(score)
        ql_agent.completed_tasks.append(episode_deliveries)
        deliveries_per_episode.append(episode_deliveries)
        last_10_deliveries.append(episode_deliveries)
        
        # Calculate rolling statistics
        avg_deliveries = sum(last_10_deliveries) / len(last_10_deliveries)
        avg_deliveries_history.append(avg_deliveries)
        
        # Print progress
        ql_agent.debug(DEBUG_CRITICAL, 
            f"Episode {episode:4}/{n_episodes:<4} | "
            f"Score: {score:8.1f} | "
            f"Deliveries: {episode_deliveries:3} | "
            f"10-ep Avg: {avg_deliveries:5.1f} | "
            f"Epsilon: {ql_agent.epsilon:.4f}")
        
        # Log to TensorBoard
        ql_agent.writer.add_scalar("Performance/Reward", score, episode)
        ql_agent.writer.add_scalar("Performance/Deliveries", episode_deliveries, episode)
        ql_agent.writer.add_scalar("Training/Epsilon", ql_agent.epsilon, episode)
        ql_agent.writer.add_scalar("Training/NoImprovement", no_improvement, episode)
        
        if episode >= 10:
            avg_reward = sum(scores[-10:]) / 10
            ql_agent.writer.add_scalar("Performance/Avg10Reward", avg_reward, episode)
            ql_agent.writer.add_scalar("Performance/Avg10Deliveries", avg_deliveries, episode)
        
        # Log action distribution
        ql_agent.log_action_distribution(episode_actions, episode)
        
        # Check for early stopping
        if use_early_stopping:
            no_improvement, should_stop = check_early_stopping(
                ql_agent, env, episode, metrics_history, no_improvement
            )
            if should_stop:
                print(f"\nEarly stopping at episode {episode} based on learning stability.")
                exit_reason = "earlystopping"
                break
        
        # Save model periodically
        if episode % save_every == 0:
            ql_agent.save_model(ql_agent.model_path)
            
        # Print timing info occasionally
        if episode % 10 == 0:
            print(f"\nTiming breakdown (avg per episode):")
            print(f"  Environment time: {timing['env']/episode:.4f}s")
            print(f"  Action selection time: {timing['select']/episode:.4f}s")
            print(f"  Agent learning time: {timing['learn']/episode:.4f}s")
            print(f"  Episode duration: {timing['total']/episode:.4f}s")
            print(f"  Press 'n' to move to next phase, 'e' to evaluate, 'q' to quit training\n")
        
        # Check control flags ONE FINAL TIME before incrementing episode
        # This provides maximum responsiveness to keyboard input
        if control['next_phase'] or control['quit_training'] or control['evaluate_model']:
            continue  # Don't increment episode, go back to top of loop
            
        # Increment episode counter
        episode += 1
        
    # End of training
    ql_agent.save_model(ql_agent.model_path)
    
    # Run final evaluation if appropriate
    if episode > eval_every and exit_reason != "quit":
        print("\nRunning final evaluation...")
        final_benchmark = benchmark_environment(
            env_phase=phase, n_steps=max_steps, 
            debug_level=debug_level, model_path=ql_agent.model_path,
            eval_episodes=eval_episodes
        )
        
        if final_benchmark:
            print("\nFinal evaluation results:")
            print(f"  A* completed tasks: {final_benchmark['astar_tasks']}")
            print(f"  RL completed tasks: {final_benchmark['rl_tasks']}")
            print(f"  Performance ratio (RL/A*): {final_benchmark['performance_ratio']:.2f}")
    
    # Clean up
    try:
        env.close()
    except Exception as e:
        print(f"Error closing environment: {e}")
    
    return ql_agent, exit_reason, eval_history

def create_stage1_env():
    """Create Stage 1 basic navigation environment."""
    return env(grid_size=(5, 5), n_agents=1, n_humans=0, num_shelves=0, 
              num_pickup_points=1, num_dropoff_points=1, render_mode="human", env_name="stage1")

def create_stage2_env():
    """Create Stage 2 environment with obstacles."""
    return env(grid_size=(10, 8), n_agents=1, n_humans=0, num_shelves=16, 
              num_pickup_points=1, num_dropoff_points=1, render_mode="human", env_name="stage2")

def create_stage3_env():
    """Create Stage 3 multi-agent warehouse environment."""
    return env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=6, n_humans=10, 
              num_shelves=2048, num_pickup_points=3, num_dropoff_points=2, render_mode="human", env_name="stage3")

def initialize_multi_agent_networks(agent, new_env, source_agent_id="agent_0"):
    """Copy network architecture and weights from one agent to all others."""

    if agent.use_parameter_sharing:
        # No need to copy networks if using parameter sharing, all agents share the same network
        return agent
    
    # Extract network architecture parameters from source agent
    source_network = agent.q_networks[source_agent_id]
    source_state_dict = source_network.state_dict()
    input_channels = source_network.actual_channels
    height = source_network.height
    width = source_network.width
    frame_history = source_network.frame_history
    hidden_size = source_network.fc1.out_features // 2  # Divide by 2 since it uses hidden_size*2
        
    # Initialize all new agent networks with the source knowledge
    for agent_id in new_env.agents:
        if agent_id != source_agent_id:
            agent.q_networks[agent_id] = QNetwork(
                (input_channels//frame_history, height, width), 
                7, hidden_size, frame_history
            ).to(device)
            
            agent.target_networks[agent_id] = QNetwork(
                (input_channels//frame_history, height, width), 
                7, hidden_size, frame_history
            ).to(device)

            agent.q_networks[agent_id].load_state_dict(source_state_dict)
            agent.target_networks[agent_id].load_state_dict(source_state_dict)

            agent.optimizers[agent_id] = torch.optim.AdamW(
                agent.q_networks[agent_id].parameters(), 
                lr=agent.alpha, 
                eps=1e-5, 
                weight_decay=0.0001
            )
            
    return agent

def train_stage(stage_num, stage_names, stage_env, agent, max_steps, save_every, use_qmix=False):
    """Train a specific curriculum stage and handle its outcomes."""
    print(f"\n=== STAGE {stage_num}: {stage_names[stage_num]} ===")
    print("Press 'n' during training to advance to next phase, 'e' to evaluate current model, 'q' to quit training completely")
    
    # Configure agent parameters for this stage
    if agent is not None:
        # Update the agent's phase to match the current stage
        agent.phase = stage_num
        agent.epsilon = 0.9
        
        # Update model name for this phase
        agent.model_name = f"phase_{stage_num}_dqn_{agent.start_time}.pth"
        agent.model_path = os.path.join(get_model_path(), agent.model_name)
        
        # Update TensorBoard writer for the new phase
        if agent.use_tensorboard and hasattr(agent, 'writer'):
            agent.writer.close()
            agent.writer = SummaryWriter(log_dir=f"runs/{agent.model_name}")
            
            # Log phase transition
            agent.writer.add_text("Training", 
                                f"Entered Stage {stage_num}: {stage_names[stage_num]} with QMIX={use_qmix}")
    
    # Train for this stage (use_qmix only used if agent is None)
    agent, exit_reason, eval_history = train_DQN(
        stage_env, 
        agent=agent, 
        n_episodes=200000,
        max_steps=max_steps, 
        save_every=save_every, 
        phase=stage_num,  # Pass the current stage number
        use_qmix=use_qmix
    )
    
    # Get model path for this stage
    model_path = agent.model_path
    
    if exit_reason != "quit":
        # Run benchmark
        print(f"\nBenchmarking Stage {stage_num} against A*...")
        benchmark_results = benchmark_environment(
            env_phase=stage_num, 
            n_steps=max_steps, 
            debug_level=DEBUG_NONE, 
            model_path=model_path
        )
    else:
        benchmark_results = None
    
    # Modified behavior: If exit_reason is "nextphase", proceed automatically without confirmation
    if exit_reason == "nextphase" and stage_num < 3:
        # Skip confirmation and just print info
        print(f"\nMoving to next phase at episode {agent.t_step}")
        print(f"\nProceeding to Stage {stage_num + 1}: {stage_names[stage_num + 1]}...")
        # exit_reason already set to "nextphase", no need to change
    
    return agent, exit_reason, eval_history, benchmark_results, model_path

def train_DQN_curriculum(target_env, n_episodes=1000, max_steps=1000, debug_level=DEBUG_CRITICAL, 
                        save_every=100, model_path=get_model_path()):
    """
    Train Q-learning agent with curriculum learning across multiple stages.
    Uses QMIX for all stages to leverage global state information even in single-agent settings.
    """
    # Setup 
    STAGE_NAMES = {
        1: "Basic Navigation Training (with QMIX)",
        2: "Navigation with Obstacles (with QMIX)",
        3: "Multi-agent with Dynamic Obstacles (with QMIX)"
    }
    
    STAGE_STEPS = {
        1: 100,
        2: 150,
        3: 300
    }
    
    STAGE_SAVE_FREQ = {
        1: 50,
        2: 25,
        3: 25
    }
    
    # Create models directory if it doesn't exist
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created models directory: {model_dir}")
    print(f"Using model dir: {model_dir}")
    
    # Initialize result tracking
    results = {
        "stage1_benchmark": None,
        "stage2_benchmark": None,
        "stage3_benchmark": None,
        "stage1_eval_history": None,
        "stage2_eval_history": None,
        "stage3_eval_history": None,
    }
    
    # STAGE 1: Basic Navigation - repeatable if requested
    current_stage = 1
    agent = None
    
    while current_stage <= 3:
        if current_stage == 1:
            # STAGE 1: Basic Navigation with QMIX
            stage_env = create_stage1_env()
            
            # If agent already exists, configure it for QMIX
            if agent is not None:
                configure_agent_for_qmix(agent, stage_env)
                
            agent, exit_reason, eval_history, benchmark, stage_model_path = train_stage(
                1, STAGE_NAMES, stage_env, agent, STAGE_STEPS[1], STAGE_SAVE_FREQ[1], use_qmix=True
            )
            
            results["stage1_eval_history"] = eval_history
            results["stage1_benchmark"] = benchmark
            
        elif current_stage == 2:
            # STAGE 2: Navigation with obstacles with QMIX
            stage_env = create_stage2_env()
            
            # If transitioning between stages, ensure QMIX is properly configured
            if agent is not None:
                configure_agent_for_qmix(agent, stage_env)
                
            agent, exit_reason, eval_history, benchmark, stage_model_path = train_stage(
                2, STAGE_NAMES, stage_env, agent, STAGE_STEPS[2], STAGE_SAVE_FREQ[2], use_qmix=True
            )
            
            results["stage2_eval_history"] = eval_history
            results["stage2_benchmark"] = benchmark
            
        else:  # current_stage == 3
            # STAGE 3: Multi-agent with dynamic obstacles with QMIX
            stage_env = create_stage3_env()
            
            # Initialize multi-agent networks from stage 2 knowledge
            agent = initialize_multi_agent_networks(agent, stage_env)
            
            # Configure QMIX for multiple agents
            configure_agent_for_qmix(agent, stage_env)
            
            agent, exit_reason, eval_history, benchmark, stage_model_path = train_stage(
                3, STAGE_NAMES, stage_env, agent, STAGE_STEPS[3], STAGE_SAVE_FREQ[3], use_qmix=True
            )
            
            results["stage3_eval_history"] = eval_history
            results["stage3_benchmark"] = benchmark
        
        # Handle exit reason
        if exit_reason == "quit":
            print(f"Training terminated by user request after Stage {current_stage}")
            break
        elif exit_reason == "nextphase":
            current_stage += 1  # Move to next stage
            if agent is not None:
                # Explicitly update phase attribute when moving to next stage
                agent.phase = current_stage
                print(f"Updated agent phase to {current_stage}")
        elif exit_reason == "continue":
            # Stay in current stage - do nothing
            pass
        else:  # completed or early_stopping
            current_stage += 1  # Move to next stage by default

    # Save final model
    if agent is not None:
        agent.save_model(f"dqn_{agent.start_time}_final.pth")
        print(f"Final model saved as: dqn_{agent.start_time}_final.pth")
    
    # Print summary results
    print("\n=== CURRICULUM LEARNING COMPLETE ===")
    print("Performance Summary:")
    for stage in range(1, 4):
        benchmark = results[f"stage{stage}_benchmark"]
        if benchmark:
            print(f"Stage {stage}: A* Tasks: {benchmark['astar_tasks']}, " +
                  f"RL Tasks: {benchmark['rl_tasks']}, " +
                  f"Ratio: {benchmark['performance_ratio']:.2f}")

    return results

def configure_agent_for_qmix(agent, env):
    """
    Configure an existing agent to use QMIX with the given environment.
    This handles both initial setup and reconfiguration when changing environments.
    
    Args:
        agent: The QLAgent to configure
        env: The environment to use for QMIX configuration
    """
    print(f"Configuring agent for QMIX with {len(env.agents)} agents")
    
    # Enable QMIX
    agent.use_qmix = True

    # Update agent's phase if environment has changed significantly
    # (This is important when transitioning between stages)
    if (hasattr(agent, 'prev_env_size') and 
        agent.prev_env_size != env.grid_size):
        print(f"Environment size changed from {agent.prev_env_size} to {env.grid_size}")
    
    # Get sample global state to determine dimensions
    dummy_state, _ = env.get_global_state()
    state_dim = dummy_state.shape[0]
    
    # Check if we need to create or resize the mixer
    needs_new_mixer = (
        not hasattr(agent, 'mixer') or
        agent.mixer is None or
        # Check if number of agents changed
        agent.mixer.num_agents != len(env.agents) or
        # Check if state dimension changed
        agent.mixer.state_dim != state_dim
    )
    
    if needs_new_mixer:
        print(f"Creating new mixer network: {len(env.agents)} agents, state_dim={state_dim}")
        # Initialize QMIX networks
        from .qnet import QMixNetwork
        agent.mixer = QMixNetwork(len(env.agents), state_dim).to(device)
        agent.target_mixer = QMixNetwork(len(env.agents), state_dim).to(device)
        agent.target_mixer.load_state_dict(agent.mixer.state_dict())
        agent.mixer_optimizer = torch.optim.AdamW(
            agent.mixer.parameters(), lr=agent.alpha, eps=1e-5, weight_decay=0.0001
        )
    
    # Always recreate the replay buffer when changing environments
    # This ensures we don't keep experiences from a different environment
    from .replay import QMIXReplayBuffer
    buffer_size = agent.memory.buffer_size if hasattr(agent, 'memory') else 50000
    batch_size = agent.batch_size if hasattr(agent, 'batch_size') else 64
    use_per = agent.use_per if hasattr(agent, 'use_per') else True
    
    # Initialize QMIX replay buffer
    agent.qmix_memory = QMIXReplayBuffer(buffer_size, batch_size, use_priority=use_per)
    
    # Store agent order for consistent batch processing
    agent.agent_ids = sorted(list(env.agents))
    
    # Log QMIX configuration
    if hasattr(agent, 'writer') and agent.writer is not None:
        if len(env.agents) == 1:
            agent.writer.add_text("Training", f"Using QMIX for single agent with global state ({state_dim} dims)")
        else:
            agent.writer.add_text("Training", f"Using QMIX for {len(env.agents)} agents with global state ({state_dim} dims)")


