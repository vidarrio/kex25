import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from collections import namedtuple, deque
from subprocess import call
import time
from torch.cuda.amp import autocast
from environment import env
from algorithms import run_a_star

# Debug levels
DEBUG_NONE = 0
DEBUG_CRITICAL = 1
DEBUG_INFO = 2
DEBUG_VERBOSE = 3
DEBUG_ALL = 4
DEBUG_SPECIFIC = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Get path to models directory (src/models)
def get_model_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, '..', 'models')
    model_name = 'q_learning_model.pth'
    model_path = os.path.join(models_dir, model_name)
    return model_path

# Define experience tuple type
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class QNetwork(nn.Module):
    """Neural network for Q-value approximation."""

    def __init__(self, input_size, action_size, hidden_size=64):
        """
        Initialize network parameters.

        Args:
            input_size: Size of the input state.
            action_size: Number of possible actions.
            hidden_size: Size of the hidden layer.
        """

        super(QNetwork, self).__init__()

        # Extract dimensions from input size
        channels, height, width = input_size

        # Input layer
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Calculate flattened size after convolutional layers
        flattened_size = 32 * height * width

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x: Input state.
        """

        # Apply convolutions 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
            
        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer (no activation, we get raw Q-values)
        return self.fc3(x)
        
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """
        Initialize the replay buffer.
        
        Args:
            buffer_size: Maximum size of the buffer.
            batch_size: Size of training batch.
        """

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """
        Add new experience to the buffer memory.
        """

        e = Experience(state, action, reward, next_state, done)
        
        # Prioritize successful experiences
        if reward > 10.0: # task completion
            # Add experience multiple times
            for _ in range(3):
                self.memory.append(e)
        else:
            # Add experience once
            self.memory.append(e)

    def sample(self):
        """
        Randomly sample a batch of experiences from the buffer.
        """

        experiences = random.sample(self.memory, k=self.batch_size)

        # Process all states together before transferring to GPU
        states = np.stack([e.state for e in experiences])
        actions = np.vstack([e.action for e in experiences])
        rewards = np.vstack([e.reward for e in experiences])
        next_states = np.stack([e.next_state for e in experiences])
        dones = np.vstack([e.done for e in experiences]).astype(np.uint8)

        # Single transfer to GPU using pinned memory
        states = torch.from_numpy(states).pin_memory().to(device, non_blocking=True)
        actions = torch.from_numpy(actions).long().pin_memory().to(device, non_blocking=True)
        rewards = torch.from_numpy(rewards).float().pin_memory().to(device, non_blocking=True)
        next_states = torch.from_numpy(next_states).pin_memory().to(device, non_blocking=True)
        dones = torch.from_numpy(dones).float().pin_memory().to(device, non_blocking=True)


        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        Return the current size of the buffer.
        """

        return len(self.memory)
        
class QLAgent:
    """
    Q-learning agent for warehouse navigation.
    """



    def __init__(self, env, debug_level=DEBUG_NONE,
                 alpha=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1,
                 epsilon_decay=0.99, hidden_size=64, buffer_size=20000, batch_size=64,
                 update_freq=8, tau=0.001):
        """
        Initialize the Q-learning agent.

        Args:
            env: Warehouse environment.
            debug_level: Debug level (default: DEBUG_NONE).
            alpha: Learning rate (default: 0.01).
            gamma: Discount factor (default: 0.99).
            epsilon_start: Initial exploration rate (default: 1.0).
            epsilon_end: Final exploration rate (default: 0.1).
            epsilon_decay: Decay rate for exploration (default: 0.995).
            hidden_size: Size of the hidden layer (default: 128).
            buffer_size: Size of the replay buffer (default: 10000).
            batch_size: Size of the training batch (default: 64).
            update_freq: Frequency of model updates (default: 4).
            tau: Target network update rate (default: 0.001).
        """
        
        self.env = env
        self.debug_level = debug_level

        # Get observation shape from environment
        observation_shape = env.observation_size
        observation_channels = 10

        # State and action dimentions
        self.state_size = (observation_channels, *observation_shape)
        self.action_size = 7 # 4 movement, 3 actions (pickup, dropoff, wait)

        # Initialize Q-networks for each agent
        self.q_networks = {}
        self.target_networks = {}
        self.optimizers = {}

        for agent in env.agents:
            self.q_networks[agent] = QNetwork(self.state_size, self.action_size, hidden_size).to(device)
            self.target_networks[agent] = QNetwork(self.state_size, self.action_size, hidden_size).to(device)
            # Copy weights from Q-network to target network
            self.target_networks[agent].load_state_dict(self.q_networks[agent].state_dict())
            self.optimizers[agent] = torch.optim.Adam(self.q_networks[agent].parameters(), lr=alpha)

        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size, batch_size)

        # Initialize hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_freq = update_freq
        self.tau = tau
        self.batch_size = batch_size

        # Initilize step counter
        self.t_step = 0

        # Tracking training progress
        self.completed_tasks = []
        self.episode_rewards = []

    def debug(self, level, message):
        """
        Print debug messages based on the debug level.
        """

        if self.debug_level >= level:
            print(f"[DEBUG {level}] {message}")

    def step(self, states, action, reward, next_state, done):
        """
        Store experience to memory, sample from memory, and learn.
        """
        # Add experience to replay buffer for each agent
        for agent in self.env.agents:
            if agent in states and agent in next_state:
                self.memory.add(
                    states[agent], 
                    action[agent],
                    reward[agent],
                    next_state[agent],
                    done[agent]
                )

        # Increment step counter
        self.t_step += 1
        
        # Update model less often for multiple agents
        update_freq = 4 if len(self.env.agents) == 1 else self.update_freq
        
        # Learn every update_freq steps
        if self.t_step % update_freq == 0 and len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            agent = self.env.agents[self.t_step % len(self.env.agents)]
            self._learn(agent, experiences)

    def select_action(self, state, agent, eval_mode=False):
        """Return action for given state using epsilon-greedy policy with proper masking."""
        
        # Early decision for pickup/dropoff at goal states
        is_at_pickup = state[5, 2, 2] > 0.5  # Center cell is pickup
        is_at_dropoff = state[6, 2, 2] > 0.5  # Center cell is dropoff
        is_carrying = state[7, 2, 2] > 0.5
        
        # For explicit goal states, return deterministic action
        if is_at_pickup and not is_carrying:
            return 4  # pickup
        if is_at_dropoff and is_carrying:
            return 5  # dropoff
        
        # Boundary detection from state observation
        # In the observation, channels 0-3 of cells outside bounds will be 0
        at_left_edge = np.sum(state[2, 2, 0]) > 0  # Shelf or edge to the left
        at_right_edge = np.sum(state[2, 2, 4]) > 0  # Shelf or edge to the right 
        at_top_edge = np.sum(state[2, 0, 2]) > 0  # Shelf or edge above
        at_bottom_edge = np.sum(state[2, 4, 2]) > 0  # Shelf or edge below
        
        # Create a mask for valid actions (1 for valid, 0 for invalid)
        valid_action_mask = np.ones(7)
        if at_left_edge:
            valid_action_mask[0] = 0  # Block LEFT movement
        if at_bottom_edge:
            valid_action_mask[1] = 0  # Block DOWN movement
        if at_right_edge:
            valid_action_mask[2] = 0  # Block RIGHT movement
        if at_top_edge:
            valid_action_mask[3] = 0  # Block UP movement
        
        # Also mask pickup/dropoff if not at valid locations
        if not is_at_pickup or is_carrying:
            valid_action_mask[4] = 0
        if not is_at_dropoff or not is_carrying:
            valid_action_mask[5] = 0
        
        # Standard exploration/exploitation
        if not eval_mode and random.random() < self.epsilon:
            # Find valid actions for exploration
            valid_movement_actions = [a for a in range(4) if valid_action_mask[a] == 1]
            if valid_movement_actions:
                # Prefer movement actions with probability 0.9, wait with 0.1
                if random.random() < 0.9:
                    return random.choice(valid_movement_actions)
                else:
                    return 6  # Wait action
            else:
                # If no valid movement actions, wait
                return 6
        
        # Get Q-values
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            q_values = self.q_networks[agent](state_tensor).cpu().numpy().flatten()
        
        # Apply the mask by setting invalid action values to negative infinity
        masked_q_values = q_values.copy()
        for a in range(7):
            if valid_action_mask[a] == 0:
                masked_q_values[a] = float('-inf')
        
        # If all actions are masked (extreme edge case), default to wait
        if np.all(masked_q_values == float('-inf')):
            return 6  # Wait
        
        return np.argmax(masked_q_values)
    
    def select_action_batch(self, observations):
        """
        Select actions for all agents in a single batch with proper action masking.
        """
        actions = {}
        
        # Prepare batch for network inference
        batch_states = []
        batch_agents = []
        
        # First, handle direct goal actions (pickup/dropoff) and exploration
        for agent in self.env.agents:
            state = observations[agent]
            is_at_pickup = state[5, 2, 2] > 0.5
            is_at_dropoff = state[6, 2, 2] > 0.5
            is_carrying = state[7, 2, 2] > 0.5
            
            # When at goal states use deterministic action
            if is_at_pickup and not is_carrying:
                actions[agent] = 4  # pickup
                continue
            elif is_at_dropoff and is_carrying:
                actions[agent] = 5  # dropoff
                continue
            
            # Boundary detection from state observation
            # In the observation, channels 0-3 of cells outside bounds will be 0
            at_left_edge = np.sum(state[2, 2, 0]) > 0  # Shelf or edge to the left
            at_right_edge = np.sum(state[2, 2, 4]) > 0  # Shelf or edge to the right 
            at_top_edge = np.sum(state[2, 0, 2]) > 0  # Shelf or edge above
            at_bottom_edge = np.sum(state[2, 4, 2]) > 0  # Shelf or edge below
            
            # Create a mask for valid actions (1 for valid, 0 for invalid)
            valid_action_mask = np.ones(7)
            if at_left_edge:
                valid_action_mask[0] = 0  # Block LEFT movement
            if at_bottom_edge:
                valid_action_mask[1] = 0  # Block DOWN movement
            if at_right_edge:
                valid_action_mask[2] = 0  # Block RIGHT movement
            if at_top_edge:
                valid_action_mask[3] = 0  # Block UP movement
            
            # Also mask pickup/dropoff if not at valid locations
            if not is_at_pickup or is_carrying:
                valid_action_mask[4] = 0
            if not is_at_dropoff or not is_carrying:
                valid_action_mask[5] = 0
            
            # For exploration
            if random.random() < self.epsilon:
                # Find valid actions for exploration
                valid_movement_actions = [a for a in range(4) if valid_action_mask[a] == 1]
                if valid_movement_actions:
                    # Prefer movement actions with probability 0.9, wait with 0.1
                    if random.random() < 0.9:
                        actions[agent] = random.choice(valid_movement_actions)
                    else:
                        actions[agent] = 6  # Wait action
                else:
                    # If no valid movement actions, wait
                    actions[agent] = 6
            else:
                # Need network evaluation
                batch_agents.append(agent)
                batch_states.append(state)
                # Store the valid action mask for later use
                if not hasattr(self, 'batch_action_masks'):
                    self.batch_action_masks = {}
                self.batch_action_masks[agent] = valid_action_mask
        
        # Only run network if we have agents needing evaluation
        if batch_states:
            # Convert to tensor and process in one batch
            with torch.no_grad():
                state_batch = torch.from_numpy(np.stack(batch_states)).float().to(device)
                # Process all states in a single batch
                q_values_batch = self.q_networks[batch_agents[0]](state_batch)
                
                # Apply action masking to the q-values for all agents in batch
                for i, agent in enumerate(batch_agents):
                    q_values = q_values_batch[i].cpu().numpy()
                    
                    # Apply the mask by setting invalid action values to negative infinity
                    masked_q_values = q_values.copy()
                    valid_action_mask = self.batch_action_masks[agent]
                    for a in range(7):
                        if valid_action_mask[a] == 0:
                            masked_q_values[a] = float('-inf')
                    
                    # If all actions are masked (extreme edge case), default to wait
                    if np.all(masked_q_values == float('-inf')):
                        actions[agent] = 6  # Wait
                    else:
                        # Get best valid action
                        actions[agent] = np.argmax(masked_q_values)
        
        return actions

        
    def update_epsilon(self):
        """
        Update epsilon
        """
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def save_model(self, model_path=get_model_path()):
        """
        Save the Q-network model to the specified path.
        """

        model_data = {
            agent: self.q_networks[agent].state_dict()
            for agent in self.env.agents
        }
        torch.save(model_data, model_path)

    def load_model(self, filepath=get_model_path()):
        """
        Load the Q-network model from the specified path.
        """

        model_data = torch.load(filepath, map_location=device)
        for agent in self.env.agents:
            self.q_networks[agent].load_state_dict(model_data[agent])
            self.target_networks[agent].load_state_dict(model_data[agent])
        
    def _learn(self, agent, experiences=None):
        """
        Update Q-network using sampled experiences.
        """

        # If sample is not provied
        if experiences is None:
            # Only learn if enough samples are available
            if len(self.memory) < self.batch_size:
                return
            # Sample a batch of experiences
            states, actions, rewards, next_states, dones = self.memory.sample()
        else:
            states, actions, rewards, next_states, dones = experiences
        
        # Get Q-values from local and target networks - batch processing
        q_values = self.q_networks[agent](states).gather(1, actions)
        
        # Clip rewards for stability (-10 to +10 range)
        rewards = torch.clamp(rewards, min=-10.0, max=10.0)

        # Only compute next_q_values if gamma > 0
        if self.gamma > 0:
            # Get Q-values from target network
            with torch.no_grad():
                # Get actions from local network
                next_actions = self.q_networks[agent](next_states).argmax(1, keepdim=True)
                # Get Q-values from target network for those actions
                next_q_values = self.target_networks[agent](next_states).gather(1, next_actions)
                # Compute targets using Bellman equation
                targets = rewards + (self.gamma * next_q_values * (1 - dones))
        else:
            # If gamma is 0, use rewards directly
            targets = rewards

        # Compute loss and update weights
        loss = F.mse_loss(q_values, targets)

        # Zero gradients (reset gradients)
        self.optimizers[agent].zero_grad(set_to_none=True)

        # Compute gradients (backpropagation)
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_networks[agent].parameters(), max_norm=1.0)

        # Update based on gradients
        self.optimizers[agent].step()

        # Update target network
        if self.t_step % self.update_freq == 0:
            self._soft_update(self.q_networks[agent], self.target_networks[agent], self.tau)

    def _soft_update(self, local_model, target_model, tau):
        """
        Soft update target newtwork weights.
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        
# Monitor GPU utilization during training
def check_gpu_usage():
    if torch.cuda.is_available():
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

def train_DQN(env, agent=None, n_episodes=1000, max_steps=1000, debug_level=DEBUG_CRITICAL, save_every=100, model_path=get_model_path()):
    """
    Train Q-learning agent in the warehouse environment.
    
    Args:
        env: Warehouse environment.
        n_episodes: Number of training episodes (default: 1000).
        max_steps: Maximum steps per episode (default: 1000).
        debug_level: Debug level (default: DEBUG_NONE).
        save_every: Save model every n episodes (default: 100).
        model_path: Path to save the trained model.
    
    Returns:
        agent: Trained Q-learning agent.
    """

    # Verify GPU usage
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    # Set pytorch to use more cpu threads for data loading
    if hasattr(torch, 'set_num_threads'):
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        print(f"Using {num_threads} CPU threads for data loading.")

    # Initialize agent
    if agent is None:
        # Create new agent if not provided
        ql_agent = QLAgent(env, debug_level=debug_level)
    else:
        # Use provided agent
        ql_agent = agent
        ql_agent.env = env
    
    # Track scores
    scores = []

    # Add timing variables
    total_env_time = 0
    total_agent_time = 0
    total_select_action_time = 0
    total_step_time = 0
    
    # In your training loop, add these to track stats:
    deliveries_per_episode = []
    last_10_deliveries = deque(maxlen=10)
    
    # Add early stopping variables
    best_performance = -float('inf')
    best_model_state = None
    patience = 100  # Number of episodes to wait before early stopping
    no_improvement = 0
    best_episode = 0
    
    # Training loop
    for episode in range(1, n_episodes + 1):
        episode_start = time.time()
        
        # Reset environment
        env_reset_start = time.time()
        observations, _ = env.reset()
        total_env_time += time.time() - env_reset_start
        score = 0

        # Run episode
        for step in range(max_steps):
            # Time action selection
            select_action_start = time.time()
            actions = ql_agent.select_action_batch(observations)
            total_select_action_time += time.time() - select_action_start

            # Time environment step
            env_step_start = time.time()
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            total_env_time += time.time() - env_step_start
            
            # Time agent learning step
            agent_step_start = time.time()
            ql_agent.step(observations, actions, rewards, next_observations, terminations)
            total_step_time += time.time() - agent_step_start
            
            # Update score and observations
            score += sum(rewards.values())
            observations = next_observations
            
            # Render environment
            # env.render()

            # Debug info
            ql_agent.debug(DEBUG_INFO, f"Episode {episode}, Step {step}")
            ql_agent.debug(DEBUG_INFO, f"Actions: {actions}")
            ql_agent.debug(DEBUG_INFO, f"Rewards: {rewards}")
            ql_agent.debug(DEBUG_INFO, f"Completed tasks: {env.completed_tasks}")
            ql_agent.debug(DEBUG_INFO, f"Score: {score}")

            # Check if episode is done
            if all(terminations.values() or all(truncations.values())):
                break

        # Update epsilon for exploration
        ql_agent.update_epsilon()

        # Save score
        scores.append(score)

        # Track performance metrics
        ql_agent.episode_rewards.append(score)
        ql_agent.completed_tasks.append(env.completed_tasks)

        # After each episode, add:
        episode_deliveries = sum(env.completed_tasks.values())
        deliveries_per_episode.append(episode_deliveries)
        last_10_deliveries.append(episode_deliveries)
    
        # Print rolling stats every episode:
        avg_deliveries = sum(last_10_deliveries) / len(last_10_deliveries) if last_10_deliveries else 0
        ql_agent.debug(DEBUG_CRITICAL, 
            f"Episode {episode:4}/{n_episodes:<4} | "
            f"Score: {score:8.1f} | "
            f"Deliveries: {episode_deliveries:3} | "
            f"10-ep Avg: {avg_deliveries:5.1f} | "
            f"Epsilon: {ql_agent.epsilon:.4f}")

        # Early stopping check
        if episode > 10:  # After we have enough data
            # Compare to best performance so far
            if avg_deliveries > best_performance:
                best_performance = avg_deliveries
                best_episode = episode
                no_improvement = 0
                
                # Save best model state
                best_model_state = {
                    agent: ql_agent.q_networks[agent].state_dict().copy() 
                    for agent in env.agents
                }
                
                # Also save to disk
                ql_agent.save_model(model_path.replace('.pth', '_best.pth'))
                ql_agent.debug(DEBUG_INFO, f"New best model saved (episode {episode}, deliveries: {avg_deliveries:.1f})")
            else:
                no_improvement += 1
            
            # If no improvement for 'patience' episodes, stop training
            if no_improvement >= patience:
                print(f"Early stopping at episode {episode}. No improvement for {patience} episodes.")
                
                # Restore best model
                for agent in env.agents:
                    ql_agent.q_networks[agent].load_state_dict(best_model_state[agent])
                    ql_agent.target_networks[agent].load_state_dict(best_model_state[agent])
                    
                print(f"Restored best model from episode {best_episode}")
                break

        # Save model periodically
        if episode % save_every == 0:
            ql_agent.save_model(model_path)
            
        # Print timing info every 10 episodes
        if episode % 10 == 0:
            print(f"\nTiming breakdown (avg per episode):")
            print(f"  Environment time: {total_env_time/episode:.4f}s")
            print(f"  Action selection time: {total_select_action_time/episode:.4f}s")
            print(f"  Agent learning time: {total_step_time/episode:.4f}s")
            print(f"  Episode duration: {(time.time() - episode_start):.4f}s\n")
            
    # Save final model
    ql_agent.save_model(model_path)

    # Return trained agent
    return ql_agent

def train_DQN_curriculum(target_env, n_episodes=1000, max_steps=1000, debug_level=DEBUG_CRITICAL, save_every=100, model_path=get_model_path()):
    """
    Train Q-learning agent with improved curriculum learning.
    """
    # Create models directory if it doesn't exist
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created models directory: {model_dir}")
    
    print(f"Using model path: {model_path}")
    
    # STAGE 1: Basic Navigation (5x5, fully observable)
    stage1_env = env(grid_size=(5, 5), n_agents=1, n_humans=0, num_shelves=0, 
                    num_pickup_points=1, num_dropoff_points=1, render_mode="human")
    agent = train_DQN(stage1_env, n_episodes=250, max_steps=100, save_every=50, model_path=model_path)
    
    # Explicitly save model after stage 1
    agent.save_model(model_path)
    print(f"Stage 1 complete - model saved to {model_path}")
    
    # Create copy of Stage 1 network
    stage1_model_path = model_path.replace('.pth', '_stage1.pth')
    agent.save_model(stage1_model_path)
    print(f"Stage 1 model backup saved to {stage1_model_path}")
    
    # Benchmark stage 1 with 10 evaluation runs
    print("\nEvaluating Stage 1 Performance (10 runs)...")
    stage1_evaluations = []
    for i in range(10):
        print(f"\nEvaluation Run {i+1}/10")
        # Create fresh environment for each evaluation
        eval_env = env(grid_size=(5, 5), n_agents=1, n_humans=0, num_shelves=0, 
                      num_pickup_points=1, num_dropoff_points=1, render_mode="human")
        completed_tasks = run_q_learning(eval_env, model_path=model_path, n_steps=100, debug_level=DEBUG_NONE)
        stage1_evaluations.append(completed_tasks)
        
        # Close environment to free resources
        eval_env.close()
        import matplotlib.pyplot as plt
        plt.close('all')
    
    avg_stage1_performance = sum(stage1_evaluations) / len(stage1_evaluations)
    print(f"\nStage 1 Performance Summary:")
    print(f"Individual runs: {stage1_evaluations}")
    print(f"Average completed tasks: {avg_stage1_performance:.2f}")
    
    # Full benchmark (just once) to see A* comparison
    print("\nBenchmarking Stage 1 against A*...")
    benchmark1 = benchmark_environment(stage1_env, n_steps=100, debug_level=DEBUG_NONE, model_path=model_path)
    
    # STAGE 2: Navigation with simple obstacles
    print("\nStarting Stage 2 Training...")
    agent.epsilon = 0.6
    agent.epsilon_decay = 0.992
    
    for agent_name in agent.optimizers:
        for param_group in agent.optimizers[agent_name].param_groups:
            param_group['lr'] *= 0.5
    
    stage2_env = env(grid_size=(10, 8), n_agents=1, n_humans=0, num_shelves=16, 
                    num_pickup_points=1, num_dropoff_points=1, render_mode="human")
    agent = train_DQN(stage2_env, agent=agent, n_episodes=200, max_steps=150, 
                     save_every=25, model_path=model_path)
    
    # Explicitly save model after stage 2
    agent.save_model(model_path)
    print(f"Stage 2 complete - model saved to {model_path}")
    
    # Benchmark stage 2 with 10 evaluation runs
    print("\nEvaluating Stage 2 Performance (10 runs)...")
    stage2_evaluations = []
    for i in range(10):
        print(f"\nEvaluation Run {i+1}/10")
        # Create fresh environment for each evaluation
        eval_env = env(grid_size=(10, 8), n_agents=1, n_humans=0, num_shelves=16, 
                      num_pickup_points=1, num_dropoff_points=1, render_mode="human")
        completed_tasks = run_q_learning(eval_env, model_path=model_path, n_steps=150, debug_level=DEBUG_NONE)
        stage2_evaluations.append(completed_tasks)
        
        # Close environment to free resources
        eval_env.close()
        import matplotlib.pyplot as plt
        plt.close('all')
    
    avg_stage2_performance = sum(stage2_evaluations) / len(stage2_evaluations)
    print(f"\nStage 2 Performance Summary:")
    print(f"Individual runs: {stage2_evaluations}")
    print(f"Average completed tasks: {avg_stage2_performance:.2f}")
    
    # Full benchmark (just once) to see A* comparison
    print("\nBenchmarking Stage 2 against A*...")
    benchmark2 = benchmark_environment(stage2_env, n_steps=150, debug_level=DEBUG_NONE, model_path=model_path)
    
    # Return performance metrics
    return {
        "stage1_evals": stage1_evaluations,
        "stage1_avg": avg_stage1_performance,
        "stage2_evals": stage2_evaluations, 
        "stage2_avg": avg_stage2_performance,
        "stage1_benchmark": benchmark1,
        "stage2_benchmark": benchmark2
    }

def run_q_learning(env, model_path=get_model_path(), n_steps=1000, debug_level=DEBUG_NONE):
    """
    Run trained Q-learning agent in the warehouse environment.
    """
    
    # Initialize agent in evaluation mode
    QL_agent = QLAgent(env, debug_level=debug_level)
    QL_agent.epsilon = 0.05  # Small epsilon for minimal exploration during evaluation
    
    # Load the trained model if it exists
    try:
        QL_agent.load_model(model_path)
        print(f"Loaded trained model from {model_path}")
    except FileNotFoundError:
        print(f"Model file not found at {model_path}. Using untrained agent.")

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
            state = observations[agent]
            # Force evaluation mode
            action = QL_agent.select_action(state, agent, eval_mode=True)
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
        print(f"Step {step}, Tasks: {completed_tasks}")
        print(f"Actions: {actions}")
        print(f"Rewards: {rewards}")

        # Print action distribution every 20 steps
        if step > 0 and step % 20 == 0:
            print("\nAction distribution so far:")
            total = sum(action_counts.values())
            for action, count in action_counts.items():
                action_name = ["LEFT", "DOWN", "RIGHT", "UP", "PICKUP", "DROPOFF", "WAIT"][action]
                percentage = (count / total) * 100
                print(f"  {action_name}: {count} ({percentage:.1f}%)")

        # Check if done
        if all(terminations.values()) or all(truncations.values()):
            break
    
    # Print final results
    print(f"Final score: {total_rewards}")
    print(f"Completed tasks: {completed_tasks}")
    
    # Print final action distribution
    print("\nFinal action distribution:")
    total = sum(action_counts.values())
    for action, count in action_counts.items():
        action_name = ["LEFT", "DOWN", "RIGHT", "UP", "PICKUP", "DROPOFF", "WAIT"][action]
        percentage = (count / total) * 100
        print(f"  {action_name}: {count} ({percentage:.1f}%)")
    
    # Clean up
    import matplotlib.pyplot as plt
    plt.close('all')
    env.close()
    
    return completed_tasks

def benchmark_environment(env, n_steps=200, debug_level=DEBUG_NONE, model_path=get_model_path()):
    """
    Run both A* and RL agents on the same environment to establish performance benchmarks.
    """
    import matplotlib.pyplot as plt
    from copy import deepcopy
    import time
    
    # Create completely separate environment instances
    env_copy_astar = deepcopy(env)
    env_copy_rl = deepcopy(env)
    
    # Ensure environments are properly configured
    env_copy_astar.render_mode = "human"
    env_copy_rl.render_mode = "human"
    
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
            print("No model found! Benchmark will use untrained agent.")
    else:
        print(f"Using model from {model_path}")
    
    # Run A* agent
    print("Running A* benchmark...")
    astar_tasks = run_a_star(env_copy_astar, n_steps=n_steps, debug_level=debug_level)
    
    # Close any open plots and delay to ensure resources are released
    plt.close('all')
    time.sleep(1)
    
    # Run RL agent with explicit model loading
    print("Running RL benchmark...")
    rl_tasks = run_q_learning(env_copy_rl, model_path=model_path, n_steps=n_steps, debug_level=debug_level)
    
    # Close any open plots
    plt.close('all')
    
    # Calculate performance ratio (RL / A*)
    if astar_tasks and astar_tasks > 0:  # Ensure it's not None and positive
        performance_ratio = rl_tasks / astar_tasks
    else:
        performance_ratio = 0
        
    # Print benchmark results
    print("\n=== BENCHMARK RESULTS ===")
    print(f"A* completed tasks: {astar_tasks}")
    print(f"RL completed tasks: {rl_tasks}")
    print(f"Performance ratio (RL/A*): {performance_ratio:.2f}")
    
    return {
        "astar_tasks": astar_tasks,
        "rl_tasks": rl_tasks,
        "performance_ratio": performance_ratio
    }



