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
import matplotlib.pyplot as plt  # add import for plotting near the other imports
import json  # add this import near the other imports

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
def get_model_path(path=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, '..', 'models')
    if path:
        model_name = path
    else:
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

        # Single transfer to GPU
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

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
                 alpha=0.00005, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1,
                 epsilon_decay=0.9985, hidden_size=64, buffer_size=1000000, batch_size=128,
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
            tau: Target network update rate (default: 0.01).
        """
        
        self.env = env
        self.debug_level = debug_level

        # Get observation shape from environment
        observation_shape = env.observation_size
        observation_channels = 10

        # State and action dimentions
        self.state_size = (observation_channels, *observation_shape)
        self.action_size = 7 # 4 movement, 3 actions (pickup, dropoff, wait)

        # Shared Q-network and target network
        self.q_network = QNetwork(self.state_size, self.action_size, hidden_size).to(device)
        self.target_network = QNetwork(self.state_size, self.action_size, hidden_size).to(device)
        # Initialize target with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        # Single optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=alpha)

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

        # Add attribute to store loss values for the current episode
        self.episode_losses = []

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

        # Learn every update_freq steps
        self.t_step += 1
        if self.t_step % self.update_freq == 0 and len(self.memory) >= self.batch_size:
            # Sample a batch of experiences
            experiences = self.memory.sample()
            # Learn using the shared network
            self._learn(experiences)

    def select_action(self, state, eval_mode=False):
        """
        Return action for given state using epsilon-greedy policy.
        """

        # Check if agent is at a pickup or dropoff point
        is_at_pickup = state[5, 2, 2] > 0.5
        is_at_dropoff = state[6, 2, 2] > 0.5
        is_carrying = state[7, 2, 2] > 0.5

        # When at goal states, reduce exploration
        if eval_mode or self.epsilon < 0.3:
            if is_at_pickup and not is_carrying:
                return 4
            elif is_at_dropoff and is_carrying:
                return 5

        # Check if we should explore
        if not eval_mode and random.random() < self.epsilon:
            # Explore: select random action
            return random.randint(0, self.action_size - 1)

        # Convert state to tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)

        with torch.no_grad():
            action_values = self.q_network(state_tensor)

        # Exploit: select action with highest Q-value
        return int(torch.argmax(action_values).item())
    
    def select_action_batch(self, observations):
        """
        Select actions for all agents in a single batch.
        """

        actions = {}
        states = []
        agents = []

        # For each agent, check if we're on a pickup or dropoff point
        # and if we are carrying an item
        is_at_pickup = {}
        is_at_dropoff = {}
        is_carrying = {}
        for agent in self.env.agents:
            is_at_pickup[agent] = observations[agent][5, 2, 2] > 0.5
            is_at_dropoff[agent] = observations[agent][6, 2, 2] > 0.5
            is_carrying[agent] = observations[agent][7, 2, 2] > 0.5

        # When at goal states, reduce exploration
        for agent in self.env.agents:
            if self.epsilon < 0.3:
                if is_at_pickup[agent] and not is_carrying[agent]:
                    actions[agent] = 4
                elif is_at_dropoff[agent] and is_carrying[agent]:
                    actions[agent] = 5

        # Determine which agents need network evaluation vs random actions when agent is not in actions
        for agent in self.env.agents:
            # If agent already has an action, skip it
            if agent in actions:
                continue

            # Check if exploring
            if random.random() < self.epsilon:
                # Explore: select random action
                actions[agent] = random.randint(0, self.action_size - 1)
            else:
                # Process state with network evaluation
                agents.append(agent)
                states.append(observations[agent])

        # For any agents that need network evaluation
        if states:
            state_batch = torch.from_numpy(np.stack(states)).float().to(device)

            with torch.no_grad():
                # Batch forward on shared network
                action_values_batch = self.q_network(state_batch)
                for i, agent in enumerate(agents):
                    actions[agent] = int(torch.argmax(action_values_batch[i:i+1]).item())

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
        torch.save(self.q_network.state_dict(), model_path)

    def load_model(self, filepath=get_model_path()):
        """
        Load the Q-network model from the specified path.
        """
        state_dict = torch.load(filepath, map_location=device)
        self.q_network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)

    def _learn(self, experiences=None):
        """
        Update Q-network using sampled experiences.
        """

        # If sample is not provided
        if experiences is None:
            # Only learn if enough samples are available
            if len(self.memory) < self.batch_size:
                return
            # Sample a batch of experiences
            states, actions, rewards, next_states, dones = self.memory.sample()
        else:
            states, actions, rewards, next_states, dones = experiences

        # Get Q-values from local and target networks - batch processing
        q_values = self.q_network(states).gather(1, actions)

        # Only compute next_q_values if gamma > 0
        if self.gamma > 0:
            # Get Q-values from target network
            with torch.no_grad():
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
                targets = rewards + (self.gamma * next_q_values * (1 - dones))
        else:
            # If gamma is 0, use rewards directly
            targets = rewards

        # Compute loss and update weights
        loss = F.mse_loss(q_values, targets)
        # Append current loss to the episode_losses for plotting later
        self.episode_losses.append(loss.item())

        # Zero gradients (reset gradients)
        self.optimizer.zero_grad(set_to_none=True)

        # Compute gradients (backpropagation)
        loss.backward()
        # gradient clipping to stabilize updates
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=5.0)

        # Update based on gradients
        self.optimizer.step()

        # Update target network
        if self.t_step % (self.update_freq * 10) == 0:
            self._soft_update(self.q_network, self.target_network, self.tau)

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

def train_DQN(env, n_episodes=1000, max_steps=1000, debug_level=DEBUG_CRITICAL, save_every=100, model_path=get_model_path(), load_path=None):
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
    ql_agent = QLAgent(env, debug_level=debug_level)
    
    #load model if it exists
    try:
        ql_agent.load_model(load_path)
        print(f"Loaded trained model from {load_path}")
    except FileNotFoundError:
        print(f"Model file not found at {load_path}. Using untrained agent.")

    # Track scores
    scores = []
    avg_losses = []  # list to store average loss per episode
    avg_score = []
    avg_deliveries = []  # list to store average deliveries per robot per episode
    epsilons = []        # list to store epsilon value at end of each episode

    # Add timing variables
    total_env_time = 0
    total_agent_time = 0
    total_select_action_time = 0
    total_step_time = 0
    
    # Training loop
    try:
        for episode in range(1, n_episodes + 1):
            episode_start = time.time()
            
            # Reset environment
            env_reset_start = time.time()
            observations, _ = env.reset()
            total_env_time += time.time() - env_reset_start
            ql_agent.episode_losses = []  # reset loss list for new episode
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
                # if episode > 600:
                #     env.render()
                    
                
                    
                

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
            # Record average deliveries and epsilon after this episode
            avg_deliveries.append(sum(env.completed_tasks.values()) / len(env.agents))
            epsilons.append(ql_agent.epsilon)

            # Save score
            scores.append(score)
            
            avg_score.append(score)

            # Track performance metrics
            ql_agent.episode_rewards.append(score)
            ql_agent.completed_tasks.append(env.completed_tasks)

            # Compute average loss for the episode (if any loss was recorded)
            if ql_agent.episode_losses:
                episode_avg_loss = np.mean(ql_agent.episode_losses)
                avg_losses.append(episode_avg_loss)
                print(f"Episode {episode} average loss: {episode_avg_loss:.4f}")
            else:
                avg_losses.append(0)
                print(f"Episode {episode} average loss: 0.0000")
                
            # Log episode results
            ql_agent.debug(DEBUG_CRITICAL, f"Episode {episode}/{n_episodes}, Score: {score}, Completed tasks: {env.completed_tasks}, Epsilon: {ql_agent.epsilon:.4f}")

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
    except Exception as e:
        print(e)
        pass
    finally:
        # Save final model
        ql_agent.save_model(model_path)
        
        # Save training metrics (average losses and scores) to a JSON file
        metrics = {
            "avg_losses": avg_losses,
            "avg_score": avg_score,
            "avg_deliveries": avg_deliveries,
            "epsilons": epsilons
        }
        current_time = time.strftime("%Y%m%d-%H%M%S")
        metrics_file = f'{model_path}_training_metrics__{current_time}.json'
        with open(metrics_file, "w") as f:
            json.dump(metrics, f)
        print(f"Training metrics saved to {metrics_file}")
        
        # Plot loss, score, deliveries, and epsilon over episodes
        plt.figure(figsize=(10, 12))
        plt.subplot(4, 1, 1)
        plt.plot(avg_losses)
        plt.title("Average Q Loss per Episode")
        plt.subplot(4, 1, 2)
        plt.plot(avg_score)
        plt.title("Average Score per Episode")
        plt.subplot(4, 1, 3)
        plt.plot(avg_deliveries)
        plt.title("Average Deliveries per Robot per Episode")
        plt.subplot(4, 1, 4)
        plt.plot(epsilons)
        plt.title("Epsilon Value per Episode")
        plt.tight_layout()
        plot_file = f'{model_path}_training_metrics__{current_time}.png'
        plt.savefig(plot_file)
        plt.close()
        print(f"Training plots saved to {plot_file}")

        # Return trained agent
        return ql_agent

def run_q_learning(env, model_path=get_model_path(), n_steps=1000, debug_level=DEBUG_NONE):
    """
    Run trained Q-learning agent in the warehouse environment.
    
    Args:
        env: Warehouse environment.
        model_path: Path to the trained model.
        n_steps: Number of steps to run (default: 1000).
        debug_level: Debug level (default: DEBUG_NONE).
    """
    
    # Initialize agent
    QL_agent = QLAgent(env, debug_level=debug_level)

    # Load the trained model if it exists
    try:
        QL_agent.load_model(model_path)
        print(f"Loaded trained model from {model_path}")
    except FileNotFoundError:
        print(f"Model file not found at {model_path}. Using untrained agent.")

    # Reset environment
    observations, _ = env.reset()

    # Run for n_steps
    for step in range(n_steps):
        # Get actions for each agent
        actions = {}
        for agent in env.agents:
            state = observations[agent]
            actions[agent] = QL_agent.select_action(state, eval_mode=True)

        # Take actions in the environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Render the environment
        env.render()

        # Debug info
        print(f"Step {step}")
        print(f"Actions: {actions}")
        print(f"Rewards: {rewards}")
        print(f"Completed tasks: {env.completed_tasks}")

        if all(terminations.values()) or all(truncations.values()):
            break

    # Close the environment
    env.close()
    # Print final score
    final_score = sum(rewards.values())
    print(f"Final score: {final_score}")
    return final_score



