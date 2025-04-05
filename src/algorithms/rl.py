import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from collections import namedtuple, deque
from subprocess import call

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
                 alpha=0.01, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1,
                 epsilon_decay=0.995, hidden_size=64, buffer_size=10000, batch_size=256,
                 update_freq=16, tau=0.001):
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
        observation_channels = 9

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

        # Learn every update_freq steps
        self.t_step += 1
        if self.t_step % self.update_freq == 0:
            # Learn individually for each agent
            for agent in self.env.agents:
                self._learn(agent)

    def select_action(self, state, agent, eval_mode=False):
        """
        Return action for given state using epsilon-greedy policy.
        """

        # Convert state to tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # Set Q-network to evaluation mode
        self.q_networks[agent].eval()

        with torch.no_grad():
            action_values = self.q_networks[agent](state_tensor)

        # Switch back to training mode
        self.q_networks[agent].train()

        # Epsilon-greedy action selection
        if random.random() > self.epsilon or eval_mode:
            # Exploit: select action with highest Q-value
            return np.argmax(action_values.cpu().numpy())
        else:
            # Explore: select random action
            return random.choice(range(self.action_size))
        
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
        
    def _learn(self, agent):
        """
        Update Q-network using sampled experiences.
        """

        # Only learn if enough samples are available
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample()

        # Get Q-values from local and target networks
        q_values = self.q_networks[agent](states).gather(1, actions)
        next_q_values = self.target_networks[agent](next_states).detach().max(1)[0].unsqueeze(1)

        # Compute target Q-values
        targets = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss and update weights
        loss = F.mse_loss(q_values, targets)
        # Zero gradients (reset gradients)
        self.optimizers[agent].zero_grad()
        # Compute gradients (backpropagation)
        loss.backward()
        # Update based on gradients
        self.optimizers[agent].step()

        # Update target network
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

def train_DQN(env, n_episodes=1000, max_steps=1000, debug_level=DEBUG_CRITICAL, save_every=100, model_path=get_model_path()):
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

    # Add this at the beginning of train_DQN to verify GPU usage
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

    # Initialize agent
    ql_agent = QLAgent(env, debug_level=debug_level)

    # Track scores
    scores = []

    # Training loop
    for episode in range(1, n_episodes + 1):
        # Track gpu usage
        check_gpu_usage()

        # Reset environment
        observations, _ = env.reset()
        score = 0

        # Run episode
        for step in range(max_steps):
            # Get actions for each agent
            actions = {}
            for agent in env.agents:
                state = observations[agent]
                action = ql_agent.select_action(state, agent)
                actions[agent] = action

            # Take actions in the environment
            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            # Store experience and learn
            ql_agent.step(observations, actions, rewards, next_observations, terminations)

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

        # Log episode results
        ql_agent.debug(DEBUG_CRITICAL, f"Episode {episode}/{n_episodes}, Score: {score}, Completed tasks: {env.completed_tasks}, Epsilon: {ql_agent.epsilon:.4f}")

        # Save model periodically
        if episode % save_every == 0:
            ql_agent.save_model(model_path)
            
    # Save final model
    ql_agent.save_model(model_path)

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
        agent.load_model(model_path)
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
            actions[agent] = QL_agent.select_action(state, agent, eval_mode=True)

        # Take actions in the environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Render the environment
        env.render()

        # Debug info
        print(f"Step {step}")
        print(f"Actions: {actions}")
        print(f"Rewards: {rewards}")
        print(f"Completed tasks: {env.completed_tasks}")

        # Check if done
        if all(terminations.values()) or all(truncations.values()):
            break

    # Close the environment
    env.close()
    # Print final score
    final_score = sum(rewards.values())
    print(f"Final score: {final_score}")
    return final_score


        
        