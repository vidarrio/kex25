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
from torch.utils.tensorboard import SummaryWriter

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
    return models_dir

# Define experience tuple type
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class QNetwork(nn.Module):
    """Neural network for Q-value approximation."""

    def __init__(self, input_size, action_size, hidden_size=64, frame_history=4):
        """
        Initialize network parameters.

        Args:
            input_size: Size of the input state.
            action_size: Number of possible actions.
            hidden_size: Size of the hidden layer.
        """

        super(QNetwork, self).__init__()

        # Extract input dimensions
        self.channels, self.height, self.width = input_size
        self.actual_channels = self.channels * frame_history
        self.frame_history = frame_history
        self.temperature = 1.0 # Softmax temperature for action probabilities

        # Temporal convolutional layer to learn frame-stacked features
        # if frame_history > 1:
        #     self.temporal_conv = nn.Conv3d(
        #         in_channels=self.channels,
        #         out_channels=16,
        #         kernel_size=(frame_history, 3, 3),
        #         stride=(1, 1, 1),
        #         padding=(0, 1, 1)
        #     )

        #     # Iniitialize temporal convolutional layer weights to emphasize recent frames
        #     with torch.no_grad():
        #         weight_scale = torch.linspace(0.5, 1.0, steps=frame_history).view(-1, 1, 1)

        #         # Apply weights to all filters in the temporal convolution
        #         for i in range(self.temporal_conv.weight.size(0)): # output channels
        #             for j in range(self.temporal_conv.weight.size(1)): # input channels
        #                 self.temporal_conv.weight[i, j] *= weight_scale

        #     # Adjust input to first standard convolutional layer
        #     self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # else:
        #     self.conv1 = nn.Conv2d(in_channels=self.actual_channels, out_channels=32, kernel_size=3, padding=1)

        # # Second standard convolutional layer
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

        # # Add spatial attention
        # self.attention = nn.Sequential(
        #     nn.Conv2d(32, 1, kernel_size=1),
        #     nn.Sigmoid()
        # )

        # # Planning module (simplified value iteration network)
        # self.planning = nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 16, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        # )

        # Calculate flattened size after convolutional layers
        flattened_size = 32 * self.height * self.width

        # Calculate flattened size without convolutional layers
        flattened_size_no_conv = self.actual_channels * self.height * self.width

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size_no_conv, hidden_size*2)
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)

        # Output layer
        self.fc3 = nn.Linear(hidden_size, action_size)

        # Initialize weights (kaiming initialization)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, nonlinearity='relu', mode='fan_in')
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
                    
                

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x: Input state.
        """

        # Process with temporal convolution if using frame stacking
        batch_size = x.size(0)
        # if hasattr(self, 'temporal_conv'):
            # Reshape for 3D convolution [batch, channels, frames, height, width]
            # x = x.view(batch_size, self.channels, self.frame_history, self.height, self.width)
            # Process only temporal dimension with 3d convolution
            # x = F.relu(self.temporal_conv(x))
            # Reshape back to 2D shape [batch, channels, height, width]
            # x = x.view(batch_size, 16, self.height, self.width)

        # Apply convolutions 
        # identity = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(identity)))
        # x = x + identity  # Residual connection

        # Apply spacial attention
        # attention_weights = self.attention(x) # [batch, 1, height, width]
        # x = x * attention_weights

        # Apply planning module
        # planning_features = self.planning(x) # [batch, 16, height, width]

        # Combine features from convolutional and planning modules
        # x = x.view(batch_size, 32, self.height * self.width)
        # planning_features = planning_features.view(batch_size, 16, self.height * self.width)
            
        # Flatten both completely
        # x = x.reshape(batch_size, -1) # [batch, 32 * height * width]

        # Flatten without convolutional layers
        x = x.view(batch_size, -1)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))


        # Output layer (no activation, we get raw Q-values)
        # return self.fc3(x)

        # Output layer with softmax for action probabilities
        return F.softmax(self.fc3(x) / self.temperature, dim=1)

class SumTree:
    """
    A binary sum tree data structure for efficient sampling based on priorities.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1) # Total nodes in binary tree
        self.data = np.zeros(capacity, dtype=object) # Data storage
        self.write_index = 0 # Current writing index
        self.n_entries = 0 # Number of entries in buffer

    def _propagate(self, idx, change):
        """
        Propagate priority change up the tree.
        
        Args:
            idx: Index of the leaf node.
            change: Change in priority.
        """

        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, value):
        """
        Retrieve the index of the leaf node with the given priority value.

        Args:
            idx: Index of the current node.
            value: Priority value to search for.
        """

        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree): # If we are at a leaf node
            return idx
        
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])
        
    def total(self):
        """
        Return the sum of all priorities in the tree.
        """

        return self.tree[0]

    def add(self, priority, data):
        """
        Add a new experience to the tree, replacing lowest priority if full.
        
        Args:
            priority: Priority of the experience.
            data: Experience data.
        """

        if self.n_entries < self.capacity:
            # If not full, add normally
            idx = self.write_index + self.capacity - 1
            self.data[self.write_index] = data
            self.update(idx, priority)
            self.write_index = (self.write_index + 1) % self.capacity
            self.n_entries += 1
        else:
            # If full, replace the lowest priority experience 80% of the time
            if random.random() < 0.8:
                leaf_start = self.capacity - 1
                leaf_end = len(self.tree)
                min_priority_idx = np.argmin(self.tree[leaf_start:leaf_end]) + leaf_start
                # Replace the experience
                self.data[min_priority_idx - self.capacity + 1] = data
                self.update(min_priority_idx, priority)
            else:
                # Otherwise, replace oldest experience (FIFO)
                idx = self.write_index + self.capacity - 1
                self.data[self.write_index] = data
                self.update(idx, priority)
                self.write_index = (self.write_index + 1) % self.capacity

    def update(self, idx, priority):
        """
        Update the priority of an experience.
        
        Args:
            idx: Index of the experience.
            priority: New priority value.
        """

        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, value):
        """
        Get experience based on priority value.
        
        args:
            value: Priority value to search for.
        """

        idx = self._retrieve(0, value)
        data_idx = idx - self.capacity + 1
        
        return idx, self.tree[idx], self.data[data_idx]
    
class PrioritizedReplayBuffer:
    """
    Prioritized replay buffer for experience replay using SumTree.
    """

    def __init__(self, capacity, batch_size, PER_alpha=0.6, PER_beta_start=0.4, PER_beta_end=1.0, PER_beta_frames=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer_size = capacity
        self.alpha = PER_alpha # How much prioritization to use (0.0 = no prioritization, 1.0 = full prioritization)
        self.beta = PER_beta_start # Importance sampling correction (0.0 = no correction, 1.0 = full correction)
        self.beta_start = PER_beta_start
        self.beta_end = PER_beta_end 
        self.beta_frames = PER_beta_frames
        self.frame = 1 # Current frame for beta annealing
        self.epsilon = 1e-6 # Small constant to avoid zero priorities
        self.max_priority = 1.0 # Maximum priority for new experiences

    def beta_by_frame(self):
        """
        Linearly anneal beta parameter
        """

        return min(self.beta_end, self.beta_start + (self.beta_end - self.beta_start) * (self.frame / self.beta_frames))
    
    def add(self, experience, error=None):
        """
        Add experience to buffer with priority based on TD error. New experiences are given max priority.
        """

        # Use max priority for new experiences if no error is provided
        if error is None:
            priority = self.max_priority
        else:
            # Calculate priority based on TD error
            priority = (abs(error) + self.epsilon) ** self.alpha
        
        self.tree.add(priority, experience)

    def sample(self):
        """
        Sample a batch of experiences based on priorities.
        """

        batch = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []

        self.beta = self.beta_by_frame()
        self.frame += 1

        for i in range(self.batch_size):
            value = random.uniform(i * segment, (i + 1) * segment)
            idx, priority, data = self.tree.get(value)
            priorities.append(priority)
            batch.append(data)
            idxs.append(idx)

        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = (self.tree.n_entries * sampling_probabilities) ** (-self.beta)
        weights /= weights.max()  # Normalize weights for stability

        # Convert weights to tensor
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)

        return states, actions, rewards, next_states, dones, weights, idxs
    
    def update_priorities(self, idxs, errors):
        """
        Update priorities based on new TD errors.

        Args:
            idxs: Indices of the experiences to update.
            errors: New TD errors for the experiences.
        """

        # Clip extreme errors
        errors = np.clip(errors, 1e-4, 10)
        for idx, error in zip(idxs, errors):
            # Calculate new priority based on TD error
            priority = (abs(error) + self.epsilon) ** self.alpha
            # Update the priority in the tree
            self.tree.update(idx, priority)
            # Update the maximum priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        """
        Return the current size of the buffer.
        """

        return self.tree.n_entries        
        
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
        self.buffer_size = buffer_size

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

    def __init__(self, env, debug_level=DEBUG_NONE, phase=1,
                 alpha=0.00025, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1,
                 epsilon_decay=0.995, hidden_size=128, buffer_size=20000, batch_size=32,
                 update_freq=8, tau=0.0005, use_tensorboard=False, use_per=True):
        """
        Initialize the Q-learning agent.

        Args:
            env: Warehouse environment.
            debug_level: Debug level (default: DEBUG_NONE).
            phase: Phase of training (default: 1).
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
            use_tensorboard: Whether to use TensorBoard for logging (default: False).
            use_per: Whether to use prioritized experience replay (default: True).
        """
        
        self.env = env
        self.debug_level = debug_level
        self.phase = phase
        self.use_tensorboard = use_tensorboard
        self.use_per = use_per
        self.model_name = f"phase_{self.phase}_dqn_{time.strftime('%Y%m%d-%H%M%S')}.pth"
        self.model_path = os.path.join(get_model_path(), self.model_name)
        self.epsilon_start = epsilon_start
        
        # Check if frame stacking is used
        if env.use_frame_stack:
            self.frame_history = env.n_frames
        else:
            self.frame_history = 1

        # Get observation shape from environment
        observation_shape = env.observation_size
        observation_channels = env.observation_channels

        # State and action dimensions
        self.state_size = (observation_channels, *observation_shape)
        self.action_size = 7 # 4 movement, 3 actions (pickup, dropoff, wait)

        # Initialize Q-networks for each agent
        self.q_networks = {}
        self.target_networks = {}
        self.optimizers = {}

        for agent in env.agents:
            self.q_networks[agent] = QNetwork(self.state_size, self.action_size, hidden_size, self.frame_history).to(device)
            self.target_networks[agent] = QNetwork(self.state_size, self.action_size, hidden_size, self.frame_history).to(device)
            # Copy weights from Q-network to target network
            self.target_networks[agent].load_state_dict(self.q_networks[agent].state_dict())
            self.optimizers[agent] = torch.optim.AdamW(self.q_networks[agent].parameters(), lr=alpha, eps=1e-5, weight_decay=0.0001)

        # Initialize replay buffer
        if use_per:
            self.memory = PrioritizedReplayBuffer(buffer_size, batch_size)
        else:
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

        # Initialize TensorBoard writer if enabled
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=f"runs/{self.model_name}")

            # log hyperparameters
            hp_dict = {
                'phase': self.phase,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon_start': self.epsilon,
                'epsilon_end': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'hidden_size': hidden_size,
                'buffer_size': buffer_size,
                'batch_size': batch_size,
                'update_freq': update_freq,
                'tau': tau,
                'frame_history': self.frame_history,
            }

            # Add as text summary
            param_str = "\n".join([f"{k}: {v}" for k, v in hp_dict.items()])
            self.writer.add_text("Hyperparameters", param_str)

            # Log reward hyperparameters
            reward_hyperparams = {
                'collision_penalty': self.env.collision_penalty,
                'task_reward': self.env.task_reward,
                'step_cost': self.env.step_cost,
                'progress_reward': self.env.progress_reward,
                'wait_penalty': self.env.wait_penalty,
                'revisit_penalty': self.env.revisit_penalty,
            }

            reward_str = "\n".join([f"{k}: {v}" for k, v in reward_hyperparams.items()])
            self.writer.add_text("Reward Hyperparameters", reward_str)

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
        # Add experience to Replay for each agent
        for agent in self.env.agents:
            if self.use_per:
                # Initially add no error / max priority
                experience = (states[agent], action[agent], reward[agent], next_state[agent], done[agent])
                self.memory.add(experience)
            else:
                # Add experience to normal replay buffer
                self.memory.add(states[agent], action[agent], reward[agent], next_state[agent], done[agent])

        # Increment step counter
        self.t_step += 1
        
        # Update model less often for multiple agents
        update_freq = 4 if len(self.env.agents) == 1 else self.update_freq
        
        # Learn every update_freq steps as long as we are out of initial frame history
        if self.frame_history < self.t_step and self.t_step % update_freq == 0 and len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            agent = self.env.agents[self.t_step % len(self.env.agents)]
            self._learn(agent, experiences)

    def select_action(self, state, agent, eval_mode=False, sampling_mode='argmax'):
        """Return action using epsilon-greedy with softmax probabilities and proper masking."""
        
        indices = self.get_current_frame_indices(state)

        # Early decision for pickup/dropoff at goal states for more reliable goal actions
        is_at_pickup = state[indices['pickup'], 2, 2] > 0.5  # Center cell is pickup
        is_at_dropoff = state[indices['dropoff'], 2, 2] > 0.5  # Center cell is dropoff
        is_carrying = state[indices['carrying'], 2, 2] > 0.5 # Center cell is carrying
        
        # For explicit goal states, bias heavily toward correct action but not 100%
        if is_at_pickup and not is_carrying:
            # 95% chance of pickup during evaluation, 90% during training
            if random.random() < (0.95 if eval_mode else 0.9):
                return 4  # pickup
        if is_at_dropoff and is_carrying:
            # 95% chance of dropoff during evaluation, 90% during training
            if random.random() < (0.95 if eval_mode else 0.9):
                return 5  # dropoff
        
        # Boundary detection from state observation
        at_left_edge = np.sum(state[indices['boundary'], 2, 0]) > 0
        at_right_edge = np.sum(state[indices['boundary'], 2, 4]) > 0
        at_top_edge = np.sum(state[indices['boundary'], 0, 2]) > 0
        at_bottom_edge = np.sum(state[indices['boundary'], 4, 2]) > 0
        
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
        
        # Epsilon-greedy exploration: pure random actions with probability epsilon
        if not eval_mode and random.random() < self.epsilon:
            # Find valid actions for exploration
            valid_actions = [a for a in range(7) if valid_action_mask[a] == 1]
            if valid_actions:
                return random.choice(valid_actions)
            else:
                return 6  # Wait action as fallback
        
        # Exploitation: Use softmax probabilities from network
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action_probs = self.q_networks[agent](state_tensor)
            q_values = action_probs.cpu().numpy().flatten()
        
        # Apply action masking to probabilities
        masked_q_values = q_values.copy()
        for a in range(7):
            if valid_action_mask[a] == 0:
                masked_q_values[a] = 0.0
        
        # Renormalize probabilities after masking
        if np.sum(masked_q_values) > 0:
            masked_q_values /= np.sum(masked_q_values)
        else:
            return 6  # Wait action as fallback
        
        # During evaluation, always take most probable action
        if eval_mode:
            return np.argmax(masked_q_values)
        
        # During training, use specified sampling mode
        if sampling_mode == 'argmax':
            # Deterministic action - exploit
            return np.argmax(masked_q_values)
        elif sampling_mode == 'sample':
            # Stochastic action - sample from softmax distribution
            try:
                return np.random.choice(range(7), p=masked_q_values)
            except ValueError:
                # If probabilites do not sum to 1, fallback to argmax
                return np.argmax(masked_q_values)
    
    def select_action_batch(self, observations, sampling_mode='argmax'):
        """
        Select actions for all agents using epsilon-greedy with softmax and proper masking.
        """
        indices = self.get_current_frame_indices(observations)
        actions = {}
        
        # Prepare batch for network inference
        batch_states = []
        batch_agents = []
        batch_masks = {}
        
        # Process each agent
        for agent in self.env.agents:
            state = observations[agent]
            is_at_pickup = state[indices['pickup'], 2, 2] > 0.5
            is_at_dropoff = state[indices['dropoff'], 2, 2] > 0.5
            is_carrying = state[indices['carrying'], 2, 2] > 0.5
            
            # When at goal states, strongly bias to pickup/dropoff but still allow exploration
            # if is_at_pickup and not is_carrying:
            #     if random.random() < 0.9:
            #         actions[agent] = 4  # pickup
            #         continue
            # elif is_at_dropoff and is_carrying:
            #     if random.random() < 0.9:
            #         actions[agent] = 5  # dropoff
            #         continue
            
            # Boundary detection
            # at_left_edge = np.sum(state[indices['boundary'], 2, 0]) > 0
            # at_right_edge = np.sum(state[indices['boundary'], 2, 4]) > 0
            # at_top_edge = np.sum(state[indices['boundary'], 0, 2]) > 0
            # at_bottom_edge = np.sum(state[indices['boundary'], 4, 2]) > 0
            
            # Create valid action mask
            valid_action_mask = np.ones(7)
            # if at_left_edge:
            #     valid_action_mask[0] = 0
            # if at_bottom_edge:
            #     valid_action_mask[1] = 0
            # if at_right_edge:
            #     valid_action_mask[2] = 0
            # if at_top_edge:
            #     valid_action_mask[3] = 0
            
            # Mask goal actions
            # if not is_at_pickup or is_carrying:
            #     valid_action_mask[4] = 0
            # if not is_at_dropoff or not is_carrying:
            #     valid_action_mask[5] = 0
            
            # Epsilon-greedy exploration
            if random.random() < self.epsilon:
                valid_actions = [a for a in range(7) if valid_action_mask[a] == 1]
                if valid_actions:
                    actions[agent] = random.choice(valid_actions)
                else:
                    actions[agent] = 6  # Wait action
            else:
                # Need network evaluation
                batch_agents.append(agent)
                batch_states.append(state)
                batch_masks[agent] = valid_action_mask
        
        # Only run network if we have agents needing evaluation
        if batch_states:
            # Convert to tensor and process in one batch
            with torch.no_grad():
                state_batch = torch.from_numpy(np.stack(batch_states)).float().to(device)
                
                # Process each agent with its own network
                for i, agent in enumerate(batch_agents):
                    action_probs = self.q_networks[agent](state_batch[i:i+1])
                    q_values = action_probs.cpu().numpy().flatten()
                    
                    # Apply action masking
                    mask = batch_masks[agent]
                    masked_q_values = q_values.copy()
                    for a in range(7):
                        if mask[a] == 0:
                            masked_q_values[a] = 0.0
                    
                    # Renormalize probabilities after masking
                    if np.sum(masked_q_values) > 0:
                        masked_q_values /= np.sum(masked_q_values)
                        
                        # Use specified sampling mode
                        if sampling_mode == 'argmax':
                            # Deterministic action - exploit
                            actions[agent] = np.argmax(masked_q_values)
                        elif sampling_mode == 'sample':
                            # Stochastic action - sample from softmax distribution
                            try:
                                actions[agent] = np.random.choice(range(7), p=masked_q_values)
                            except ValueError:
                                # If probabilities don't sum to 1, fallback to argmax
                                actions[agent] = np.argmax(masked_q_values)
                    else:
                        actions[agent] = 6  # Wait action
        
        return actions

    def get_current_frame_indices(self, state):
        """
        Get the indices of the most recent frame in the state.
        """

        if self.frame_history <= 1:
            return {
                'pickup': 5,
                'dropoff': 6,
                'carrying': 7,
                'boundary': 2,
            }
        else:
            # With frame stacking, the new frame is at the end of the stack
            channel_offset = (self.frame_history - 1) * self.env.observation_channels 
            return {
                'pickup': 5 + channel_offset,
                'dropoff': 6 + channel_offset,
                'carrying': 7 + channel_offset,
                'boundary': 2 + channel_offset,
            }

    def update_epsilon(self):
        """
        Update epsilon
        """
        

        if self.t_step < 10000: # linear warmup period
            self.epsilon = max(self.epsilon_min, self.epsilon_start - (self.epsilon_start - 0.5) * (self.t_step / 10000))
        else: # Exponential decay
            self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def update_temperature(self):
        """Anneal temperature based on learning progress rather than steps"""
        min_temp = 0.2  # Minimum temperature
        
        # Early in training, use high temperature
        if self.t_step < 10000:
            new_temp = 1.0
        else:
            # Calculate performance trend from completed tasks
            if len(self.completed_tasks) > 20:
                recent = sum(self.completed_tasks[-10:])
                older = sum(self.completed_tasks[-20:-10])
                
                # If performance is improving rapidly, maintain higher temperature
                if recent > older * 1.2:  # 20% improvement
                    new_temp = 0.8
                # If performance has plateaued, reduce temperature
                elif recent <= older * 1.05:  # <5% improvement
                    new_temp = 0.6
                # Moderate improvement
                else:
                    new_temp = 0.7
            else:
                # Gradual decrease based on step count as fallback
                new_temp = max(min_temp, 1.0 - (0.5 * min(1.0, self.t_step / 500000)))
        
        # Update temperature in all networks
        for agent in self.env.agents:
            self.q_networks[agent].temperature = new_temp
            self.target_networks[agent].temperature = new_temp
            
        # Log the temperature
        if hasattr(self, 'writer'):
            self.writer.add_scalar("Training/Temperature", new_temp, self.t_step)

    def save_model(self, model_path=None):
        """
        Save the Q-network model to the specified path.
        """

        model_data = {
            agent: self.q_networks[agent].state_dict()
            for agent in self.env.agents
        }

        if model_path is None:
            model_path = get_model_path()
        

        if model_path.endswith('.pth'):
            save_path = model_path
        else:
            os.makedirs(model_path, exist_ok=True)
            save_path = os.path.join(model_path, self.model_name)

        torch.save(model_data, save_path)

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
        # If sample is not provided
        if experiences is None:
            # Only learn if enough samples are available
            if len(self.memory) < self.batch_size:
                return
            # Sample a batch of experiences
            if self.use_per:
                # Sample from PER
                states, actions, rewards, next_states, dones, weights, indices = self.memory.sample()
            else:
                # Sample from normal replay buffer
                states, actions, rewards, next_states, dones = self.memory.sample()
        else:
            if self.use_per:
                # Unpack experiences
                states, actions, rewards, next_states, dones, weights, indices = experiences
            else:
                # Unpack experiences
                states, actions, rewards, next_states, dones = experiences
        
        # Get probabilities from the network
        action_probs = self.q_networks[agent](states)
        
        # Use probabilities for TD error calculation - gather the probabilities for the actions taken
        q_values = action_probs.gather(1, actions)
        
        # Log average Q-value to TensorBoard
        avg_q_value = q_values.mean().item()
        self.writer.add_scalar(f'Training/{agent}/Avg_Q_Value', avg_q_value, self.t_step)

        # Log max Q-value to TensorBoard
        max_q_value = q_values.max().item()
        self.writer.add_scalar(f'Training/{agent}/Max_Q_Value', max_q_value, self.t_step)

        # Log Q-value difference to TensorBoard
        with torch.no_grad():
            target_probs = self.target_networks[agent](states)
            target_q_values = target_probs.gather(1, actions)
        
        # Compute the absolute difference between Q-values and target Q-values
        q_value_diff = (q_values - target_q_values).abs()

        # Log the mean Q-value difference to TensorBoard
        mean_q_value_diff = q_value_diff.mean().item()
        self.writer.add_scalar(f'Training/{agent}/Mean_Q_Value_Difference', mean_q_value_diff, self.t_step)

        # Log the max Q-value difference as well
        max_q_value_diff = q_value_diff.max().item()
        self.writer.add_scalar(f'Training/{agent}/Max_Q_Value_Difference', max_q_value_diff, self.t_step)

        # Clip rewards for stability (-1 to +1 range)
        rewards = torch.clamp(rewards, min=-1.0, max=1.0)

        # Double DQN with probabilities
        with torch.no_grad():
            # Use local Q-network to select the best action for the next state
            next_probs = self.q_networks[agent](next_states)
            next_actions = next_probs.argmax(dim=1, keepdim=True)
            
            # Use target Q-network to evaluate the Q-value of the selected action
            target_next_probs = self.target_networks[agent](next_states)
            next_q_values = target_next_probs.gather(1, next_actions)
            
            # Compute targets using the Double DQN formula
            targets = rewards + (self.gamma * next_q_values * (1 - dones))

        # Calculate TD (Temporal Difference) errors using probabilities
        td_errors = (targets - q_values).detach()
        
        # For PER priority updates, convert to numpy
        td_errors_numpy = td_errors.cpu().numpy()

        # Log TD errors to TensorBoard
        self.writer.add_scalar(f'Training/{agent}/TD_Errors', torch.mean(td_errors.abs()).item(), self.t_step)

        # Store metrics for early stopping
        if not hasattr(self, 'last_loss'):
            self.last_loss = {}
        if not hasattr(self, 'last_td_error'):
            self.last_td_error = {}

        # Store the most recent metrics by agent
        loss_value = None  # Will be set after computing loss

        self.last_td_error[agent] = torch.mean(td_errors.abs()).item()

        # If using PER, update priorities based on TD errors
        if self.use_per:
            # Update priorities in the replay buffer
            if indices is not None:
                self.memory.update_priorities(indices, td_errors_numpy)

            # Log priorities in the replay buffer every 1000 steps
            if self.t_step % 1000 == 0:
                priority_distribution = np.array([self.memory.tree.tree[idx] for idx in range(self.memory.tree.capacity - 1, len(self.memory.tree.tree))])
                self.writer.add_histogram(f'ReplayBuffer/Priorities', priority_distribution, self.t_step) 

            # Compute loss (huber) and weight it by the importance sampling weights
            loss = self.compute_loss(q_values, targets, weights=weights)
        else:
            # Compute loss (huber)
            loss = self.compute_loss(q_values, targets)
        
        # Store loss value for early stopping
        self.last_loss[agent] = loss.item()
            
        # Log loss to TensorBoard
        self.writer.add_scalar(f'Training/{agent}/Loss', loss.item(), self.t_step)

        # Periodically log weight histograms
        if self.t_step % 1000 == 0:
            for name, param in self.q_networks[agent].named_parameters():
                self.writer.add_histogram(f'Weights/{agent}/{name}', param, self.t_step)
                
            # Also log the action probability distribution
            self.writer.add_histogram(f'ActionProbs/{agent}', action_probs, self.t_step)

        # Zero gradients (reset gradients)
        self.optimizers[agent].zero_grad(set_to_none=True)

        # Compute gradients (backpropagation)
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_networks[agent].parameters(), max_norm=1.0)

        # Update based on gradients
        self.optimizers[agent].step()

        # Update target network (soft update)
        self._soft_update(self.q_networks[agent], self.target_networks[agent], self.tau)

    def log_action_distribution(self, actions, episode):
        """Log distribution of actions taken in an episode."""
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        
        # Handle either format of actions (dict of lists or single actions)
        if isinstance(actions, dict) and actions and isinstance(next(iter(actions.values())), list):
            # Format: {agent_id: [actions...]}
            for agent_id, agent_actions in actions.items():
                for action in agent_actions:
                    if isinstance(action, (int, np.integer)):
                        action_counts[action] += 1
        else:
            # Format: {agent_id: action}
            for agent_id, action in actions.items():
                if isinstance(action, (int, np.integer)):
                    action_counts[action] += 1
        
        total = sum(action_counts.values())
        if total > 0:  # Avoid division by zero
            for action_id, count in action_counts.items():
                action_name = ["LEFT", "DOWN", "RIGHT", "UP", "PICKUP", "DROPOFF", "WAIT"][action_id]
                percentage = (count / total) * 100
                self.writer.add_scalar(f'Actions/{action_name}', percentage, episode)  

    def _soft_update(self, local_model, target_model, tau):
        """
        Soft update target newtwork weights.
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def compute_loss(self, q_values, targets, weights=None):
        # Huber loss (with or without PER weights)
        if weights is not None:
            td_loss = (weights * F.smooth_l1_loss(q_values, targets, reduction='none')).mean()
        else:
            td_loss = F.smooth_l1_loss(q_values, targets)
        
        # No output regularization needed with softmax (inherent normalization)
        return td_loss
        
    def determine_sampling_mode(self, episode, n_episodes, performance_history, sampling_mode='auto'):
        """
        Determine sampling mode based on learning curve rather than episode number
        
        Args:
            episode: Current episode
            n_episodes: Max episodes (used as fallback)
            performance_history: List of recent performance metrics
            sampling_mode: The base strategy ('auto', 'sample', or 'argmax')
        """
        if sampling_mode != 'auto':
            return sampling_mode
        
        # For first 50 episodes, always use 'sample' for better diversity
        if episode < 50:
            return 'sample'
        else:
            # After that, prefer argmax
            return 'argmax'
        
# Monitor GPU utilization during training
def check_gpu_usage():
    if torch.cuda.is_available():
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

def train_DQN(env, agent=None, n_episodes=1000, max_steps=1000, debug_level=DEBUG_CRITICAL, 
             save_every=100, phase=1, sampling_mode='auto', use_early_stopping=False):
    """
    Train Q-learning agent in the warehouse environment with interactive controls.
    
    Args:
        env: Warehouse environment.
        agent: Existing agent to continue training (default: None).
        n_episodes: Number of training episodes (default: 1000).
        max_steps: Maximum steps per episode (default: 1000).
        debug_level: Debug level (default: DEBUG_CRITICAL).
        save_every: Save model every n episodes (default: 100).
        phase: Phase of training (default: 1).
        sampling_mode: Action selection mode ('argmax', 'sample', or 'auto') (default: 'auto').
                      'auto' will use 'sample' for first 70% of training, then 'argmax'.
    
    Returns:
        agent: Trained Q-learning agent.
        exit_reason: String indicating why training stopped (completed, interrupted, nextphase)
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
        ql_agent = QLAgent(env, debug_level=debug_level, use_tensorboard=True, phase=phase)

    else:
        # Use provided agent
        old_phase = agent.phase

        ql_agent = agent
        ql_agent.env = env
        ql_agent.phase = phase
        ql_agent.model_name = f"phase_{phase}_dqn_{time.strftime('%Y%m%d-%H%M%S')}.pth"
        ql_agent.model_path = os.path.join(get_model_path(), ql_agent.model_name)

        # If phase is different, reset the TensorBoard writer
        if ql_agent.phase != old_phase:
            ql_agent.writer.close()
            ql_agent.writer = SummaryWriter(log_dir=f"runs/{ql_agent.model_name}")
            ql_agent.debug(DEBUG_INFO, f"TensorBoard writer reset for phase {phase}")
            # log hyperparameters
            hp_dict = {
                'phase': ql_agent.phase,
                'alpha': ql_agent.alpha,
                'gamma': ql_agent.gamma,
                'epsilon_start': ql_agent.epsilon,
                'epsilon_end': ql_agent.epsilon_min,
                'epsilon_decay': ql_agent.epsilon_decay,
                'hidden_size': ql_agent.q_networks[env.agents[0]].fc1.out_features,
                'buffer_size': ql_agent.memory.buffer_size,
                'batch_size': ql_agent.batch_size,
                'update_freq': ql_agent.update_freq,
                'tau': ql_agent.tau,
                'frame_history': ql_agent.frame_history,
                'sampling_mode': sampling_mode
            }
            # Add as text summary
            param_str = "\n".join([f"{k}: {v}" for k, v in hp_dict.items()])
            ql_agent.writer.add_text("Hyperparameters", param_str)
            ql_agent.debug(DEBUG_INFO, f"Hyperparameters logged for phase {phase}")
            # Reset completed tasks and episode rewards
            ql_agent.completed_tasks = []
            ql_agent.episode_rewards = []
            ql_agent.t_step = 0
            ql_agent.epsilon = ql_agent.epsilon_start
            ql_agent.memory = PrioritizedReplayBuffer(ql_agent.memory.buffer_size, ql_agent.batch_size)
            ql_agent.debug(DEBUG_INFO, f"Step counter reset for phase {phase}")

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
    
    # Add tracking for learning metrics
    metrics_history = {
        'loss': [],
        'td_error': []
    }
    best_learning_score = float('inf')  # Lower is better for loss/TD metrics
    best_model_state = None
    patience = 500  # Number of episodes to wait before early stopping
    no_improvement = 0
    best_episode = 0
    
    # Track recent performance for adaptive decisions
    avg_deliveries_history = []
    
    # Set up keyboard interrupt handling for interactive control
    import sys
    import select
    import termios
    import tty
    import threading
    
    # Flag to indicate if we should proceed to next phase or evaluate
    next_phase = False
    quit_training = False
    evaluate_model = False
    
    # Function to check for keypresses without blocking
    def check_input():
        nonlocal next_phase, quit_training, evaluate_model
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while True:
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    if key == 'n':
                        print("\nNext phase requested! Saving current model and proceeding to next phase...")
                        next_phase = True
                    elif key == 'q':
                        print("\nQuit requested! Saving current model and stopping training...")
                        quit_training = True
                    elif key == 'e':
                        print("\nEvaluation requested! Saving current model and running benchmark...")
                        evaluate_model = True
                time.sleep(0.1)
        except Exception as e:
            print(f"Input thread error: {e}")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    # Start input checking thread
    input_thread = threading.Thread(target=check_input, daemon=True)
    input_thread.start()
    
    print("\nTraining control: Press 'n' to move to next phase, 'e' to evaluate current model, 'q' to quit training")

    # Training loop
    exit_reason = "completed"  # Default exit reason
    episode = 1
    while episode <= n_episodes:
        # Check for phase change request
        if next_phase:
            # Save the current model and exit loop
            ql_agent.save_model(ql_agent.model_path)
            print(f"Moving to next phase at episode {episode}")
            exit_reason = "nextphase"
            break
            
        # Check for quit request
        if quit_training:
            # Save the current model and exit training completely
            ql_agent.save_model(ql_agent.model_path)
            print(f"Quitting training at episode {episode}")
            exit_reason = "quit"
            break
        
        # Check for evaluation request
        if evaluate_model:
            # Save the current model
            temp_model_path = ql_agent.model_path.replace('.pth', f'_temp_eval_{episode}.pth')
            ql_agent.save_model(temp_model_path)
            print(f"\nEvaluating current model at episode {episode}...")
            
            # Run benchmark
            benchmark_results = benchmark_environment(env_phase=phase, n_steps=max_steps, 
                                                     debug_level=debug_level, model_path=temp_model_path)
            
            if benchmark_results:
                print("\n=== BENCHMARK RESULTS ===")
                print(f"A* completed tasks: {benchmark_results['astar_tasks']}")
                print(f"RL completed tasks: {benchmark_results['rl_tasks']}")
                print(f"Performance ratio (RL/A*): {benchmark_results['performance_ratio']:.2f}")
                
                # Log benchmark results to TensorBoard
                if hasattr(ql_agent, 'writer'):
                    ql_agent.writer.add_scalar("Benchmark/A*_Tasks", benchmark_results['astar_tasks'], episode)
                    ql_agent.writer.add_scalar("Benchmark/RL_Tasks", benchmark_results['rl_tasks'], episode)
                    ql_agent.writer.add_scalar("Benchmark/Performance_Ratio", benchmark_results['performance_ratio'], episode)
            
            # Reset evaluation flag and continue training
            evaluate_model = False
            
            # Ask if user wants to continue or move to next phase
            user_input = input("\nEvaluation complete. Press Enter to continue training, 'n' for next phase, 'q' to quit: ")
            if user_input.lower() == 'n':
                print("Moving to next phase...")
                exit_reason = "nextphase"
                break
            elif user_input.lower() == 'q':
                print("Quitting training...")
                exit_reason = "quit"
                break
            
            print("\nResuming training...")
            print("Training control: Press 'n' to move to next phase, 'e' to evaluate current model, 'q' to quit training")
            continue
        
        # Run a single episode
        episode_start = time.time()
        
        # Reset environment
        env_reset_start = time.time()
        observations, _ = env.reset()
        total_env_time += time.time() - env_reset_start
        score = 0

        # Track actions for this episode
        episode_actions = {}
        
        # Determine which sampling mode to use for this episode
        current_sampling_mode = sampling_mode
        if sampling_mode == 'auto':
            current_sampling_mode = ql_agent.determine_sampling_mode(
                episode, 
                n_episodes,
                avg_deliveries_history, 
                sampling_mode
            )

        # Run episode
        for step in range(max_steps):
            # Check for interactive control requests
            if next_phase or quit_training or evaluate_model:
                break
                
            # Time action selection
            select_action_start = time.time()
            actions = ql_agent.select_action_batch(observations, sampling_mode=current_sampling_mode)
            total_select_action_time += time.time() - select_action_start

            # Store actions for this episode
            for agent_id, action in actions.items():
                if agent_id not in episode_actions:
                    episode_actions[agent_id] = []
                episode_actions[agent_id].append(action)

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
            
            # Debug info
            ql_agent.debug(DEBUG_INFO, f"Episode {episode}, Step {step}")
            ql_agent.debug(DEBUG_INFO, f"Actions: {actions}")
            ql_agent.debug(DEBUG_INFO, f"Rewards: {rewards}")
            ql_agent.debug(DEBUG_INFO, f"Completed tasks: {env.completed_tasks}")
            ql_agent.debug(DEBUG_INFO, f"Score: {score}")

            # Check if episode is done
            if all(terminations.values() or all(truncations.values())):
                break

        # Skip the rest of episode processing if we received a control request
        if next_phase or quit_training or evaluate_model:
            continue

        # Update epsilon for exploration
        ql_agent.update_epsilon()

        # Update temperature for action probabilities
        ql_agent.update_temperature()

        # Save score
        scores.append(score)

        # Track performance metrics
        ql_agent.episode_rewards.append(score)
        ql_agent.completed_tasks.append(sum(env.completed_tasks.values()))

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
            f"Epsilon: {ql_agent.epsilon:.4f} | "
            f"Mode: {current_sampling_mode}")
        
        # Log individual episode data to TensorBoard
        ql_agent.writer.add_scalar("Performance/Reward", score, episode)
        ql_agent.writer.add_scalar("Performance/Deliveries", episode_deliveries, episode)
        ql_agent.writer.add_scalar("Training/Epsilon", ql_agent.epsilon, episode)
        ql_agent.writer.add_scalar("Training/NoImprovement", no_improvement, episode)
        ql_agent.writer.add_scalar("Training/SamplingMode", 1 if current_sampling_mode == 'sample' else 0, episode)

        # Log average performance over last 10 episodes
        if episode >= 10:
            avg_reward = sum(scores[-10:]) / 10
            ql_agent.writer.add_scalar("Performance/Avg10Reward", avg_reward, episode)
            ql_agent.writer.add_scalar("Performance/Avg10Deliveries", avg_deliveries, episode)

        # Track action distribution for the episode
        ql_agent.log_action_distribution(episode_actions, episode)

        # Early stopping based on learning metrics stabilization
        if use_early_stopping == True and episode > 10:  # After we have enough data
            # Get the most recent loss and TD error values
            recent_losses = []
            recent_td_errors = []
            
            for agent_id in env.agents:
                # Access metrics directly from the agent
                if hasattr(ql_agent, 'last_loss') and agent_id in ql_agent.last_loss:
                    recent_losses.append(ql_agent.last_loss[agent_id])
                if hasattr(ql_agent, 'last_td_error') and agent_id in ql_agent.last_td_error:
                    recent_td_errors.append(ql_agent.last_td_error[agent_id])
            
            # If we have fresh metrics
            if recent_losses and recent_td_errors:
                # Calculate average across agents
                avg_loss = sum(recent_losses) / len(recent_losses)
                avg_td_error = sum(recent_td_errors) / len(recent_td_errors)
                
                # Store in history
                metrics_history['loss'].append(avg_loss)
                metrics_history['td_error'].append(avg_td_error)
                
                # Use exponential moving average for smoother signals
                if 'ema_loss' not in metrics_history:
                    metrics_history['ema_loss'] = avg_loss
                    metrics_history['ema_td_error'] = avg_td_error
                else:
                    # Update EMA with 0.95 weight for previous value, 0.05 for new value
                    metrics_history['ema_loss'] = 0.95 * metrics_history['ema_loss'] + 0.05 * avg_loss
                    metrics_history['ema_td_error'] = 0.95 * metrics_history['ema_td_error'] + 0.05 * avg_td_error
                
                # Calculate learning score using smoothed metrics
                learning_score = (0.6 * metrics_history['ema_loss'] + 0.4 * metrics_history['ema_td_error'])
                
                # Log the learning score
                ql_agent.writer.add_scalar("Training/LearningScore", learning_score, episode)
                
                # Check for stabilization (convergence)
                if episode > 1000:  # Only check after significant training
                    # Calculate the rate of change in learning score
                    if 'recent_scores' not in metrics_history:
                        metrics_history['recent_scores'] = deque(maxlen=300)
                    
                    metrics_history['recent_scores'].append(learning_score)
                    
                    if len(metrics_history['recent_scores']) >= 200:
                        # Calculate slope of recent learning scores
                        window_size = 50
                        smoothed_scores = []
                        recent = list(metrics_history['recent_scores'])

                        for i in range(len(recent) - window_size + 1):
                            smoothed_scores.append(sum(recent[i:i+window_size]) / window_size)

                        x = np.array(range(len(smoothed_scores)))
                        y = np.array(smoothed_scores)
                        
                        # Simple linear regression to get slope
                        slope = np.polyfit(x, y, 1)[0]
                        
                        # Log the slope
                        ql_agent.writer.add_scalar("Training/LearningScoreSlope", slope, episode)
                        
                        # Check if learning has stabilized (slope is near zero)
                        # We use absolute value since we care about stability, not direction
                        slope_threshold = 0.00002  # Very small slope indicates stabilization
                        
                        if abs(slope) < slope_threshold:
                            no_improvement += 1
                        else:
                            no_improvement = 0

                        if episode > 1500:
                            # Check for absolute stability regardless of slope
                            recent_100 = list(metrics_history['recent_scores'])[-100:]
                            max_val = max(recent_100)
                            min_val = min(recent_100)
                            relative_range = (max_val - min_val) / np.mean(recent_100)
                            
                            # If variation is less than 3%, consider stable
                            if relative_range < 0.03:
                                no_improvement += 3  # Count more heavily toward patience
                            
                        # Save model state when learning is still changing significantly
                        if episode > best_episode + 500:  # Don't save too frequently
                            best_episode = episode
                            best_model_state = {
                                agent: ql_agent.q_networks[agent].state_dict().copy() 
                                for agent in env.agents
                            }
                            
                            # Save to disk
                            ql_agent.save_model(ql_agent.model_path.replace('.pth', '_best.pth'))
                            ql_agent.debug(DEBUG_INFO, f"New best model saved (episode {episode}, learning still progressing)")
                        
                        # Stop when learning has stabilized for 'patience' episodes
                        if no_improvement >= patience:
                            print(f"Early stopping at episode {episode}. Learning stabilized for {patience} episodes.")
                            print(f"Best model was saved at episode {best_episode}")
                            
                            # Restore best model if available
                            if best_model_state:
                                for agent in env.agents:
                                    ql_agent.q_networks[agent].load_state_dict(best_model_state[agent])
                                    ql_agent.target_networks[agent].load_state_dict(best_model_state[agent])
                                    
                                print(f"Restored model from episode {best_episode}")
                            exit_reason = "earlystopping"
                            break

        # Save model periodically
        if episode % save_every == 0:
            ql_agent.save_model(ql_agent.model_path)
            
        # Print timing info every 10 episodes
        if episode % 10 == 0:
            print(f"\nTiming breakdown (avg per episode):")
            print(f"  Environment time: {total_env_time/episode:.4f}s")
            print(f"  Action selection time: {total_select_action_time/episode:.4f}s")
            print(f"  Agent learning time: {total_step_time/episode:.4f}s")
            print(f"  Episode duration: {(time.time() - episode_start):.4f}s")
            print(f"  Press 'n' to move to next phase, 'e' to evaluate, 'q' to quit training\n")

        # Log action probability visualizations
        if episode % 10 == 0:
            with torch.no_grad():
                # Sample a batch from replay buffer if available
                if len(ql_agent.memory) >= ql_agent.batch_size:
                    if ql_agent.use_per:
                        states, _, _, _, _, _, _ = ql_agent.memory.sample()
                    else:
                        states, _, _, _, _ = ql_agent.memory.sample()
                    
                    # Get probabilities for visualization
                    for agent in env.agents:
                        probs = ql_agent.q_networks[agent](states)
                        average_probs = probs.mean(dim=0).cpu().numpy()
                        
                        # Log probability distribution
                        action_names = ["LEFT", "DOWN", "RIGHT", "UP", "PICKUP", "DROPOFF", "WAIT"]
                        for i, name in enumerate(action_names):
                            ql_agent.writer.add_scalar(f"ActionProbs/{agent}/{name}", average_probs[i], episode)
        
        # After calculating avg_deliveries:
        avg_deliveries_history.append(avg_deliveries)
        
        # Increment episode counter at the end of each successful episode
        episode += 1

    # Save final model
    ql_agent.save_model(ql_agent.model_path)

    try:
        env.close()
    except Exception as e:
        print(f"Error closing environment: {e}")

    # Return trained agent and exit reason
    return ql_agent, exit_reason

def train_DQN_curriculum(target_env, n_episodes=1000, max_steps=1000, debug_level=DEBUG_CRITICAL, save_every=100, model_path=get_model_path()):
    """
    Train Q-learning agent with improved curriculum learning.
    """
    # Create models directory if it doesn't exist
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created models directory: {model_dir}")
    
    print(f"Using model dir: {model_dir}")
    
    # STAGE 1: Basic Navigation (5x5, fully observable)
    print("\n=== STAGE 1: Basic Navigation Training ===")
    print("Press 'n' during training to advance to next phase, 'e' to evaluate current model, 'q' to quit training completely")
    
    stage1_env = env(grid_size=(5, 5), n_agents=1, n_humans=0, num_shelves=0, 
                    num_pickup_points=1, num_dropoff_points=1, render_mode="human")
    agent, exit_reason = train_DQN(stage1_env, n_episodes=200000, max_steps=100, save_every=50, phase=1)

    p1_model_path = agent.model_path
    
    # Check if training should continue
    if exit_reason == "quit":
        print("Training terminated by user request after Stage 1")
        return None
    
    # Full benchmark (just once) to see A* comparison
    print("\nBenchmarking Stage 1 against A*...")
    benchmark1 = benchmark_environment(env_phase=1, n_steps=100, debug_level=DEBUG_NONE, model_path=p1_model_path)
    
    # STAGE 2: Navigation with simple obstacles
    print("\n=== STAGE 2: Navigation with Obstacles ===")
    print("Press 'n' during training to advance to next phase, 'e' to evaluate current model, 'q' to quit training completely")
    
    agent.epsilon = 0.9
    
    stage2_env = env(grid_size=(10, 8), n_agents=1, n_humans=0, num_shelves=16, 
                    num_pickup_points=1, num_dropoff_points=1, render_mode="human")
    agent, exit_reason = train_DQN(stage2_env, agent=agent, n_episodes=200000, max_steps=150, 
                     save_every=25, phase=2)

    p2_model_path = agent.model_path
    
    # Check if training should continue
    if exit_reason == "quit":
        print("Training terminated by user request after Stage 2")
        return {
            "stage1_benchmark": benchmark1,
            "stage2_benchmark": None,
            "stage3_benchmark": None,
        }
    
    # Full benchmark (just once) to see A* comparison
    print("\nBenchmarking Stage 2 against A*...")
    benchmark2 = benchmark_environment(env_phase=2, n_steps=150, debug_level=DEBUG_NONE, model_path=p2_model_path)

    # Stage 3: Final training, multi-agent with dynamic obstacles
    print("\n=== STAGE 3: Multi-agent with Dynamic Obstacles ===")
    print("Press 'n' during training to advance to next phase, 'e' to evaluate current model, 'q' to quit training completely")
    
    agent.epsilon = 0.9

    warehouse_env = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=6, n_humans=10, num_shelves=2048, num_pickup_points=3,
                            num_dropoff_points=2, render_mode="human")
    
    # Extract network architecture parameters from Phase 2
    phase2_network = agent.q_networks['agent_0']
    phase2_state_dict = phase2_network.state_dict()
    input_channels = phase2_network.actual_channels
    height = phase2_network.height
    width = phase2_network.width
    frame_history = phase2_network.frame_history
    hidden_size = phase2_network.fc1.out_features // 2  # Divide by 2 since it uses hidden_size*2

    # Initialize all new agent networks with the phase 2 knowledge
    for agent_id in warehouse_env.agents:
        if agent_id != 'agent_0':
            agent.q_networks[agent_id] = QNetwork((input_channels//frame_history, height, width), 7, hidden_size, frame_history).to(device)
            agent.target_networks[agent_id] = QNetwork((input_channels//frame_history, height, width), 7, hidden_size, frame_history).to(device)

            agent.q_networks[agent_id].load_state_dict(phase2_state_dict)
            agent.target_networks[agent_id].load_state_dict(phase2_state_dict)

            agent.optimizers[agent_id] = torch.optim.AdamW(agent.q_networks[agent_id].parameters(), lr=agent.alpha, eps=1e-5, weight_decay=0.0001)

    # Train Phase 3
    agent, exit_reason = train_DQN(warehouse_env, agent=agent, n_episodes=200000, max_steps=300, 
                     save_every=25, phase=3)
    
    p3_model_path = agent.model_path

    # Full benchmark (just once) to see A* comparison
    print("\nBenchmarking Stage 3 against A*...")
    benchmark3 = benchmark_environment(env_phase=3, n_steps=300, debug_level=DEBUG_NONE, model_path=p3_model_path)

    # Save final model
    agent.save_model(model_path)
    
    # Print summary results
    print("\n=== CURRICULUM LEARNING COMPLETE ===")
    print("Performance Summary:")
    print(f"Stage 1: A* Tasks: {benchmark1['astar_tasks']}, RL Tasks: {benchmark1['rl_tasks']}, Ratio: {benchmark1['performance_ratio']:.2f}")
    print(f"Stage 2: A* Tasks: {benchmark2['astar_tasks']}, RL Tasks: {benchmark2['rl_tasks']}, Ratio: {benchmark2['performance_ratio']:.2f}")
    print(f"Stage 3: A* Tasks: {benchmark3['astar_tasks']}, RL Tasks: {benchmark3['rl_tasks']}, Ratio: {benchmark3['performance_ratio']:.2f}")

    # Return performance metrics
    return {
        "stage1_benchmark": benchmark1,
        "stage2_benchmark": benchmark2,
        "stage3_benchmark": benchmark3,
    }

def run_q_learning(env, full_model_path, n_steps=1000, debug_level=DEBUG_NONE, sampling_mode='argmax'):
    """
    Run trained Q-learning agent in the warehouse environment.
    """
    
    # Initialize agent in evaluation mode
    QL_agent = QLAgent(env, debug_level=debug_level)
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

def benchmark_environment(env_phase, n_steps=200, debug_level=DEBUG_NONE, model_path=get_model_path()):
    """
    Run both A* and RL agents on the same environment to establish performance benchmarks.
    """
    import matplotlib.pyplot as plt
    from copy import deepcopy
    import time

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
    
    total_a_star_completed_tasks = 0
    total_dqn_completed_tasks = 0
    
    # For 100 episodes, run both agents with the same seed, creating fresh environments each time
    for i in range(100):
        # Create a new seed for this episode
        seed = random.randint(0, 10000)
        
        # Create fresh environments for each episode with the same seed
        if env_phase == 1:
            env_a_star = env(grid_size=(5, 5), n_agents=1, n_humans=0, num_shelves=0, 
                        num_pickup_points=1, num_dropoff_points=1, render_mode=None, seed=seed)
            env_dqn = env(grid_size=(5, 5), n_agents=1, n_humans=0, num_shelves=0, 
                        num_pickup_points=1, num_dropoff_points=1, render_mode=None, seed=seed)
        elif env_phase == 2:
            env_a_star = env(grid_size=(10, 8), n_agents=1, n_humans=0, num_shelves=16, 
                        num_pickup_points=1, num_dropoff_points=1, render_mode=None, seed=seed)
            env_dqn = env(grid_size=(10, 8), n_agents=1, n_humans=0, num_shelves=16, 
                        num_pickup_points=1, num_dropoff_points=1, render_mode=None, seed=seed)
        elif env_phase == 3:
            env_a_star = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=6, n_humans=10, num_shelves=2048, num_pickup_points=3,
                            num_dropoff_points=2, render_mode=None, seed=seed)
            env_dqn = env(grid_size=(34, 32), human_grid_size=(34, 32), n_agents=6, n_humans=10, num_shelves=2048, num_pickup_points=3,
                            num_dropoff_points=2, render_mode=None, seed=seed)
        else:
            raise ValueError("Invalid phase. Must be 1, 2, or 3.")
        
        # Reset environments
        env_a_star.reset()
        env_dqn.reset()
        
        # Run A* agent
        astar_tasks = run_a_star(env_a_star, n_steps=n_steps, debug_level=DEBUG_NONE)
        
        # Close environment and plots
        env_a_star.close()
        plt.close('all')
        time.sleep(1)
    
        # Run RL agent with explicit model loading
        rl_tasks = run_q_learning(env_dqn, full_model_path=model_path, n_steps=n_steps, debug_level=DEBUG_NONE)
        
        # Close environment and plots
        env_dqn.close()
        plt.close('all')

        # Update total completed tasks
        total_a_star_completed_tasks += astar_tasks
        total_dqn_completed_tasks += rl_tasks
        
        # Print progress every 10 episodes
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/100 benchmark episodes")
            print(f"Current totals - A*: {total_a_star_completed_tasks}, RL: {total_dqn_completed_tasks}")
    
    # Calculate performance ratio (RL / A*)
    if total_a_star_completed_tasks and total_a_star_completed_tasks > 0:
        performance_ratio = total_dqn_completed_tasks / total_a_star_completed_tasks
    else:
        performance_ratio = 0
        
    # Print benchmark results
    print("\n=== BENCHMARK RESULTS ===")
    print(f"A* completed tasks: {total_a_star_completed_tasks}")
    print(f"RL completed tasks: {total_dqn_completed_tasks}")
    print(f"Performance ratio (RL/A*): {performance_ratio:.2f}")
    
    return {
        "astar_tasks": total_a_star_completed_tasks,
        "rl_tasks": total_dqn_completed_tasks,
        "performance_ratio": performance_ratio
    }
