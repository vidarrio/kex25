import random
import numpy as np
import torch
from collections import deque
from collections import namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define experience tuple type
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

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
        self.max_priority = 1.0  # Track max priority for new samples
        self.min_priority = 1.0  # Track min priority for IS weights

    def _propagate(self, idx, change):
        """
        Propagate priority change up the tree.
        
        Args:
            idx: Index of the leaf node.
            change: Change in priority.
        """
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            if parent == 0:  # Reached root
                break
            parent = (parent - 1) // 2

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
        
    def total_priority(self):
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
        # Calculate leaf index
        leaf_idx = self.write_index + self.capacity - 1
        
        # Update data and priority
        self.data[self.write_index] = data
        self.update(leaf_idx, priority)
        
        # Update min/max priorities
        self.max_priority = max(self.max_priority, priority)
        self.min_priority = min(self.min_priority, priority)
        
        # Update write index and entry count
        self.write_index = (self.write_index + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        """
        Update the priority of an experience.
        
        Args:
            idx: Index of the experience in the tree.
            priority: New priority value.
        """
        # Calculate change in priority
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        
        # Update min/max priorities
        if priority > 0:  # Only track valid priorities
            self.max_priority = max(self.max_priority, priority)
            self.min_priority = min(self.min_priority, priority)
        
        # Propagate change up the tree
        self._propagate(idx, change)

    def batch_update(self, idxs, priorities):
        """
        Update priorities for a batch of experiences.
        
        Args:
            idxs: Indices of the experiences to update.
            priorities: New priority values.
        """
        for idx, priority in zip(idxs, priorities):
            change = priority - self.tree[idx]
            self.tree[idx] = priority
            self._propagate(idx, change)

        # Update min/max priorities
        if len(priorities) > 0:
            self.max_priority = max(self.max_priority, max(priorities))
            valid_priorities = [p for p in priorities if p > 0]
            if valid_priorities:
                self.min_priority = min(self.min_priority, min(valid_priorities))

    def get(self, value):
        """
        Get experience based on priority value.
        
        Args:
            value: Priority value to search for.
        """
        # Get leaf node index
        idx = self._retrieve(0, value)
        
        # Calculate data index (relative to data array)
        data_idx = idx - (self.capacity - 1)
        
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
        self.epsilon = 1e-5 # Small constant to avoid zero priorities

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
            priority = self.tree.max_priority
        else:
            # Calculate priority based on TD error
            priority = (abs(error) + self.epsilon) ** self.alpha
        
        self.tree.add(priority, experience)
        self.frame += 1  # Increment frame counter for beta annealing
        self.beta = self.beta_by_frame()  # Update beta

    def sample(self):
        """
        Sample a batch of experiences based on their priorities using the SumTree.
        Returns:
            states (torch.Tensor): Batch of states
            actions (torch.Tensor): Batch of actions
            rewards (torch.Tensor): Batch of rewards
            next_states (torch.Tensor): Batch of next states
            dones (torch.Tensor): Batch of done flags
            weights (torch.Tensor): Importance sampling weights
            indices (np.array): Array of sampled indices
        """
        # Handle edge case: not enough samples
        if self.tree.n_entries == 0:
            # Return default tensors
            empty_state_shape = (0,) + next(iter(self.tree.data)).state.shape
            return (torch.zeros(empty_state_shape, device=device),
                    torch.zeros((0, 1), dtype=torch.long, device=device),
                    torch.zeros((0, 1), device=device),
                    torch.zeros(empty_state_shape, device=device),
                    torch.zeros((0, 1), device=device),
                    torch.zeros((0, 1), device=device),
                    np.array([]))
        
        # Use vectorized numpy operations for sampling
        total_priority = self.tree.total_priority()
        
        # Create segment boundaries
        segment_size = total_priority / self.batch_size
        segment_bounds = np.linspace(0, total_priority, self.batch_size + 1)
        
        # Sample from each segment with slight randomness
        values = np.random.uniform(segment_bounds[:-1], segment_bounds[1:])
        
        # Preallocate arrays
        indices = np.empty(self.batch_size, dtype=np.int32)
        priorities = np.empty(self.batch_size, dtype=np.float32)
        batch = []
        
        # Get samples based on priority values (could be parallelized further if needed)
        for i, value in enumerate(values):
            idx = self.tree._retrieve(0, value)
            data_idx = idx - (self.tree.capacity - 1)
            
            indices[i] = idx
            priorities[i] = self.tree.tree[idx]
            batch.append(self.tree.data[data_idx])
        
        # Calculate importance sampling weights in a vectorized way
        # Current beta value for importance sampling
        beta = self.beta
        
        # Normalize priorities to probabilities
        probs = priorities / total_priority
        
        # Calculate weights with minimal operations - vectorized
        weights = (probs * self.tree.n_entries) ** (-beta)
        
        # Normalize weights to max_weight for stability
        max_weight = ((self.tree.min_priority / total_priority) * self.tree.n_entries) ** (-beta)
        weights = weights / max_weight
        
        # Collect experience elements using numpy operations for efficiency
        states = np.stack([exp.state for exp in batch])
        actions = np.vstack([exp.action for exp in batch])
        rewards = np.vstack([exp.reward for exp in batch])
        next_states = np.stack([exp.next_state for exp in batch])
        dones = np.vstack([exp.done for exp in batch]).astype(np.float32)
        
        # Transfer to GPU in a single batch rather than individually
        states_tensor = torch.from_numpy(states).float().to(device, non_blocking=True)
        actions_tensor = torch.from_numpy(actions).long().to(device, non_blocking=True)
        rewards_tensor = torch.from_numpy(rewards).float().to(device, non_blocking=True)
        next_states_tensor = torch.from_numpy(next_states).float().to(device, non_blocking=True)
        dones_tensor = torch.from_numpy(dones).float().to(device, non_blocking=True)
        weights_tensor = torch.from_numpy(weights).float().unsqueeze(1).to(device, non_blocking=True)
        
        return (states_tensor, actions_tensor, rewards_tensor, 
                next_states_tensor, dones_tensor, weights_tensor, indices)
    
    def update_priorities(self, idxs, errors):
        """
        Update priorities based on new TD errors.

        Args:
            idxs: Indices of the experiences to update.
            errors: New TD errors for the experiences.
        """
        # Clip very small errors to avoid numerical instability
        errors = np.clip(np.abs(errors), self.epsilon, 10.0)
        priorities = (errors + self.epsilon) ** self.alpha
        
        # Update priorities based on TD errors
        self.tree.batch_update(idxs, priorities)
    
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

    def add(self, experience):
        """
        Add new experience to the buffer memory.
        """
        
        # Prioritize successful experiences
        if experience.reward > 10.0: # task completion
            # Add experience multiple times
            for _ in range(3):
                self.memory.append(experience)
        else:
            # Add experience once
            self.memory.append(experience)

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

class QMIXReplayBuffer:
    """
    Replay buffer for QMIX algorithm that stores individual agent experiences 
    alongside global state information.
    """

    def __init__(self, buffer_size, batch_size, use_priority=False, 
                 alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=100000):
        """
        Initialize QMIX replay buffer.
        
        Args:
            buffer_size: Maximum size of buffer
            batch_size: Size of training batch
            use_priority: Whether to use prioritized experience replay
            alpha: PER exponent (0 = uniform sampling, higher = more prioritization)
            beta_start: Starting importance sampling correction
            beta_end: Final importance sampling correction
            beta_frames: Number of frames for beta annealing
        """
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.use_priority = use_priority
        
        # Define QMIX experience tuple type with global state
        self.QMIXExperience = namedtuple(
            'QMIXExperience', 
            ['agent_states', 'agent_actions', 'agent_rewards', 
             'agent_next_states', 'agent_dones', 
             'global_state', 'next_global_state']
        )

        # Create appropriate storage based on prioritization choice
        if use_priority:
            self.tree = SumTree(buffer_size)
            self.alpha = alpha
            self.beta = beta_start
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.beta_frames = beta_frames
            self.frame = 1
            self.epsilon = 1e-5
        else:
            self.memory = deque(maxlen=buffer_size)

    def add(self, agent_states, agent_actions, agent_rewards, 
            agent_next_states, agent_dones, global_state, next_global_state, error=None):
        """
        Add experience to buffer
        
        Args:
            agent_states: States for each agent [num_agents, state_shape]
            agent_actions: Actions for each agent [num_agents]
            agent_rewards: Rewards for each agent [num_agents]
            agent_next_states: Next states for each agent [num_agents, state_shape]
            agent_dones: Done flags for each agent [num_agents]
            global_state: Global state representation [global_state_shape]
            next_global_state: Next global state [global_state_shape]
            error: TD error for prioritization (optional)
        """
        # Create experience tuple
        experience = self.QMIXExperience(
            agent_states, agent_actions, agent_rewards, 
            agent_next_states, agent_dones, 
            global_state, next_global_state
        )
        
        # Add to appropriate storage
        if self.use_priority:
            if error is None:
                priority = self.tree.max_priority
            else:
                priority = (abs(error) + self.epsilon) ** self.alpha
            
            self.tree.add(priority, experience)
            self.frame += 1
            self.beta = min(self.beta_end, self.beta_start + 
                           (self.beta_end - self.beta_start) * 
                           (self.frame / self.beta_frames))
        else:
            self.memory.append(experience)

    def sample(self):
        """
        Sample a batch of experiences from buffer
        
        Returns:
            Tuple of tensors for QMIX training:
            - agent_states: [batch_size, num_agents, state_shape]
            - agent_actions: [batch_size, num_agents]
            - agent_rewards: [batch_size, num_agents]
            - agent_next_states: [batch_size, num_agents, state_shape]
            - agent_dones: [batch_size, num_agents]
            - global_states: [batch_size, global_state_shape]
            - next_global_states: [batch_size, global_state_shape]
            - weights: Importance sampling weights (if using PER)
            - indices: Sampled indices (if using PER)
        """
        if self.use_priority:
            return self._sample_prioritized()
        else:
            return self._sample_uniform()

    def _sample_uniform(self):
        """Sample uniformly from replay buffer"""
        if len(self.memory) < self.batch_size:
            # Return empty tensors if not enough samples
            return None
            
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # Extract and batch experience components
        agent_states = np.stack([e.agent_states for e in experiences])
        agent_actions = np.stack([e.agent_actions for e in experiences])
        agent_rewards = np.stack([e.agent_rewards for e in experiences])
        agent_next_states = np.stack([e.agent_next_states for e in experiences])
        agent_dones = np.stack([e.agent_dones for e in experiences]).astype(np.float32)
        global_states = np.stack([e.global_state for e in experiences])
        next_global_states = np.stack([e.next_global_state for e in experiences])
        
        # Transfer to device
        agent_states = torch.from_numpy(agent_states).float().to(device, non_blocking=True)
        agent_actions = torch.from_numpy(agent_actions).long().to(device, non_blocking=True)
        agent_rewards = torch.from_numpy(agent_rewards).float().to(device, non_blocking=True)
        agent_next_states = torch.from_numpy(agent_next_states).float().to(device, non_blocking=True)
        agent_dones = torch.from_numpy(agent_dones).float().to(device, non_blocking=True)
        global_states = torch.from_numpy(global_states).float().to(device, non_blocking=True)
        next_global_states = torch.from_numpy(next_global_states).float().to(device, non_blocking=True)
        
        # For uniform sampling, weights are all 1s and indices are None
        weights = torch.ones((self.batch_size, 1), device=device)
        indices = None
        
        return (agent_states, agent_actions, agent_rewards, 
                agent_next_states, agent_dones, 
                global_states, next_global_states, 
                weights, indices)

    def _sample_prioritized(self):
        """Sample based on priorities using SumTree"""
        if self.tree.n_entries == 0 or self.tree.n_entries < self.batch_size:
            # Return empty tensors if not enough samples
            return None
        
        # Use vectorized numpy operations for sampling
        total_priority = self.tree.total_priority()
        
        # Create segment boundaries
        segment_size = total_priority / self.batch_size
        segment_bounds = np.linspace(0, total_priority, self.batch_size + 1)
        
        # Sample from each segment with slight randomness
        values = np.random.uniform(segment_bounds[:-1], segment_bounds[1:])
        
        # Preallocate arrays
        indices = np.empty(self.batch_size, dtype=np.int32)
        priorities = np.empty(self.batch_size, dtype=np.float32)
        batch = []
        
        # Get samples based on priority values
        for i, value in enumerate(values):
            idx, priority, experience = self.tree.get(value)
            
            indices[i] = idx
            priorities[i] = priority
            batch.append(experience)
        
        # Calculate importance sampling weights
        probs = priorities / total_priority
        weights = (probs * self.tree.n_entries) ** (-self.beta)
        max_weight = ((self.tree.min_priority / total_priority) * self.tree.n_entries) ** (-self.beta)
        weights = weights / max_weight
        
        # Extract experience components
        agent_states = np.stack([e.agent_states for e in batch])
        agent_actions = np.stack([e.agent_actions for e in batch])
        agent_rewards = np.stack([e.agent_rewards for e in batch])
        agent_next_states = np.stack([e.agent_next_states for e in batch])
        agent_dones = np.stack([e.agent_dones for e in batch]).astype(np.float32)
        global_states = np.stack([e.global_state for e in batch])
        next_global_states = np.stack([e.next_global_state for e in batch])
        
        # Transfer to device
        agent_states = torch.from_numpy(agent_states).float().to(device)
        agent_actions = torch.from_numpy(agent_actions).long().to(device)
        agent_rewards = torch.from_numpy(agent_rewards).float().to(device)
        agent_next_states = torch.from_numpy(agent_next_states).float().to(device)
        agent_dones = torch.from_numpy(agent_dones).float().to(device)
        global_states = torch.from_numpy(global_states).float().to(device)
        next_global_states = torch.from_numpy(next_global_states).float().to(device)
        weights = torch.from_numpy(weights).float().unsqueeze(1).to(device)
        
        return (agent_states, agent_actions, agent_rewards, 
                agent_next_states, agent_dones, 
                global_states, next_global_states, 
                weights, indices)

    def update_priorities(self, indices, errors):
        """Update priorities for prioritized replay based on TD errors"""
        if not self.use_priority or indices is None:
            return
            
        # Clip very small errors to avoid numerical instability
        errors = np.clip(np.abs(errors), self.epsilon, 10.0)
        priorities = (errors + self.epsilon) ** self.alpha
        
        # Update priorities based on TD errors
        self.tree.batch_update(indices, priorities)
    
    def __len__(self):
        """Return the current size of the buffer."""
        if self.use_priority:
            return self.tree.n_entries
        else:
            return len(self.memory)