# Standard library imports
import os
import random
import time

# Third party imports
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Local application imports
from .qnet import QNetwork, QMixNetwork
from .replay import ReplayBuffer, PrioritizedReplayBuffer, Experience, QMIXReplayBuffer
from .common import device, get_model_path, DEBUG_NONE, DEBUG_CRITICAL, DEBUG_INFO, DEBUG_VERBOSE, DEBUG_ALL, DEBUG_SPECIFIC

class QLAgent:
    """
    Q-learning agent for warehouse navigation.
    """

    def __init__(self, env, debug_level=DEBUG_NONE, phase=1,
                 alpha=0.00025, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1,
                 epsilon_decay=0.995, hidden_size=128, buffer_size=200000, batch_size=64,
                 update_freq=8, tau=0.0005, use_tensorboard=True, use_per=True, use_softmax=False,
                 use_qmix=False, use_popart=False, use_parameter_sharing=True):
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
            use_softmax: Whether to use softmax action selection (default: False).
            use_qmix: Whether to use QMIX for multi-agent training (default: False).
            use_popart: Whether to use PopArt for normalization (default: False).
            use_parameter_sharing: Whether to use parameter sharing for multi-agent training (default: True).
        """
        
        self.env = env
        self.debug_level = debug_level
        self.phase = phase
        self.use_tensorboard = use_tensorboard
        self.use_per = use_per
        self.use_softmax = use_softmax
        self.use_popart = use_popart
        self.use_parameter_sharing = use_parameter_sharing
        if phase == 1:
            self.start_time = time.strftime("%Y%m%d-%H%M%S")
        self.model_name = f"phase_{self.phase}_dqn_{self.start_time}.pth"
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

        if use_parameter_sharing:
            # Use shared Q-network for all agents
            self.q_networks['shared'] = QNetwork(self.state_size, self.action_size, hidden_size, self.frame_history).to(device)
            self.target_networks['shared'] = QNetwork(self.state_size, self.action_size, hidden_size, self.frame_history).to(device)
            # Copy weights from Q-network to target network
            self.target_networks['shared'].load_state_dict(self.q_networks['shared'].state_dict())
            self.optimizers['shared'] = torch.optim.AdamW(self.q_networks['shared'].parameters(), lr=alpha, eps=1e-5, weight_decay=0.0001)
        else:
            # Initialize separate Q-networks for each agent
            for agent in env.agents:
                self.q_networks[agent] = QNetwork(self.state_size, self.action_size, hidden_size, self.frame_history).to(device)
                self.target_networks[agent] = QNetwork(self.state_size, self.action_size, hidden_size, self.frame_history).to(device)
                # Copy weights from Q-network to target network
                self.target_networks[agent].load_state_dict(self.q_networks[agent].state_dict())
                self.optimizers[agent] = torch.optim.AdamW(self.q_networks[agent].parameters(), lr=alpha, eps=1e-5, weight_decay=0.0001)

        # Add QMIX-specific components if enabled
        self.use_qmix = use_qmix
        if self.use_qmix:
            # Calculate state dimension based on global state shape
            # Get a sample global state to determine dimensions
            dummy_state, _ = env.get_global_state()
            state_dim = dummy_state.shape[0]  # Flattened state size
            
            # Initialize QMIX networks with PopArt
            self.mixer = QMixNetwork(len(env.agents), state_dim, use_popart=self.use_popart).to(device)
            self.target_mixer = QMixNetwork(len(env.agents), state_dim, use_popart=self.use_popart).to(device)
            # Copy weights from mixer to target mixer
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.mixer_optimizer = torch.optim.AdamW(
                self.mixer.parameters(), lr=alpha, eps=1e-5, weight_decay=0.0001
            )
            
            # If using PER, the QMIX buffer will handle it
            self.qmix_memory = QMIXReplayBuffer(buffer_size, batch_size, use_priority=self.use_per)
        else:
            # Standard replay buffer initialization (your existing code)
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
                'use_per': use_per,
                'use_softmax': use_softmax,
                'use_qmix': use_qmix,
                'use_popart': use_popart,
                'use_parameter_sharing': use_parameter_sharing,
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
            experience = Experience(states[agent], action[agent], reward[agent], next_state[agent], done[agent])
            self.memory.add(experience)

        # Increment step counter
        self.t_step += 1
        
        # Update model less often for multiple agents
        update_freq = 4 if len(self.env.agents) == 1 else self.update_freq
        
        # Learn every update_freq steps as long as we are out of initial frame history
        if self.frame_history < self.t_step and self.t_step % update_freq == 0 and len(self.memory) >= self.batch_size:
            agent = self.env.agents[self.t_step % len(self.env.agents)]
            self._learn(agent)

    def select_action(self, state, agent, eval_mode=False, sampling_mode='argmax'):
        """Return action using epsilon-greedy without masking."""
        
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
        
        # Epsilon-greedy exploration: pure random actions with probability epsilon
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, 6)  # Random action from all possible actions
        
        # Exploitation: Use network to get action values
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            # Use shared network if parameter sharing is enabled, otherwise use agent-specific network
            if self.use_parameter_sharing:
                q_values = self.q_networks['shared'](state_tensor).cpu().numpy().flatten()
            else:
                q_values = self.q_networks[agent](state_tensor).cpu().numpy().flatten()
        
        # Different handling based on value type
        if self.use_softmax:
            # During evaluation or if using argmax sampling, take most probable action
            if eval_mode or sampling_mode == 'argmax':
                return np.argmax(q_values)
            
            # For sampling mode, sample from softmax distribution
            elif sampling_mode == 'sample':
                try:
                    return np.random.choice(range(7), p=q_values)
                except ValueError:
                    # If probabilities don't sum to 1, fallback to argmax
                    return np.argmax(q_values)
        else:
            # For raw Q-values, just use argmax
            return np.argmax(q_values)

    def select_action_batch(self, observations, sampling_mode='argmax'):
        """
        Select actions for all agents using epsilon-greedy without masking.
        """
        indices = self.get_current_frame_indices(observations)
        actions = {}
        
        # Prepare batch for network inference
        batch_states = []
        batch_agents = []
        
        # Process each agent
        for agent in self.env.agents:
            state = observations[agent]
            is_at_pickup = state[indices['pickup'], 2, 2] > 0.5
            is_at_dropoff = state[indices['dropoff'], 2, 2] > 0.5
            is_carrying = state[indices['carrying'], 2, 2] > 0.5
            
            # When at goal states, strongly bias to pickup/dropoff but still allow exploration
            if is_at_pickup and not is_carrying:
                if random.random() < 0.9:
                    actions[agent] = 4  # pickup
                    continue
            elif is_at_dropoff and is_carrying:
                if random.random() < 0.9:
                    actions[agent] = 5  # dropoff
                    continue
            
            # Epsilon-greedy exploration
            if random.random() < self.epsilon:
                actions[agent] = random.randint(0, 6)  # Random action from all possible actions
            else:
                # Need network evaluation
                batch_agents.append(agent)
                batch_states.append(state)
        
        # Only run network if we have agents needing evaluation
        if batch_states:
            # Convert to tensor and process in one batch
            with torch.no_grad():
                state_batch = torch.from_numpy(np.stack(batch_states)).float().to(device)
                
                if self.use_parameter_sharing:
                    # When parameter sharing is enabled, evaluate all states in a single batch
                    q_values_batch = self.q_networks['shared'](state_batch).cpu().numpy()
                    
                    for i, agent in enumerate(batch_agents):
                        q_values = q_values_batch[i]
                        
                        # Different handling based on value type
                        if self.use_softmax:
                            # Use specified sampling mode
                            if sampling_mode == 'argmax':
                                actions[agent] = np.argmax(q_values)
                            elif sampling_mode == 'sample':
                                try:
                                    actions[agent] = np.random.choice(range(7), p=q_values)
                                except ValueError:
                                    actions[agent] = np.argmax(q_values)
                        else:
                            actions[agent] = np.argmax(q_values)
                else:
                    # When using individual networks, process each agent separately
                    for i, agent in enumerate(batch_agents):
                        q_values = self.q_networks[agent](state_batch[i:i+1]).cpu().numpy().flatten()
                        
                        # Different handling based on value type
                        if self.use_softmax:
                            # Use specified sampling mode
                            if sampling_mode == 'argmax':
                                actions[agent] = np.argmax(q_values)
                            elif sampling_mode == 'sample':
                                try:
                                    actions[agent] = np.random.choice(range(7), p=q_values)
                                except ValueError:
                                    actions[agent] = np.argmax(q_values)
                        else:
                            actions[agent] = np.argmax(q_values)
        
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
        
        # Update temperature in networks based on parameter sharing
        if self.use_parameter_sharing:
            self.q_networks['shared'].temperature = new_temp
            self.target_networks['shared'].temperature = new_temp
        else:
            for agent in self.env.agents:
                self.q_networks[agent].temperature = new_temp
                self.target_networks[agent].temperature = new_temp
            
        # Log the temperature
        if hasattr(self, 'writer'):
            self.writer.add_scalar("Training/Temperature", new_temp, self.t_step)

    def save_model(self, model_path=None):
        """
        Save the Q-network model and QMIX mixer (if used) to the specified path.
        """
        # Save networks based on parameter sharing mode
        if self.use_parameter_sharing:
            model_data = {
                'shared': self.q_networks['shared'].state_dict()
            }
        else:
            model_data = {
                agent: self.q_networks[agent].state_dict()
                for agent in self.env.agents
            }
        
        # Add parameter sharing flag to the saved model
        model_data['use_parameter_sharing'] = self.use_parameter_sharing
        
        # If using QMIX, also save the mixer
        if self.use_qmix and len(self.env.agents) > 1:
            model_data['mixer'] = self.mixer.state_dict()

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
        Load the Q-network model and QMIX mixer (if available) from the specified path.
        """
        model_data = torch.load(filepath, map_location=device)
        
        # Check if the model was saved with parameter sharing
        saved_with_parameter_sharing = model_data.get('use_parameter_sharing', False)
        
        # Load networks based on how the model was saved
        if saved_with_parameter_sharing:
            if 'shared' in model_data:
                self.q_networks['shared'].load_state_dict(model_data['shared'])
                self.target_networks['shared'].load_state_dict(model_data['shared'])
            else:
                print("Warning: Model saved with parameter sharing but no shared model found")
        else:
            # Load individual agent networks
            for agent in self.env.agents:
                if agent in model_data:
                    self.q_networks[agent].load_state_dict(model_data[agent])
                    self.target_networks[agent].load_state_dict(model_data[agent])
                else:
                    print(f"Warning: No model data found for agent {agent}")
        
        # Load mixer if using QMIX and it's in the saved model
        if self.use_qmix and 'mixer' in model_data:
            self.mixer.load_state_dict(model_data['mixer'])
            self.target_mixer.load_state_dict(model_data['mixer'])
            print("Loaded QMIX mixer from saved model")
        
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

        # Use the shared network if parameter sharing is enabled
        network_to_use = 'shared' if self.use_parameter_sharing else agent

        if self.use_softmax:
            # Get probabilities from the network
            action_probs = self.q_networks[network_to_use](states)
            
            # Use probabilities for TD error calculation - gather the probabilities for the actions taken
            q_values = action_probs.gather(1, actions)

        if not self.use_softmax:
            # Get Q-values from the network
            q_values = self.q_networks[network_to_use](states).gather(1, actions)
        
        # Log average Q-value to TensorBoard
        avg_q_value = q_values.mean().item()
        self.writer.add_scalar(f'Training/{network_to_use}/Avg_Q_Value', avg_q_value, self.t_step)

        # Log max Q-value to TensorBoard
        max_q_value = q_values.max().item()
        self.writer.add_scalar(f'Training/{network_to_use}/Max_Q_Value', max_q_value, self.t_step)

        # Log Q-value difference to TensorBoard
        if self.use_softmax:
            with torch.no_grad():
                target_probs = self.target_networks[network_to_use](states)
                target_q_values = target_probs.gather(1, actions)
        
        if not self.use_softmax:
            with torch.no_grad():
                # Get target Q-values from the target network
                target_q_values = self.target_networks[network_to_use](states).gather(1, actions)
        
        # Compute the absolute difference between Q-values and target Q-values
        q_value_diff = (q_values - target_q_values).abs()

        # Log the mean Q-value difference to TensorBoard
        mean_q_value_diff = q_value_diff.mean().item()
        self.writer.add_scalar(f'Training/{network_to_use}/Mean_Q_Value_Difference', mean_q_value_diff, self.t_step)

        # Log the max Q-value difference as well
        max_q_value_diff = q_value_diff.max().item()
        self.writer.add_scalar(f'Training/{network_to_use}/Max_Q_Value_Difference', max_q_value_diff, self.t_step)

        # Double DQN with probabilities (softmax)
        if self.use_softmax:
            with torch.no_grad():
                # Use local Q-network to select the best action for the next state
                next_probs = self.q_networks[network_to_use](next_states)
                next_actions = next_probs.argmax(dim=1, keepdim=True)
                
                # Use target Q-network to evaluate the Q-value of the selected action
                target_next_probs = self.target_networks[network_to_use](next_states)
                next_q_values = target_next_probs.gather(1, next_actions)
                
                # Compute targets using the Double DQN formula
                targets = rewards + (self.gamma * next_q_values * (1 - dones))

        # Double DQN with raw Q-values (argmax)
        if not self.use_softmax:
            with torch.no_grad():
                # Use local Q-network to select the best action for the next state
                next_actions = self.q_networks[network_to_use](next_states).argmax(dim=1, keepdim=True)
                
                # Use target Q-network to evaluate the Q-value of the selected action
                target_next_q_values = self.target_networks[network_to_use](next_states).gather(1, next_actions)
                
                # Compute targets using the Double DQN formula
                targets = rewards + (self.gamma * target_next_q_values * (1 - dones))

        # Calculate TD (Temporal Difference) errors
        td_errors = (targets - q_values).abs().detach()
        
        # For PER priority updates, convert to numpy
        td_errors_numpy = td_errors.cpu().numpy()

        # Log TD errors to TensorBoard
        self.writer.add_scalar(f'Training/{network_to_use}/TD_Errors', torch.mean(td_errors.abs()).item(), self.t_step)

        # Store metrics for early stopping
        if not hasattr(self, 'last_loss'):
            self.last_loss = {}
        if not hasattr(self, 'last_td_error'):
            self.last_td_error = {}

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
        self.last_loss[network_to_use] = loss.item()
            
        # Log loss to TensorBoard
        self.writer.add_scalar(f'Training/{network_to_use}/Loss', loss.item(), self.t_step)

        # Periodically log weight histograms
        if self.t_step % 1000 == 0:
            for name, param in self.q_networks[network_to_use].named_parameters():
                self.writer.add_histogram(f'Weights/{network_to_use}/{name}', param, self.t_step)
            
            if self.use_softmax:
                # Also log the action probability distribution
                self.writer.add_histogram(f'ActionProbs/{network_to_use}', action_probs, self.t_step)

        # Zero gradients (reset gradients)
        self.optimizers[network_to_use].zero_grad(set_to_none=True)

        # Compute gradients (backpropagation)
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_networks[network_to_use].parameters(), max_norm=1.0)

        # Update based on gradients
        self.optimizers[network_to_use].step()

        # Update target network (soft update)
        self._soft_update(self.q_networks[network_to_use], self.target_networks[network_to_use], self.tau)

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
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Args:
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau: Interpolation parameter 
        """
        # Handle PopArt statistics separately if present
        has_popart = hasattr(local_model, 'popart') and local_model.popart is not None
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        
        # Special handling for PopArt buffers
        if has_popart:
            # Directly copy PopArt statistics
            target_model.popart.mu.copy_(local_model.popart.mu)
            target_model.popart.sigma.copy_(local_model.popart.sigma)
            target_model.popart.old_mu.copy_(local_model.popart.old_mu)
            target_model.popart.old_sigma.copy_(local_model.popart.old_sigma)
    
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

    def step_qmix(self, states, actions, rewards, next_states, dones, global_state, next_global_state):
        """
        Store experience with global state information and perform QMIX learning.
        Works for both single and multi-agent scenarios.
        """

        current_agents = list(self.env.agents)

        # Handle case based on number of agents
        if len(current_agents) == 1:
            # For single agent, still use QMIX but as a state-augmentation technique
            agent = current_agents[0]  # Get the single agent
            
            # Create agent arrays with shape [1, ...] for the single agent
            agent_states = np.expand_dims(states[agent], axis=0)
            agent_actions = np.expand_dims(actions[agent], axis=0)
            agent_rewards = np.expand_dims(rewards[agent], axis=0)
            agent_next_states = np.expand_dims(next_states[agent], axis=0)
            agent_dones = np.expand_dims(dones[agent], axis=0)
        else:
            # Use consistent ordering of agents
            sorted_agents = sorted(current_agents)
            # Multi-agent case - convert dictionaries to arrays for batch processing
            agent_states = np.array([states[agent] for agent in sorted_agents])
            agent_actions = np.array([actions[agent] for agent in sorted_agents])
            agent_rewards = np.array([rewards[agent] for agent in sorted_agents])
            agent_next_states = np.array([next_states[agent] for agent in sorted_agents])
            agent_dones = np.array([dones[agent] for agent in sorted_agents])
        
        # Add to QMIX replay buffer
        self.qmix_memory.add(
            agent_states, agent_actions, agent_rewards,
            agent_next_states, agent_dones,
            global_state, next_global_state
        )
        
        # Increment step counter
        self.t_step += 1
        
        # Learn every update_freq steps as long as we have enough samples
        if self.t_step % self.update_freq == 0 and len(self.qmix_memory) >= self.batch_size:
            self._learn_qmix()

    def _learn_qmix(self):
        """
        Update Q-networks and mixer using QMIX algorithm with PopArt normalization.
        """
        # Sample batch of experiences
        batch = self.qmix_memory.sample()
        if batch is None:  # Not enough samples
            return
            
        if self.use_per:
            agent_states, agent_actions, agent_rewards, agent_next_states, agent_dones, \
            global_states, next_global_states, weights, indices = batch
        else:
            agent_states, agent_actions, agent_rewards, agent_next_states, agent_dones, \
            global_states, next_global_states, weights, indices = batch
        
        batch_size = agent_states.shape[0]
        
        # Always use current environment agents
        current_agents = sorted(list(self.env.agents))
        num_agents = len(current_agents)
        
        # Calculate chosen Q-values for each agent
        chosen_action_qvals = []
        target_max_qvals = []
        
        # Determine which network to use
        network_to_use = 'shared' if self.use_parameter_sharing else None
        
        for agent_idx, agent in enumerate(current_agents):
            if agent_idx >= agent_states.shape[1]:
                # Skip agents that exceed the batch shape
                continue
                
            # Get the appropriate network based on parameter sharing setting
            if self.use_parameter_sharing:
                q_network = self.q_networks['shared']
                target_network = self.target_networks['shared']
            else:
                q_network = self.q_networks[agent]
                target_network = self.target_networks[agent]
                
            # Get Q-values for chosen actions from online network
            q_values = q_network(agent_states[:, agent_idx])
            chosen_action_qval = q_values.gather(1, agent_actions[:, agent_idx].unsqueeze(1)).squeeze(1)
            chosen_action_qvals.append(chosen_action_qval)
            
            # Get max Q-values for next states from target network
            with torch.no_grad():
                target_q_values = target_network(agent_next_states[:, agent_idx])
                target_max_qval = target_q_values.max(dim=1)[0]
                target_max_qvals.append(target_max_qval)
        
        # Stack Q-values for all agents
        chosen_action_qvals = torch.stack(chosen_action_qvals, dim=1)  # [batch_size, num_agents]
        target_max_qvals = torch.stack(target_max_qvals, dim=1)  # [batch_size, num_agents]
        
        # Mix individual agent Q-values using the mixer network
        chosen_mixed_qvals = self.mixer(chosen_action_qvals, global_states, denormalize=False)
        
        with torch.no_grad():
            target_mixed_qvals = self.target_mixer(target_max_qvals, next_global_states, denormalize=False)
        
        # Calculate targets (bootstrapped Q-values)
        # Sum rewards across agents for joint reward
        total_rewards = agent_rewards.sum(dim=1)
        
        # If any agent is done, mark the joint state as done
        joint_dones = agent_dones.max(dim=1)[0]
        
        # Calculate target for centralized critic
        targets_raw = total_rewards + (1 - joint_dones) * self.gamma * target_mixed_qvals
        
        # PopArt normalization if enabled
        if self.use_popart:
            # First denormalize for proper logging
            targets_denorm = targets_raw
            if self.mixer.popart is not None:
                mu, sigma = self.mixer.popart.update_stats(targets_denorm.detach().unsqueeze(1))
                
                # Log statistics
                if self.use_tensorboard and hasattr(self, 'writer') and self.writer:
                    self.writer.add_scalar('PopArt/Mean', mu, self.t_step)
                    self.writer.add_scalar('PopArt/StdDev', sigma, self.t_step)
                
            # Normalize both targets and predictions
            targets = self.mixer.normalize_targets(targets_raw)
            chosen_mixed_norm = self.mixer.normalize_targets(chosen_mixed_qvals)
            
            # Loss
            td_error = targets.detach() - chosen_mixed_norm
            loss = (weights * td_error.pow(2)).mean()
        else:
            # Standard QMIX loss without normalization
            targets = targets_raw
            td_error = targets.detach() - chosen_mixed_qvals
            loss = (weights * td_error.pow(2)).mean()
        
        # Logging
        if self.use_tensorboard and hasattr(self, 'writer') and self.writer:
            self.writer.add_scalar('QMIX/Loss', loss.item(), self.t_step)
            self.writer.add_scalar('QMIX/Mean_Q', chosen_mixed_qvals.mean().item(), self.t_step)
            self.writer.add_scalar('QMIX/Target_Value', targets.mean().item(), self.t_step)
        
        # Zero gradients
        self.mixer_optimizer.zero_grad()
        
        # Zero gradients for agent networks based on parameter sharing
        if self.use_parameter_sharing:
            self.optimizers['shared'].zero_grad()
        else:
            for agent in current_agents:
                if agent in self.optimizers:
                    self.optimizers[agent].zero_grad()
        
        # Compute gradients
        loss.backward()
        
        # Clip gradients to prevent explosion
        if self.use_parameter_sharing:
            torch.nn.utils.clip_grad_norm_(
                self.q_networks['shared'].parameters(), max_norm=1.0
            )
        else:
            for agent in current_agents:
                if agent in self.q_networks:
                    torch.nn.utils.clip_grad_norm_(
                        self.q_networks[agent].parameters(), max_norm=1.0
                    )
        
        torch.nn.utils.clip_grad_norm_(
            self.mixer.parameters(), max_norm=1.0
        )
        
        # Apply gradients
        self.mixer_optimizer.step()
        
        if self.use_parameter_sharing:
            self.optimizers['shared'].step()
        else:
            for agent in current_agents:
                if agent in self.optimizers:
                    self.optimizers[agent].step()
        
        # Update target networks
        #if self.t_step % (self.update_freq * 10) == 0:
        self._soft_update(self.mixer, self.target_mixer, self.tau)
            
        if self.use_parameter_sharing:
            self._soft_update(self.q_networks['shared'], self.target_networks['shared'], self.tau)
        else:
            for agent in current_agents:
                if agent in self.q_networks and agent in self.target_networks:
                    self._soft_update(self.q_networks[agent], self.target_networks[agent], self.tau)
        
        # Update priorities in replay buffer if using PER
        if self.use_per and indices is not None:
            # Calculate absolute TD errors and take mean over agents for each experience
            td_error_abs = td_error.abs().detach()
            
            # Convert to numpy and ensure it's 1D
            td_error_numpy = td_error_abs.cpu().numpy()
            
            # Use this 1D array for priority updates
            self.qmix_memory.update_priorities(indices, td_error_numpy)
        
        return loss.item()