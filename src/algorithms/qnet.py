import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Neural network for Q-value approximation."""

    def __init__(self, input_size, action_size, hidden_size=64, frame_history=4, use_softmax=False):
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
        self.use_softmax = use_softmax

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
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu', mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
                

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


        # Output layer, either probabilities or Q-values
        if self.use_softmax:
            return F.softmax(self.fc3(x) / self.temperature, dim=1)
        else:
            return self.fc3(x)
        
class QMixNetwork(nn.Module):
    """
    Mixing network for QMIX algorithm.
    Takes individual Q-values and produces a joint Q-value.
    """

    def __init__(self, num_agents, state_dim, mixing_embed_dim=32, use_popart=False):
        """
        Initialize network parameters.

        Args:
            num_agents: Number of agents.
            state_dim: Size of the state.
            mixing_embed_dim: Size of the mixing network embedding.
        """
        
        super(QMixNetwork, self).__init__()
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim
        self.use_popart = use_popart

        # Hypernetworks that produce weights and biases for the mixing network
        self.hyper_w1 = nn.Linear(state_dim, num_agents * mixing_embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)

        # Second layer (output)
        self.hyper_w2 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b2 = nn.Linear(state_dim, 1)

        if use_popart:
            # PopArt normalization layer
            self.popart = PopArtLayer(1, 1, beta=0.0003)
        else:
            self.popart = None
        
        # Initialize weights with Glorot
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier/Glorot initialization for better stability
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, agent_qs, state, denormalize=True):
        """
        Forward pass through the mixing network.

        Args:
            agent_qs: Individual Q-values from agents [batch_size, num_agents].
            state: Global state information [batch_size, state_dim].
            denormalize: Whether to denormalize the output using PopArt.
        """

        batch_size = agent_qs.size(0)

        # First layer
        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        # Second layer
        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        # Output
        joint_q = torch.bmm(hidden, w2) + b2
        joint_q = joint_q.view(batch_size, 1)

        # Apply PopArt normalization if enabled
        if self.use_popart and self.popart is not None and denormalize:
            joint_q = self.popart.denormalize(joint_q) # Don't use linear layer

        return joint_q.squeeze(-1)
    
    def normalize_targets(self, targets):
        """Normalize targets using PopArt statistics."""
        if self.use_popart and self.popart is not None:
            return self.popart.normalize(targets)
        return targets
    
    def update_popart_stats(self, targets):
        """Update PopArt statistics with new targets."""
        if self.use_popart and self.popart is not None:
            self.popart.update_stats(targets)
    
class PopArtLayer(nn.Module):
    """
    PopArt normalization layer that normalizes targets during training
    and denormalizes predictions during inference.
    """
    def __init__(self, input_size, output_size=1, beta=0.0001):
        """
        Initialize PopArt layer.
        """
        super(PopArtLayer, self).__init__()
        
        # Linear layer for value prediction
        self.linear = nn.Linear(input_size, output_size)
        
        # Use scalar statistics for simplicity
        self.register_buffer('mu', torch.zeros(1))
        self.register_buffer('sigma', torch.ones(1))
        self.register_buffer('old_mu', torch.zeros(1))
        self.register_buffer('old_sigma', torch.ones(1))
        
        self.beta = beta
        self.output_size = output_size
    
    def normalize(self, y):
        """Normalize targets using current statistics."""
        return (y - self.mu) / (self.sigma + 1e-8)

    def denormalize(self, y):
        """Denormalize predictions using current statistics."""
        return y * self.sigma + self.mu
    
    def update_stats(self, targets):
        """Update running statistics with new batch of targets."""
        # Save old statistics
        self.old_mu.copy_(self.mu)
        self.old_sigma.copy_(self.sigma)
        
        # Compute batch statistics - reduce to scalar 
        batch_mean = targets.mean().detach()
        batch_var = targets.var().detach()
        batch_std = torch.sqrt(batch_var + 1e-8)
        
        # Update running statistics (scalars)
        self.mu = self.beta * batch_mean + (1 - self.beta) * self.mu
        self.sigma = self.beta * batch_std + (1 - self.beta) * self.sigma
        
        # Update weights and bias to preserve outputs
        self._update_parameters()

        # Return current statistics for logging
        return self.mu.item(), self.sigma.item()

    def _update_parameters(self):
        """
        Update linear layer parameters to preserve outputs after statistics update.
        This ensures that denormalize(normalize(x)) = x even after statistics change.
        """
        # Get the ratio of old to new standard deviations
        ratio = self.old_sigma / (self.sigma + 1e-8)
        
        # Update weights and bias
        self.linear.weight.data.mul_(ratio)
        self.linear.bias.data.mul_(ratio)
        self.linear.bias.data.add_(self.old_mu - self.mu * ratio)

