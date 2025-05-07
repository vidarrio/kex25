

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import matplotlib.pyplot as plt
import torch

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO, sharing convolutional backbone."""
    def __init__(self, input_size, action_size, hidden_size=64):
        super(ActorCritic, self).__init__()
        channels, height, width = input_size
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        flattened = 32 * height * width
        # Shared fully connected
        self.fc_shared = nn.Linear(flattened, hidden_size)
        # Actor head
        self.fc_actor = nn.Linear(hidden_size, action_size)
        # Critic head
        self.fc_critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))
        logits = self.fc_actor(x)
        value = self.fc_critic(x)
        return logits, value

class PPOMemory:
    """Simple memory for on-policy PPO."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        self.__init__()

class PPOAgent:
    """Proximal Policy Optimization agent."""
    def __init__(self, input_size, action_size, hidden_size=64,
                 lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=10,
                 value_coef=0.5, entropy_coef=0.01):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Actor-Critic networks
        self.policy = ActorCritic(input_size, action_size, hidden_size).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        # Clone for old policy
        self.policy_old = ActorCritic(input_size, action_size, hidden_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def select_action_batch(self, observations, memory):
        """Select actions for all agents, store in memory."""
        states = torch.FloatTensor(np.stack(list(observations.values()))).to(device)
        logits, _ = self.policy_old(states)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        logprobs = dist.log_prob(actions)

        memory.states.append(states)
        memory.actions.append(actions)
        memory.logprobs.append(logprobs)

        action_dict = {agent: actions[i].item() for i, agent in enumerate(observations.keys())}
        return action_dict

    def update(self, memory):
        states = torch.cat(memory.states, dim=0)
        actions = torch.cat(memory.actions, dim=0)
        old_logprobs = torch.cat(memory.logprobs, dim=0).detach()

        rewards = [r for step in memory.rewards for r in step]
        dones = [d for term in memory.is_terminals for d in term]
        returns = []
        discounted = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted = 0
            discounted = reward + self.gamma * discounted
            returns.insert(0, discounted)
        # Convert to tensor and normalize returns to stabilize training
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        _, values = self.policy(states)
        values = values.squeeze()
        # Compute and normalize advantages for better training stability
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.K_epochs):
            logits, val = self.policy(states)
            dist = Categorical(logits=logits)
            logprobs = dist.log_prob(actions)
            entropy = dist.entropy()

            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = (-torch.min(surr1, surr2) +
                    self.value_coef * F.mse_loss(val.squeeze(), returns) -
                    self.entropy_coef * entropy).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

def train_PPO(env, n_episodes=1000, max_steps=1000, save_prefix='ppo_model', **kwargs):
    """
    Train PPO agent on the given environment.
    Save model and plots using save_prefix.
    """
    # --- 1) Prepare to log ---
    episode_rewards = []
    deliveries_per_episode = []
    collisions_per_episode = []

    # (existing initialization code)
    input_size = (10, *env.observation_size)
    action_size = env.action_space(env.possible_agents[0]).n
    agent = PPOAgent(input_size, action_size, **kwargs)
    memory = PPOMemory()
    try:
        for episode in range(1, n_episodes+1):
            # start each episode
            ep_reward = 0
            observations, _ = env.reset()

            for step in range(max_steps):
                # select & apply action
                actions = agent.select_action_batch(observations, memory)
                next_obs, rewards, terms, truns, _ = env.step(actions)

                # accumulate this step’s reward
                ep_reward += sum(rewards.values())

                # store for PPO update
                memory.rewards.append([rewards[a] for a in env.agents])
                memory.is_terminals.append([terms[a]   for a in env.agents])

                observations = next_obs
                if all(terms.values()):
                    break

            # record and run your PPO update
            episode_rewards.append(ep_reward)
            # Record deliveries and collisions
            deliveries_per_episode.append(sum(env.completed_tasks.values()))
            collisions_per_episode.append(sum(env.collisions.values()))
            agent.update(memory)
            memory.clear()

            if episode % 10 == 0:
                print(f"PPO Episode {episode} — total reward: {ep_reward}")
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving model and rewards...")
    finally:
        # Save trained policy weights
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f'{save_prefix}.pth')
        torch.save(agent.policy.state_dict(), model_path)
        print(f"PPO policy saved to {model_path}")

        # Plot training metrics: rewards, deliveries, collisions
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))

        axs[0].plot(episode_rewards)
        axs[0].set_title("PPO Episode Rewards")
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Total Reward")
        axs[0].grid(True)

        axs[1].plot(deliveries_per_episode)
        axs[1].set_title("Deliveries per Episode")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Number of Deliveries")
        axs[1].grid(True)

        axs[2].plot(collisions_per_episode)
        axs[2].set_title("Collisions per Episode")
        axs[2].set_xlabel("Episode")
        axs[2].set_ylabel("Number of Collisions")
        axs[2].grid(True)

        fig.tight_layout()
        plot_path = os.path.join(models_dir, f'{save_prefix}_metrics.png')
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Training metrics plot saved to {plot_path}")

    return agent


def evaluate_PPO(env, model_path, n_episodes=10, max_steps=1000, render=True):
    """
    Run a trained PPO policy in the environment and optionally render each episode.
    """
    # Reconstruct agent architecture
    input_size  = (10, *env.observation_size)
    action_size = env.action_space(env.possible_agents[0]).n
    agent = PPOAgent(input_size, action_size)
    # Load trained weights
    agent.policy.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy.eval()

    for ep in range(1, n_episodes+1):
        obs, _ = env.reset()
        total_reward = 0

        for t in range(max_steps):
            # Batch all agent observations into a tensor
            states = torch.FloatTensor(np.stack(list(obs.values()))).to(device)
            with torch.no_grad():
                logits, _ = agent.policy(states)
            # Greedy action selection
            action_idxs = torch.argmax(logits, dim=-1).cpu().numpy()
            actions = {
                agent_name: int(action_idxs[i])
                for i, agent_name in enumerate(obs.keys())
            }

            obs, rewards, terminations, truncations, info = env.step(actions)
            total_reward += sum(rewards.values())

            if render:
                env.render()
                print(f"Step {t}: Actions={actions}, Rewards={rewards}")

            if all(terminations.values()):
                break

        deliveries = sum(env.completed_tasks.values())
        print(f"Eval Ep {ep}: TotalReward={total_reward:.1f}, Deliveries={deliveries}")

    env.close()