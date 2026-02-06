"""
Reinforcement Learning Bidder
Implements PPO and DQN algorithms for optimal bid optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random
from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class Experience:
    """Single experience tuple"""
    state: np.ndarray
    action: float
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class BiddingEnvironment:
    """
    Simulated bidding environment for RL training
    """
    
    def __init__(
        self,
        ctr_model,
        initial_budget: float = 1000.0,
        episode_length: int = 1000
    ):
        self.ctr_model = ctr_model
        self.initial_budget = initial_budget
        self.episode_length = episode_length
        
        # State space: [budget_remaining, time_remaining, avg_ctr, avg_win_rate, current_bid_landscape]
        self.state_dim = 5
        
        # Action space: bid multiplier (continuous, 0.1 to 2.0)
        self.action_dim = 1
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.budget_remaining = self.initial_budget
        self.time_step = 0
        self.total_impressions = 0
        self.total_clicks = 0
        self.total_spent = 0.0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state"""
        budget_fraction = self.budget_remaining / self.initial_budget
        time_fraction = self.time_step / self.episode_length
        
        avg_ctr = self.total_clicks / max(self.total_impressions, 1)
        avg_win_rate = self.total_impressions / max(self.time_step, 1)
        
        # Simulate market competition (random walk)
        bid_landscape = 0.5 + np.random.randn() * 0.1
        
        return np.array([
            budget_fraction,
            time_fraction,
            avg_ctr,
            avg_win_rate,
            bid_landscape
        ], dtype=np.float32)
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action and return next state, reward, done, info
        
        Args:
            action: Bid multiplier (0.1 to 2.0)
        
        Returns:
            next_state, reward, done, info
        """
        # Clip action
        action = np.clip(action, 0.1, 2.0)
        
        # Simulate auction
        base_bid = 1.0
        bid_amount = base_bid * action
        
        # Simulate CTR
        true_ctr = np.random.beta(2, 50)  # Avg ~0.04
        
        # Simulate competition (higher bid = higher win prob)
        market_price = np.random.lognormal(0.5, 0.5)
        win_prob = 1.0 / (1.0 + np.exp(-2 * (bid_amount - market_price)))
        
        won = np.random.random() < win_prob
        
        # Calculate costs and outcomes
        if won:
            # Second-price: pay market price
            price_paid = market_price * 0.9  # Pay slightly less
            
            if price_paid > self.budget_remaining:
                won = False
                price_paid = 0.0
            else:
                self.budget_remaining -= price_paid
                self.total_spent += price_paid
                self.total_impressions += 1
                
                # Check for click
                clicked = np.random.random() < true_ctr
                if clicked:
                    self.total_clicks += 1
                    revenue = 1.0  # $1 per click
                else:
                    revenue = 0.0
        else:
            price_paid = 0.0
            revenue = 0.0
        
        # Calculate reward
        if won:
            # Reward = revenue - cost + efficiency bonus
            immediate_reward = revenue - price_paid
            
            # Bonus for staying on budget pace
            ideal_budget_fraction = 1.0 - (self.time_step / self.episode_length)
            actual_budget_fraction = self.budget_remaining / self.initial_budget
            pacing_bonus = -abs(ideal_budget_fraction - actual_budget_fraction)
            
            reward = immediate_reward + 0.1 * pacing_bonus
        else:
            # Small penalty for not bidding (opportunity cost)
            reward = -0.01
        
        # Update time
        self.time_step += 1
        
        # Check if episode is done
        done = (self.time_step >= self.episode_length) or (self.budget_remaining <= 0)
        
        # Get next state
        next_state = self._get_state()
        
        info = {
            'won': won,
            'price_paid': price_paid,
            'revenue': revenue,
            'budget_remaining': self.budget_remaining,
            'total_clicks': self.total_clicks,
            'total_impressions': self.total_impressions
        }
        
        return next_state, reward, done, info


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for PPO
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Linear(64, 1)
    
    def forward(self, state):
        """Forward pass"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Actor output
        action_mean = torch.sigmoid(self.actor_mean(x)) * 1.9 + 0.1  # Scale to [0.1, 2.0]
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
        
        # Critic output
        value = self.critic(x)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        """Get action from policy"""
        action_mean, action_std, value = self.forward(state)
        
        if deterministic:
            action = action_mean
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
        
        action = torch.clamp(action, 0.1, 2.0)
        
        return action, value


class PPOBidder:
    """
    Proximal Policy Optimization (PPO) Bidder
    State-of-the-art RL algorithm for continuous control
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.network = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        self.training_step = 0
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> float:
        """Select action given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, _ = self.network.get_action(state_tensor, deterministic)
        
        return action.item()
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_val * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, dtype=torch.float32)
    
    def update(self, trajectories: List[Dict], epochs: int = 4):
        """Update policy using PPO"""
        # Prepare batch data
        states = torch.FloatTensor([t['state'] for t in trajectories])
        actions = torch.FloatTensor([t['action'] for t in trajectories])
        rewards = [t['reward'] for t in trajectories]
        dones = [t['done'] for t in trajectories]
        
        # Compute values and advantages
        with torch.no_grad():
            _, _, values = self.network(states)
            values = values.squeeze(-1).numpy()
            next_value = 0 if dones[-1] else values[-1]
        
        advantages = self.compute_gae(rewards, values, dones, next_value)
        returns = advantages + torch.tensor(values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(epochs):
            # Forward pass
            action_mean, action_std, current_values = self.network(states)
            current_values = current_values.squeeze(-1)
            
            # Policy loss
            dist = Normal(action_mean, action_std)
            log_probs = dist.log_prob(actions.unsqueeze(1)).squeeze(1)
            
            # Compute old log probs (should be stored in practice)
            with torch.no_grad():
                old_dist = Normal(action_mean.detach(), action_std.detach())
                old_log_probs = old_dist.log_prob(actions.unsqueeze(1)).squeeze(1)
            
            # Importance ratio
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(current_values, returns)
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        self.training_step += 1
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for discrete action space
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, state):
        """Forward pass"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class DQNBidder:
    """
    Deep Q-Network (DQN) Bidder
    Classic RL algorithm adapted for bidding
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        n_actions: int = 10,  # Discretize action space
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 10000
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Action space: discrete bid multipliers
        self.actions = np.linspace(0.1, 2.0, n_actions)
        
        # Networks
        self.q_network = DQNNetwork(state_dim, n_actions)
        self.target_network = DQNNetwork(state_dim, n_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        
        self.steps = 0
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> float:
        """Epsilon-greedy action selection"""
        if not deterministic and np.random.random() < self.epsilon:
            # Explore
            action_idx = np.random.randint(self.n_actions)
        else:
            # Exploit
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax(1).item()
        
        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay
        )
        
        return self.actions[action_idx]
    
    def update(self, batch_size: int = 64):
        """Update Q-network"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch
        experiences = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([
            np.argmin(np.abs(self.actions - e.action))
            for e in experiences
        ])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        if self.steps % 1000 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.steps += 1
        
        return {'loss': loss.item(), 'epsilon': self.epsilon}


def train_rl_bidder(
    algorithm: str = 'ppo',
    n_episodes: int = 1000,
    save_path: str = '/home/claude/rtb_system/models/rl_bidder.pt'
):
    """
    Train RL bidder
    
    Args:
        algorithm: 'ppo' or 'dqn'
        n_episodes: Number of training episodes
        save_path: Path to save trained model
    """
    print(f"\n🎓 Training {algorithm.upper()} Bidder...")
    
    # Create environment (mock for demonstration)
    env = BiddingEnvironment(ctr_model=None)
    
    # Create agent
    if algorithm == 'ppo':
        agent = PPOBidder()
    else:
        agent = DQNBidder()
    
    # Training loop
    episode_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        trajectories = []
        
        done = False
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            
            # Store experience
            if algorithm == 'ppo':
                trajectories.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'done': done
                })
            else:
                experience = Experience(state, action, reward, next_state, done)
                agent.replay_buffer.push(experience)
                agent.update(batch_size=64)
            
            state = next_state
        
        # Update policy (PPO)
        if algorithm == 'ppo' and len(trajectories) > 0:
            agent.update(trajectories)
        
        episode_rewards.append(episode_reward)
        
        # Logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}/{n_episodes}, Avg Reward: {avg_reward:.2f}")
    
    # Save model
    if algorithm == 'ppo':
        torch.save(agent.network.state_dict(), save_path)
    else:
        torch.save(agent.q_network.state_dict(), save_path)
    
    print(f"✅ Model saved to {save_path}")
    
    return agent, episode_rewards


if __name__ == "__main__":
    # Train PPO agent
    ppo_agent, ppo_rewards = train_rl_bidder(
        algorithm='ppo',
        n_episodes=500,
        save_path='/home/claude/rtb_system/models/ppo_bidder.pt'
    )
    
    print(f"\n📊 PPO Training Results:")
    print(f"  Final Avg Reward: {np.mean(ppo_rewards[-100:]):.2f}")
    print(f"  Best Reward: {max(ppo_rewards):.2f}")
