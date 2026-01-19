#!/usr/bin/env python3
"""
TD7 (Temporal Difference 7) Agent for continuous control
Combines TD3, LAP prioritized replay, encoder networks, and value clipping
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class Encoder(nn.Module):
    """
    State-action encoder that learns latent representations
    Outputs: zs (state embedding), zsa (state-action embedding)
    """
    def __init__(self, state_dim=7, action_dim=2, zs_dim=256, zsa_dim=256, hidden_dim=256):
        super(Encoder, self).__init__()
        
        self.zs_dim = zs_dim
        self.zsa_dim = zsa_dim
        
        # State encoder: s -> zs
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, zs_dim)
        )
        
        # State-action encoder: (s, a) -> zsa
        self.state_action_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, zsa_dim)
        )
        
        # Predictor: predicts next state embedding from zsa
        self.predictor = nn.Sequential(
            nn.Linear(zsa_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, zs_dim)
        )
    
    def forward(self, state, action=None):
        """
        Forward pass
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim) optional
        Returns:
            zs: state embedding
            zsa: state-action embedding (if action provided)
        """
        zs = self.state_encoder(state)
        
        if action is not None:
            sa = torch.cat([state, action], dim=1)
            zsa = self.state_action_encoder(sa)
            return zs, zsa
        
        return zs, None
    
    def predict_next_zs(self, zsa):
        """Predict next state embedding from state-action embedding"""
        return self.predictor(zsa)


class Critic(nn.Module):
    """
    Twin Q-networks (TD3 style) with encoder embeddings
    Takes state, action, and encoder embeddings
    """
    def __init__(self, state_dim=7, action_dim=2, zs_dim=256, zsa_dim=256, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim + zs_dim + zsa_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 network (twin)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim + zs_dim + zsa_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action, zs, zsa):
        """
        Forward pass through both Q-networks
        Returns: q1, q2
        """
        x = torch.cat([state, action, zs, zsa], dim=1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2


class Actor(nn.Module):
    """
    Policy network that outputs deterministic actions conditioned on state and embeddings
    """
    def __init__(self, state_dim=7, action_dim=2, zs_dim=256, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + zs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        self.action_dim = action_dim
    
    def forward(self, state, zs):
        """
        Forward pass
        Args:
            state: (batch_size, state_dim)
            zs: (batch_size, zs_dim) state embedding
        Returns:
            action: (batch_size, action_dim) in [-1, 1]
        """
        x = torch.cat([state, zs], dim=1)
        return self.network(x)


class LAPReplayBuffer:
    """
    LAP (Largest-Alpha-Priority) Prioritized Replay Buffer
    Prioritizes experiences by TD error
    """
    def __init__(self, capacity=100000, alpha=0.4, min_priority=1.0):
        self.capacity = capacity
        self.alpha = alpha
        self.min_priority = min_priority
        self.buffer = []
        self.priorities = []
        self.position = 0
    
    def add(self, state, action, reward, next_state, done, priority=None):
        """Add experience to buffer"""
        if priority is None:
            priority = max(self.priorities) if self.priorities else self.min_priority
        
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample batch with priority-based sampling"""
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices
        )
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = max(priority, self.min_priority)
    
    def __len__(self):
        return len(self.buffer)


class TD7Agent:
    """
    TD7 Agent - Advanced actor-critic algorithm
    Components:
    - Twin Q-networks (TD3)
    - State-action encoder
    - LAP prioritized replay
    - Value clipping
    - Delayed policy updates
    """
    def __init__(
        self,
        state_dim=10,
        action_dim=2,
        hidden_dim=128,
        zs_dim=128,
        zsa_dim=128,
        # actor_lr=3e-4,
        actor_lr=1e-4,
        critic_lr=3e-4,
        encoder_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_freq=2,
        target_update_rate=1,
        buffer_capacity=100000,
        batch_size=256,
        alpha=0.4,
        min_priority=1.0,
        exploration_noise=0.1,
        policy_noise=0.2,
        noise_clip=0.5
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_freq = policy_freq
        self.target_update_rate = target_update_rate
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"TD7 Agent using device: {self.device}")
        
        # Networks
        self.encoder = Encoder(state_dim, action_dim, zs_dim, zsa_dim, hidden_dim).to(self.device)
        self.encoder_target = Encoder(state_dim, action_dim, zs_dim, zsa_dim, hidden_dim).to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        
        self.actor = Actor(state_dim, action_dim, zs_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, zs_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim, zs_dim, zsa_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, zs_dim, zsa_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Replay buffer
        self.replay_buffer = LAPReplayBuffer(buffer_capacity, alpha, min_priority)
        
        # Training statistics
        self.training_step = 0
        self.episode_count = 0
        
        # Value clipping
        self.q_min = float('inf')
        self.q_max = float('-inf')
        self.value_clip_percentile = 5
    
    def select_action(self, state, training=True):
        """
        Select action using actor network with optional exploration noise
        Args:
            state: numpy array (state_dim,)
            training: whether to add exploration noise
        Returns:
            action: numpy array (action_dim,)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            zs, _ = self.encoder(state_tensor)
            action_tensor = self.actor(state_tensor, zs)
            action = action_tensor.cpu().numpy()[0]
        
        # Add exploration noise during training
        if training:
            # Use annealed noise instead of fixed
            current_noise = self.get_exploration_noise(self.episode_count, 10000)
            noise = np.random.normal(0, current_noise, size=self.action_dim)
            action = np.clip(action + noise, -1, 1)
        
        # Scale to action ranges
        scaled_action = action.copy()
        scaled_action[0] = (action[0] + 1.0) * 0.5   # Linear: [-0.5, 0.5]
        scaled_action[1] = action[1] * 1.0   # Angular: [-1.0, 1.0]
        
        return scaled_action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        # Unscale action back to [-1, 1]
        unscaled_action = action.copy()
        unscaled_action[0] = (action[0] / 0.5) - 1.0
        unscaled_action[1] = action[1] / 1.0
        
        self.replay_buffer.add(state, unscaled_action, reward, next_state, done)
    
    def update(self):
        """
        TD7 update step
        Returns: encoder_loss, critic_loss, actor_loss
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0, 0.0
        
        self.training_step += 1
        
        # Sample batch
        states, actions, rewards, next_states, dones, indices = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # === 1. Update Encoder ===
        zs, zsa = self.encoder(states, actions)
        next_zs_pred = self.encoder.predict_next_zs(zsa)
        
        with torch.no_grad():
            next_zs_target, _ = self.encoder_target(next_states)
        
        encoder_loss = nn.MSELoss()(next_zs_pred, next_zs_target)
        
        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()
        
        # === 2. Update Critic ===
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_zs_t, _ = self.encoder_target(next_states)
            next_actions = (self.actor_target(next_states, next_zs_t) + noise).clamp(-1, 1)
            
            # Compute target Q-values
            next_zs_full, next_zsa = self.encoder_target(next_states, next_actions)
            target_q1, target_q2 = self.critic_target(next_states, next_actions, next_zs_full, next_zsa)
            target_q = torch.min(target_q1, target_q2)
            
            # Value clipping
            target_q_clipped = self._clip_q_values(target_q)
            target_q_final = rewards + (1 - dones) * self.gamma * target_q_clipped
        
        # Current Q-values
        zs_current, zsa_current = self.encoder(states, actions)
        current_q1, current_q2 = self.critic(states, actions, zs_current.detach(), zsa_current.detach())
        
        # Critic loss (Huber loss for stability)
        critic_loss = nn.SmoothL1Loss()(current_q1, target_q_final) + nn.SmoothL1Loss()(current_q2, target_q_final)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update priorities in replay buffer
        with torch.no_grad():
            td_errors = torch.abs(current_q1 - target_q_final).cpu().numpy().flatten()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # === 3. Update Actor (delayed) ===
        actor_loss_value = 0.0
        if self.training_step % self.policy_freq == 0:
            zs_actor, _ = self.encoder(states)
            actor_actions = self.actor(states, zs_actor.detach())
            _, zsa_actor = self.encoder(states, actor_actions)
            
            actor_q, _ = self.critic(states, actor_actions, zs_actor.detach(), zsa_actor.detach())
            actor_loss = -actor_q.mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            actor_loss_value = actor_loss.item()
        
        # === 4. Update target networks ===
        if self.training_step % self.target_update_rate == 0:
            self._soft_update(self.encoder, self.encoder_target)
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
        
        return encoder_loss.item(), critic_loss.item(), actor_loss_value
    
    def _clip_q_values(self, q_values):
        """Apply value clipping to Q-targets"""
        # Update running min/max
        self.q_min = min(self.q_min, q_values.min().item())
        self.q_max = max(self.q_max, q_values.max().item())
        
        # Compute clipping range
        q_range = self.q_max - self.q_min
        clip_min = self.q_min - (q_range * self.value_clip_percentile / 100)
        clip_max = self.q_max + (q_range * self.value_clip_percentile / 100)
        
        return torch.clamp(q_values, clip_min, clip_max)
    
    def _soft_update(self, source, target):
        """Soft update target network parameters"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath):
        """Save all networks and training state"""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'encoder_target': self.encoder_target.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'q_min': self.q_min,
            'q_max': self.q_max
        }, filepath)
        print(f"TD7 model saved to {filepath}")
    
    def load(self, filepath):
        """Load all networks and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.encoder_target.load_state_dict(checkpoint['encoder_target'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        self.q_min = checkpoint['q_min']
        self.q_max = checkpoint['q_max']
        
        print(f"TD7 model loaded from {filepath}")
        print(f"Episodes: {self.episode_count}, Steps: {self.training_step}")


    def get_exploration_noise(self, episode, max_episodes):
        """Anneal exploration noise over training"""
        initial = 0.2
        final = 0.02
        decay = 1.5 #3.0
        progress = min(episode / max_episodes, 1.0)
        # return 0.99
        return final + (initial - final) * np.exp(-decay * progress)


