#!/usr/bin/env python3
"""
Neural Network-based SARSA for continuous action space
Uses PyTorch for function approximation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class QNetwork(nn.Module):
    """
    Q-Network for continuous state and action spaces
    Takes state and action as input, outputs Q-value
    """
    def __init__(self, state_dim=6, action_dim=2, hidden_dim=128):
        super(QNetwork, self).__init__()

        # Network architecture: state + action -> Q-value
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output single Q-value

        self.relu = nn.ReLU()

    def forward(self, state, action):
        """
        Forward pass
        Args:
            state: (batch_size, state_dim) tensor
            action: (batch_size, action_dim) tensor
        Returns:
            q_value: (batch_size, 1) tensor
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q_value = self.fc3(x)

        return q_value


class PolicyNetwork(nn.Module):
    """
    Policy network for selecting actions
    Takes state as input, outputs action
    """
    def __init__(self, state_dim=6, action_dim=2, hidden_dim=128):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # Output in [-1, 1]

        self.action_dim = action_dim

    def forward(self, state):
        """
        Forward pass
        Args:
            state: (batch_size, state_dim) tensor
        Returns:
            action: (batch_size, action_dim) tensor in [-1, 1]
        """
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action = self.tanh(self.fc3(x))

        return action


class SARSAAgent:
    """
    SARSA agent with neural network function approximation
    On-policy learning with epsilon-greedy exploration
    """
    def __init__(
        self,
        state_dim=6,
        action_dim=2,
        hidden_dim=128,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_network = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        # Loss function
        self.criterion = nn.MSELoss()

        # Training statistics
        self.episode_count = 0
        self.training_step = 0

    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        Args:
            state: numpy array (state_dim,)
            training: whether in training mode (use exploration)
        Returns:
            action: numpy array (action_dim,)
                    [linear_velocity, angular_velocity]
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            linear_vel = np.random.uniform(-0.5, 0.5)
            angular_vel = np.random.uniform(-1.0, 1.0)
            action = np.array([linear_vel, angular_vel], dtype=np.float32)
        else:
            # Exploitation: use policy network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action_tensor = self.policy_network(state_tensor)
                action = action_tensor.cpu().numpy()[0]

                # Scale to action ranges
                # Network outputs [-1, 1], scale to actual ranges
                action[0] = action[0] * 0.5   # Linear: [-0.5, 0.5]
                action[1] = action[1] * 1.0   # Angular: [-1.0, 1.0]

        return action

    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA update: Q(s,a) <- Q(s,a) + α[r + γQ(s',a') - Q(s,a)]

        Args:
            state: current state
            action: action taken
            reward: reward received
            next_state: next state
            next_action: next action (on-policy!)
            done: whether episode ended
        """
        self.training_step += 1

        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        next_action_tensor = torch.FloatTensor(next_action).unsqueeze(0).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([done]).to(self.device)

        # Current Q-value
        current_q = self.q_network(state_tensor, action_tensor)

        # SARSA target: r + γ * Q(s', a')
        with torch.no_grad():
            next_q = self.q_network(next_state_tensor, next_action_tensor)
            target_q = reward_tensor + (1 - done_tensor) * self.gamma * next_q

        # Update Q-network
        q_loss = self.criterion(current_q, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update policy network to maximize Q-value
        policy_action = self.policy_network(state_tensor)
        # Scale policy action
        scaled_policy_action = policy_action.clone()
        scaled_policy_action[:, 0] = scaled_policy_action[:, 0] * 0.5
        scaled_policy_action[:, 1] = scaled_policy_action[:, 1] * 1.0

        policy_q = self.q_network(state_tensor, scaled_policy_action)
        policy_loss = -policy_q.mean()  # Maximize Q-value

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return q_loss.item(), policy_loss.item()

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save networks and training state"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'policy_network': self.policy_network.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'training_step': self.training_step
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load networks and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.q_network.load_state_dict(checkpoint['q_network'])
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        self.training_step = checkpoint['training_step']

        print(f"Model loaded from {filepath}")
        print(f"Episodes: {self.episode_count}, Steps: {self.training_step}")


def test_networks():
    """Test the networks"""
    print("Testing SARSA Networks...")

    # Create agent
    agent = SARSAAgent(state_dim=6, action_dim=2)

    # Test state and action
    state = np.array([0.5, 0.5, 4.5, 4.5, 1.0, 0.0], dtype=np.float32)

    print("\n1. Testing action selection:")
    print(f"   State: {state}")

    action = agent.select_action(state, training=True)
    print(f"   Action (with exploration): {action}")

    action = agent.select_action(state, training=False)
    print(f"   Action (greedy): {action}")

    print("\n2. Testing Q-network:")
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action_tensor = torch.FloatTensor(action).unsqueeze(0)

    q_value = agent.q_network(state_tensor, action_tensor)
    print(f"   Q(s, a) = {q_value.item():.4f}")

    print("\n3. Testing SARSA update:")
    next_state = np.array([0.6, 0.5, 4.5, 4.5, 1.0, 0.0], dtype=np.float32)
    next_action = agent.select_action(next_state, training=True)
    reward = -1.5
    done = False

    q_loss, policy_loss = agent.update(state, action, reward, next_state, next_action, done)
    print(f"   Q-loss: {q_loss:.4f}")
    print(f"   Policy loss: {policy_loss:.4f}")

    print("\n4. Testing save/load:")
    agent.save('test_model.pt')
    agent.load('test_model.pt')

    print("\nAll tests passed! ✓")


if __name__ == '__main__':
    test_networks()
