#!/usr/bin/env python3
"""
Train Deep SARSA agent in the maze environment
"""

import rclpy
from maze_robot_qlearning.maze_environment import MazeEnvironment
from maze_robot_qlearning.sarsa_network import SARSAAgent
import numpy as np
import time
import json
import os
from datetime import datetime


class DeepSARSATrainer:
    def __init__(
        self,
        num_episodes=500,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        save_interval=50
    ):
        # Initialize ROS
        rclpy.init()
        self.env = MazeEnvironment()

        # Wait for environment to initialize
        time.sleep(2.0)

        # Create agent
        self.agent = SARSAAgent(
            state_dim=7, # after adding the lidar sensor update the input size of neurons
            action_dim=2,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay
        )

        # Training parameters
        self.num_episodes = num_episodes
        self.save_interval = save_interval

        # Create directories
        os.makedirs('sarsa_models', exist_ok=True)
        os.makedirs('training_logs', exist_ok=True)

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.q_losses = []
        self.policy_losses = []

        self.env.get_logger().info('Deep SARSA Trainer initialized!')
        self.env.get_logger().info(f'Training for {num_episodes} episodes')

    def train(self):
        """Train the agent"""
        best_reward = -float('inf')

        self.env.get_logger().info('='*70)
        self.env.get_logger().info('STARTING DEEP SARSA TRAINING')
        self.env.get_logger().info('='*70)

        for episode in range(1, self.num_episodes + 1):
            episode_start_time = time.time()

            # Reset environment
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            # Select initial action (epsilon-greedy)
            action = self.agent.select_action(state, training=True)

            episode_q_losses = []
            episode_policy_losses = []

            # Episode loop
            while not done:
                # Execute action
                next_state, reward, done, info = self.env.step(action)

                # Select next action (on-policy SARSA!)
                next_action = self.agent.select_action(next_state, training=True)

                # Update agent
                q_loss, policy_loss = self.agent.update(
                    state, action, reward, next_state, next_action, done
                )

                episode_q_losses.append(q_loss)
                episode_policy_losses.append(policy_loss)

                # Update for next iteration
                state = next_state
                action = next_action

                episode_reward += reward
                episode_length += 1

            # Decay epsilon after each episode
            self.agent.decay_epsilon()
            self.agent.episode_count += 1

            # Record statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.q_losses.append(np.mean(episode_q_losses))
            self.policy_losses.append(np.mean(episode_policy_losses))

            if info.get('success', False):
                self.success_count += 1

            # Calculate episode time
            episode_time = time.time() - episode_start_time

            # Logging
            if episode % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                recent_success = sum(1 for i in range(max(0, episode-10), episode)
                                   if i < len(self.episode_rewards) and
                                   self.episode_rewards[i] > 50)

                self.env.get_logger().info('')
                self.env.get_logger().info(f'Episode {episode}/{self.num_episodes}')
                self.env.get_logger().info(f'  Reward: {episode_reward:.2f}')
                self.env.get_logger().info(f'  Length: {episode_length} steps')
                self.env.get_logger().info(f'  Success: {info.get("reason", "unknown")}')
                self.env.get_logger().info(f'  Epsilon: {self.agent.epsilon:.3f}')
                self.env.get_logger().info(f'  Avg reward (last 10): {np.mean(recent_rewards):.2f}')
                self.env.get_logger().info(f'  Success rate (last 10): {recent_success}/10')
                self.env.get_logger().info(f'  Time: {episode_time:.2f}s')

            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.agent.save('sarsa_models/sarsa_best.pt')
                self.env.get_logger().info(f'  → New best model saved! Reward: {best_reward:.2f}')

            # Save checkpoint
            if episode % self.save_interval == 0:
                self.agent.save(f'sarsa_models/sarsa_episode_{episode}.pt')
                self.save_training_stats()
                self.env.get_logger().info(f'  → Checkpoint saved at episode {episode}')

        # Final save
        self.agent.save('sarsa_models/sarsa_final.pt')
        self.save_training_stats()

        # Training summary
        self.print_summary()

    def save_training_stats(self):
        """Save training statistics to JSON"""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_count': self.success_count,
            'success_rate': self.success_count / len(self.episode_rewards) if self.episode_rewards else 0,
            'q_losses': self.q_losses,
            'policy_losses': self.policy_losses,
            'final_epsilon': self.agent.epsilon,
            'total_episodes': len(self.episode_rewards),
            'timestamp': datetime.now().isoformat()
        }

        with open('training_logs/training_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

    def print_summary(self):
        """Print training summary"""
        self.env.get_logger().info('')
        self.env.get_logger().info('='*70)
        self.env.get_logger().info('TRAINING COMPLETE')
        self.env.get_logger().info('='*70)
        self.env.get_logger().info(f'Total episodes: {len(self.episode_rewards)}')
        self.env.get_logger().info(f'Total successes: {self.success_count}')
        self.env.get_logger().info(f'Success rate: {self.success_count/len(self.episode_rewards)*100:.1f}%')
        self.env.get_logger().info(f'Average reward: {np.mean(self.episode_rewards):.2f}')
        self.env.get_logger().info(f'Best reward: {max(self.episode_rewards):.2f}')
        self.env.get_logger().info(f'Average episode length: {np.mean(self.episode_lengths):.1f} steps')

        # Last 100 episodes statistics
        if len(self.episode_rewards) >= 100:
            last_100_rewards = self.episode_rewards[-100:]
            last_100_success = sum(1 for r in last_100_rewards if r > 50)
            self.env.get_logger().info('')
            self.env.get_logger().info('Last 100 episodes:')
            self.env.get_logger().info(f'  Success rate: {last_100_success}%')
            self.env.get_logger().info(f'  Average reward: {np.mean(last_100_rewards):.2f}')

        self.env.get_logger().info('')
        self.env.get_logger().info('Models saved in: sarsa_models/')
        self.env.get_logger().info('Training logs saved in: training_logs/')
        self.env.get_logger().info('='*70)

    def close(self):
        """Clean up"""
        self.env.close()
        rclpy.shutdown()


def main():
    """Main training function"""
    try:
        # Training parameters
        trainer = DeepSARSATrainer(
            num_episodes=500,          # Total episodes
            learning_rate=1e-3,        # Learning rate
            gamma=0.99,                # Discount factor
            epsilon_start=1.0,         # Initial exploration
            epsilon_end=0.01,          # Final exploration
            epsilon_decay=0.995,       # Exploration decay
            save_interval=50           # Save every N episodes
        )

        # Train
        trainer.train()

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    finally:
        if 'trainer' in locals():
            trainer.close()


if __name__ == '__main__':
    main()
