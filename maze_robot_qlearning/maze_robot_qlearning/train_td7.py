#!/usr/bin/env python3
"""
Train TD7 agent in the maze environment
"""

import rclpy
from maze_robot_qlearning.maze_environment import MazeEnvironment
from maze_robot_qlearning.td7_network import TD7Agent  # CHANGE THIS LINE
import numpy as np
import time
import json
import os
from datetime import datetime


class TD7Trainer:
    def __init__(
        self,
        num_episodes=500,
        actor_lr=3e-4,
        critic_lr=3e-4,
        encoder_lr=3e-4,
        gamma=0.99,
        batch_size=256,
        buffer_capacity=100000,
        save_interval=50,
        update_after=500  # Start training after this many steps
    ):
        # Initialize ROS
        rclpy.init()
        self.env = MazeEnvironment()

        # Wait for environment to initialize
        time.sleep(2.0)
        self.warmup_steps = 3_000


        # Create TD7 agent - UPDATE THIS SECTION (lines 35-43)
        self.agent = TD7Agent(
            state_dim=10,  # Updated for lidar sensor
            action_dim=2,
            hidden_dim=128,
            zs_dim=128,
            zsa_dim=128,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            encoder_lr=encoder_lr,
            gamma=gamma,
            batch_size=batch_size,
            buffer_capacity=buffer_capacity,
            policy_freq=4,
            target_update_rate=1,
            exploration_noise=0.1
        )

        # Training parameters
        self.num_episodes = num_episodes
        self.save_interval = save_interval
        self.update_after = update_after

        # Create directories
        os.makedirs('td7_models', exist_ok=True)
        os.makedirs('training_logs', exist_ok=True)

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.encoder_losses = []
        self.critic_losses = []
        self.actor_losses = []
        self.total_steps = 0

        self.env.get_logger().info('TD7 Trainer initialized!')
        self.env.get_logger().info(f'Training for {num_episodes} episodes')

    def train(self):
        """Train the agent"""
        best_reward = -float('inf')

        self.env.get_logger().info('='*70)
        self.env.get_logger().info('STARTING TD7 TRAINING')
        self.env.get_logger().info('='*70)

        for episode in range(1, self.num_episodes + 1):
            episode_start_time = time.time()

            # Reset environment
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            episode_encoder_losses = []
            episode_critic_losses = []
            episode_actor_losses = []

            while not done:
                # WARM START: Use random actions for first N episodes
                # if episode <= WARMUP_EPISODES:
                #     action = np.random.uniform([0.0, -1.0], [1.0, 1.0], size=2)
                # else:
                #     # Select action
                #     action = self.agent.select_action(state, training=True)

                if self.total_steps < self.warmup_steps:
                    action = np.random.uniform(
                        low=[0.0, -1.0],
                        high=[1.0, 1.0],
                        size=2
                    )
                else:
                    action = self.agent.select_action(state, training=True)

                # Execute action
                next_state, reward, done, info = self.env.step(action)
                self.total_steps += 1

                # Store transition in replay buffer
                self.agent.store_transition(state, action, reward, next_state, done)

                # Update agent (only after collecting initial experiences)
                if self.total_steps >= self.warmup_steps:
                    encoder_loss, critic_loss, actor_loss = self.agent.update()
                    episode_encoder_losses.append(encoder_loss)
                    episode_critic_losses.append(critic_loss)
                    if actor_loss is not None:
                        episode_actor_losses.append(actor_loss)

                # Update for next iteration
                state = next_state
                episode_reward += reward
                episode_length += 1

            # Update episode count
            self.agent.episode_count += 1

            # Record statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            if episode_encoder_losses:
                self.encoder_losses.append(np.mean(episode_encoder_losses))
                self.critic_losses.append(np.mean(episode_critic_losses))
            if episode_actor_losses:
                self.actor_losses.append(np.mean(episode_actor_losses))

            if info.get('success', False):
                self.success_count += 1

            # Calculate episode time
            episode_time = time.time() - episode_start_time

            # Determine success status with emoji
            success_status = info.get("success", None)
            reason = info.get("reason", "unknown")

            if success_status is True:
                status_msg = f"✅ SUCCESS - {reason}"
            elif reason == "collision":
                status_msg = "❌ COLLISION"
            elif reason == "timeout":
                status_msg = "⏱️  TIMEOUT"
            elif reason == "out_of_bounds":
                status_msg = "🚫 OUT OF BOUNDS"
            else:
                status_msg = f"❓ {reason}"

            # Log every episode outcome
            self.env.get_logger().info(
                f'{status_msg} Episode {episode}: reward={episode_reward:.2f}, '
                f'steps={episode_length}, reason={reason}'
            )

            # Detailed logging every 10 episodes
            if episode % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                recent_success = sum(1 for i in range(max(0, episode-10), episode)
                                if i < len(self.episode_rewards) and
                                self.episode_rewards[i] > 50)

                self.env.get_logger().info('')
                self.env.get_logger().info(f'Episode {episode}/{self.num_episodes}')
                self.env.get_logger().info(f'  Reward: {episode_reward:.2f}')
                self.env.get_logger().info(f'  Length: {episode_length} steps')
                self.env.get_logger().info(f'  Status: {status_msg}')  # NOW DEFINED
                self.env.get_logger().info(f'  Buffer size: {len(self.agent.replay_buffer)}')
                self.env.get_logger().info(f'  Total steps: {self.total_steps}')
                self.env.get_logger().info(f'  Avg reward (last 10): {np.mean(recent_rewards):.2f}')
                self.env.get_logger().info(f'  Success rate (last 10): {recent_success}/10')
                self.env.get_logger().info(f'  Time: {episode_time:.2f}s')

            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.agent.save('td7_models/td7_best.pt')
                self.env.get_logger().info(f'  → New best model saved! Reward: {best_reward:.2f}')

            # Save checkpoint
            if episode % self.save_interval == 0:
                self.agent.save(f'td7_models/td7_episode_{episode}.pt')
                self.save_training_stats()
                self.env.get_logger().info(f'  → Checkpoint saved at episode {episode}')

        # Final save
        self.agent.save('td7_models/td7_final.pt')
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
            'encoder_losses': self.encoder_losses,
            'critic_losses': self.critic_losses,
            'actor_losses': self.actor_losses,
            'total_steps': self.total_steps,
            'buffer_size': len(self.agent.replay_buffer),
            'total_episodes': len(self.episode_rewards),
            'timestamp': datetime.now().isoformat()
        }

        with open('training_logs/td7_training_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

    def print_summary(self):
        """Print training summary"""
        self.env.get_logger().info('')
        self.env.get_logger().info('='*70)
        self.env.get_logger().info('TD7 TRAINING COMPLETE')
        self.env.get_logger().info('='*70)
        self.env.get_logger().info(f'Total episodes: {len(self.episode_rewards)}')
        self.env.get_logger().info(f'Total steps: {self.total_steps}')
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
        self.env.get_logger().info('Models saved in: td7_models/')
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
        trainer = TD7Trainer(
            num_episodes=10000,
            actor_lr=1e-4,
            critic_lr=3e-4,
            encoder_lr=3e-4,
            gamma=0.99,
            batch_size=256,
            buffer_capacity=100000,
            save_interval=50)

        # Train
        trainer.train()

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    finally:
        if 'trainer' in locals():
            trainer.close()


if __name__ == '__main__':
    main()
