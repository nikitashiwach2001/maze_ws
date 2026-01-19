#!/usr/bin/env python3
"""
Test trained Deep SARSA agent in the maze environment
"""

import rclpy
from maze_robot_qlearning.maze_environment import MazeEnvironment
from maze_robot_qlearning.sarsa_network import SARSAAgent
import numpy as np
import time
import os


class SARSAAgentTester:
    def __init__(self, model_path, num_test_episodes=10):
        """
        Initialize the test environment
        
        Args:
            model_path: Path to the saved model file (e.g., 'sarsa_models/sarsa_best.pt')
            num_test_episodes: Number of episodes to test
        """
        # Initialize ROS
        rclpy.init()
        self.env = MazeEnvironment()
        
        # Wait for environment to initialize
        time.sleep(2.0)
        
        # Create agent with same parameters as training
        self.agent = SARSAAgent(
            state_dim=7,  # x, y, goal_x, goal_y, cos(theta), sin(theta), min_obstacle_distance
            action_dim=2,  # linear_vel, angular_vel
            learning_rate=1e-3,
            gamma=0.99,
            epsilon_start=0.0,  # No exploration during testing
            epsilon_end=0.0,
            epsilon_decay=1.0
        )
        
        # Load trained model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        self.agent.load(model_path)
        self.agent.epsilon = 0.0  # Force greedy policy for testing
        
        self.num_test_episodes = num_test_episodes
        self.env.get_logger().info(f'Loaded model from: {model_path}')
        self.env.get_logger().info(f'Testing for {num_test_episodes} episodes')
        
    def test(self):
        """Test the agent and collect statistics"""
        test_rewards = []
        test_lengths = []
        success_count = 0
        
        self.env.get_logger().info('='*70)
        self.env.get_logger().info('STARTING AGENT TESTING')
        self.env.get_logger().info('='*70)
        
        for episode in range(1, self.num_test_episodes + 1):
            episode_start_time = time.time()
            
            # Reset environment
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            self.env.get_logger().info(f'\n--- Episode {episode}/{self.num_test_episodes} ---')
            self.env.get_logger().info(f'Starting position: ({state[0]:.2f}, {state[1]:.2f})')
            
            # Episode loop - greedy policy (no exploration)
            while not done:
                # Select action greedily
                action = self.agent.select_action(state, training=False)
                
                # Execute action
                next_state, reward, done, info = self.env.step(action)
                
                # Log step details
                if episode_length % 5 == 0:  # Log every 5 steps to reduce clutter
                    self.env.get_logger().info(
                        f'Step {episode_length}: '
                        f'Pos=({next_state[0]:.2f}, {next_state[1]:.2f}), '
                        f'Action=[{action[0]:.2f}, {action[1]:.2f}], '
                        f'Reward={reward:.2f}'
                    )
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # Safety check - prevent infinite loops
                if episode_length > 200:
                    self.env.get_logger().warning('Max steps exceeded! Terminating episode.')
                    done = True
                    info = {'success': False, 'reason': 'safety_timeout'}
            
            # Record episode results
            test_rewards.append(episode_reward)
            test_lengths.append(episode_length)
            
            if info.get('success', False):
                success_count += 1
            
            episode_time = time.time() - episode_start_time
            
            # Log episode summary
            self.env.get_logger().info(f'\nEpisode {episode} Summary:')
            self.env.get_logger().info(f'  Total Reward: {episode_reward:.2f}')
            self.env.get_logger().info(f'  Episode Length: {episode_length} steps')
            self.env.get_logger().info(f'  Result: {info.get("reason", "unknown")}')
            self.env.get_logger().info(f'  Success: {"✓" if info.get("success", False) else "✗"}')
            self.env.get_logger().info(f'  Time: {episode_time:.2f}s')
            self.env.get_logger().info(f'  Final position: ({state[0]:.2f}, {state[1]:.2f})')
            
            # Small delay between episodes
            time.sleep(1.0)
        
        # Print final statistics
        self.print_statistics(test_rewards, test_lengths, success_count)
        
        return test_rewards, test_lengths, success_count
    
    def print_statistics(self, rewards, lengths, successes):
        """Print testing statistics"""
        self.env.get_logger().info('')
        self.env.get_logger().info('='*70)
        self.env.get_logger().info('TESTING COMPLETE')
        self.env.get_logger().info('='*70)
        self.env.get_logger().info(f'Total test episodes: {len(rewards)}')
        self.env.get_logger().info(f'Successes: {successes}/{len(rewards)}')
        self.env.get_logger().info(f'Success rate: {successes/len(rewards)*100:.1f}%')
        self.env.get_logger().info('')
        self.env.get_logger().info('Reward Statistics:')
        self.env.get_logger().info(f'  Mean: {np.mean(rewards):.2f}')
        self.env.get_logger().info(f'  Std: {np.std(rewards):.2f}')
        self.env.get_logger().info(f'  Min: {np.min(rewards):.2f}')
        self.env.get_logger().info(f'  Max: {np.max(rewards):.2f}')
        self.env.get_logger().info('')
        self.env.get_logger().info('Episode Length Statistics:')
        self.env.get_logger().info(f'  Mean: {np.mean(lengths):.1f} steps')
        self.env.get_logger().info(f'  Std: {np.std(lengths):.1f} steps')
        self.env.get_logger().info(f'  Min: {np.min(lengths)} steps')
        self.env.get_logger().info(f'  Max: {np.max(lengths)} steps')
        self.env.get_logger().info('='*70)
    
    def close(self):
        """Clean up resources"""
        self.env.close()
        rclpy.shutdown()


def main():
    """Main testing function"""
    import sys
    
    # Default model path
    default_model = 'sarsa_models/sarsa_best.pt'
    
    # Check if model path provided as command line argument
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = default_model
    
    # Check if number of episodes provided
    if len(sys.argv) > 2:
        num_episodes = int(sys.argv[2])
    else:
        num_episodes = 10
    
    try:
        print(f"\n{'='*70}")
        print(f"Testing SARSA Agent")
        print(f"Model: {model_path}")
        print(f"Episodes: {num_episodes}")
        print(f"{'='*70}\n")
        
        # Create tester
        tester = SARSAAgentTester(
            model_path=model_path,
            num_test_episodes=num_episodes
        )
        
        # Run tests
        rewards, lengths, successes = tester.test()
        
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user")
    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'tester' in locals():
            tester.close()


if __name__ == '__main__':
    main()
