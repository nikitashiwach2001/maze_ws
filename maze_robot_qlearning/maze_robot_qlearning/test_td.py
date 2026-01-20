#!/usr/bin/env python3
"""
Test a trained TD7 agent in Gazebo with visualization
"""

import rclpy
import time
import numpy as np
from maze_robot_qlearning.maze_environment import MazeEnvironment
from maze_robot_qlearning.td7_network import TD7Agent


class TD7Tester:
    def __init__(self, model_path):
        # Initialize ROS
        rclpy.init()

        # Create environment (Gazebo will stay visible)
        self.env = MazeEnvironment()
        time.sleep(2.0)

        # Create agent (SAME CONFIG AS TRAINING)
        self.agent = TD7Agent(
            state_dim=10,
            action_dim=2,
            hidden_dim=128,
            zs_dim=128,
            zsa_dim=128,
            actor_lr=1e-4,      # lr not used in test
            critic_lr=3e-4,
            encoder_lr=3e-4,
            gamma=0.99,
            batch_size=256,
            buffer_capacity=1,  # not used
            policy_freq=4,
            target_update_rate=1,
            exploration_noise=0.0   # IMPORTANT: no noise
        )

        # Load trained weights
        self.agent.load(model_path)

        self.env.get_logger().info(f"Loaded TD7 model from: {model_path}")

    def run(self, num_episodes=5, max_steps=500):
        """Run trained policy"""

        for ep in range(1, num_episodes + 1):
            state = self.env.reset()
            done = False
            episode_reward = 0
            step = 0

            self.env.get_logger().info(f"▶️  TEST EPISODE {ep} STARTED")

            while not done and step < max_steps:
                # Select deterministic action
                action = self.agent.select_action(state, training=False)

                # Step environment
                next_state, reward, done, info = self.env.step(action)

                episode_reward += reward
                state = next_state
                step += 1

                # Slow down for visualization
                time.sleep(0.05)

            # Episode summary
            reason = info.get("reason", "unknown")
            success = info.get("success", False)

            if success:
                status = "✅ SUCCESS"
            else:
                status = f"❌ FAILED ({reason})"

            self.env.get_logger().info(
                f"{status} | Episode {ep} | Reward: {episode_reward:.2f} | Steps: {step}"
            )

            time.sleep(1.0)

    def close(self):
        self.env.close()
        rclpy.shutdown()


def main():
    # MODEL_PATH = "td7_models/td7_episode_4450.pt"

    MODEL_PATH = "td7_models/td7_best.pt"
    # OR
    # MODEL_PATH = "td7_models/td7_episode_4450.pt"

    tester = TD7Tester(MODEL_PATH)

    try:
        tester.run(num_episodes=10, max_steps=200)
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    finally:
        tester.close()


if __name__ == "__main__":
    main()
