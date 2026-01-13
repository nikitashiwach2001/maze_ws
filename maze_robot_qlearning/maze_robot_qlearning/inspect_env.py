#!/usr/bin/env python3
"""
Inspect the maze environment - check observation space, action space, and dynamics
"""

import rclpy
from maze_robot_qlearning.maze_environment import MazeEnvironment
import numpy as np
import time


def inspect_environment():
    """Inspect environment properties"""
    rclpy.init()
    env = MazeEnvironment()

    try:
        # Wait for initialization
        time.sleep(2.0)

        print("\n" + "="*60)
        print("MAZE ENVIRONMENT INSPECTION")
        print("="*60)

        # Reset and get initial observation
        obs = env.reset()

        print("\n1. OBSERVATION SPACE:")
        print(f"   Shape: {obs.shape}")
        print(f"   Type: {obs.dtype}")
        print(f"   Dimensions: {len(obs)}")
        print(f"   Format: [x, y, goal_x, goal_y, cos(theta), sin(theta)]")
        print(f"   Example: {obs}")
        print(f"\n   Breakdown:")
        print(f"   - Robot position (x, y): ({obs[0]:.3f}, {obs[1]:.3f})")
        print(f"   - Goal position: ({obs[2]:.3f}, {obs[3]:.3f})")
        print(f"   - Orientation (cos, sin): ({obs[4]:.3f}, {obs[5]:.3f})")

        print("\n2. ACTION SPACE:")
        print(f"   Type: Continuous")
        print(f"   Dimensions: 2")
        print(f"   Format: [linear_velocity, angular_velocity]")
        print(f"   Ranges:")
        print(f"   - Linear velocity:  [-0.5, 0.5] m/s")
        print(f"   - Angular velocity: [-1.0, 1.0] rad/s")
        print(f"\n   Action examples:")
        print(f"   - Move forward:  [0.3, 0.0]")
        print(f"   - Turn left:     [0.0, 0.5]")
        print(f"   - Turn right:    [0.0, -0.5]")
        print(f"   - Move backward: [-0.3, 0.0]")
        print(f"   - Arc motion:    [0.3, 0.3]")

        print("\n3. REWARD STRUCTURE:")
        print(f"   - Goal reached (+100): Distance < {env.goal_tolerance}m")
        print(f"   - Out of bounds (-50): Outside 0-{env.maze_size}m")
        print(f"   - Timeout (-10): > {env.max_steps} steps")
        print(f"   - Step penalty (-1.0): Each action")
        print(f"   - Distance penalty (-0.1 × distance): Closer = better")

        print("\n4. EPISODE SETTINGS:")
        print(f"   - Max steps per episode: {env.max_steps}")
        print(f"   - Start position: {env.start_position}")
        print(f"   - Goal position: {env.goal_position}")
        print(f"   - Maze size: {env.maze_size}x{env.maze_size}m")

        # Test action execution
        print("\n5. TESTING ACTION EXECUTION:")
        print("   Executing 3 sample actions...")

        test_actions = [
            ([0.3, 0.0], "Forward"),
            ([0.0, 0.5], "Turn left"),
            ([0.3, 0.3], "Arc forward-left")
        ]

        for i, (action, description) in enumerate(test_actions):
            print(f"\n   Action {i+1}: {description} = {action}")
            obs, reward, done, info = env.step(action)
            print(f"   → Position: ({obs[0]:.3f}, {obs[1]:.3f})")
            print(f"   → Reward: {reward:.3f}")
            print(f"   → Done: {done}")

            if done:
                print(f"   → Episode ended: {info['reason']}")
                break

        print("\n6. FOR NEURAL NETWORK:")
        print(f"   - Input layer size: {len(obs)} (observation dimensions)")
        print(f"   - Output layer size: 2 (action dimensions)")
        print(f"   - Action space: Continuous (use tanh activation)")
        print(f"   - State space: Continuous (normalize inputs)")

        print("\n" + "="*60)
        print("INSPECTION COMPLETE")
        print("="*60 + "\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()
        rclpy.shutdown()


if __name__ == '__main__':
    inspect_environment()
