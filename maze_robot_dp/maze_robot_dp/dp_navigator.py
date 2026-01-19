#!/usr/bin/env python3
"""
DP Navigator - Execute optimal policy from Value Iteration in Gazebo
Uses the same environment as SARSA training
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import json
import math
import time


class DPNavigator(Node):
    def __init__(self):
        super().__init__('dp_navigator')

        # Load DP solution
        self.load_solution('dp_solution.json')

        # Robot state
        self.current_position = np.array([0.5, 0.5])
        self.current_orientation = 0.0

        # Goal
        self.goal_position = np.array([4.5, 4.5])
        self.goal_tolerance = 0.3

        # ROS 2 publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odometry_callback, 10
        )

        # Navigation state
        self.steps = 0
        self.max_steps = 200

        self.get_logger().info('DP Navigator initialized!')
        self.get_logger().info('Executing optimal policy from Value Iteration')

    def load_solution(self, filepath):
        """Load DP solution from file"""
        try:
            with open(filepath, 'r') as f:
                solution = json.load(f)

            self.policy = np.array(solution['policy'])
            self.grid_size = solution['grid_size']
            self.grid_resolution = solution['grid_resolution']

            print(f"Loaded DP solution from {filepath}")
            print(f"Grid size: {self.grid_size}x{self.grid_size}")
        except FileNotFoundError:
            print(f"Error: {filepath} not found!")
            print("Please run 'value_iteration' first to generate the solution")
            raise

    def odometry_callback(self, msg):
        """Update robot position from odometry"""
        self.current_position[0] = msg.pose.pose.position.x
        self.current_position[1] = msg.pose.pose.position.y

        # Extract yaw from quaternion
        quat = msg.pose.pose.orientation
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        self.current_orientation = math.atan2(siny_cosp, cosy_cosp)

    def get_policy_action(self):
        """Get action from DP policy for current position"""
        # Convert continuous position to grid coordinates
        # World coordinates: (x, y) where x is horizontal, y is vertical
        # Grid coordinates: policy[grid_x][grid_y] where grid_x corresponds to world x, grid_y to world y
        grid_x = int(self.current_position[0] / self.grid_resolution)
        grid_y = int(self.current_position[1] / self.grid_resolution)

        # Clip to valid range
        grid_x = np.clip(grid_x, 0, self.grid_size - 1)
        grid_y = np.clip(grid_y, 0, self.grid_size - 1)

        # Get policy direction - policy is stored as [grid_x][grid_y][direction]
        # The policy gives [dx, dy] in grid space where:
        # dx > 0 means move in +X direction (right/forward in world)
        # dy > 0 means move in +Y direction (up/left in world)
        policy_dx, policy_dy = self.policy[grid_x, grid_y]

        # Check if we're at a zero-action cell (wall or goal)
        if policy_dx == 0 and policy_dy == 0:
            # No movement commanded, check if we're at goal
            distance_to_goal = np.linalg.norm(self.current_position - self.goal_position)
            if distance_to_goal < self.goal_tolerance:
                return np.array([0.0, 0.0])
            else:
                # Shouldn't be here - move toward goal directly
                dx = self.goal_position[0] - self.current_position[0]
                dy = self.goal_position[1] - self.current_position[1]
                desired_heading = math.atan2(dy, dx)
                heading_error = desired_heading - self.current_orientation
                while heading_error > math.pi:
                    heading_error -= 2 * math.pi
                while heading_error < -math.pi:
                    heading_error += 2 * math.pi
                return np.array([0.2, 2.0 * heading_error])

        # Calculate desired heading from policy direction
        # policy_dx and policy_dy are in grid units, they represent direction to move
        desired_heading = math.atan2(policy_dy, policy_dx)

        # Calculate heading error
        heading_error = desired_heading - self.current_orientation

        # Normalize angle to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        # Adaptive controller: slow down for sharp turns, speed up when aligned
        if abs(heading_error) < 0.3:  # Well aligned (< 17 degrees)
            linear_vel = 0.4
            angular_vel = 1.5 * heading_error
        elif abs(heading_error) < 0.8:  # Moderate turn (< 46 degrees)
            linear_vel = 0.2
            angular_vel = 2.5 * heading_error
        else:  # Sharp turn (> 46 degrees)
            linear_vel = 0.05
            angular_vel = 3.0 * heading_error

        return np.array([linear_vel, angular_vel])

    def navigate(self):
        """Execute navigation using DP policy"""
        self.get_logger().info('Starting navigation...')

        rate = self.create_rate(10)  # 10 Hz

        while rclpy.ok() and self.steps < self.max_steps:
            # Check if goal reached
            distance_to_goal = np.linalg.norm(self.current_position - self.goal_position)

            if distance_to_goal < self.goal_tolerance:
                self.get_logger().info(f'Goal reached in {self.steps} steps!')
                break

            # Get action from policy
            action = self.get_policy_action()

            # Execute action
            twist = Twist()
            twist.linear.x = float(action[0])
            twist.angular.z = float(action[1])
            self.cmd_vel_pub.publish(twist)

            self.steps += 1

            if self.steps % 20 == 0:
                grid_x = int(self.current_position[0] / self.grid_resolution)
                grid_y = int(self.current_position[1] / self.grid_resolution)
                policy_dx, policy_dy = self.policy[np.clip(grid_x, 0, self.grid_size-1),
                                                    np.clip(grid_y, 0, self.grid_size-1)]
                self.get_logger().info(
                    f'Step {self.steps}: Pos ({self.current_position[0]:.2f}, '
                    f'{self.current_position[1]:.2f}), Grid ({grid_x}, {grid_y}), '
                    f'Policy ({policy_dx:.1f}, {policy_dy:.1f}), Dist: {distance_to_goal:.2f}m'
                )

            rate.sleep()

        # Stop robot
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

        if self.steps >= self.max_steps:
            self.get_logger().warn('Max steps reached without reaching goal')


def main():
    rclpy.init()

    try:
        navigator = DPNavigator()

        # Wait for odometry
        time.sleep(2.0)

        # Execute navigation
        navigator.navigate()

    except KeyboardInterrupt:
        print("\nNavigation interrupted by user")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
