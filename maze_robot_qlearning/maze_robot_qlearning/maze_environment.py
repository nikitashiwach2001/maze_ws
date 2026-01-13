#!/usr/bin/env python3
"""
Continuous state space maze environment for neural network-based RL
State: (x, y, goal_x, goal_y) - continuous positions
Actions: Linear and angular velocity commands
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
# from gazebo_msgs.srv import SetEntityState
# from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import DeleteEntity, SpawnEntity, SetEntityState
import numpy as np
import time
import math
from sensor_msgs.msg import LaserScan


class MazeEnvironment(Node):
    def __init__(self):
        super().__init__('maze_environment')

        # Environment parameters
        self.maze_size = 5.0  # 5x5 meter maze
        self.goal_position = np.array([4.5, 4.5])  # Top-right corner
        self.start_position = np.array([0.5, 0.5])  # Bottom-left corner
        self.goal_tolerance = 0.3  # Distance to goal for success

        # Robot state
        self.current_position = np.array([0.5, 0.5])
        self.current_orientation = 0.0  # yaw angle

        # Episode tracking
        self.episode_step = 0
        self.max_steps = 200  # Maximum steps per episode

        # ROS 2 publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odometry_callback, 10
        )

        ############################# Lidar Scan ##############################################
        self.scan_sub = self.create_subscription(
                LaserScan, '/scan', self.scan_callback, 10
            )
        # Add LiDAR state variables
        self.min_obstacle_distance = float('inf')
        self.scan_ranges = None
        self.collision_threshold = 0.3  # meters - distance to consider as "too close"

        ############################# Lidar Scan ##############################################

        # Gazebo services for resetting robot position
        self.set_entity_client = self.create_client(SetEntityState, '/set_entity_state')
        self.delete_entity_client = self.create_client(DeleteEntity, '/delete_entity')
        self.spawn_entity_client = self.create_client(SpawnEntity, '/spawn_entity')

        # Load robot URDF for respawning
        import os
        from ament_index_python.packages import get_package_share_directory
        urdf_file = os.path.join(
            get_package_share_directory('maze_robot_qlearning'),
            'urdf',
            'simple_robot.urdf'
        )
        with open(urdf_file, 'r') as f:
            self.robot_urdf = f.read()

        self.get_logger().info('Maze Environment initialized!')
        self.get_logger().info(f'Maze size: {self.maze_size}x{self.maze_size}m')
        self.get_logger().info(f'Goal position: ({self.goal_position[0]}, {self.goal_position[1]})')
        self.get_logger().info(f'Start position: ({self.start_position[0]}, {self.start_position[1]})')

    def odometry_callback(self, msg):
        """Update robot position from odometry"""
        self.current_position[0] = msg.pose.pose.position.x
        self.current_position[1] = msg.pose.pose.position.y

        # Extract yaw from quaternion
        quat = msg.pose.pose.orientation
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        self.current_orientation = math.atan2(siny_cosp, cosy_cosp)

    
    # def scan_callback(self, msg):
    #     """Update LiDAR data"""
    #     # Filter out invalid readings (inf, nan)
    #     valid_ranges = [r for r in msg.ranges if not np.isinf(r) and not np.isnan(r) and r > 0]
        
    #     if valid_ranges:
    #         self.min_obstacle_distance = min(valid_ranges)
    #         self.scan_ranges = np.array(msg.ranges)
    #     else:
    #         self.min_obstacle_distance = float('inf')
    
    def scan_callback(self, msg):
        """Update LiDAR data - ONLY CHECK FRONT HEMISPHERE"""
        # LiDAR scan is typically 360 degrees
        # We only want to check the front ~180 degrees
        
        num_readings = len(msg.ranges)
        
        # Front hemisphere: take middle half of readings
        # For 360 degree scan, this is roughly -90° to +90° in front
        front_start = num_readings // 4      # 90 degrees on left
        front_end = 3 * num_readings // 4    # 90 degrees on right
        
        # Extract front readings only
        front_ranges = msg.ranges[front_start:front_end]
        
        # Filter out invalid readings (inf, nan)
        valid_ranges = [r for r in front_ranges if not np.isinf(r) and not np.isnan(r) and r > 0]
        
        if valid_ranges:
            self.min_obstacle_distance = min(valid_ranges)
            self.scan_ranges = np.array(msg.ranges)  # Keep full scan for potential future use
        else:
            self.min_obstacle_distance = float('inf')



    def get_observation(self):
        """
        Get current state observation
        Returns: numpy array [x, y, goal_x, goal_y, cos(theta), sin(theta)]
        - x, y: current position
        - goal_x, goal_y: goal position
        - cos(theta), sin(theta): current orientation (better for NN than raw angle)
        """
        obs = np.array([
            self.current_position[0],
            self.current_position[1],
            self.goal_position[0],
            self.goal_position[1],
            math.cos(self.current_orientation),
            math.sin(self.current_orientation),
            self.min_obstacle_distance  # Add LiDAR info
        ], dtype=np.float32)

        return obs

    def reset(self):
        """
        Reset environment to starting position using set_entity_state
        Returns: initial observation
        """
        self.get_logger().info('Resetting environment...')
        
        # Stop robot first
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        time.sleep(0.1)

        # Wait for Gazebo service
        if not self.set_entity_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gazebo set_entity_state service not available')
            return self.get_observation()

        # Reset robot to start position
        self.teleport_robot(self.start_position[0], self.start_position[1], 0.0)

        # Wait for physics to stabilize
        time.sleep(0.3)
        
        # Update odometry
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.1)

        # Reset episode tracking
        self.episode_step = 0

        obs = self.get_observation()
        self.get_logger().info(f'Reset complete. Position: ({obs[0]:.2f}, {obs[1]:.2f})')

        return obs

    # def reset(self):
        """
        Reset environment by respawning robot at start position
        Returns: initial observation
        """
        self.get_logger().info('Resetting environment...')

        # Stop robot first
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        time.sleep(0.2)

        # Delete existing robot
        if self.delete_entity_client.wait_for_service(timeout_sec=2.0):
            delete_req = DeleteEntity.Request()
            delete_req.name = 'simple_robot'
            delete_future = self.delete_entity_client.call_async(delete_req)
            rclpy.spin_until_future_complete(self, delete_future, timeout_sec=2.0)
            time.sleep(0.5)

        # Spawn robot at start position
        if self.spawn_entity_client.wait_for_service(timeout_sec=2.0):
            spawn_req = SpawnEntity.Request()
            spawn_req.name = 'simple_robot'
            spawn_req.xml = self.robot_urdf
            spawn_req.initial_pose.position.x = self.start_position[0]
            spawn_req.initial_pose.position.y = self.start_position[1]
            spawn_req.initial_pose.position.z = 0.05
            spawn_req.initial_pose.orientation.w = 1.0

            spawn_future = self.spawn_entity_client.call_async(spawn_req)
            rclpy.spin_until_future_complete(self, spawn_future, timeout_sec=2.0)

            # Wait for robot to be ready and odometry to start
            time.sleep(1.5)

            # Spin to get odometry updates
            for _ in range(10):
                rclpy.spin_once(self, timeout_sec=0.1)

        # Reset episode tracking
        self.episode_step = 0

        obs = self.get_observation()
        self.get_logger().info(f'Reset complete. Position: ({obs[0]:.2f}, {obs[1]:.2f})')

        return obs

    
    # def reset(self):
        # """
        # Reset environment - drive robot back to start position
        # More reliable than delete/spawn
        # """
        # self.get_logger().info('Resetting environment...')
        
        # # Stop robot first
        # twist = Twist()
        # self.cmd_vel_pub.publish(twist)
        # time.sleep(0.3)
        
        # # Get current position
        # for _ in range(5):
        #     rclpy.spin_once(self, timeout_sec=0.1)
        
        # current_x = self.current_position[0]
        # current_y = self.current_position[1]
        
        # # Calculate distance to start
        # distance_to_start = np.sqrt(
        #     (current_x - self.start_position[0])**2 + 
        #     (current_y - self.start_position[1])**2
        # )
        
        # self.get_logger().info(f'Current position: ({current_x:.2f}, {current_y:.2f}), Distance to start: {distance_to_start:.2f}m')
        
        # # If far from start, drive back (more reliable than delete/spawn)
        # if distance_to_start > 0.3:
        #     self.drive_to_start()
        
        # # Reset episode tracking
        # self.episode_step = 0
        
        # # Small delay to stabilize
        # time.sleep(0.5)
        # for _ in range(5):
        #     rclpy.spin_once(self, timeout_sec=0.1)
        
        # obs = self.get_observation()
        # self.get_logger().info(f'Reset complete. Position: ({obs[0]:.2f}, {obs[1]:.2f})')
        
        # return obs

    # def drive_to_start(self):
        # """Drive robot back to start position"""
        # max_time = 15.0  # Maximum 15 seconds to return
        # start_time = time.time()
        
        # while (time.time() - start_time) < max_time:
        #     # Update position
        #     rclpy.spin_once(self, timeout_sec=0.01)
            
        #     # Calculate direction to start
        #     dx = self.start_position[0] - self.current_position[0]
        #     dy = self.start_position[1] - self.current_position[1]
        #     distance = np.sqrt(dx**2 + dy**2)
            
        #     # If close enough, stop
        #     if distance < 0.02:
        #         twist = Twist()
        #         self.cmd_vel_pub.publish(twist)
        #         break
            
        #     # Calculate desired heading
        #     target_angle = np.arctan2(dy, dx)
        #     angle_diff = target_angle - self.current_orientation
            
        #     # Normalize angle
        #     while angle_diff > np.pi:
        #         angle_diff -= 2 * np.pi
        #     while angle_diff < -np.pi:
        #         angle_diff += 2 * np.pi
            
        #     # Simple controller
        #     twist = Twist()
        #     twist.linear.x = min(0.5, distance * 0.8)
        #     twist.angular.z = angle_diff * 2.0
            
        #     self.cmd_vel_pub.publish(twist)
        #     time.sleep(0.05)
        
        # # Final stop
        # twist = Twist()
        # self.cmd_vel_pub.publish(twist)


    def step(self, action):
        """
        Execute action and return next state, reward, done, info

        Args:
            action: [linear_velocity, angular_velocity]
                    linear: -0.5 to 0.5 m/s
                    angular: -1.0 to 1.0 rad/s

        Returns:
            observation: current state
            reward: reward for this step
            done: whether episode is finished
            info: additional information
        """
        self.episode_step += 1

        # Clip actions to safe ranges
        linear_vel = np.clip(action[0], -0.5, 1.0)
        angular_vel = np.clip(action[1], -1.5, 1.5)

        # Execute action for fixed duration
        twist = Twist()
        twist.linear.x = float(linear_vel)
        twist.angular.z = float(angular_vel)

        # Execute action for 0.5 seconds
        action_duration = 1.0
        start_time = time.time()
        while (time.time() - start_time) < action_duration:
            self.cmd_vel_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.05)

        # Stop robot
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

        # Get new observation
        obs = self.get_observation()

        # Calculate reward and check termination
        reward, done, info = self.compute_reward()

        return obs, reward, done, info

    def compute_reward(self):
        """
        Compute reward based on current state

        Returns:
            reward: scalar reward
            done: whether episode should terminate
            info: dict with additional information
        """
        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(self.current_position - self.goal_position)

        # Check if goal reached
        if distance_to_goal < self.goal_tolerance:
            return 100.0, True, {'success': True, 'reason': 'goal_reached'}
        
        # Check for collision/too close to obstacle
        if self.min_obstacle_distance < self.collision_threshold:
            collision_penalty = -20.0  # Harsh penalty for getting too close
            # Optionally terminate episode on collision
            # return collision_penalty, True, {'success': False, 'reason': 'collision'}
        else:
            collision_penalty = 0.0

        # Check if out of bounds
        if (self.current_position[0] < 0 or self.current_position[0] > self.maze_size or
            self.current_position[1] < 0 or self.current_position[1] > self.maze_size):
            return -50.0, True, {'success': False, 'reason': 'out_of_bounds'}

        # Check if max steps reached
        if self.episode_step >= self.max_steps:
            return -10.0, True, {'success': False, 'reason': 'timeout'}

        # Step penalty + reward shaping (negative distance to goal)
        step_penalty = -1.0
        distance_reward = -0.1 * distance_to_goal

        reward = step_penalty + distance_reward + collision_penalty
        done = False
        info = {'success': None, 'reason': 'ongoing'}

        return reward, done, info
    
    def teleport_robot(self, x, y, yaw):
        """Teleport robot to specified pose"""
        import math
        
        req = SetEntityState.Request()
        req.state.name = 'simple_robot'
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = 0.05
        
        # Convert yaw to quaternion
        req.state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state.pose.orientation.w = math.cos(yaw / 2.0)
        
        future = self.set_entity_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        if future.result() is not None:
            return future.result().success
        return False

    def close(self):
        """Clean up resources"""
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        self.destroy_node()


def main():
    """Test the environment"""
    rclpy.init()
    env = MazeEnvironment()

    try:
        # Wait for things to initialize
        time.sleep(2.0)

        # Test reset
        print("\n=== Testing Environment ===")
        obs = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial observation: {obs}")
        print(f"Observation format: [x, y, goal_x, goal_y, cos(theta), sin(theta)]")

        # Test a few random actions
        print("\n=== Testing Random Actions ===")
        for i in range(5):
            # Random action: [linear_vel, angular_vel]
            action = np.random.uniform([-0.3, -0.5], [0.3, 0.5], size=2)
            print(f"\nStep {i+1}: Action = [{action[0]:.2f}, {action[1]:.2f}]")

            obs, reward, done, info = env.step(action)
            print(f"  Position: ({obs[0]:.2f}, {obs[1]:.2f})")
            print(f"  Reward: {reward:.2f}")
            print(f"  Done: {done}")
            print(f"  Info: {info}")

            if done:
                print(f"\nEpisode finished: {info['reason']}")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
