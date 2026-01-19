#!/usr/bin/env python3
"""
Continuous state space maze environment for neural network-based RL
State: (x, y, goal_x, goal_y) - continuous positions
Actions: Linear and angular velocity commands
"""

from collections import deque
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
        self.goal_position = np.array([3.5, 3.5])  # Top-right corner
        self.start_position = np.array([0.5, 0.5])  # Bottom-left corner
        self.goal_tolerance = 0.4  # Distance to goal for success

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
        # self.collision_threshold = 0.19 #0.3  # meters - distance to consider as "too close"

        self.lidar_history = deque(maxlen=3)  # Store last 3 readings
        self.collision_threshold = 0.22  # Slightly higher for safety margin
        

        self.odom_updated = False
        self.lidar_updated = False

        self.collision_steps = 0

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

        self.odom_updated = True
        
    def scan_callback(self, msg):
        """Update LiDAR data - ONLY CHECK FRONT HEMISPHERE"""
        # LiDAR scan is typically 360 degrees
        # We only want to check the front ~180 degrees
        
        desired_angle = 120  # degrees
        num_readings = len(msg.ranges)
        half_span = int(num_readings * (desired_angle / 360) / 2)
        center = num_readings // 2
        start = center - half_span
        end = center + half_span
        front_ranges = msg.ranges[start:end]

        valid_ranges = [r for r in front_ranges if not np.isinf(r) and not np.isnan(r) and r > 0.05] 

        if len(valid_ranges) > 0:
            # Use 5th percentile for current reading
            current_min = np.percentile(valid_ranges, 5)
            
            # Add to history
            self.lidar_history.append(current_min)
            
            # Use median of recent readings to filter noise
            if len(self.lidar_history) >= 2:
                self.min_obstacle_distance = np.median(list(self.lidar_history))
            else:
                self.min_obstacle_distance = current_min
            self.lidar_updated = True
        else:
            self.min_obstacle_distance = float('inf')

        
        # Front hemisphere: take middle half of readings
        # For 360 degree scan, this is roughly -90° to +90° in front
        # front_start = num_readings // 4      # 90 degrees on left
        # front_end = 3 * num_readings // 4    # 90 degrees on right
        
        # # Extract front readings only
        # front_ranges = msg.ranges[front_start:front_end]
        
        # Filter out invalid readings (inf, nan)
        # valid_ranges = [r for r in front_ranges if not np.isinf(r) and not np.isnan(r) and r > 0]
        # valid_ranges = [r for r in front_ranges if not np.isinf(r) and not np.isnan(r) and r > 0.05] 
        
        # if valid_ranges:
        #     # self.min_obstacle_distance = max(0.15, min(valid_ranges))
        #     self.min_obstacle_distance = max(0.15, np.percentile(valid_ranges, 5))
        #     self.scan_ranges = np.array(msg.ranges)  # Keep full scan for potential future use
        #     self.lidar_updated = True
        # else:
        #     self.min_obstacle_distance = float('inf')



    def get_observation(self):
        """
        Enhanced state with goal-relative information
        Returns: [x, y, goal_x, goal_y, cos(theta), sin(theta), lidar, 
                distance_to_goal, angle_to_goal_cos, angle_to_goal_sin]
        """
        # Current state
        x, y = self.current_position
        gx, gy = self.goal_position
        
        # Distance to goal
        distance_to_goal = np.linalg.norm(self.current_position - self.goal_position)
        
        # Angle to goal (in robot frame)
        angle_to_goal_world = np.arctan2(gy - y, gx - x)
        angle_to_goal_relative = angle_to_goal_world - self.current_orientation
        
        obs = np.array([
            x, y, gx, gy,
            math.cos(self.current_orientation),
            math.sin(self.current_orientation),
            self.min_obstacle_distance,
            distance_to_goal / (np.sqrt(2) * self.maze_size),  # Normalized
            math.cos(angle_to_goal_relative),
            math.sin(angle_to_goal_relative)
        ], dtype=np.float32)
        
        return obs
    
        # """
        # Get current state observation
        # Returns: numpy array [x, y, goal_x, goal_y, cos(theta), sin(theta)]
        # - x, y: current position
        # - goal_x, goal_y: goal position
        # - cos(theta), sin(theta): current orientation (better for NN than raw angle)
        # """
        # obs = np.array([
        #     self.current_position[0],
        #     self.current_position[1],
        #     self.goal_position[0],
        #     self.goal_position[1],
        #     math.cos(self.current_orientation),
        #     math.sin(self.current_orientation),
        #     self.min_obstacle_distance  # Add LiDAR info
        # ], dtype=np.float32)

        # return obs

    def reset(self):    

        """
        Reset environment to starting position using set_entity_state
        Returns: initial observation
        """
        self.get_logger().info('Resetting environment...')
        self.position_buffer = []


        # self.prev_distance_to_goal = None
        # self.no_progress_steps = 0

        
        # Stop robot first
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        time.sleep(0.05)

        # Wait for Gazebo service
        if not self.set_entity_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gazebo set_entity_state service not available')
            return self.get_observation()

        # Reset robot to start position
        self.teleport_robot(self.start_position[0], self.start_position[1], 0.0)

        # Wait for physics to stabilize
        time.sleep(0.3)
        
        # Wait for fresh odometry with callback flag
        

        self.odom_updated = False
        timeout_start = time.time()
        while not self.odom_updated and (time.time() - timeout_start) < 0.3:
            rclpy.spin_once(self, timeout_sec=0.01)
        
        # Reset episode tracking
        self.episode_step = 0

        # Verify we have valid sensor data

        # Wait for fresh LiDAR data with callback flag
        self.lidar_history.clear()
        self.collision_steps = 0
        self.prev_distance_to_goal = None  
        self.progress_step_count = 0        
        self.min_obstacle_distance = float('inf')
        self.lidar_updated = False



        self.get_logger().info(f'LiDAR distance after reset: {self.min_obstacle_distance:.2f}m')
        self.get_logger().info(f'LiDAR history size: {len(self.lidar_history)}')  # Should be 3

        for _ in range(3):
            self.lidar_updated = False
            timeout_start = time.time()
            while not self.lidar_updated and (time.time() - timeout_start) < 0.2:
                rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.1)  # Small delay between readings
        
        # Verify we have valid sensor data
        self.get_logger().info(f'LiDAR distance after reset: {self.min_obstacle_distance:.2f}m')
        
        obs = self.get_observation()
        self.get_logger().info(f'Reset complete. Position: ({obs[0]:.2f}, {obs[1]:.2f})')
        return obs

 
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
        linear_vel = np.clip(action[0], 0.0, 1.0) # forward only
        angular_vel = np.clip(action[1], -1.0, 1.0)

        # Execute action for fixed duration
        twist = Twist()
        twist.linear.x = float(linear_vel)
        twist.angular.z = float(angular_vel)

        # Execute action for 0.5 seconds
        action_duration = 0.5  # seconds
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

        # DEBUG: Log each step (especially useful for first step)
        if self.episode_step <= 3:  # Log first 3 steps of each episode
            self.get_logger().info(
                f'  Step {self.episode_step}: '
                f'action=[{linear_vel:.3f}, {angular_vel:.3f}], '
                f'pos=({obs[0]:.2f}, {obs[1]:.2f}), '
                f'lidar={self.min_obstacle_distance:.2f}m, '
                f'reward={reward:.2f}'
            )
        return obs, reward, done, info

    # def compute_reward(self):
    #     """
    #     Compute reward based on current state (NO OBSTACLES version)

    #     Returns:
    #         reward: scalar reward
    #         done: whether episode should terminate
    #         info: dict with additional information
    #     """
    #     # Calculate distance to goal
    #     distance_to_goal = np.linalg.norm(self.current_position - self.goal_position)

    #     # Check if goal reached
    #     if distance_to_goal < self.goal_tolerance:
    #         return 100.0, True, {'success': True, 'reason': 'goal_reached'}

    #     # Check if out of bounds
    #     if (self.current_position[0] < 0 or self.current_position[0] > self.maze_size or
    #         self.current_position[1] < 0 or self.current_position[1] > self.maze_size):
    #         return -50.0, True, {'success': False, 'reason': 'out_of_bounds'}

    #     # Check if max steps reached
    #     if self.episode_step >= self.max_steps:
    #         return -10.0, True, {'success': False, 'reason': 'timeout'}

    #     # Reward shaping: encourage moving toward goal
    #     # Higher reward for being closer to goal
    #     step_penalty = -0.5  # Reduced penalty since no obstacles
    #     distance_reward = -0.2 * distance_to_goal  # Encourage getting closer

    #     reward = step_penalty + distance_reward
    #     done = False
    #     info = {'success': None, 'reason': 'ongoing'}

    #     return reward, done, info

    def compute_reward(self):
        """TD7-optimized reward function with strong wall avoidance"""

        self.position_buffer.append(self.current_position.copy())

        window = 30        # ~1.5 seconds
        min_displacement = 0.20  # meters

        if len(self.position_buffer) >= window:
            displacement = np.linalg.norm(
                self.position_buffer[-1] - self.position_buffer[-window]
            )

            if displacement < min_displacement:
                return -15.0, True, {
                    'success': False,
                    'reason': 'stuck_spinning'
                }

        distance_to_goal = np.linalg.norm(self.current_position - self.goal_position)

        self.get_logger().info(f'Distance to goal: {distance_to_goal:.2f}m')
        
        # === TERMINAL REWARDS ===
        
        # Success: reached goal
        if distance_to_goal < self.goal_tolerance:
            self.get_logger().info("Success- goal_reached.")
            return 100.0, True, {'success': True, 'reason': 'goal_reached'}
        
        # Collision with obstacle (LiDAR-based)
        if self.min_obstacle_distance <= self.collision_threshold:
            self.collision_steps += 1
        else:
            self.collision_steps = 0

        if self.collision_steps >= 4:
            self.get_logger().error("COLLISION - Episode terminating.")
            return -20.0, True, {'success': False, 'reason': 'collision'}
        
                
        # Out of bounds
        if (self.current_position[0] < 0 or self.current_position[0] > self.maze_size or
            self.current_position[1] < 0 or self.current_position[1] > self.maze_size):
            return -20.0, True, {'success': False, 'reason': 'out_of_bounds'}
        
        # Timeout
        if self.episode_step >= self.max_steps:
            return -5.0, True, {'success': False, 'reason': 'timeout'}
        
        # === STEP REWARDS ===
        
        # 1. Progress reward (encourage moving toward goal)
        max_dist = np.sqrt(2) * self.maze_size
        normalized_distance = distance_to_goal / max_dist
        progress_reward = 2.0 * (1.0 - normalized_distance)  # Range: 0 to 2.0
        
        # 2. Time penalty (encourage efficiency)
        time_penalty = -0.05
        
        # 3. STRONG wall avoidance penalty (IMPROVED)
        # Progressive penalty based on distance to nearest obstacle
        safe_distance = 0.5  # Ideal minimum distance from walls (meters)
        danger_distance = 0.25  # Getting dangerous

        if self.min_obstacle_distance < danger_distance:
            normalized = (danger_distance - self.min_obstacle_distance) / (danger_distance - self.collision_threshold)
            wall_penalty = -2.0 * (normalized ** 2)  # Quadratic penalty
        elif self.min_obstacle_distance < safe_distance:
            wall_penalty = -0.5 * (safe_distance - self.min_obstacle_distance) / (safe_distance - danger_distance)
        else:
            wall_penalty = 0.0

        # 4. NEW: Goal proximity bonus (override wall penalty when very close to goal)
        if distance_to_goal < 0.5:  # Within 0.5m of goal
            # Strong bonus that grows as robot approaches goal
            goal_proximity_bonus = 3.0 * (1.0 - distance_to_goal / 0.5)  # Range: 0 to 3.0
            wall_penalty = 0.0  # Negate wall penalty when very close to goal
        else:
            goal_proximity_bonus = 0.0

        
        # 5. Progress-based reward (reward for moving toward goal)
        if not hasattr(self, 'prev_distance_to_goal') or self.prev_distance_to_goal is None:
            self.prev_distance_to_goal = distance_to_goal
            progress_bonus = 0.0
        else:
            # Positive reward if getting closer, negative if moving away
            distance_improvement = self.prev_distance_to_goal - distance_to_goal
            
            if distance_improvement > 0:  # Moving toward goal
                # Track consecutive progress steps
                if not hasattr(self, 'progress_step_count'):
                    self.progress_step_count = 0
                self.progress_step_count += 1
                
                # Bonus grows with consecutive progress (up to 4 steps)
                consecutive_bonus = min(self.progress_step_count, 4) * 0.3  # 0.3, 0.6, 0.9, 1.2
                progress_bonus = 0.5 + consecutive_bonus  # Range: 0.5 to 1.7
            else:
                # Reset counter if moving away or not making progress
                self.progress_step_count = 0
                progress_bonus = 0.0
            
            self.prev_distance_to_goal = distance_to_goal

        # 5. Forward motion bonus (encourage exploration when safe)
        if self.min_obstacle_distance > safe_distance:
            forward_bonus = 0.3
        else:
            forward_bonus = 0.0

        
        # progress_threshold = 0.02   # 2 cm
        # max_no_progress_steps = 25  # ~1–2 seconds

        # if self.prev_distance_to_goal is None:
        #     self.prev_distance_to_goal = distance_to_goal
        # else:
        #     if abs(self.prev_distance_to_goal - distance_to_goal) < progress_threshold:
        #         self.no_progress_steps += 1
        #     else:
        #         self.no_progress_steps = 0

        #     self.prev_distance_to_goal = distance_to_goal

        # # Terminate if no meaningful progress
        # if self.no_progress_steps >= max_no_progress_steps:
        #     return -10.0, True, {
        #         'success': False,
        #         'reason': 'no_progress_stuck'
        #     }

        # Total step reward
        reward = progress_reward + time_penalty + wall_penalty + forward_bonus + goal_proximity_bonus + progress_bonus
        
        done = False
        info = {'success': None, 'reason': 'ongoing', 'distance': distance_to_goal}
        
        return reward, done, info


    # def compute_reward(self):
    #     """TD7-optimized reward function"""
    #     distance_to_goal = np.linalg.norm(self.current_position - self.goal_position)
        
    #     # === TERMINAL REWARDS ===
    #     if distance_to_goal < self.goal_tolerance:
    #         return 100.0, True, {'success': True, 'reason': 'goal_reached'}
        
    #     if self.min_obstacle_distance < self.collision_threshold:
    #         return -20.0, True, {'success': False, 'reason': 'collision'}
        
    #     if (self.current_position[0] < 0 or self.current_position[0] > self.maze_size or
    #         self.current_position[1] < 0 or self.current_position[1] > self.maze_size):
    #         return -20.0, True, {'success': False, 'reason': 'out_of_bounds'}
        
    #     if self.episode_step >= self.max_steps:
    #         return -5.0, True, {'success': False, 'reason': 'timeout'}
        
    #     # === STEP REWARDS ===
    #     # 1. Progress reward (exponential to emphasize getting closer)
    #     max_dist = np.sqrt(2) * self.maze_size
    #     progress_reward = 2.0 * (1.0 - distance_to_goal / max_dist)  # Range: 0 to 2.0
        
    #     # 2. Reduced time penalty (was -0.1)
    #     time_penalty = -0.05  # Less harsh
        
    #     # 3. Safety margin reward (avoid obstacles)
    #     if self.min_obstacle_distance < 0.8:
    #         safety_penalty = -0.5 * (0.8 - self.min_obstacle_distance)
    #     else:
    #         safety_penalty = 0.0
        
    #     reward = progress_reward + time_penalty + safety_penalty
        
    #     done = False
    #     info = {'success': None, 'reason': 'ongoing', 'distance': distance_to_goal}
        
    #     return reward, done, info


    
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
