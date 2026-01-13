#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import termios
import tty


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        self.linear_speed = 0.3
        self.angular_speed = 0.8
        
        self.get_logger().info('Keyboard Teleop Started!')
        self.get_logger().info('Use WASD keys to control:')
        self.get_logger().info('  W - Forward')
        self.get_logger().info('  S - Backward')
        self.get_logger().info('  A - Turn Left')
        self.get_logger().info('  D - Turn Right')
        self.get_logger().info('  X - Stop')
        self.get_logger().info('  Q - Quit')
    
    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key
    
    def run(self):
        self.settings = termios.tcgetattr(sys.stdin)
        
        try:
            while True:
                key = self.get_key()
                
                twist = Twist()
                
                if key == 'w':
                    twist.linear.x = self.linear_speed
                    self.get_logger().info('Moving Forward')
                elif key == 's':
                    twist.linear.x = -self.linear_speed
                    self.get_logger().info('Moving Backward')
                elif key == 'a':
                    twist.angular.z = self.angular_speed
                    self.get_logger().info('Turning Left')
                elif key == 'd':
                    twist.angular.z = -self.angular_speed
                    self.get_logger().info('Turning Right')
                elif key == 'x':
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.get_logger().info('Stop')
                elif key == 'q':
                    self.get_logger().info('Quitting...')
                    break
                else:
                    continue
                
                self.publisher.publish(twist)
                
        except Exception as e:
            self.get_logger().error(f'Error: {e}')
        finally:
            # Stop the robot
            twist = Twist()
            self.publisher.publish(twist)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


def main(args=None):
    rclpy.init(args=args)
    teleop = KeyboardTeleop()
    
    try:
        teleop.run()
    except KeyboardInterrupt:
        pass
    finally:
        teleop.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
