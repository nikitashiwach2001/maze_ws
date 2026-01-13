import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    world_file = os.path.join(
        get_package_share_directory('maze_robot_qlearning'),
        'worlds',
        'maze_world.sdf'
    )
    
    gzserver = ExecuteProcess(
        cmd=[
            'gzserver',
            '--verbose',
            world_file,
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so',
        ],
        output='screen'
    )
    
    gzclient = ExecuteProcess(
        cmd=['gzclient'],
        output='screen'
    )
    
    return LaunchDescription([
        gzserver,
        gzclient
    ])