import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import TimerAction


def generate_launch_description():
    
    # Get paths
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    

    world_file = os.path.join(
        get_package_share_directory('maze_robot_qlearning'),
        'worlds',
        'maze_world.sdf'
    )

    urdf_file = os.path.join(
        get_package_share_directory('maze_robot_qlearning'),
        'urdf',
        'simple_robot.urdf'
    )
    
    # Launch configuration variables
    x_pose = LaunchConfiguration('x_pose', default='0.5')
    y_pose = LaunchConfiguration('y_pose', default='0.5')
    z_pose = LaunchConfiguration('z_pose', default='0.05')
    # z_pose = LaunchConfiguration('z_pose', default='0.01')

    
    # Gazebo launch
    # gazebo = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
    #     ),
    #     launch_arguments={'world': world_file}.items()
    # )
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world_file,
            'verbose': 'true'
        }.items()
    )
        
    # gazebo = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(
    #             get_package_share_directory('maze_robot_qlearning'),
    #             'launch',
    #             'gzserver_launch.py'
    #         )
    #     )
    # )

    # Robot State Publisher
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()
    
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )
    
    # Spawn entity
    # spawn_entity = Node(
    #     package='gazebo_ros',
    #     executable='spawn_entity.py',
    #     arguments=[
    #         '-entity', 'simple_robot',
    #         '-topic', 'robot_description',
    #         '-x', x_pose,
    #         '-y', y_pose,
    #         '-z', z_pose
    #     ],
    #     output='screen'
    # )

    spawn_entity = TimerAction(
        period=3.0,  # Wait 3 seconds after Gazebo starts
        actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-entity', 'simple_robot',
                    '-topic', 'robot_description',
                    '-x', x_pose,
                    '-y', y_pose,
                    '-z', z_pose
                ],
                output='screen'
            )
        ]
    )
    
    return LaunchDescription([
        DeclareLaunchArgument('x_pose', default_value='0.5'),
        DeclareLaunchArgument('y_pose', default_value='0.5'),
        DeclareLaunchArgument('z_pose', default_value='0.05'),
        gazebo,
        robot_state_publisher,
        spawn_entity,
    ])
