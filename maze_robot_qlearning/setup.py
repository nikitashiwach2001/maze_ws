from setuptools import setup
import os
from glob import glob

package_name = 'maze_robot_qlearning'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Q-Learning maze navigation robot',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'keyboard_teleop = maze_robot_qlearning.keyboard_teleop:main',
            'maze_env = maze_robot_qlearning.maze_environment:main',
            'inspect_env = maze_robot_qlearning.inspect_env:inspect_environment',
            'train_deep_sarsa = maze_robot_qlearning.train_deep_sarsa:main',
            'train_td7 = maze_robot_qlearning.train_td7:main',
            'test_agent = maze_robot_qlearning.test_agent:main',
        ],
    },
)