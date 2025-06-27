from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    rviz_config = os.path.join(get_package_share_directory('rgbd_proc'), 'config', 'rviz', 'helhest.rviz')

    ld = LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument('rviz', default_value='false', description='Launch RVIZ2 or not'),
        DeclareLaunchArgument('bag_path', description='Full path to the ROS 2 bag file to play'),

        SetParameter(name='use_sim_time', value=True),

        # Start playing the bag file
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', LaunchConfiguration('bag_path'),
                 '--clock',
                 '--start-offset', '0.0'],
        ),

        # Launch RVIZ2 with the specified configuration
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ])

    return ld
