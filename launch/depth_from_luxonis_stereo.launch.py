from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_stereo_image_proc = get_package_share_directory('stereo_image_proc')

    # Paths
    stereo_image_proc_launch = PathJoinSubstitution(
        [pkg_stereo_image_proc, 'launch', 'stereo_image_proc.launch.py'])

    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument('rviz', default_value='true', description='Launch RVIZ (optional).'),

        SetParameter(name='use_sim_time', value=True),

        # Uncompress images for stereo_image_rect and remap to expected names from stereo_image_proc
        Node(
            package='image_transport', executable='republish', name='republish_left', output='screen',
            namespace='luxonis/oak',
            parameters=[{
                'in_transport': 'compressed',
                'out_transport': 'raw',
            }],
            remappings=[('in/compressed', 'left/image_raw/compressed'),
                        ('out', 'left/image_raw')]),
        Node(
            package='image_transport', executable='republish', name='republish_right', output='screen',
            namespace='luxonis/oak',
            parameters=[{
                'in_transport': 'compressed',
                'out_transport': 'raw',
            }],
            remappings=[('in/compressed', 'right/image_raw/compressed'),
                        ('out', 'right/image_raw')]),

        # Run the ROS package stereo_image_proc for image rectification
        GroupAction(
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([stereo_image_proc_launch]),
                    launch_arguments=[
                        ('left_namespace', 'luxonis/oak/left'),
                        ('right_namespace', 'luxonis/oak/right'),
                        ('disparity_range', '128'),
                        ('texture_threshold', '10'),
                        ('approximate_sync', 'true'),
                    ]
                ),
            ]
        ),
    ])
