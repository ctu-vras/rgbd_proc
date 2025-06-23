from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    rviz_config = os.path.join(get_package_share_directory('dataproc'), 'config', 'rviz', 'helhest.rviz')
    bag_launch_path = os.path.join(get_package_share_directory('dataproc'),
                                   'launch',
                                   'helhest_play.launch.py')
    depth_from_luxonis_stereo = os.path.join(get_package_share_directory('dataproc'),
                                             'launch',
                                             'depth_from_luxonis_stereo.launch.py')

    ld = LaunchDescription([
        # Bag file playback
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(bag_launch_path),
            launch_arguments=[
                ('bag_path', '/home/ruslan/data/bags/helhest/helhest_2025_04_29-16_10_48/'),
            ]
        ),

        # Depth from Luxonis stereo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(depth_from_luxonis_stereo),
        ),

        # Disparity visualization
        Node(
            package='image_view',
            executable='disparity_view',
            name='disparity_view',
            remappings=[
                ('image', '/disparity'),
            ],
            parameters=[{
                'autosize': True,
            }],
        ),

        # RVIZ2 visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
        )
    ])

    return ld
