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

        # Uncompress depth image and publish it as raw
        Node(
            package='image_transport',
            executable='republish',
            name='depth_uncompressor',
            remappings=[
                ('in/compressedDepth', '/luxonis/oak/stereo/image_raw/compressedDepth'),
                ('out', '/luxonis/oak/stereo/image_raw/depth'),
                ('out', '/luxonis_tof_front/tof_front/tof/image_raw/depth'),
            ],
            parameters=[{
                'in_transport': 'compressedDepth',
                'out_transport': 'raw',
            }],
        ),

        # Uncompress the rgb image and publish it as raw
        Node(
            package='image_transport',
            executable='republish',
            name='rgb_uncompressor',
            remappings=[
                ('in/compressed', '/luxonis/oak/right/image_rect/compressed'),
                ('out', '/luxonis/oak/right/image_rect'),
            ],
            parameters=[{
                'in_transport': 'compressed',
                'out_transport': 'raw',
            }],
        ),

        # Convert depth image to point cloud
        Node(
            package='depth_image_proc',
            executable='point_cloud_xyzrgb_node',
            name='depth_image_to_colored_point_cloud',
            remappings=[
                ('depth_registered/image_rect', '/luxonis/oak/stereo/image_raw/depth'),
                ('rgb/image_rect_color', '/luxonis/oak/right/image_rect'),
                ('rgb/camera_info', '/luxonis/oak/right/camera_info'),
                ('points', '/luxonis/oak/depth/points'),
            ],
            parameters=[{
                'approximate_sync': True,
                'queue_size': 10,
            }],
        ),
        # Node(
        #     package='depth_image_proc',
        #     executable='point_cloud_xyz_node',
        #     name='depth_image_to_point_cloud',
        #     remappings=[
        #         ('image_rect', '/luxonis_tof_front/tof_front/tof/image_raw/compressedDepth'),
        #         ('camera_info', '/luxonis_tof_front/tof_front/tof/camera_info'),
        #         ('points', '/luxonis/oak/depth/points'),
        #     ],
        #     parameters=[{
        #         'depth_image_transport': 'compressedDepth',
        #         'queue_size': 10,
        #     }],
        # ),

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
