#!/usr/bin/env python3

import numpy as np
import cv2

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo


def cloud_from_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    height, width = depth.shape

    # Generate pixel coordinates
    vu = np.indices(depth.shape).reshape((2, -1))  # shape (2, H*W)
    uv = vu[::-1, :]  # shape (2, H*W)
    coords = np.vstack([
        uv[0, :],  # u (x-coordinate)
        uv[1, :],  # v (y-coordinate)
        np.ones(height * width)
    ])  # shape (3, H*W)

    K_inv = np.linalg.inv(K)
    depth_flat = depth.flatten()
    points = (K_inv @ coords) * depth_flat  # shape (3, H*W)
    point3d_coords = points.T  # shape (H*W, 3)

    # # Filter out invalid points (depth == 0)
    # valid = depth_flat > 0
    # point3d_coords = point3d_coords[valid]

    return point3d_coords


class DepthFilter(Node):
    def __init__(self):
        super().__init__('depth_filter')
        self.vis = self.declare_parameter('vis', False).value

        # fov filter parameters
        self.declare_parameter('fov_filter.enabled', True)
        self.declare_parameter('fov_filter.min_angle', -15.0)
        self.declare_parameter('fov_filter.max_angle', 15.0)
        self.fov_filter_enabled = self.get_parameter('fov_filter.enabled').get_parameter_value().bool_value
        self.min_angle = self.get_parameter('fov_filter.min_angle').get_parameter_value().double_value
        self.max_angle = self.get_parameter('fov_filter.max_angle').get_parameter_value().double_value

        # median filter parameters
        self.declare_parameter('median_filter.enabled', False)
        self.declare_parameter('median_filter.kernel_size', 5)
        self.median_filter_enabled = self.get_parameter('median_filter.enabled').get_parameter_value().bool_value
        self.median_kernel_size = self.get_parameter('median_filter.kernel_size').get_parameter_value().integer_value

        # distance filter parameters
        self.declare_parameter('dist_filter.enabled', False)
        self.declare_parameter('dist_filter.min_dist', 1.0)
        self.declare_parameter('dist_filter.max_dist', 10.0)
        self.dist_filter_enabled = self.get_parameter('dist_filter.enabled').get_parameter_value().bool_value
        self.min_dist = self.get_parameter('dist_filter.min_dist').get_parameter_value().double_value
        self.max_dist = self.get_parameter('dist_filter.max_dist').get_parameter_value().double_value

        self.cv_bridge = CvBridge()
        self.filtered_depth_pub = self.create_publisher(
            msg_type=Image,
            topic='/filtered_depth',
            qos_profile=10,
        )
        self.depth_subscription = self.create_subscription(
            msg_type=Image,
            topic='/depth',
            callback=self.depth_callback,
            qos_profile=10,
        )
        # subscribe to camera info once
        self.K = None
        self.camera_info_subscription = self.create_subscription(
            msg_type=CameraInfo,
            topic='/camera_info',
            callback=self.camera_info_callback,
            qos_profile=10,
        )

    def camera_info_callback(self, msg: CameraInfo):
        self._logger.info(f"Received camera info message")
        self.K = np.array(msg.k).reshape(3, 3)
        # Unsubscribe from the camera info topic after receiving the first message
        self.destroy_subscription(self.camera_info_subscription)

    def depth_callback(self, msg: Image):
        self._logger.debug(f"Received depth message")

        # Convert the depth image to a numpy array
        depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        depth_filtered = depth.copy()
        if self.median_filter_enabled:
            # median filter to remove noise
            depth_filtered = cv2.medianBlur(depth_filtered, self.median_kernel_size)

        # Filter based on elevation angle
        if self.K is not None and self.fov_filter_enabled:
            # Get the camera intrinsic parameters
            fy = self.K[1, 1]
            cy = self.K[1, 2]

            # Create a meshgrid of pixel coordinates
            min_angle = self.min_angle * np.pi / 180
            max_angle = self.max_angle * np.pi / 180

            # Convert to row range
            min_row = int(cy + fy * np.tan(min_angle))
            max_row = int(cy + fy * np.tan(max_angle))
            self._logger.debug(f"min_row={min_row}, max_row={max_row}")

            depth_filtered[:min_row] = 0
            depth_filtered[max_row:] = 0

        if self.K is not None and self.dist_filter_enabled:
            # points = cloud_from_depth(depth_filtered, self.K)
            depth_filtered[depth_filtered < self.min_dist] = 0
            depth_filtered[depth_filtered > self.max_dist] = 0

        if self.vis:
            # show the colorized result
            result = np.concatenate((depth, depth_filtered), axis=1)
            result_vis = result.copy()
            result_vis = cv2.normalize(result_vis, None, 0, 255, cv2.NORM_MINMAX)
            result_vis = np.uint8(result_vis)
            result_vis = cv2.applyColorMap(result_vis, cv2.COLORMAP_JET)
            result_vis = cv2.cvtColor(result_vis, cv2.COLOR_RGB2BGR)
            cv2.imshow("Before vs After", result_vis)
            cv2.waitKey(1)

        # Convert the filtered depth image back to a ROS message
        filtered_depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_filtered, encoding='passthrough')
        filtered_depth_msg.header = msg.header
        filtered_depth_msg.header.frame_id = msg.header.frame_id
        filtered_depth_msg.header.stamp = self.get_clock().now().to_msg()
        self.filtered_depth_pub.publish(filtered_depth_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DepthFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
