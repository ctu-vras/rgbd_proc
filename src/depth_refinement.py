#!/usr/bin/env python3

import cv2
import numpy as np
import torch
from time import time
from disp_refine.linknet import DispRef
from disp_refine.utils import get_disp_from_depth, get_disp_l2r_from_depth_right
from disp_refine.vis import colorize_img

import rclpy
import rclpy.time
from rclpy.executors import ExternalShutdownException
from rclpy.impl.logging_severity import LoggingSeverity
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf2_ros


class DepthRefinementNode(Node):
    normal_mean_var = {'mean': 0.3611, 'std': 0.2979}

    def __init__(self):
        super().__init__('depth_refinement_node')
        self.declare_parameter('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.declare_parameter('max_msgs_delay', 0.1)
        self.declare_parameter('max_age', np.inf)
        self.declare_parameter('weights', '')
        self.declare_parameter('max_disp', 100)
        self.declare_parameter('vis', False)

        self.device = torch.device(self.get_parameter('device').value)
        self.max_disp = self.get_parameter('max_disp').get_parameter_value().integer_value
        self.vis = self.get_parameter('vis').get_parameter_value().bool_value

        self._logger.set_level(LoggingSeverity.DEBUG)

        self.model = self.load_model()

        self.img_topics = ['/camera_left/image_rect', '/camera_right/image_rect']
        self.left_camera_info_topic = '/camera_left/camera_info'
        self.depth_topic = '/depth_in'

        self.cv_bridge = CvBridge()
        self._tf_buffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self.cams_baseline = 0.15  # TODO: hardcoded for now as it is not correct in the tfs

        self.max_msgs_delay = self.get_parameter('max_msgs_delay').get_parameter_value().double_value
        self.max_age = self.get_parameter('max_age').get_parameter_value().double_value

        # depth publisher
        self.depth_pub = self.create_publisher(Image, f'/depth_refined', 10)

    def safe_lookup_transform(self, target_frame, source_frame, time):
        try:
            return self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                time
            )
        except tf2_ros.ExtrapolationException:
            return self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time()
            )

    def get_cams_baseline(self, imgL_msg, imgR_msg):
        transform = self.safe_lookup_transform(imgL_msg.header.frame_id, imgR_msg.header.frame_id,
                                               imgL_msg.header.stamp)
        baseline = abs(transform.transform.translation.x)  # in meters
        self._logger.info(f'Baseline: {baseline:.3f} m')
        return baseline

    def load_model(self):
        model = DispRef()
        weights = self.get_parameter('weights').get_parameter_value().string_value
        # load pretrained weights
        if weights is not None:
            self._logger.info(f'Loading weights from {weights}')
            state_dict = torch.load(weights, map_location=self.device)
            model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        # optimize the model for inference
        if self.device.type == 'cpu':
            dummpy_input = torch.zeros((1, 2, 480, 768), dtype=torch.float32).to(self.device)
            model = torch.jit.trace(model, dummpy_input)
            self._logger.info('Model traced for CPU inference')
        return model

    def spin(self):
        try:
            rclpy.spin(self)
        except (KeyboardInterrupt, ExternalShutdownException):
            self.get_logger().info('Keyboard interrupt, shutting down...')
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def start(self):
        # subscribe to topics with approximate time synchronization
        subs = []
        for topic in self.img_topics + [self.depth_topic]:
            self._logger.info('Subscribing to %s' % topic)
            subs.append(Subscriber(self, Image, topic))

        self._logger.info('Subscribing to %s' % self.left_camera_info_topic)
        subs.append(Subscriber(self, CameraInfo, self.left_camera_info_topic))

        sync = ApproximateTimeSynchronizer(subs, queue_size=10, slop=self.max_msgs_delay)
        sync.registerCallback(self.callback)

    def callback(self, *msgs):
        self._logger.debug('Received %d messages' % len(msgs))
        # if a message is stale, do not process it
        t_now = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec / 1e9
        t_msg = msgs[0].header.stamp.sec + msgs[0].header.stamp.nanosec / 1e9
        dt = abs(t_now - t_msg)
        if dt > self.max_age:
            self._logger.debug(f'Message is stale (time diff: {dt:.3f} > {self.max_age} s), skipping...')
        else:
            # process the messages
            self.proc(*msgs)

    @torch.inference_mode()
    def proc(self, *msgs):
        imgL_msg, imgR_msg, depth_msg, left_cam_info_msg = msgs
        self._logger.info('Processing images')

        if self.cams_baseline is None:
            self.cams_baseline = self.get_cams_baseline(imgL_msg, imgR_msg)
        self._logger.debug(f'Camera baseline is {self.cams_baseline:.3f} m')

        imgL = self.cv_bridge.imgmsg_to_cv2(imgL_msg, desired_encoding='passthrough')
        depth_in = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        mean, std = self.normal_mean_var["mean"], self.normal_mean_var["std"]
        img_in_norm = torch.from_numpy(imgL / 255.).to(self.device)
        img_in_norm = (img_in_norm - mean) / std
        self._logger.debug(f"Left image shape: {img_in_norm.shape}")

        T_left_from_right = torch.eye(4, dtype=torch.float32).to(self.device)
        T_left_from_right[0, 3] = self.cams_baseline
        K = torch.as_tensor(left_cam_info_msg.k, dtype=torch.float32).reshape(3, 3).to(self.device)
        depth_in = torch.as_tensor(depth_in, dtype=torch.float32).to(self.device) / 1000.  # meters
        # disp_in = get_disp_l2r_from_depth_right(depth_in, T_left_from_right, K)
        disp_in = get_disp_from_depth(depth_in, T_left_from_right, K)
        self._logger.debug(f"L2R disparity shape: {disp_in.shape}")

        disp_in_norm = disp_in / self.max_disp
        inputs = torch.cat([img_in_norm.unsqueeze(0).unsqueeze(0),
                            disp_in_norm.unsqueeze(0).unsqueeze(0)], dim=1).float()

        # forward pass
        t2 = time()
        disp_cor = self.model(inputs)
        disp = disp_in + disp_cor * self.max_disp
        self._logger.info(f'Inference time: {time() - t2:.3f} ms')
        self._logger.debug(f'Predicted disparity shape: {disp.shape}')
        disp_np = disp.squeeze().cpu().numpy()
        self._logger.info(f'Disp values: {np.min(disp_np):.3f}, .., {np.max(disp_np):.3f}')

        # convert disparity to depth
        focal_length = left_cam_info_msg.k[0]  # assuming fx is the first element in the camera matrix
        self._logger.debug(f'Focal length: {focal_length:.3f} pixels')
        depth = np.zeros_like(disp_np)
        valid = disp_np > 2.  # valid disparity values
        depth[valid] = (self.cams_baseline * focal_length) / disp_np[valid]  # [m] * [pixels] / [pixels] = [m]
        self._logger.info(f'Depth values: {np.min(depth):.3f} m, .., {np.max(depth):.3f} m')

        # publish depth message
        depth_msg = self.cv_bridge.cv2_to_imgmsg(depth, encoding='passthrough')
        depth_msg.header = imgL_msg.header
        self.depth_pub.publish(depth_msg)

        if self.vis:
            disp_in_vis = colorize_img(disp_in.cpu().numpy())
            cv2.imshow('Input disparity', disp_in_vis)
            disp_vis = colorize_img(disp_np)
            cv2.imshow('Refined disparity', disp_vis)
            depth_vis = colorize_img(depth)
            cv2.imshow('Refined depth', depth_vis)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = DepthRefinementNode()
    node.start()
    node.spin()


if __name__ == '__main__':
    main()
