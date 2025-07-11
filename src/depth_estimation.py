#!/usr/bin/env python3

import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image as pillow
from time import time
from psmnet import basic, stackhourglass

import rclpy
import rclpy.time
from rclpy.executors import ExternalShutdownException
from rclpy.impl.logging_severity import LoggingSeverity
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
# from stereo_msgs.msg import DisparityImage
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf2_ros


class DepthEstimationNode(Node):
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(**normal_mean_var)])

    def __init__(self):
        super().__init__('depth_estimation_node')
        self.declare_parameter('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.declare_parameter('img_topics',
                               ['/camera_left/image_rect', '/camera_right/image_rect'])
        self.declare_parameter('left_camera_info_topic', '/camera_left/camera_info')
        self.declare_parameter('max_msgs_delay', 0.1)
        self.declare_parameter('max_age', np.inf)
        self.declare_parameter('model_name', 'basic')
        self.declare_parameter('weights', '')
        self.declare_parameter('max_disp', 192)
        self.declare_parameter('vis', False)

        self.device = torch.device(self.get_parameter('device').value)
        self.max_disp = self.get_parameter('max_disp').get_parameter_value().integer_value
        self.vis = self.get_parameter('vis').get_parameter_value().bool_value

        self._logger.set_level(LoggingSeverity.DEBUG)

        self.model = self.load_model()

        self.img_topics = self.get_parameter('img_topics').get_parameter_value().string_array_value
        assert len(self.img_topics) == 2, "Two image topics must be provided for stereo depth estimation."
        self.left_camera_info_topic = self.get_parameter('left_camera_info_topic').get_parameter_value().string_value

        self.cv_bridge = CvBridge()
        self._tf_buffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self.cams_baseline = None

        self.max_msgs_delay = self.get_parameter('max_msgs_delay').get_parameter_value().double_value
        self.max_age = self.get_parameter('max_age').get_parameter_value().double_value

        # depth publisher
        self.depth_pub = self.create_publisher(Image, '/depth', 10)

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
        model_name = self.get_parameter('model_name').get_parameter_value().string_value
        weights = self.get_parameter('weights').get_parameter_value().string_value
        if model_name == 'basic':
            model = basic(maxdisp=self.max_disp)
        elif model_name == 'stackhourglass':
            model = stackhourglass(maxdisp=self.max_disp)
        else:
            raise ValueError(f'Unknown model type: {model_name}')
        self._logger.info(f'Loading depth estimator model: {model_name}')
        model = torch.nn.DataParallel(model)

        # load pretrained weights
        if weights is not None:
            self._logger.info(f'Loading weights from {weights}')
            state_dict = torch.load(weights, map_location=self.device)
            model.load_state_dict(state_dict['state_dict'])

        model.to(self.device)
        model.eval()
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
        for topic in self.img_topics:
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
        t0 = time()
        imgL_msg, imgR_msg, left_cam_info_msg = msgs
        self._logger.info('Processing images')

        if self.cams_baseline is None:
            self.cams_baseline = self.get_cams_baseline(imgL_msg, imgR_msg)
        self._logger.debug(f'Camera baseline is {self.cams_baseline:.3f} m')

        imgL = self.cv_bridge.imgmsg_to_cv2(imgL_msg, desired_encoding='passthrough')
        imgL = pillow.fromarray(imgL).convert('RGB')
        imgL = self.infer_transform(imgL)

        imgR = self.cv_bridge.imgmsg_to_cv2(imgR_msg, desired_encoding='passthrough')
        imgR = pillow.fromarray(imgR).convert('RGB')
        imgR = self.infer_transform(imgR)

        # pad to width and height to 16 times
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1] // 16
            top_pad = (times + 1) * 16 - imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2] // 16
            right_pad = (times + 1) * 16 - imgL.shape[2]
        else:
            right_pad = 0

        imgL = F.pad(imgL, (0, right_pad, top_pad, 0)).unsqueeze(0)
        imgR = F.pad(imgR, (0, right_pad, top_pad, 0)).unsqueeze(0)
        self._logger.debug(f'imgL shape: {imgL.shape}')
        self._logger.debug(f'imgR shape: {imgR.shape}')

        imgL = imgL.to(self.device)
        imgR = imgR.to(self.device)
        t1 = time()
        self._logger.info(f'Preprocessing time: {t1 - t0:.3f} ms')

        # forward pass
        t2 = time()
        disp = self.model(imgL, imgR)
        self._logger.info(f'Inference time: {time() - t2:.3f} ms')
        self._logger.debug(f'Predicted disparity shape: {disp.shape}')
        disp_np = disp.squeeze().cpu().numpy()

        # remove padding
        if top_pad != 0 and right_pad != 0:
            disp_np = disp_np[top_pad:, :-right_pad]
        elif top_pad == 0 and right_pad != 0:
            disp_np = disp_np[:, :-right_pad]
        elif top_pad != 0 and right_pad == 0:
            disp_np = disp_np[top_pad:, :]

        # convert disparity to depth
        focal_length = left_cam_info_msg.k[0]  # assuming fx is the first element in the camera matrix
        self._logger.debug(f'Focal length: {focal_length:.3f} pixels')
        depth = (self.cams_baseline * focal_length) / (disp_np + 1e-6)  # [m] * [pixels] / [pixels] = [m]
        self._logger.info(f'Depth values: {np.min(depth):.3f} m - {np.max(depth):.3f} m')

        # publish depth message
        depth_msg = self.cv_bridge.cv2_to_imgmsg(depth, encoding='passthrough')
        depth_msg.header = imgL_msg.header
        self.depth_pub.publish(depth_msg)

        if self.vis:
            # visualize the depth map with a colormap
            depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255.0 / np.max(depth)), cv2.COLORMAP_JET)
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)
            cv2.imshow('PSMNet Depth', depth_vis)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = DepthEstimationNode()
    node.start()
    node.spin()


if __name__ == '__main__':
    main()
