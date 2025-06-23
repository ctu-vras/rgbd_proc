import os
import cv2
import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage


def extract_images(bag_path, image_topic, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

    reader.open(storage_options, converter_options)

    bridge = CvBridge()
    image_count = 0

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == image_topic:
            msg = deserialize_message(data, CompressedImage)
            cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            filename = os.path.join(output_dir, f"{timestamp:.6f}.png")
            cv2.imwrite(filename, cv_image)
            image_count += 1
            print(f"Saved image: {filename}")

    print(f"Finished extracting {image_count} images to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract images from a ROS 2 bag (MCAP format).")
    parser.add_argument("bag_path", help="Path to the ROS 2 bag directory")
    parser.add_argument("image_topic", help="Image topic to extract, e.g., /camera/image_raw")
    parser.add_argument("output_dir", help="Directory to save extracted images")

    args = parser.parse_args()
    rclpy.init()
    extract_images(args.bag_path, args.image_topic, args.output_dir)
    rclpy.shutdown()
