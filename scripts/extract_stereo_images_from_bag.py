import os
import cv2
import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage


def extract_stereo_images(bag_path, left_topic, right_topic, output_dir, interval_sec=1.0, time_tolerance=0.05):
    os.makedirs(output_dir, exist_ok=True)

    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)

    bridge = CvBridge()
    left_buffer = []
    right_buffer = []
    last_saved_time = -float('inf')
    pair_count = 0

    def to_stamp(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    while reader.has_next():
        topic, data, t = reader.read_next()

        if topic == left_topic:
            # left_msg = deserialize_message(data, Image)
            left_msg = deserialize_message(data, CompressedImage)
            left_stamp = to_stamp(left_msg)
            left_buffer.append((left_stamp, left_msg))

        elif topic == right_topic:
            # right_msg = deserialize_message(data, Image)
            right_msg = deserialize_message(data, CompressedImage)
            right_stamp = to_stamp(right_msg)
            right_buffer.append((right_stamp, right_msg))

        # Try to match left and right images
        while left_buffer and right_buffer:
            left_time, left_msg = left_buffer[0]
            best_match = None
            min_diff = float('inf')

            for right_time, right_msg in right_buffer:
                diff = abs(right_time - left_time)
                if diff < min_diff and diff <= time_tolerance:
                    min_diff = diff
                    best_match = (right_time, right_msg)

            if best_match:
                if left_time - last_saved_time >= interval_sec:
                    # Save pair
                    # left_cv = bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
                    # right_cv = bridge.imgmsg_to_cv2(best_match[1], desired_encoding='bgr8')
                    left_cv = bridge.compressed_imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
                    right_cv = bridge.compressed_imgmsg_to_cv2(best_match[1], desired_encoding='bgr8')

                    ts_str = f"{left_time:.6f}"
                    cv2.imwrite(os.path.join(output_dir, f"left_{ts_str}.png"), left_cv)
                    cv2.imwrite(os.path.join(output_dir, f"right_{ts_str}.png"), right_cv)

                    print(f"Saved stereo pair at {ts_str}")
                    last_saved_time = left_time
                    pair_count += 1

                # Remove used messages
                left_buffer.pop(0)
                right_buffer = [pair for pair in right_buffer if pair[0] != best_match[0]]
            else:
                # No match found, remove oldest one to avoid memory leak
                if left_time < right_buffer[0][0]:
                    left_buffer.pop(0)
                else:
                    right_buffer.pop(0)

    print(f"Finished saving {pair_count} stereo pairs to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract stereo image pairs every 1 sec from a ROS 2 bag (MCAP format).")
    parser.add_argument("bag_path", help="Path to the ROS 2 bag directory")
    parser.add_argument("left_topic", help="Left camera image topic (e.g., /stereo/left/image_raw)")
    parser.add_argument("right_topic", help="Right camera image topic (e.g., /stereo/right/image_raw)")
    parser.add_argument("output_dir", help="Directory to save image pairs")
    args = parser.parse_args()

    rclpy.init()
    extract_stereo_images(args.bag_path, args.left_topic, args.right_topic, args.output_dir)
    rclpy.shutdown()
