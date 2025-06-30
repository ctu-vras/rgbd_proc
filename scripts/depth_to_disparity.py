import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))
import numpy as np
import cv2
import yaml
from tqdm import tqdm
from glob import glob
import open3d as o3d
import torch
import argparse
from disp_refine.utils import get_disp_l2r_from_depth_right, get_cloud_from_depth, get_disp_from_depth
from disp_refine.vis import colorize_img


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Depth to Disparity')
    parser.add_argument('--seq_path', type=str, default='../data/Helhest/helhest_2025_06_13-16_00_06',
                        help='Path to the sequence directory')
    parser.add_argument('--vis', type=str2bool, default='False',
                        help='Whether to visualize the depth map')
    return parser.parse_args()


def main():
    args = parse_args()
    seq_path = args.seq_path
    depth_files = sorted(glob(os.path.join(seq_path, 'luxonis', 'depth', '*.png')))
    calibration_path = os.path.join(seq_path, 'calibration')

    # read camera intrinsics
    calib_intr = yaml.safe_load(open(os.path.join(calibration_path, 'cameras', 'camera_left.yaml')))
    K = np.array(calib_intr['camera_matrix']['data']).reshape(3, 3)

    calib_extr = yaml.safe_load(open(os.path.join(calibration_path, 'transformations.yaml')))
    Tr_camera_left__robot = np.array(calib_extr['Tr_camera_left__robot']['data'], dtype=float).reshape(4, 4)
    Tr_camera_right__robot = np.array(calib_extr['Tr_camera_right__robot']['data'], dtype=float).reshape(4, 4)
    T_left_from_right = np.linalg.inv(Tr_camera_left__robot) @ Tr_camera_right__robot  # (4, 4)

    os.makedirs(os.path.join(seq_path, 'luxonis', 'disparity'), exist_ok=True)
    for i in tqdm(range(len(depth_files))):
        depth = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED) / 1000.0  # convert mm to m
        depth = torch.as_tensor(depth, dtype=torch.float32)
        T_left_from_right = torch.as_tensor(T_left_from_right, dtype=torch.float32)
        K = torch.as_tensor(K, dtype=torch.float32)
        # disp = get_disp_l2r_from_depth_right(depth, T_left_from_right, K)
        disp = get_disp_from_depth(depth, T_left_from_right, K)

        # save disparity map
        disp_file = depth_files[i].replace('depth', 'disparity')
        disp_file_np = disp_file.replace('.png', '.npy')
        disp_vis = colorize_img(disp.numpy())
        cv2.imwrite(disp_file, disp_vis)
        np.save(disp_file_np, disp)

        if args.vis:
            # visualize disparity and point cloud
            disp_vis = colorize_img(disp.numpy())
            cv2.imshow("Disparity", disp_vis)
            cv2.waitKey(0)
            cv2.destroyWindow("Disparity")

            depth = depth * (depth > 0)
            pcd = get_cloud_from_depth(depth.numpy(), K.numpy())
            o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()