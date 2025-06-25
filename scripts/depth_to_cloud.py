import os
import numpy as np
from PIL import Image
import yaml
import open3d as o3d
from disp_refine.utils import get_cloud_from_depth


def main():
    seq_path = '../data/ROUGH/helhest_2025_06_13-15_01_10'
    depth_files = sorted(os.listdir(os.path.join(seq_path, 'luxonis', 'depth')))
    calibration_path = os.path.join(seq_path, 'calibration')

    # read camera intrinsics
    calib = yaml.safe_load(open(os.path.join(calibration_path, 'cameras', 'camera_right.yaml')))
    K = np.array(calib['camera_matrix']['data']).reshape(3, 3)
    f = K[0, 0]  # assuming fx is the first element in the camera matrix

    calib_extr = yaml.safe_load(open(os.path.join(calibration_path, 'transformations.yaml')))
    Tr_camera_left__robot = np.array(calib_extr['Tr_camera_left__robot']['data'], dtype=float).reshape(4, 4)
    Tr_camera_right__robot = np.array(calib_extr['Tr_camera_right__robot']['data'], dtype=float).reshape(4, 4)
    T_left_from_right = np.linalg.inv(Tr_camera_left__robot) @ Tr_camera_right__robot
    B = np.linalg.norm(Tr_camera_left__robot[:3, 3] - Tr_camera_right__robot[:3, 3])  # cameras baseline [m]

    ind = 150
    depth_path = os.path.join(seq_path, 'luxonis', 'depth', depth_files[ind])
    depth = Image.open(depth_path)
    # depth.show()

    depth_gt_path = os.path.join(seq_path, 'defom-stereo', 'depth', depth_files[ind].replace('.png', '.npy'))
    depth_gt = np.load(depth_gt_path)

    valid_mask = np.ones(depth_gt.shape, dtype=bool)
    valid_mask[:7, :] = False
    valid_mask[:, :7] = False
    valid_mask = valid_mask & (depth_gt < (B * f * 1000 / 2))  # filter out points that are too far
    depth_gt = depth_gt * valid_mask

    pcd_luxonis = get_cloud_from_depth(np.asarray(depth), K, rgb=None)
    pcd_luxonis.transform(T_left_from_right)
    pcd_luxonis.paint_uniform_color([0, 0, 1])  # green for Luxonis depth cloud

    pcd_defom = get_cloud_from_depth(np.asarray(depth_gt), K, rgb=None)
    pcd_defom.paint_uniform_color([1, 0, 0])  # red for DEFOM-Stereo depth cloud
    pcd_defom, _ = pcd_defom.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    o3d.visualization.draw_geometries([pcd_luxonis, pcd_defom])


if __name__ == '__main__':
    main()