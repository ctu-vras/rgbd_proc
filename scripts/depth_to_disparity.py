import os
import numpy as np
import cv2
import yaml
from tqdm import tqdm
from glob import glob
import open3d as o3d

from disp_refine.utils import get_disp_l2r_from_depth_right, get_cloud_from_depth
from disp_refine.vis import colorize_disp


def main():
    seq_path = '../data/ROUGH/helhest_2025_06_13-15_01_10'
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
        depth_right = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED) / 1000.0  # convert mm to m
        disp = get_disp_l2r_from_depth_right(depth_right, T_left_from_right, K)

        # save disparity map
        disp_file = depth_files[i].replace('depth', 'disparity')
        disp_file_np = disp_file.replace('.png', '.npy')
        disp_vis = cv2.applyColorMap(cv2.convertScaleAbs(disp, alpha=255/disp.max()), cv2.COLORMAP_JET)
        cv2.imwrite(disp_file, disp_vis)
        np.save(disp_file_np, disp)

        # # visualize disparity and point cloud
        # disp_vis = colorize_disp(disp)
        # cv2.imshow("Disparity", disp_vis)
        # cv2.waitKey(0)
        # cv2.destroyWindow("Disparity")
        #
        # depth_right = depth_right * (depth_right > 0)
        # pcd = get_cloud_from_depth(depth_right, K)
        # o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()