import os
import numpy as np
import cv2
import yaml
from tqdm import tqdm
from glob import glob
from depth_to_cloud import get_cloud_from_depth
import open3d as o3d

seq_path = '../data/ROUGH/helhest_2025_06_13-15_01_10'
image_files = {
    'left': sorted(glob(os.path.join(seq_path, 'images', '*.png'))),
    'right': sorted(glob(os.path.join(seq_path, 'images', '*.png')))
}
depth_files = sorted(glob(os.path.join(seq_path, 'luxonis', 'depth', '*.png')))
calibration_path = os.path.join(seq_path, 'calibration')

# read camera intrinsics
calib_intr = yaml.safe_load(open(os.path.join(calibration_path, 'cameras', 'camera_left.yaml')))
K = np.array(calib_intr['camera_matrix']['data']).reshape(3, 3)
f = K[0, 0]  # assuming fx is the first element in the camera matrix

calib_extr = yaml.safe_load(open(os.path.join(calibration_path, 'transformations.yaml')))
Tr_camera_left__robot = np.array(calib_extr['Tr_camera_left__robot']['data'], dtype=float).reshape(4, 4)
Tr_camera_right__robot = np.array(calib_extr['Tr_camera_right__robot']['data'], dtype=float).reshape(4, 4)
B = np.linalg.norm(Tr_camera_left__robot[:3, 3] - Tr_camera_right__robot[:3, 3])  # cameras baseline [m]


def get_disparity_from_depth(ind=None, show_disp=False, show_pcd=False):
    if ind is None:
        ind = np.random.randint(0, len(depth_files))

    depth = cv2.imread(depth_files[ind], cv2.IMREAD_UNCHANGED)
    non_zero = depth > 0
    disp = np.zeros_like(depth, dtype=np.float32)
    disp[non_zero] = (B * f) / (depth[non_zero])  # [m] * [pixels] / [pixels] = [m]

    if show_disp:
        # apply colormap to disparity
        disp_vis = cv2.convertScaleAbs(disp, alpha=255/disp.max())
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        cv2.imshow("Disparity", disp_vis)
        cv2.waitKey(0)
        cv2.destroyWindow("Disparity")

    if show_pcd:
        valid_mask = np.ones(depth.shape, dtype=bool)
        valid_mask[:7, :] = False
        valid_mask[:, :7] = False
        valid_mask = valid_mask & (depth < (B*f/2))  # filter out points that are too far
        depth = depth * valid_mask
        pcd = get_cloud_from_depth(depth, K)
        o3d.visualization.draw_geometries([pcd])

    return disp


def main():
    os.makedirs(os.path.join(seq_path, 'luxonis', 'disparity'), exist_ok=True)
    for i in tqdm(range(len(depth_files))):
        disp = get_disparity_from_depth(i)
        disp_file = depth_files[i].replace('depth', 'disparity')
        disp_file_np = disp_file.replace('.png', '.npy')
        disp_vis = cv2.applyColorMap(cv2.convertScaleAbs(disp, alpha=255/disp.max()), cv2.COLORMAP_JET)
        cv2.imwrite(disp_file, disp_vis)
        np.save(disp_file_np, disp)


if __name__ == '__main__':
    main()