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


def get_disp_l2r_from_depth_right(i=None, show_disp=False, show_pcd=False):
    if i is None:
        i = np.random.randint(0, len(depth_files))

    depth_right = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED) / 1000.0  # convert mm to m
    # valid = depth_right > 0
    # disp_r2l = np.zeros_like(depth_right, dtype=np.float32)
    # disp_r2l[valid] = (B * f) / (depth_right[valid])  # [m] * [pixels] / [pixels] = [m]

    H, W = depth_right.shape
    # Create pixel coordinate grid (x, y)
    u, v = np.meshgrid(np.arange(W), np.arange(H))  # shape: (H, W)

    # Flatten and form homogeneous image coordinates
    ones = np.ones_like(u)
    pixels_hom = np.stack([u, v, ones], axis=0).reshape(3, -1)  # (3, H*W)

    # Back-project to 3D in right camera frame
    depth_flat = depth_right.flatten()
    valid = depth_flat > 0
    K_inv = np.linalg.inv(K)

    points_right = K_inv @ pixels_hom * depth_flat  # (3, N)

    # Add homogeneous coordinate for transformation
    points_right_hom = np.vstack([points_right, np.ones((1, points_right.shape[1]))])  # (4, N)

    # Transform to left camera frame
    # T_left_from_right = Tr_camera_left__robot @ np.linalg.inv(Tr_camera_right__robot)  # (4, 4)
    T_left_from_right = np.linalg.inv(Tr_camera_left__robot) @ Tr_camera_right__robot  # (4, 4)
    points_left_hom = T_left_from_right @ points_right_hom
    points_left = points_left_hom[:3, :]  # (3, N)

    # Project back to pixel coordinates in left camera frame
    vu_left_hom = K @ points_left  # (3, N)
    vu_left = vu_left_hom[:2, :] / vu_left_hom[2, :]  # (2, N)
    vu_left = np.round(vu_left).astype(int)  # round to nearest pixel

    # Create disparity map
    disp_l2r = np.zeros_like(depth_right, dtype=np.float32)
    valid_pixels = (vu_left[0, :] >= 0) & (vu_left[0, :] < W) & (vu_left[1, :] >= 0) & (vu_left[1, :] < H)
    u, v = vu_left[1, valid_pixels], vu_left[0, valid_pixels]
    disp_l2r[u, v] = (B * f) / vu_left_hom[2][valid_pixels]  # [m] * [pixels] / [pixels] = [m]

    if show_disp:
        # apply colormap to disparity
        disp_vis = cv2.convertScaleAbs(disp_l2r, alpha=255/disp_l2r.max())
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        cv2.imshow("Disparity", disp_vis)
        cv2.waitKey(0)
        cv2.destroyWindow("Disparity")

    if show_pcd:
        depth_right = depth_right * valid.reshape(H, W)
        pcd = get_cloud_from_depth(depth_right, K)
        o3d.visualization.draw_geometries([pcd])

    return disp_l2r


def main():
    os.makedirs(os.path.join(seq_path, 'luxonis', 'disparity'), exist_ok=True)
    for i in tqdm(range(len(depth_files))):
        disp = get_disp_l2r_from_depth_right(i=i)
        disp_file = depth_files[i].replace('depth', 'disparity')
        disp_file_np = disp_file.replace('.png', '.npy')
        disp_vis = cv2.applyColorMap(cv2.convertScaleAbs(disp, alpha=255/disp.max()), cv2.COLORMAP_JET)
        cv2.imwrite(disp_file, disp_vis)
        np.save(disp_file_np, disp)


if __name__ == '__main__':
    main()