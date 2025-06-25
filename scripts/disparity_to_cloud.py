import os
import numpy as np
import cv2
import yaml
import open3d as o3d
from tqdm import tqdm
from disp_refine.utils import get_cloud_from_depth

seq_path = '../data/ROUGH/helhest_2025_06_13-15_01_10'
image_files = sorted(os.listdir(os.path.join(seq_path, 'images', 'left')))
disp_files = sorted([f for f in os.listdir(os.path.join(seq_path, 'defom-stereo', 'disparity')) if f.endswith('.npy')])
points_files = sorted(os.listdir(os.path.join(seq_path, 'luxonis', 'clouds')))
calibration_path = os.path.join(seq_path, 'calibration')


def get_pcd_from_disparity(ind=None, show_disp=False, show_cloud=False, colorize_cloud=False):
    if ind is None:
        ind = np.random.randint(0, len(image_files))

    disp_path = os.path.join(seq_path, 'defom-stereo', 'disparity', disp_files[ind])
    disp = np.load(disp_path)
    disp[:7, :] = 0  # set invalid pixels to 0
    disp[:, :7] = 0  # set invalid pixels to 0

    if show_disp:
        # apply colormap to disparity
        disp_vis = cv2.convertScaleAbs(disp, alpha=255/disp.max())
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        cv2.imshow("Disparity", disp_vis)
        cv2.waitKey(0)
        cv2.destroyWindow("Disparity")

    # read camera intrinsics
    calib_intr = yaml.safe_load(open(os.path.join(calibration_path, 'cameras', 'camera_left.yaml')))
    K = np.array(calib_intr['camera_matrix']['data']).reshape(3, 3)
    focal_length = K[0, 0]  # assuming fx is the first element in the camera matrix

    calib_extr = yaml.safe_load(open(os.path.join(calibration_path, 'transformations.yaml')))
    Tr_camera_left__robot = np.array(calib_extr['Tr_camera_left__robot']['data'], dtype=float).reshape(4, 4)
    Tr_camera_right__robot = np.array(calib_extr['Tr_camera_right__robot']['data'], dtype=float).reshape(4, 4)
    cams_baseline = np.linalg.norm(Tr_camera_left__robot[:3, 3] - Tr_camera_right__robot[:3, 3])  # [m]

    # depth = (cams_baseline * focal_length) / (disp + 1e-6)  # [m] * [pixels] / [pixels] = [m]
    valid_mask = disp > 2.0
    depth = np.zeros_like(disp)
    depth[valid_mask] = (cams_baseline * focal_length) / disp[valid_mask]

    if colorize_cloud:
        img_path = os.path.join(seq_path, 'images', 'left', image_files[ind])
        image = np.asarray(cv2.imread(img_path))
    else:
        image = None

    pcd_disp = get_cloud_from_depth(depth_mm=np.asarray(depth) * 1000., K=K, rgb=image)
    pcd_disp, _ = pcd_disp.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    if show_cloud:
        # points_path = os.path.join(seq_path, 'luxonis', 'clouds', points_files[ind])
        # points = np.load(points_path)['points']
        # points = points[~np.isnan(points).any(axis=1)]
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # o3d.visualization.draw_geometries([pcd_disp, pcd])
        o3d.visualization.draw_geometries([pcd_disp])

    return pcd_disp


def main():
    os.makedirs(os.path.join(seq_path, 'defom-stereo', 'clouds'), exist_ok=True)
    for i in tqdm(range(len(disp_files))):
        pcd = get_pcd_from_disparity(i)
        # save point cloud
        pcd_path = os.path.join(seq_path, 'defom-stereo', 'clouds', f'{os.path.basename(disp_files[i])[:-4]}.npz')
        points = np.asarray(pcd.points)
        np.savez(pcd_path, points=points)


if __name__ == '__main__':
    main()