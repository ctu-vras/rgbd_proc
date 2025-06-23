import os
import numpy as np
from PIL import Image
import yaml
import open3d as o3d


seq_path = '../data/ROUGH/helhest_2025_06_13-15_01_10'
image_files = sorted(os.listdir(os.path.join(seq_path, 'images', 'left')))
depth_files = sorted(os.listdir(os.path.join(seq_path, 'luxonis', 'depth')))
clouds_files = sorted(os.listdir(os.path.join(seq_path, 'luxonis', 'clouds')))
calibration_path = os.path.join(seq_path, 'calibration')

# read camera intrinsics
calib = yaml.safe_load(open(os.path.join(calibration_path, 'cameras', 'camera_right.yaml')))
K = np.array(calib['camera_matrix']['data']).reshape(3, 3)

f = K[0, 0]  # assuming fx is the first element in the camera matrix

calib_extr = yaml.safe_load(open(os.path.join(calibration_path, 'transformations.yaml')))
Tr_camera_left__robot = np.array(calib_extr['Tr_camera_left__robot']['data'], dtype=float).reshape(4, 4)
Tr_camera_right__robot = np.array(calib_extr['Tr_camera_right__robot']['data'], dtype=float).reshape(4, 4)
B = np.linalg.norm(Tr_camera_left__robot[:3, 3] - Tr_camera_right__robot[:3, 3])  # cameras baseline [m]


def get_cloud_from_depth(depth_mm: np.ndarray, K: np.ndarray, rgb=None) -> o3d.geometry.PointCloud:
    height, width = depth_mm.shape

    # Generate pixel coordinates
    vu = np.indices(depth_mm.shape).reshape((2, -1))  # shape (2, H*W)
    uv = vu[::-1, :]  # shape (2, H*W)
    coords = np.vstack([
        uv[0, :],  # u (x-coordinate)
        uv[1, :],  # v (y-coordinate)
        np.ones(height * width)
    ])  # shape (3, H*W)

    K_inv = np.linalg.inv(K)
    depth_flat = depth_mm.flatten()
    points = (K_inv @ coords) * depth_flat  # shape (3, H*W)
    point3d_coords = points.T / 1000.0  # convert to meters, shape (H*W, 3)

    # Filter out invalid points (depth == 0)
    valid = depth_flat > 0
    point3d_coords = point3d_coords[valid]

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point3d_coords)

    if rgb is not None:
        point3d_colors = rgb.reshape(height * width, 3)[valid] / 255.0
        pcd.colors = o3d.utility.Vector3dVector(point3d_colors)

    return pcd


def compare_depths():
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
    pcd_luxonis.paint_uniform_color([0, 1, 0])  # green for Luxonis depth cloud
    pcd_defom = get_cloud_from_depth(np.asarray(depth_gt), K, rgb=None)
    pcd_defom.paint_uniform_color([1, 0, 0])  # red for DEFOM-Stereo depth cloud

    o3d.visualization.draw_geometries([pcd_luxonis, pcd_defom])


def main():
    # ind = np.random.randint(0, len(image_files))
    ind = 150

    img_path = os.path.join(seq_path, 'images', 'left', image_files[ind])
    image = Image.open(img_path).convert('RGB')
    # image.show()

    depth_path = os.path.join(seq_path, 'luxonis', 'depth', depth_files[ind])
    depth = Image.open(depth_path)
    # depth.show()

    points_path = os.path.join(seq_path, 'luxonis', 'clouds', clouds_files[ind])
    points = np.load(points_path)['points']
    points = points[~np.isnan(points).any(axis=1)]
    print(points.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    image = np.asarray(image)
    depth = np.asarray(depth)
    pcd_depth = get_cloud_from_depth(depth, K, rgb=image)
    o3d.visualization.draw_geometries([pcd, pcd_depth])
    # o3d.visualization.draw_geometries([pcd_depth])


if __name__ == '__main__':
    # main()
    compare_depths()