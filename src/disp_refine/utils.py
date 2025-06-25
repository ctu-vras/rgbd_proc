import numpy as np
import open3d as o3d


def get_disp_l2r_from_depth_right(depth_right, T_left_from_right, K):
    H, W = depth_right.shape
    # Create pixel coordinate grid (x, y)
    u, v = np.meshgrid(np.arange(W), np.arange(H))  # shape: (H, W)

    # Flatten and form homogeneous image coordinates
    ones = np.ones_like(u)
    pixels_hom = np.stack([u, v, ones], axis=0).reshape(3, -1)  # (3, H*W)

    # Back-project to 3D in right camera frame
    depth_flat = depth_right.flatten()
    K_inv = np.linalg.inv(K)

    points_right = K_inv @ pixels_hom * depth_flat  # (3, N)

    # Add homogeneous coordinate for transformation
    points_right_hom = np.vstack([points_right, np.ones((1, points_right.shape[1]))])  # (4, N)

    # Transform to left camera frame
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
    f = K[0, 0]  # focal length in pixels
    B = np.linalg.norm(T_left_from_right[:3, 3])  # baseline
    disp_l2r[u, v] = (B * f) / vu_left_hom[2][valid_pixels]  # [m] * [pixels] / [pixels] = [m]

    return disp_l2r


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
