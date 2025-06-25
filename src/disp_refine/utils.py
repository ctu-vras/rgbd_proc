import numpy as np
import open3d as o3d
import torch


def get_disp_l2r_from_depth_right(depth_right, T_left_from_right, K):
    """
    Compute left-to-right disparity from a depth map referenced to the right camera frame.

    Args:
        depth_right (torch.Tensor): (H, W) depth map in meters, dtype=torch.float32
        T_left_from_right (torch.Tensor): (4, 4) transform from right to left camera
        K (torch.Tensor): (3, 3) camera intrinsics

    Returns:
        disp_l2r (torch.Tensor): (H, W) disparity map, dtype=torch.float32
    """
    device = depth_right.device
    H, W = depth_right.shape

    # Create meshgrid of pixel coordinates (u, v)
    u = torch.arange(W, dtype=depth_right.dtype, device=device)
    v = torch.arange(H, dtype=depth_right.dtype, device=device)
    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')
    ones = torch.ones_like(u_grid)

    # Flatten to pixel coordinate list (3, H*W)
    pixels_hom = torch.stack([u_grid, v_grid, ones], dim=0).reshape(3, -1)

    # Back-project to 3D in right camera frame
    depth_flat = depth_right.flatten()  # (H*W,)
    K_inv = torch.inverse(K)
    points_right = (K_inv @ pixels_hom) * depth_flat  # (3, N)

    # Convert to homogeneous coordinates (4, N)
    points_right_hom = torch.cat([points_right, torch.ones((1, points_right.shape[1]), device=device)], dim=0)

    # Transform to left camera frame
    points_left_hom = T_left_from_right @ points_right_hom
    points_left = points_left_hom[:3, :]  # (3, N)

    # Project to left image
    points_left_proj = K @ points_left
    z = points_left_proj[2, :] + 1e-6  # avoid divide-by-zero
    u_left = points_left_proj[0, :] / z
    v_left = points_left_proj[1, :] / z

    # Round to nearest integer pixel locations
    u_left_round = u_left.round().long()
    v_left_round = v_left.round().long()

    # Create disparity map
    disp_l2r = torch.zeros((H, W), dtype=torch.float32, device=device)

    # Compute baseline and focal length
    f = K[0, 0]
    B = torch.norm(T_left_from_right[:3, 3])

    # Valid projection mask
    valid = (
            (u_left_round >= 0) & (u_left_round < W) &
            (v_left_round >= 0) & (v_left_round < H) &
            (z > 0)
    )

    # Compute disparity = (B * f) / z
    disparity = (B * f) / z[valid]

    # Assign disparity to output image
    disp_l2r[v_left_round[valid], u_left_round[valid]] = disparity

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
