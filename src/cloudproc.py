import torch
from .utils import position
import numpy as np

default_rng = np.random.default_rng(135)


__all__ = [
    'filter_range',
    'filter_grid',
    'filter_cylinder',
    'filter_box',
    'valid_point_mask',
    'estimate_heightmap',
    'hm_to_cloud',
]



def affine(tf, x):
    """Apply an affine transform to points."""
    tf = np.asarray(tf)
    x = np.asarray(x)
    assert tf.ndim == 2
    assert x.ndim == 2
    assert tf.shape[1] == x.shape[0] + 1
    y = np.matmul(tf[:-1, :-1], x) + tf[:-1, -1:]
    return y

def within_bounds(x, min=None, max=None, bounds=None, log_variable=None):
    """Mask of x being within bounds  min <= x <= max."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    assert isinstance(x, torch.Tensor)

    keep = torch.ones((x.numel(),), dtype=torch.bool, device=x.device)

    if bounds:
        assert min is None and max is None
        min, max = bounds

    if min is not None and min > -float('inf'):
        if not isinstance(min, torch.Tensor):
            min = torch.tensor(min)
        keep = keep & (x.flatten() >= min)
    if max is not None and max < float('inf'):
        if not isinstance(max, torch.Tensor):
            max = torch.tensor(max)
        keep = keep & (x.flatten() <= max)

    if log_variable is not None:
        print('%.3f = %i / %i points kept (%.3g <= %s <= %.3g).'
              % (keep.double().mean(), keep.sum(), keep.numel(),
                 min if min is not None else float('nan'),
                 log_variable,
                 max if max is not None else float('nan')))

    return keep


def filter_range(cloud, min, max, log=False, only_mask=False):
    """Keep points within range interval."""
    assert isinstance(cloud, np.ndarray), type(cloud)
    assert isinstance(min, (float, int)), min
    assert isinstance(max, (float, int)), max
    assert min <= max, (min, max)
    min = float(min)
    max = float(max)
    if min <= 0.0 and max == np.inf:
        return cloud
    if cloud.dtype.names:
        cloud = cloud.ravel()
    x = position(cloud)
    r = np.linalg.norm(x, axis=1)
    mask = (min <= r) & (r <= max)

    if log:
        print('%.3f = %i / %i points kept (range min %s, max %s).'
              % (mask.sum() / len(cloud), mask.sum(), len(cloud), min, max))

    if only_mask:
        return mask

    filtered = cloud[mask]
    return filtered


def filter_grid(cloud, grid_res, keep='first', log=False, rng=default_rng, only_mask=False):
    """Keep single point within each cell. Order is not preserved."""
    assert isinstance(cloud, np.ndarray), type(cloud)
    # assert cloud.dtype.names
    assert isinstance(grid_res, (float, int)) and grid_res > 0.0
    assert keep in ('first', 'random', 'last')

    if cloud.dtype.names:
        cloud = cloud.ravel()
    if keep == 'first':
        pass
    elif keep == 'random':
        rng.shuffle(cloud)
    elif keep == 'last':
        cloud = cloud[::-1]

    x = position(cloud)
    keys = np.floor(x / grid_res).astype(int)
    assert keys.size > 0
    _, ind = np.unique(keys, return_index=True, axis=0)

    if log:
        print('%.3f = %i / %i points kept (grid res. %.3f m).'
              % (len(ind) / len(keys), len(ind), len(keys), grid_res))

    if only_mask:
        return ind

    filtered = cloud[ind]
    return filtered


def filter_cylinder(cloud, radius, axis='z', log=False, only_mask=False):
    """Keep points within cylinder."""
    assert isinstance(cloud, np.ndarray), type(cloud)
    assert isinstance(radius, (float, int)) and radius > 0.0
    assert axis in ('x', 'y', 'z')

    if cloud.dtype.names:
        cloud = cloud.ravel()
    x = position(cloud)
    if axis == 'x':
        mask = np.abs(x[:, 0]) <= radius
    elif axis == 'y':
        mask = np.abs(x[:, 1]) <= radius
    elif axis == 'z':
        mask = np.abs(x[:, 2]) <= radius
    else:
        raise ValueError(axis)

    if log:
        print('%.3f = %i / %i points kept (radius %.3f m).'
              % (mask.sum() / len(cloud), mask.sum(), len(cloud), radius))

    if only_mask:
        return mask

    filtered = cloud[mask]
    return filtered


def filter_box(cloud, box_size, box_pose=None, only_mask=False):
    """Keep points with rectangular bounds."""
    assert isinstance(cloud, np.ndarray)
    assert isinstance(box_size, (tuple, list)) and len(box_size) == 3
    assert all(isinstance(s, (float, int)) and s > 0.0 for s in box_size)
    assert box_pose is None or isinstance(box_pose, np.ndarray)

    if cloud.dtype.names:
        pts = position(cloud)
    else:
        pts = cloud
    assert pts.ndim == 2, "Input points tensor dimensions is %i (only 2 is supported)" % pts.ndim
    pts = torch.from_numpy(pts)

    if box_pose is None:
        box_pose = np.eye(4)
    assert isinstance(box_pose, np.ndarray)
    assert box_pose.shape == (4, 4)
    box_center = box_pose[:3, 3]
    box_orient = box_pose[:3, :3]

    pts = (pts - box_center) @ box_orient

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    keep_x = within_bounds(x, min=-box_size[0] / 2, max=+box_size[0] / 2)
    keep_y = within_bounds(y, min=-box_size[1] / 2, max=+box_size[1] / 2)
    keep_z = within_bounds(z, min=-box_size[2] / 2, max=+box_size[2] / 2)

    keep = torch.logical_and(keep_x, keep_y)
    keep = torch.logical_and(keep, keep_z)

    if only_mask:
        return keep
    filtered = cloud[keep]
    return filtered


def valid_point_mask(arr, discard_tf=None, discard_model=None):
    assert isinstance(arr, np.ndarray)
    assert arr.dtype.names
    # Identify valid points, i.e., points with valid depth which are not part
    # of the robot (contained by the discard model).
    # x = position(arr)
    # x = x.reshape((-1, 3)).T
    x = position(arr.ravel()).T
    valid = np.isfinite(x).all(axis=0)
    valid = np.logical_and(valid, (x != 0.0).any(axis=0))
    if discard_tf is not None and discard_model is not None:
        y = affine(discard_tf, x)
        valid = np.logical_and(valid, ~discard_model.contains_point(y))
    return valid.reshape(arr.shape)

def estimate_heightmap(points, grid_res, d_max, h_max, r_min=None, h_min=None):
    # remove nans from the point cloud if any
    mask = ~torch.isnan(points).any(dim=1)
    points = points[mask]

    if r_min is not None:
        # remove points in a r_min radius
        distances = torch.norm(points[:, :2], dim=1)
        mask = distances > r_min
        points = points[mask]

    if h_min is None:
        h_min = -h_max

    mask = ((points[:, 0] > -d_max) & (points[:, 0] < d_max) &
            (points[:, 1] > -d_max) & (points[:, 1] < d_max) &
            (points[:, 2] > h_min) & (points[:, 2] < h_max))
    points = points[mask]

    # Extract X, Y, Z
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Compute grid dimensions
    x_bins = torch.arange(-d_max, d_max, grid_res)
    y_bins = torch.arange(-d_max, d_max, grid_res)

    # Digitize coordinates to find grid indices
    x_indices = torch.bucketize(x.contiguous(), x_bins) - 1
    y_indices = torch.bucketize(y.contiguous(), y_bins) - 1

    # Use scatter_reduce to populate the heightmap
    flat_indices = y_indices * len(x_bins) + x_indices  # Flattened indices
    flat_heightmap = torch.full((len(y_bins) * len(x_bins),), float('nan'))

    # Use scatter_reduce to take the maximum height per grid cell
    flat_heightmap = torch.scatter_reduce(
        flat_heightmap,
        dim=0,
        index=flat_indices,
        src=z,
        reduce="amax",
        include_self=False
    )

    # Reshape back to 2D
    heightmap = flat_heightmap.view(len(y_bins), len(x_bins))

    # Replace NaNs with a default value (e.g., 0.0)
    measurements_mask = ~torch.isnan(heightmap)
    heightmap = torch.nan_to_num(heightmap, nan=0.0)
    # heightmap = torch.nan_to_num(heightmap, nan=(h_max + h_min) / 2.)

    hm = torch.stack([heightmap, measurements_mask], dim=0)  # (2, H, W)

    return hm



def hm_to_cloud(height, d_max, mask=None):
    assert isinstance(height, np.ndarray) or isinstance(height, torch.Tensor)
    assert height.ndim == 2
    if mask is not None:
        assert isinstance(mask, (np.ndarray, torch.Tensor))
        assert mask.ndim == 2
        assert height.shape == mask.shape
        mask = mask.bool() if isinstance(mask, torch.Tensor) else mask.astype(bool)
    z_grid = height
    if isinstance(height, np.ndarray):
        x_grid = np.linspace(-d_max, d_max, z_grid.shape[0])
        y_grid = np.linspace(-d_max, d_max, z_grid.shape[1])
        x_grid, y_grid = np.meshgrid(x_grid, y_grid)
        hm_cloud = np.stack([x_grid, y_grid, z_grid], axis=2)
    else:
        x_grid = torch.linspace(-d_max, d_max, z_grid.shape[0]).to(z_grid.device)
        y_grid = torch.linspace(-d_max, d_max, z_grid.shape[1]).to(z_grid.device)
        x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
        hm_cloud = torch.stack([x_grid, y_grid, z_grid], dim=2)
    if mask is not None:
        hm_cloud = hm_cloud[mask]
    hm_cloud = hm_cloud.reshape([-1, 3])
    return hm_cloud
