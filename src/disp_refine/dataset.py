import copy
import os
from glob import glob
import yaml
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


def load_calib(calib_path):
    calib = {}
    # read camera calibration
    cams_path = os.path.join(calib_path, 'cameras')
    if not os.path.exists(cams_path):
        print('No cameras calibration found in path {}'.format(cams_path))
        return None

    for file in os.listdir(cams_path):
        if file.endswith('.yaml'):
            with open(os.path.join(cams_path, file), 'r') as f:
                cam_info = yaml.load(f, Loader=yaml.FullLoader)
                calib[file.replace('.yaml', '')] = cam_info
            f.close()
    # read cameras-lidar transformations
    trans_path = os.path.join(calib_path, 'transformations.yaml')
    with open(trans_path, 'r') as f:
        transforms = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    calib['transformations'] = transforms

    return calib


class Data(Dataset):
    """
    A dataset for disparity correction.
    """
    mean_gray = 0.3611
    std_gray = 0.2979

    def __init__(self, path, max_disp=100.0):
        super(Dataset, self).__init__()
        self.path = path
        self.max_disp = max_disp
        self.image_files = {
            'left': sorted(glob(os.path.join(path, 'images', 'left', '*.png'))),
            'right': sorted(glob(os.path.join(path, 'images', 'right', '*.png'))),
        }
        self.disp_files = {
            'luxonis': sorted(glob(os.path.join(path, 'luxonis', 'disparity', '*.npy'))),
            'defom-stereo': sorted(glob(os.path.join(path, 'defom-stereo', 'disparity', '*.npy'))),
        }
        self.calib_path = os.path.join(path, 'calibration')
        self.calib = load_calib(calib_path=self.calib_path)
        self.ids = self.get_ids()

    def __getitem__(self, i):
        if isinstance(i, (int, np.int64)):
            sample = self.get_sample(i)
            return sample
        ds = copy.deepcopy(self)
        if isinstance(i, (list, tuple, np.ndarray)):
            ds.ids = [self.ids[k] for k in i]
        else:
            assert isinstance(i, (slice, range))
            ds.ids = self.ids[i]
        return ds

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ids)

    def get_ids(self):
        ids = [f[:-4] for f in self.image_files['left']]
        ids = sorted(ids)
        return ids

    def ind_to_stamp(self, i):
        ind = self.ids[i]
        stamp = float(ind.replace('_', '.'))
        return stamp

    def calculate_img_stats(self):
        """
        Calculate mean and standard deviation of grayscale images in the dataset.
        """
        mean_gray = 0.0
        std_gray = 0.0
        n_pixels = 0
        for i in tqdm(range(len(self)), desc='Calculating image stats'):
            gray = self.get_image(i)
            gray = gray / 255.0  # normalize to 0-1 if needed
            mean_gray += gray.sum().item()
            std_gray += (gray ** 2).sum().item()
            n_pixels += gray.size
        mean_gray /= n_pixels
        std_gray = (std_gray / n_pixels - mean_gray ** 2) ** 0.5
        self.mean_gray = mean_gray
        self.std_gray = std_gray
        print(f'Mean gray: {self.mean_gray:.4f}, Std gray: {self.std_gray:.4f}')

    def get_image(self, i, camera='left'):
        img_path = self.image_files[camera][i]
        img = Image.open(img_path)
        img = np.array(img)
        return img

    def get_disp(self, i, source='luxonis'):
        disp_path = self.disp_files[source][i]
        disp = np.load(disp_path)
        return disp

    def disp_to_cloud(self, disp):
        """
        Convert disparity map to point cloud.
        :param disp: Disparity map (H, W)
        :return: Point cloud (N, 3)
        """
        K = np.array(self.calib['camera_left']['camera_matrix']['data']).reshape(3, 3)
        f = K[0, 0]  # focal length in pixels
        Tr_camera_left__robot = np.array(self.calib['transformations']['Tr_camera_left__robot']['data'], dtype=float).reshape(4, 4)
        Tr_camera_right__robot = np.array(self.calib['transformations']['Tr_camera_right__robot']['data'], dtype=float).reshape(4, 4)
        B = np.linalg.norm(Tr_camera_left__robot[:3, 3] - Tr_camera_right__robot[:3, 3])  # [m]
        valid_mask = disp > 0
        depth = np.zeros_like(disp, dtype=np.float32)
        depth[valid_mask] = (B * f) / disp[valid_mask]
        height, width = disp.shape
        vu = np.indices(disp.shape).reshape((2, -1))  # shape (2, H*W)
        uv = vu[::-1, :]  # shape (2, H*W)
        coords = np.vstack([
            uv[0, :],  # u (x-coordinate)
            uv[1, :],  # v (y-coordinate)
            np.ones(height * width)
        ])
        K_inv = np.linalg.inv(K)
        points = (K_inv @ coords) * depth.flatten()  # shape (3, H*W)
        return points.T

    def get_sample(self, i):
        img = self.get_image(i, camera='left')
        disp_input = self.get_disp(i, source='luxonis')  # l2r
        disp_label = self.get_disp(i, source='defom-stereo')  # l2r
        return img[np.newaxis], disp_input[np.newaxis], disp_label[np.newaxis]
