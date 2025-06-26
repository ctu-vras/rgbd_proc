import copy
import os
from glob import glob
import yaml
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, random_split
from tqdm import tqdm
from torchvision import transforms

data_sequences = [
    'helhest_2025_06_13-15_01_10',
    'helhest_2025_06_13-16_00_06',
]
pkg_path = os.path.join(os.path.dirname(__file__), '../../')
data_sequences = [os.path.realpath(os.path.join(pkg_path, 'data/ROUGH/', seq)) for seq in data_sequences]


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


def load_img_stats():
    """
    Load mean and std of grayscale images from a file.
    :param path: Path to the file containing mean and std values.
    :return: Tuple (mean_gray, std_gray)
    """
    path = os.path.join(pkg_path, 'config', 'dataset.yaml')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image stats file not found: {path}")
    with open(path, 'r') as f:
        stats = yaml.safe_load(f)
    return stats['img_stats']


transforms = transforms.Compose([
    transforms.ToTensor(),
])

class Data(Dataset):
    """
    A dataset for disparity correction.
    """

    def __init__(self, path):
        super(Dataset, self).__init__()
        self.path = path
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
        img_stats = load_img_stats()
        self.mean_gray = img_stats['gray']['mean']
        self.std_gray = img_stats['gray']['std']
        self.max_disp = img_stats['max_disparity']
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

    def get_image(self, i, camera='left'):
        img_path = self.image_files[camera][i]
        img = Image.open(img_path)
        return img

    def get_disp(self, i, source='luxonis'):
        disp_path = self.disp_files[source][i]
        disp = np.load(disp_path)
        disp = Image.fromarray(disp)
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
        # apply transforms
        img = transforms(img)
        disp_input = transforms(disp_input)
        disp_label = transforms(disp_label)
        return img, disp_input, disp_label


def calculate_img_stats(img_paths):
    """
    Calculate mean and standard deviation of grayscale images in the dataset.
    """
    mean_gray = 0.0
    std_gray = 0.0
    n_pixels = 0
    for img_path in tqdm(img_paths, desc='Calculating image stats'):
        gray = Image.open(img_path)
        gray = np.array(gray)
        gray = gray / 255.0  # normalize to 0-1 if needed
        mean_gray += gray.sum().item()
        std_gray += (gray ** 2).sum().item()
        n_pixels += gray.size
    mean_gray /= n_pixels
    std_gray = (std_gray / n_pixels - mean_gray ** 2) ** 0.5
    print(f'Mean gray: {mean_gray:.4f}, Std gray: {std_gray:.4f}')
    return mean_gray, std_gray


def compile_data(data_paths):
    """
    Compile multiple datasets into one.
    :param data_paths: List of dataset paths
    :return: Concatenated dataset
    """
    datasets = [Data(path) for path in data_paths]
    compiled_dataset = ConcatDataset(datasets)
    # train / val split
    n_samples = len(compiled_dataset)
    n_train = int(0.8 * n_samples)
    n_val = n_samples - n_train
    train_dataset, val_dataset = random_split(compiled_dataset, [n_train, n_val])
    print(f'Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}')
    return train_dataset, val_dataset