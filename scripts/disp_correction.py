import copy
import os
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import cv2
from PIL import Image
import yaml
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Disparity Correction Training')
    parser.add_argument('--dataset_path', type=str,
                        default='../data/ROUGH/helhest_2025_06_13-15_01_10',
                        help='Path to the dataset')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')
    parser.add_argument('--bs', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--nepochs', type=int, default=100, help='Number of epochs for training')
    return parser.parse_args()


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
        img = np.asarray(img)
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
        img = self.get_image(i, camera='right')
        disp_input = self.get_disp(i, source='luxonis')
        disp_label = self.get_disp(i, source='defom-stereo')
        return img[np.newaxis], disp_input[np.newaxis], disp_label[np.newaxis]


class DCModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Linknet(
            encoder_name='mobilenet_v2',
            # encoder_weights='imagenet',
            encoder_weights=None,  # No pre-trained weights, as we use grayscale images as input
            in_channels=2,
            classes=1,
            activation=None,
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.model(x)
        x = self.relu(x)
        return x


def data_test(args):
    dataset_path = args.dataset_path

    ds = Data(dataset_path)

    i = 150
    img, disp_in, disp_gt = ds[i]
    max_disp = min(disp_in.max(), disp_gt.max())

    cv2.imshow('img', img.squeeze())

    disp_scaled = cv2.convertScaleAbs(disp_in.squeeze(), alpha=255.0 / max_disp)
    disp_colored = cv2.applyColorMap(disp_scaled, cv2.COLORMAP_JET)
    cv2.imshow("Disp Input", disp_colored)

    disp_scaled_label = cv2.convertScaleAbs(disp_gt.squeeze(), alpha=255.0 / max_disp)
    disp_colored_label = cv2.applyColorMap(disp_scaled_label, cv2.COLORMAP_JET)
    cv2.imshow("Disp Label", disp_colored_label)

    mask_dist = (disp_in > 0) & (disp_gt < max_disp)
    mask_nan = np.isnan(disp_gt) | np.isnan(disp_gt)
    mask_valid = np.ones(disp_gt.shape, dtype=bool)
    mask_valid[:, :7, :] = False
    mask_valid[:, :, :7] = False
    mask = mask_dist & mask_valid & (~mask_nan)
    cv2.imshow("Mask", mask.squeeze().astype(np.uint8) * 255)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def train(args):
    dataset_path = args.dataset_path
    device = args.device
    bs = args.bs
    lr = args.lr
    nepochs = args.nepochs

    ds = Data(dataset_path)
    ds.calculate_img_stats()
    loader = DataLoader(ds, batch_size=bs, shuffle=True)

    model = DCModel()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    loss_min = np.inf
    for epoch in range(nepochs):
        model.train()
        loss_epoch = 0
        for img_in, disp_in, disp_gt in tqdm(loader):
            img_in = img_in.to(device)
            disp_in = disp_in.float().to(device)
            disp_gt = disp_gt.float().to(device)

            # normalize input images
            img_in = (img_in - ds.mean_gray) / ds.std_gray
            disp_in /= ds.max_disp

            optimizer.zero_grad()
            disp_corr = model(torch.cat([img_in, disp_in], dim=1))
            disp_pred = disp_in + disp_corr

            mask_dist = (disp_in > 0) & (disp_gt < ds.max_disp)
            mask_nan = torch.isnan(disp_gt) | torch.isnan(disp_gt)
            mask_valid = torch.ones(disp_gt.shape, dtype=torch.bool, device=device)
            mask_valid[..., :7, :] = False
            mask_valid[..., :, :7] = False
            mask = mask_dist & mask_valid & (~mask_nan)

            loss = criterion(disp_pred[mask], disp_gt[mask])
            loss_epoch += loss.item()

            loss.backward()
            optimizer.step()

        loss_epoch /= len(loader)
        print(f'Epoch {epoch}, Loss: {loss_epoch}')
        # save model checkpoint
        if loss_epoch < loss_min:
            loss_min = loss_epoch
            print(f'Saving model with loss {loss_min:.4f}')
            torch.save(model.state_dict(), 'dc_net.pth')


def result(args):
    import open3d as o3d

    dataset_path = args.dataset_path
    device = args.device

    ds = Data(dataset_path)
    # ds.calculate_img_stats()
    loader = DataLoader(Data(dataset_path), batch_size=1, shuffle=True)

    model = DCModel()
    model.load_state_dict(torch.load('dc_net.pth', map_location=device))
    model.eval()
    model.to(device)

    with torch.no_grad():
        img_in, disp_in, disp_gt = next(iter(loader))
        img_in = img_in.to(device)
        disp_in = disp_in.float().to(device)

        # normalize input images
        img_in_norm = (img_in - ds.mean_gray) / ds.std_gray
        disp_in_norm = disp_in / ds.max_disp

        disp_corr = model(torch.cat([img_in_norm, disp_in_norm], dim=1))
        disp_pred = disp_in + disp_corr

        # visualize colored disparities
        disp_pred = disp_pred.cpu().numpy()[0][0]
        disp_gt = disp_gt.cpu().numpy()[0][0]
        disp_in = disp_in.cpu().numpy()[0][0]

        disp_in_scaled = cv2.convertScaleAbs(disp_in, alpha=255.0 / ds.max_disp)
        disp_in_colored = cv2.applyColorMap(disp_in_scaled, cv2.COLORMAP_JET)
        cv2.imshow("Disparity Input", disp_in_colored)

        disp_scaled = cv2.convertScaleAbs(disp_pred, alpha=255.0 / ds.max_disp)
        disp_colored = cv2.applyColorMap(disp_scaled, cv2.COLORMAP_JET)
        cv2.imshow("Disparity Prediction", disp_colored)

        disp_scaled_gt = cv2.convertScaleAbs(disp_gt, alpha=255.0 / ds.max_disp)
        disp_colored_gt = cv2.applyColorMap(disp_scaled_gt, cv2.COLORMAP_JET)
        cv2.imshow("Disparity Ground Truth", disp_colored_gt)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        points = ds.disp_to_cloud(disp_pred)
        points_gt = ds.disp_to_cloud(disp_gt)
        valid_mask = disp_in > 0
        points = points[valid_mask.flatten()]
        points_gt = points_gt[valid_mask.flatten()]

        # visualize point clouds
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(points)
        pcd_pred.paint_uniform_color([0, 1, 0])  # green

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(points_gt)
        pcd_gt.paint_uniform_color([1, 0, 0])  # red

        o3d.visualization.draw_geometries([pcd_pred, pcd_gt])


def inference_test(agrs):
    from time import time
    from tqdm import tqdm

    device = 'cpu'
    model = DCModel()
    model.to(device)
    model.eval()
    img_dummy = torch.randn(1, 1, 480, 768).to(device)
    disp_dummy = torch.randn(1, 1, 480, 768).to(device)

    n_iters = 100
    t0 = time()
    for _ in tqdm(range(n_iters)):
        with torch.inference_mode():
            disp_pred = model(torch.cat([img_dummy, disp_dummy], dim=1))
    t1 = time()
    t_avg = (t1 - t0) / n_iters
    print(f'Average inference time: {t_avg:.4f} seconds per iteration on {device}')


def main():
    args = parse_args()

    train(args)
    # data_test(args)
    # result(args)
    # inference_test(args)


if __name__ == '__main__':
    main()
