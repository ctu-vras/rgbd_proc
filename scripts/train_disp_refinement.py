import copy
import os
from datetime import datetime
import numpy as np
from glob import glob
from tqdm import tqdm
import cv2
from PIL import Image
import yaml
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp


def parse_args():
    parser = argparse.ArgumentParser(description='Disparity Correction Training')
    parser.add_argument('--dataset_path', type=str,
                        default='../data/ROUGH/helhest_2025_06_13-15_01_10',
                        help='Path to the dataset')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')
    parser.add_argument('--bs', type=int, default=8, help='Batch size for training')
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


class DispRef(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Linknet(
            encoder_name='mobilenet_v2',
            encoder_weights='imagenet',
            in_channels=2,
            classes=1,
        )
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        return self.activation(self.model(x))


def colorize_disp(disp, max_disp=None):
    """
    Colorize disparity map for visualization.
    :param disp: Disparity map (H, W)
    :return: Colorized disparity map (H, W, 3)
    """
    if max_disp is None:
        max_disp = max(disp.max(), 1e-6)  # Avoid division by zero
    disp_vis = cv2.convertScaleAbs(disp, alpha=255 / max_disp)
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    return disp_vis


def train(args):
    dataset_path = args.dataset_path
    device = args.device
    bs = args.bs
    lr = args.lr
    nepochs = args.nepochs

    ds = Data(dataset_path)
    # ds.calculate_img_stats()
    loader = DataLoader(ds, batch_size=bs, shuffle=True)

    model = DispRef()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.L1Loss()

    loss_min = np.inf
    counter = 0
    log_dir = f'runs/disp_refinement_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
    os.makedirs(log_dir, exist_ok=True)
    tb_logger = SummaryWriter(log_dir=log_dir)
    for epoch in range(nepochs):
        model.train()
        loss_epoch = 0
        for img_in, disp_in, disp_gt in tqdm(loader):
            img_in = img_in.to(device)
            disp_in = disp_in.float().to(device)
            disp_gt = disp_gt.float().to(device)

            # normalize inputs
            img_in_norm = (img_in/255. - ds.mean_gray) / ds.std_gray
            disp_in_norm = disp_in / ds.max_disp
            inputs = torch.cat([img_in_norm, disp_in_norm], dim=1)

            optimizer.zero_grad()
            disp_corr = model(inputs)
            disp_pred = disp_in + disp_corr * ds.max_disp

            mask_nan = torch.isnan(disp_gt)
            mask_valid = torch.ones_like(disp_gt, dtype=torch.bool)
            mask_valid[..., :7, :] = False
            mask_valid[..., :, :7] = False
            mask = (~mask_nan) & mask_valid

            loss = criterion(disp_pred[mask], disp_gt[mask])
            tb_logger.add_scalar('loss(iter)', loss, counter)
            loss_epoch += loss.item()

            loss.backward()
            optimizer.step()

            counter += 1

        loss_epoch /= len(loader)
        tb_logger.add_scalar('loss(epoch)', loss_epoch, epoch)
        # save model checkpoint
        if loss_epoch < loss_min:
            loss_min = loss_epoch
            print(f'Epoch: {epoch}. Saving model with loss {loss_min:.4f}')
            model.eval()
            torch.save(model.state_dict(), os.path.join(log_dir, 'model.pth'))

            with torch.inference_mode():
                img_in, disp_in, disp_gt = next(iter(loader))
                img_in = img_in.to(device)
                disp_in = disp_in.float().to(device)

                # normalize inputs
                img_in_norm = (img_in / 255. - ds.mean_gray) / ds.std_gray
                disp_in_norm = disp_in / ds.max_disp
                inputs = torch.cat([img_in_norm, disp_in_norm], dim=1)

                disp_corr = model(inputs)
                disp_pred = disp_in + disp_corr * ds.max_disp

                # log input and output images to TensorBoard
                img_in = img_in.cpu().numpy()[0][0]
                disp_err = colorize_disp((disp_gt - disp_pred.cpu())[0][0].numpy())
                disp_in = colorize_disp(disp_in.cpu()[0][0].numpy())
                disp_gt = colorize_disp(disp_gt.cpu()[0][0].numpy())
                disp_pred = colorize_disp(disp_pred.cpu()[0][0].numpy())

                tb_logger.add_image('Input Image', img_in, epoch, dataformats='HW')
                tb_logger.add_image('Input Disparity', disp_in, epoch, dataformats='HWC')
                tb_logger.add_image('Ground Truth Disparity', disp_gt, epoch, dataformats='HWC')
                tb_logger.add_image('Refined Disparity', disp_pred, epoch, dataformats='HWC')
                tb_logger.add_image('Disparity Error', disp_err, epoch, dataformats='HWC')


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
