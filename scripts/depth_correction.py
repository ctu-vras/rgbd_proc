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
    parser = argparse.ArgumentParser(description='Depth Correction Training')
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
    A dataset for depth correction.
    """

    def __init__(self, path):
        super(Dataset, self).__init__()
        self.path = path
        self.max_depth = 10_000.0  # in mm
        self.image_files = {
            'left': sorted(glob(os.path.join(path, 'images', 'left', '*.png'))),
            'right': sorted(glob(os.path.join(path, 'images', 'right', '*.png'))),
        }
        self.depth_files = {
            'luxonis': sorted(glob(os.path.join(path, 'luxonis', 'depth', '*.png'))),
            'defom-stereo': sorted(glob(os.path.join(path, 'defom-stereo', 'depth', '*.npy'))),
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

    def get_image(self, i, camera='left'):
        img_path = self.image_files[camera][i]
        img = Image.open(img_path)
        img = np.asarray(img)
        return img

    def get_depth(self, i, source='luxonis'):
        depth_path = self.depth_files[source][i]
        if source == 'luxonis':
            depth = np.array(Image.open(depth_path))
        elif source == 'defom-stereo':
            depth = np.load(depth_path)
        else:
            raise ValueError("Unknown depth source: {}".format(source))
        return depth

    def get_sample(self, i):
        img = self.get_image(i, camera='right')
        depth_input = self.get_depth(i, source='luxonis')
        depth_label = self.get_depth(i, source='defom-stereo')
        return img[np.newaxis], depth_input[np.newaxis], depth_label[np.newaxis]



class DCModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Linknet(
            encoder_name='mobilenet_v2',
            encoder_weights='imagenet',
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
    img, depth_in, depth_gt = ds[i]
    max_depth = 10_000.0  # in mm

    cv2.imshow('img', img.squeeze())

    depth_scaled = cv2.convertScaleAbs(depth_in.squeeze(), alpha=255.0 / max_depth)
    depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
    cv2.imshow("Depth Input", depth_colored)

    depth_scaled_label = cv2.convertScaleAbs(depth_gt.squeeze(), alpha=255.0 / max_depth)
    depth_colored_label = cv2.applyColorMap(depth_scaled_label, cv2.COLORMAP_JET)
    cv2.imshow("Depth Label", depth_colored_label)

    mask_dist = (depth_in > 0) & (depth_gt < max_depth)
    mask_nan = np.isnan(depth_gt) | np.isnan(depth_gt)
    mask_valid = np.ones(depth_gt.shape, dtype=bool)
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
    loader = DataLoader(ds, batch_size=bs, shuffle=True)

    model = DCModel()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    loss_min = np.inf
    for epoch in range(nepochs):
        model.train()
        loss_epoch = 0
        for img_in, depth_in, depth_gt in tqdm(loader):
            img_in = img_in.to(device)
            depth_in = depth_in.float().to(device)
            depth_gt = depth_gt.float().to(device)

            optimizer.zero_grad()
            depth_pred = model(torch.cat([img_in, depth_in], dim=1))

            mask_dist = (depth_in > 0) & (depth_gt < ds.max_depth)
            mask_nan = torch.isnan(depth_gt) | torch.isnan(depth_gt)
            mask_valid = torch.ones(depth_gt.shape, dtype=torch.bool, device=device)
            mask_valid[..., :7, :] = False
            mask_valid[..., :, :7] = False
            mask = mask_dist & mask_valid & (~mask_nan)

            loss = criterion(depth_pred[mask] / ds.max_depth, depth_gt[mask] / ds.max_depth)
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
    dataset_path = args.dataset_path
    device = args.device

    loader = DataLoader(Data(dataset_path), batch_size=1, shuffle=True)

    model = DCModel()
    model.load_state_dict(torch.load('dc_net.pth', map_location=device))
    model.eval()
    model.to(device)

    # visualize predictions
    with torch.no_grad():
        img_in, depth_in, depth_gt = next(iter(loader))
        img_in = img_in.to(device)
        depth_in = depth_in.float().to(device)
        depth_pred = model(torch.cat([img_in, depth_in], dim=1))
        depth_pred = depth_pred.cpu().numpy()[0][0]
        depth_gt = depth_gt.cpu().numpy()[0][0]

        # visualize colored depth
        # depth_scaled = cv2.convertScaleAbs(depth_pred, alpha=255.0 / loader.dataset.max_depth)
        depth_scaled = cv2.convertScaleAbs(depth_pred, alpha=255.0 / depth_pred.max())
        depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
        cv2.imshow("Depth Prediction", depth_colored)

        # depth_scaled_gt = cv2.convertScaleAbs(depth_gt, alpha=255.0 / loader.dataset.max_depth)
        depth_scaled_gt = cv2.convertScaleAbs(depth_gt, alpha=255.0 / depth_gt.max())
        depth_colored_gt = cv2.applyColorMap(depth_scaled_gt, cv2.COLORMAP_JET)
        cv2.imshow("Depth Ground Truth", depth_colored_gt)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def inference_test(agrs):
    from time import time
    from tqdm import tqdm

    device = 'cpu'
    model = DCModel()
    model.to(device)
    model.eval()
    img_dummy = torch.randn(1, 1, 480, 768).to(device)
    depth_dummy = torch.randn(1, 1, 480, 768).to(device)

    n_iters = 100
    t0 = time()
    for _ in tqdm(range(n_iters)):
        with torch.inference_mode():
            depth_pred = model(torch.cat([img_dummy, depth_dummy], dim=1))
    t1 = time()
    t_avg = (t1 - t0) / n_iters
    print(f'Average inference time: {t_avg:.4f} seconds per iteration on {device}')


def main():
    args = parse_args()

    # train(args)
    data_test(args)
    # result(args)
    # inference_test(args)


if __name__ == '__main__':
    main()
