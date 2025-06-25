import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from disp_refine.dataset import Data
from disp_refine.linknet import DispRef
from disp_refine.vis import colorize_disp


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
                disp_err = colorize_disp((disp_gt - disp_pred.cpu())[0][0].numpy())[..., ::-1]  # BGR -> RGB
                disp_in = colorize_disp(disp_in.cpu()[0][0].numpy())[..., ::-1]  # BGR -> RGB
                disp_gt = colorize_disp(disp_gt.cpu()[0][0].numpy())[..., ::-1]  # BGR -> RGB
                disp_pred = colorize_disp(disp_pred.cpu()[0][0].numpy())[..., ::-1]  # BGR -> RGB

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
