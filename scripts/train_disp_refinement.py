import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))

from datetime import datetime
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from disp_refine.dataset import compile_data, data_sequences, load_img_stats
from disp_refine.linknet import DispRef
from disp_refine.vis import colorize_img


def parse_args():
    parser = argparse.ArgumentParser(description='Disparity Correction Training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--bs', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--nepochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--pretrained_weights', type=str, default=None, help='Path to pretrained model (optional)')
    return parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.device = args.device
        self.bs = args.bs
        self.lr = args.lr
        self.nepochs = args.nepochs
        self.pretrained_weights = args.pretrained_weights

        self._prepare_data()
        self._build_model()
        self._init_logging()

    def _prepare_data(self):
        img_stats = load_img_stats()
        self.mean_gray = img_stats['gray']['mean']
        self.std_gray = img_stats['gray']['std']
        self.max_disp = img_stats['max_disparity']

        train_ds, val_ds = compile_data(data_sequences)
        self.train_loader = DataLoader(train_ds, batch_size=self.bs, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.bs, shuffle=False)

    def _build_model(self):
        self.model = DispRef().to(self.device)
        if self.pretrained_weights:
            print(f'Loading pretrained weights from {self.pretrained_weights}')
            self.model.load_state_dict(torch.load(self.pretrained_weights, map_location=self.device))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.L1Loss()

    def _init_logging(self):
        log_dir = f'runs/disp_refinement_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.tb_logger = SummaryWriter(log_dir=log_dir)
        self.counter = 0
        self.best_loss = float('inf')

    def _normalize_inputs(self, img_in, disp_in):
        img_in = (img_in - self.mean_gray) / self.std_gray
        disp_in = disp_in / self.max_disp
        return torch.cat([img_in, disp_in], dim=1)

    def _compute_loss(self, disp_pred, disp_gt):
        mask_nan = ~torch.isnan(disp_gt)
        mask_valid = torch.ones_like(disp_gt, dtype=torch.bool)
        mask_valid[..., :7, :] = False
        mask_valid[..., :, :7] = False
        mask = mask_nan & mask_valid
        return self.criterion(disp_pred[mask], disp_gt[mask])

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for img_in, disp_in, disp_gt in tqdm(self.train_loader, desc=f'Train Epoch {epoch+1}', unit='batch'):
            img_in = img_in.to(self.device)
            disp_in = disp_in.float().to(self.device)
            disp_gt = disp_gt.float().to(self.device)

            inputs = self._normalize_inputs(img_in, disp_in)
            self.optimizer.zero_grad()

            disp_corr = self.model(inputs)
            disp_pred = disp_in + disp_corr * self.max_disp

            loss = self._compute_loss(disp_pred, disp_gt)
            loss.backward()
            self.optimizer.step()

            self.tb_logger.add_scalar('loss/train_iter', loss.item(), self.counter)
            self.counter += 1
            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        self.tb_logger.add_scalar('loss/train_epoch', avg_loss, epoch)
        return avg_loss

    def val_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.inference_mode():
            for img_in, disp_in, disp_gt in tqdm(self.val_loader, desc=f'Val Epoch {epoch+1}', unit='batch'):
                img_in = img_in.to(self.device)
                disp_in = disp_in.float().to(self.device)
                disp_gt = disp_gt.float().to(self.device)

                inputs = self._normalize_inputs(img_in, disp_in)
                disp_corr = self.model(inputs)
                disp_pred = disp_in + disp_corr * self.max_disp

                loss = self._compute_loss(disp_pred, disp_gt)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        self.tb_logger.add_scalar('loss/val_epoch', avg_loss, epoch)
        return avg_loss

    def log_visuals(self, epoch, img_in, disp_in, disp_gt, disp_pred):
        img_np = img_in.cpu().numpy()[0][0]
        disp_in_np = colorize_img(disp_in.cpu()[0][0].numpy(), max_val=self.max_disp)[..., ::-1]
        disp_gt_np = colorize_img(disp_gt.cpu()[0][0].numpy(), max_val=self.max_disp)[..., ::-1]
        disp_pred_np = colorize_img(disp_pred.cpu()[0][0].numpy(), max_val=self.max_disp)[..., ::-1]
        disp_err_np = colorize_img((disp_gt - disp_pred)[0][0].cpu().numpy(), max_val=self.max_disp)[..., ::-1]

        self.tb_logger.add_image(f'Epoch_{epoch}/Input Image', img_np, epoch, dataformats='HW')
        self.tb_logger.add_image(f'Epoch_{epoch}/Input Disparity', disp_in_np, epoch, dataformats='HWC')
        self.tb_logger.add_image(f'Epoch_{epoch}/Ground Truth Disparity', disp_gt_np, epoch, dataformats='HWC')
        self.tb_logger.add_image(f'Epoch_{epoch}/Refined Disparity', disp_pred_np, epoch, dataformats='HWC')
        self.tb_logger.add_image(f'Epoch_{epoch}/Disparity Error', disp_err_np, epoch, dataformats='HWC')

    def train(self):
        for epoch in range(self.nepochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                print(f'Validation improved at epoch {epoch+1}, saving model (val loss: {val_loss:.4f})')
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'model.pth'))

                with torch.inference_mode():
                    img_in, disp_in, disp_gt = next(iter(self.val_loader))
                    img_in = img_in.to(self.device)
                    disp_in = disp_in.float().to(self.device)
                    disp_gt = disp_gt.float().to(self.device)

                    inputs = self._normalize_inputs(img_in, disp_in)
                    disp_corr = self.model(inputs)
                    disp_pred = disp_in + disp_corr * self.max_disp

                    self.log_visuals(epoch, img_in, disp_in, disp_gt, disp_pred)


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
