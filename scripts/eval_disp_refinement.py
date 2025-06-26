import open3d as o3d
import numpy as np
import cv2
import argparse
import torch
from disp_refine.dataset import Data
from disp_refine.linknet import DispRef
from disp_refine.vis import colorize_img


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


def data_test(args):
    dataset_path = args.dataset_path

    ds = Data(dataset_path)

    i = 150
    img, disp_in, disp_gt = ds[i]
    max_disp = min(disp_in.max(), disp_gt.max())

    cv2.imshow('img', img[0])

    disp_colored = colorize_img(disp_in[0], max_val=max_disp)
    cv2.imshow("Disp Input", disp_colored)

    disp_colored_label = colorize_img(disp_gt[0], max_val=max_disp)
    cv2.imshow("Disp Label", disp_colored_label)

    # mask_nan = ~np.isnan(disp_gt)
    # mask_valid = np.ones(disp_gt.shape, dtype=bool)
    # mask_valid[:, :7, :] = False
    # mask_valid[:, :, :7] = False
    # mask = mask_valid & mask_nan
    # cv2.imshow("Mask", mask.squeeze().astype(np.uint8) * 255)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def result(args):
    dataset_path = args.dataset_path
    device = args.device

    ds = Data(dataset_path)
    # ds.calculate_img_stats()

    model = DispRef()
    model_path = '../config/weights/disp_refine/model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    with torch.no_grad():
        i = 0
        sample = ds[i]
        batch = [torch.from_numpy(s)[np.newaxis] for s in sample]
        img_in, disp_in, disp_gt = batch
        img_in = img_in.to(device)
        disp_in = disp_in.float().to(device)

        # normalize input images
        img_in_norm = (img_in / 255. - ds.mean_gray) / ds.std_gray
        disp_in_norm = disp_in / ds.max_disp
        inputs = torch.cat([img_in_norm, disp_in_norm], dim=1)

        disp_corr = model(inputs)
        disp_pred = disp_in + disp_corr * ds.max_disp

        # visualize colored disparities
        disp_pred = disp_pred.cpu().numpy()[0][0]
        disp_gt = disp_gt.cpu().numpy()[0][0]
        disp_in = disp_in.cpu().numpy()[0][0]

        disp_in_colored = colorize_img(disp_in, max_val=ds.max_disp)
        cv2.imshow("Disparity Input", disp_in_colored)

        disp_colored = colorize_img(disp_pred, max_val=ds.max_disp)
        cv2.imshow("Disparity Prediction", disp_colored)

        disp_colored_gt = colorize_img(disp_gt, max_val=ds.max_disp)
        cv2.imshow("Disparity Ground Truth", disp_colored_gt)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # visualize point clouds
        points = ds.disp_to_cloud(disp_pred)
        points_gt = ds.disp_to_cloud(disp_gt)
        points_in = ds.disp_to_cloud(disp_in)

        valid_mask = (disp_pred > 2) & (disp_pred < ds.max_disp) &\
                     (disp_gt > 2) & (disp_gt < ds.max_disp) & \
                     (disp_in > 2) & (disp_in < ds.max_disp)

        points = points[valid_mask.flatten()]
        points_gt = points_gt[valid_mask.flatten()]
        points_in = points_in[valid_mask.flatten()]

        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(points)
        pcd_pred.paint_uniform_color([0, 1, 0])  # green

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(points_gt)
        pcd_gt.paint_uniform_color([1, 0, 0])  # red

        pcd_in = o3d.geometry.PointCloud()
        pcd_in.points = o3d.utility.Vector3dVector(points_in)
        pcd_in.paint_uniform_color([0, 0, 1])  # blue

        # coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

        # o3d.visualization.draw_geometries([pcd_pred, pcd_in])
        o3d.visualization.draw_geometries([pcd_pred, pcd_in, pcd_gt, coord_frame])


def inference_test(agrs):
    from time import time
    from tqdm import tqdm

    device = 'cpu'
    model = DispRef()
    model.to(device)
    model.eval()
    img_dummy = torch.randn(1, 1, 480, 768).to(device)
    disp_dummy = torch.randn(1, 1, 480, 768).to(device)
    inputs = torch.cat([img_dummy, disp_dummy], dim=1)

    n_iters = 100
    t0 = time()
    for _ in tqdm(range(n_iters)):
        with torch.inference_mode():
            model(inputs)
    t1 = time()
    t_avg = (t1 - t0) / n_iters
    print(f'Average inference time: {t_avg:.4f} seconds per iteration on {device}')


def main():
    args = parse_args()

    data_test(args)
    result(args)
    inference_test(args)


if __name__ == '__main__':
    main()
