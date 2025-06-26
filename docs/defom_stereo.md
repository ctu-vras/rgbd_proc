# [DEFOM-Stereo](https://github.com/Insta360-Research-Team/DEFOM-Stereo) [CVPR 2025]

> [**DEFOM-Stereo: Depth Foundation Model Based Stereo Matching**](https://arxiv.org/abs/2501.09466)
>
> Authors: Hualie Jiang, Zhiqiang Lou, Laiyan Ding, Rui Xu, Minglang Tan, Wenjie Jiang and Rui Huang

# Preparation

### Installation

Create the environment

```bash
conda env create -f environment.yaml
conda activate defomstereo
pip install -r requirements.txt
```

# Generate disparity maps

```
cd rgbd_proc/scripts/defom_stereo
python generate_disparity_labels.py -l "/path/to/images/left/*.png" -r "/path/to/images/right/*.png" --output_directory /path/to/save/disparity/ --restore_ckpt checkpoints/defomstereo_vitl_kitti.pth
```
