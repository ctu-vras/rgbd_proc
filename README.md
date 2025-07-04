# RGB-D Data Processing

Tools to process RGB-D data collected from the
[Helhest](https://www.helhest.com/) robot equipped with
[Luxonis OAK](https://shop.luxonis.com/collections/oak-cameras-1) depth cameras.

<img src="./docs/imgs/rgb.png" height="200"/> <img src="./docs/imgs/robot_depth_cloud.png" height="200"/>


## Data

The data sequences are available at the
[subtdata:/data/disparity-refinement](http://subtdata.felk.cvut.cz/disparity-refinement/)
containing:
- disparity maps,
- depth images,
- point clouds,
- rectified left and right colored (grayscale) images,
- calibration files (extrinsics and intrinsics).

<img src="./docs/imgs/left_rect.png" height="200"/> <img src="./docs/imgs/defom_disp.png" height="200"/>

Please, put the downloaded data into the `rgbd_proc/data` folder.

## Installation

The package is organized as a
[ROS 2](https://docs.ros.org/) package and can be installed using the following commands:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone git@github.com:ctu-vras/rgbd_proc.git
cd ~/ros2_ws/
rosdep install --from-path src --ignore-src -r
colcon build --symlink-install --packages-select rgbd_proc
```


## Usage

Make sure to adjust the paths and data topics.

- To view the RGB-D data from a bag file in RViz:
    ```commandline
    ros2 launch rgbd_proc helhest_play.launch.py bag_path:=/path/to/bag rviz:=true
    ```

- To save depth images, corresponding point clouds and colored images, and calibration (extrinsics and intrinsics) from a bag file:
    ```commandline
    ros2 launch rgbd_proc save_data.launch.xml bag_path:=/path/to/bag output_path:=/path/to/save/data
    ```

### Depth Refinement

- **Architecture**: LinkNet with Mobilenet_v2 and ResNet18 encoders from [SMP](https://github.com/qubvel-org/segmentation_models.pytorch)
- **Input**: Left Gray Image + Original Disparity

  <img src="./docs/imgs/disp_in_and_img_gray.png" height="200"/>

- **Output**: Refined Disparity

  <img src="./docs/imgs/disp_refined.png" height="200"/>
  
- **Training Labels (Disparity)**: distillation from [DEFOM-Stereo](https://github.com/Insta360-Research-Team/DEFOM-Stereo)
- **Data**: [subtdata:/data/disparity-refinement](http://subtdata.felk.cvut.cz/disparity-refinement/)
- **CPU Optimization**: with `torch.jit.trace`, [docs](https://docs.pytorch.org/docs/stable/generated/torch.jit.trace.html)
- **ROS2**: node tested with Luxonis OAK camera (long-range)

  ```commandline
  ros2 launch rgbd_proc depth_refinement.launch.xml
  ```