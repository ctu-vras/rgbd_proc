# TODOs

## Depth Estimation

- [ ] Select a lightweight stereo depth estimation model using the [KITTI Stereo benchmark](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo), for example, [ESS](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/dnn_stereo_disparity).
- [ ] Create a ROS2 node for the selected model.
- [ ] Fine-tune the model on the collected dataset using the distillation from the [FoundationStereo](https://nvlabs.github.io/FoundationStereo/) or [DEFOM-Stereo](https://github.com/Insta360-Research-Team/DEFOM-Stereo) models.
- [ ] Deploy the node on the [Helhest](https://www.helhest.com/) robot.

## Terrain Mapping