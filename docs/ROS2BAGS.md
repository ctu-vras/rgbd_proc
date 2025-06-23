# ROS1 -> ROS2 bags

How to convert ROS1 bag files to ROS2 format.

1. Input: `*.bag` (ROS1 format), output: `*.db3` and `metadata.yaml` (ROS2 `sqlite3` format). The [rosbags](https://ternaris.gitlab.io/rosbags/) tools is required.
    ```bash
    pip install rosbags
    rosbags-convert --src BAG_FILE_NAME.bag --dst BAG_FILE_NAME
    ```
    The output-folder `BAG_FILE_NAME` should contain:
    ```bash
    ls BAG_FILE_NAME
    BAG_FILE_NAME.db3  metadata.yaml
    ```
2. Convert from `sqlite3` format (ROS2) to `mcap` format (ROS2).
   - create a yaml-file `out.yaml` describing the output bag format with the following content:
   ```aiignore
   ---
   output_bags:
   - uri: BAG_FILE_NAME_mcap
     storage_id: mcap
     all: true
     compression_mode: file
     compression_format: zstd
   ```
   - perform the conversion to ROS2 [MCAP](https://mcap.dev/) format:
   ```bash
   ros2 bag convert -i BAG_FILE_NAME -o out.yaml
   ```
3. As a result you should have the folder `BAG_FILE_NAME_mcap` generated with the following content:
    ```bash
    ls BAG_FILE_NAME_mcap
    BAG_FILE_NAME_mcap_0.mcap.zstd  metadata.yaml
    ```
    Make sure the conversion was correct or play the resultant bag-file for example with [Foxglove Studio](https://foxglove.dev/):
    ```bash
    ros2 bag info BAG_FILE_NAME_mcap 
    ```
