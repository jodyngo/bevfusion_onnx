# bevfusion_onnx
Convert BEVFusion to ONNX. We replaced sparse convolution by Conv3d convolution.

# Installation
1- Install the official MMdetection3D. https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion

2- Run srcipts:

python bevfusion_onnx_lidar_cam.py config/bevfusion_lidar_CAM.py [checkpoint_path] [lidar_path] [image_path]

