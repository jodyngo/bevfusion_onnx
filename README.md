# bevfusion_onnx
3D Object Detection. Convert BEVFusion to ONNX. 

# Installation
1- Install the official MMdetection3D. https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion

2- Install onnx, onnxruntime

3- Run srcipts:

python bevfusion_onnx_lidar_cam.py config/bevfusion_lidar_CAM.py [checkpoint_path] [lidar_path] [image_path]

python bevfusion_onnx_lidar.py config/bevfusion_lidar_ONLY_adverse.py [checkpoint_path] [lidar_path] [image_path]

