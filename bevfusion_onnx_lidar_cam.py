from argparse import ArgumentParser
from mmengine.config import Config
from mmdet3d.registry import MODELS
from mmengine.registry import init_default_scope
from mmdet3d.utils import register_all_modules
from mmdet3d.structures import Det3DDataSample
from mmdet3d.structures.bbox_3d.utils import get_box_type
from typing import List, Tuple, Union, Any, Dict
import torchvision.transforms as transforms
import numpy as np
import torch
from torch import nn
from PIL import Image
"""
python projects/BEVFusion/bevfusion/bevfusion_onnx_lidar2.py projects/BEVFusion/configs/bevfusion_lidar_ONLY_adverse.py /data/home/thungo/roadview-thi-demo3-collect-data/dataset_adverse_weather/0149/thi_infos_val.pkl /var/local/home/thungo/mmdetection3d/work_dirs/bevfusion_lidar_ONLY_adverse/epoch_1.pth
"""
class VoxelGenerator(object):
    """Voxel generator in numpy implementation.

    Args:
        voxel_size (list[float]): Size of a single voxel
        point_cloud_range (list[float]): Range of points
        max_num_points (int): Maximum number of points in a single voxel
        max_voxels (int, optional): Maximum number of voxels.
            Defaults to 20000.
    """

    def __init__(self,
                 voxel_size: List[float],
                 point_cloud_range: List[float],
                 max_num_points: int,
                 max_voxels: int = 20000):

        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] -
                     point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def generate(self, points: np.ndarray) -> Tuple[np.ndarray]:
        """Generate voxels given points."""
        return points_to_voxel(points, self._voxel_size,
                               self._point_cloud_range, self._max_num_points,
                               True, self._max_voxels)

    @property
    def voxel_size(self) -> List[float]:
        """list[float]: Size of a single voxel."""
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self) -> int:
        """int: Maximum number of points per voxel."""
        return self._max_num_points

    @property
    def point_cloud_range(self) -> List[float]:
        """list[float]: Range of point cloud."""
        return self._point_cloud_range

    @property
    def grid_size(self) -> np.ndarray:
        """np.ndarray: The size of grids."""
        return self._grid_size

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        indent = ' ' * (len(repr_str) + 1)
        repr_str += f'(voxel_size={self._voxel_size},\n'
        repr_str += indent + 'point_cloud_range='
        repr_str += f'{self._point_cloud_range.tolist()},\n'
        repr_str += indent + f'max_num_points={self._max_num_points},\n'
        repr_str += indent + f'max_voxels={self._max_voxels},\n'
        repr_str += indent + f'grid_size={self._grid_size.tolist()}'
        repr_str += ')'
        return repr_str


def points_to_voxel(points: np.ndarray,
                    voxel_size: Union[list, tuple, np.ndarray],
                    coors_range: Union[List[float], List[Tuple[float]],
                                       List[np.ndarray]],
                    max_points: int = 35,
                    reverse_index: bool = True,
                    max_voxels: int = 20000) -> Tuple[np.ndarray]:
    """convert kitti points(N, >=3) to voxels.

    Args:
        points (np.ndarray): [N, ndim]. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size (list, tuple, np.ndarray): [3] xyz, indicate voxel size
        coors_range (list[float | tuple[float] | ndarray]): Voxel range.
            format: xyzxyz, minmax
        max_points (int): Indicate maximum points contained in a voxel.
        reverse_index (bool): Whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels (int): Maximum number of voxels this function creates.
            For second, 20000 is a good choice. Points should be shuffled for
            randomness before this function because max_voxels drops points.

    Returns:
        tuple[np.ndarray]:
            voxels: [M, max_points, ndim] float tensor. only contain points.
            coordinates: [M, 3] int32 tensor.
            num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    if reverse_index:
        voxel_num = _points_to_voxel_reverse_kernel(
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    else:
        voxel_num = _points_to_voxel_kernel(points, voxel_size, coors_range,
                                            num_points_per_voxel,
                                            coor_to_voxelidx, voxels, coors,
                                            max_points, max_voxels)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]

    return voxels, coors, num_points_per_voxel


def _points_to_voxel_reverse_kernel(points: np.ndarray,
                                    voxel_size: Union[list, tuple, np.ndarray],
                                    coors_range: Union[List[float],
                                                       List[Tuple[float]],
                                                       List[np.ndarray]],
                                    num_points_per_voxel: int,
                                    coor_to_voxelidx: np.ndarray,
                                    voxels: np.ndarray,
                                    coors: np.ndarray,
                                    max_points: int = 35,
                                    max_voxels: int = 20000):
    """convert kitti points(N, >=3) to voxels.

    Args:
        points (np.ndarray): [N, ndim]. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size (list, tuple, np.ndarray): [3] xyz, indicate voxel size
        coors_range (list[float | tuple[float] | ndarray]): Range of voxels.
            format: xyzxyz, minmax
        num_points_per_voxel (int): Number of points per voxel.
        coor_to_voxel_idx (np.ndarray): A voxel grid of shape (D, H, W),
            which has the same shape as the complete voxel map. It indicates
            the index of each corresponding voxel.
        voxels (np.ndarray): Created empty voxels.
        coors (np.ndarray): Created coordinates of each voxel.
        max_points (int): Indicate maximum points contained in a voxel.
        max_voxels (int): Maximum number of voxels this function create.
            for second, 20000 is a good choice. Points should be shuffled for
            randomness before this function because max_voxels drops points.

    Returns:
        tuple[np.ndarray]:
            voxels: Shape [M, max_points, ndim], only contain points.
            coordinates: Shape [M, 3].
            num_points_per_voxel: Shape [M].
    """
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num


def _points_to_voxel_kernel(points: np.ndarray,
                            voxel_size: Union[list, tuple, np.ndarray],
                            coors_range: Union[List[float], List[Tuple[float]],
                                               List[np.ndarray]],
                            num_points_per_voxel: int,
                            coor_to_voxelidx: np.ndarray,
                            voxels: np.ndarray,
                            coors: np.ndarray,
                            max_points: int = 35,
                            max_voxels: int = 200000):
    """convert kitti points(N, >=3) to voxels.

    Args:
        points (np.ndarray): [N, ndim]. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size (list, tuple, np.ndarray): [3] xyz, indicate voxel size.
        coors_range (list[float | tuple[float] | ndarray]): Range of voxels.
            format: xyzxyz, minmax
        num_points_per_voxel (int): Number of points per voxel.
        coor_to_voxelidx (np.ndarray): A voxel grid of shape (D, H, W),
            which has the same shape as the complete voxel map. It indicates
            the index of each corresponding voxel.
        voxels (np.ndarray): Created empty voxels.
        coors (np.ndarray): Created coordinates of each voxel.
        max_points (int): Indicate maximum points contained in a voxel.
        max_voxels (int): Maximum number of voxels this function create.
            for second, 20000 is a good choice. Points should be shuffled for
            randomness before this function because max_voxels drops points.

    Returns:
        tuple[np.ndarray]:
            voxels: Shape [M, max_points, ndim], only contain points.
            coordinates: Shape [M, 3].
            num_points_per_voxel: Shape [M].
    """
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    # lower_bound = coors_range[:3]
    # upper_bound = coors_range[3:]
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num



class BEVTransformer:
    def __init__(self, view_transform, meta_info):
        self.view_transform = view_transform
        self.meta_info = meta_info
        self.intrinsics = torch.tensor(meta_info['cam2img'], dtype=torch.float32)  # 3x3
        self.lidar2cam = torch.tensor(meta_info['lidar2cam'], dtype=torch.float32)  # 4x4
        self.img_aug_matrix = torch.tensor(meta_info['img_aug_matrix'], dtype=torch.float32)  # 4x4
        self.compute_mapping()

    def compute_mapping(self):
        # BEV grid parameters
        xbound = self.view_transform['xbound']
        ybound = self.view_transform['ybound']
        dbound = self.view_transform['dbound']
        bev_height = int((xbound[1] - xbound[0]) / xbound[2])  # 360
        bev_width = int((ybound[1] - ybound[0]) / ybound[2])  # 360

        # Create BEV grid in LiDAR coordinates (x, y, z=0)
        x_vals = torch.arange(xbound[0], xbound[1], xbound[2])  # Shape: [360]
        y_vals = torch.arange(ybound[0], ybound[1], ybound[2])  # Shape: [360]
        bev_x, bev_y = torch.meshgrid(x_vals, y_vals, indexing='ij')  # Shapes: [360, 360]
        bev_z = torch.zeros_like(bev_x)  # Ground plane (z=0)
        bev_points = torch.stack([bev_x, bev_y, bev_z, torch.ones_like(bev_x)], dim=-1)  # Shape: [360, 360, 4]

        # Flatten for transformation
        bev_points_flat = bev_points.view(-1, 4)  # Shape: [360*360, 4]
        bev_x_ind = torch.arange(bev_width).repeat(bev_height, 1).t().flatten() + 1  # Shape: [360*360], 1-based
        bev_y_ind = torch.arange(bev_height).repeat(bev_width, 1).flatten() + 1  # Shape: [360*360], 1-based

        # Transform BEV points to camera coordinates
        cam_points = torch.matmul(bev_points_flat, self.lidar2cam.t())  # Shape: [360*360, 4]

        # Filter points in front of camera (z > dbound[0]=1.0)
        valid = cam_points[:, 2] > dbound[0]
        cam_points = cam_points[valid]  # Shape: [N, 4]
        bev_x_ind = bev_x_ind[valid]  # Shape: [N]
        bev_y_ind = bev_y_ind[valid]  # Shape: [N]

        # Project to image coordinates (homogeneous)
        cam_points_3d = cam_points[:, :3]  # Shape: [N, 3]
        img_points = torch.matmul(cam_points_3d, self.intrinsics.t())  # Shape: [N, 3]
        img_points = img_points[:, :2] / img_points[:, 2:3]  # Normalize, Shape: [N, 2]

        # Apply image augmentation
        img_points_homo = torch.cat([img_points, torch.ones_like(img_points[:, :1])], dim=-1)  # Shape: [N, 3]
        img_points_aug = torch.matmul(img_points_homo, self.img_aug_matrix[:3, :3].t())  # Shape: [N, 3]
        img_points_aug = img_points_aug[:, :2] / img_points_aug[:, 2:3]  # Shape: [N, 2]

        # Scale to feature map ([256, 704] -> [32, 88])
        scale_h = self.view_transform['feature_size'][0] / self.view_transform['image_size'][0]  # 32/256
        scale_w = self.view_transform['feature_size'][1] / self.view_transform['image_size'][1]  # 88/704
        im_u_float = img_points_aug[:, 0] * scale_w  # Feature map u-coordinates
        im_v_float = img_points_aug[:, 1] * scale_h  # Feature map v-coordinates

        # Filter valid feature coordinates
        valid = (im_u_float >= 0) & (im_u_float < self.view_transform['feature_size'][1]) & \
                (im_v_float >= 0) & (im_v_float < self.view_transform['feature_size'][0])
        self.im_u_float = im_u_float[valid]  # Shape: [M]
        self.im_v_float = im_v_float[valid]  # Shape: [M]
        self.bev_x_ind = bev_x_ind[valid]  # Shape: [M], 1-based
        self.bev_y_ind = bev_y_ind[valid]  # Shape: [M], 1-based

    def transformImage2BEV(self, inImage, out_dtype=torch.float32):
        '''
        Transform input feature map to Bird's Eye View (BEV) representation.

        Args:
            inImage (torch.Tensor): Input tensor of shape [1, 256, 32, 88].
            out_dtype (torch.dtype): Output tensor data type, default torch.float32.

        Returns:
            torch.Tensor: BEV tensor of shape [1, 80, 180, 180].
        '''
        assert inImage.shape == (1, 128, 32, 88), f"Expected input shape [1, 256, 32, 88], got {inImage.shape}"
        assert self.im_u_float is not None, "im_u_float not initialized"
        assert self.im_v_float is not None, "im_v_float not initialized"
        assert self.bev_x_ind is not None, "bev_x_ind not initialized"
        assert self.bev_y_ind is not None, "bev_y_ind not initialized"

        # Initialize output tensor
        outputData = torch.zeros((1, 80, 180, 180), dtype=out_dtype, device=inImage.device)

        # Adjust indices for 0-based indexing and downsampling
        bev_x_ind = (self.bev_x_ind / self.view_transform['downsample']).long() - 1
        bev_y_ind = (self.bev_y_ind / self.view_transform['downsample']).long() - 1
        im_u_ind = self.im_u_float.long() - 1
        im_v_ind = self.im_v_float.long() - 1

        # Filter valid indices
        valid_mask = (bev_x_ind >= 0) & (bev_x_ind < 180) & (bev_y_ind >= 0) & (bev_y_ind < 180) & \
                     (im_u_ind >= 0) & (im_u_ind < 88) & (im_v_ind >= 0) & (im_v_ind < 32)
        bev_x_ind = bev_x_ind[valid_mask]
        bev_y_ind = bev_y_ind[valid_mask]
        im_u_ind = im_u_ind[valid_mask]
        im_v_ind = im_v_ind[valid_mask]

        # Map channels (copy first 80 channels)
        for channel in range(min(80, inImage.shape[1])):
            outputData[0, channel, bev_y_ind, bev_x_ind] = inImage[0, channel, im_v_ind, im_u_ind]

        return outputData
    
class BEVFusionLidarCam(nn.Module):
    def __init__(self, config):        
        super().__init__()
        self.pts_middle_encoder = MODELS.build(config['pts_middle_encoder'])
        self.pts_backbone = MODELS.build(config['pts_backbone'])
        self.pts_neck = MODELS.build(config['pts_neck'])
        self.bbox_head = MODELS.build(config['bbox_head'])

        self.img_backbone = MODELS.build(config['img_backbone'])
        self.img_neck = MODELS.build(config['img_neck'])
        self.fusion_layer = MODELS.build(config['fusion_layer'])
        self.view_transform = MODELS.build(config['view_transform'])

        self.box_type_3d, self.box_mode_3d = get_box_type('LiDAR')

        self.lidar_aug_matrix = np.eye(4)
        self.img_aug_matrix = np.array([
                                    [4.80e-01, 0.00e+00, 0.00e+00, -5.71e+02],
                                    [0.00e+00, 4.80e-01, 0.00e+00, -7.84e+02],
                                    [0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00],
                                    [0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00]], dtype=np.float32)
        
        self.meta_info = {
                        'img_path' : '/data/home/thungo/roadview-thi-demo3-collect-data/dataset_adverse_weather/0149/val/image_all/image_15_0149.png',
                        'cam2img': [[2747.756591796875, 0.0, 2662.0], [0.0, 1548.112060546875, 1500.0], [0.0, 0.0, 1.0]],
                        'lidar_path': '/data/home/thungo/roadview-thi-demo3-collect-data/dataset_adverse_weather/0149/val/lidar_all/lidar_15_0149.bin',
                        'sample_idx': 0,
                        'lidar2cam': [[0.9999974966049194, -5.05616917507723e-05, 0.0022305184975266457, -0.00781987700611353], [0.0002293842117069289, 0.9967752695083618, -0.08024337142705917, -0.10482694953680038], [-0.002219268586486578, 0.08024367690086365, 0.996772825717926, 1.3790563344955444], [0.0, 0.0, 0.0, 1.0]],
                        'num_pts_feats': 5,
                        'box_type_3d': self.box_type_3d,
                        'box_mode_3d': self.box_mode_3d,
                        'lidar_aug_matrix': self.lidar_aug_matrix,
                        'img_aug_matrix': self.img_aug_matrix}

        self.transform = {
                            'in_channels': 128,
                            'out_channels': 80,
                            'image_size': [256, 704],
                            'feature_size': [32, 88],
                            'xbound': [-54.0, 54.0, 0.3],
                            'ybound': [-54.0, 54.0, 0.3],
                            'zbound': [-10.0, 10.0, 20.0],
                            'dbound': [1.0, 60.0, 0.5],
                            'downsample': 2
                        }
                                                     
        self.transformer = BEVTransformer(self.transform, self.meta_info)

    def forward(self, dense_grid, img): 
        # print("img", img.size()) # torch.Size([1, 3, 256, 704])
        # Lidar branch
        pts_feature = self.pts_middle_encoder(dense_grid)[0].unsqueeze(0) # torch.Size([1, 128, 180, 180])

        # Image branch
        img = img.contiguous()
        img_feature = self.img_backbone(img)
        img_ = self.img_neck(img_feature)
        if not isinstance(img_, torch.Tensor):
            img_ = img_[0]
            # print("img_",img_[0].shape) # torch.Size([1, 256, 32, 88])

        bev_tensor = self.transformer.transformImage2BEV(img_)  # Output: [1, 80, 180, 180]

        fusion_feat = self.fusion_layer(pts_feature, bev_tensor) # torch.Size([1, 256, 180, 180])
        x = self.pts_backbone(fusion_feat)
        final_feats = self.pts_neck(x) # x[0] shape: torch.Size([1, 256, 180, 180])
        scores_3d, bboxes_3d, labels_3d = self.get_boxes(final_feats, self.meta_info)
        return bboxes_3d, scores_3d, labels_3d    
    
    def get_boxes(self, feats, meta_info):   
        
        det3d_data_sample = Det3DDataSample(metainfo=meta_info)       
        batch_data_samples = [det3d_data_sample]

        batch_input_metas = [item.metainfo for item in batch_data_samples]

        outputs = self.bbox_head.predict(feats, batch_input_metas)[0]
        scores_3d = outputs.scores_3d
        bboxes_3d = outputs.bboxes_3d.tensor
        labels_3d = outputs.labels_3d
        return scores_3d, bboxes_3d, labels_3d


    def get_lidar2img(self, cam2img, lidar2cam):
        """Get the projection matrix of lidar2img.

        Args:
            cam2img (torch.Tensor): A 3x3 or 4x4 projection matrix.
            lidar2cam (torch.Tensor): A 3x3 or 4x4 projection matrix.

        Returns:
            Tensor: Transformation matrix with shape 4x4.
        """
        if cam2img.shape == (3, 3):
            temp = cam2img.new_zeros(4, 4)
            temp[:3, :3] = cam2img
            cam2img = temp

        if lidar2cam.shape == (3, 3):
            temp = lidar2cam.new_zeros(4, 4)
            temp[:3, :3] = lidar2cam
            lidar2cam = temp
        return torch.matmul(cam2img, lidar2cam)

def parse_model(model):
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

def convert_SyncBN(config):
    """Convert config's naiveSyncBN to BN.

    Args:
         config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
    """
    if isinstance(config, dict):
        for item in config:
            if item == 'norm_cfg':
                config[item]['type'] = config[item]['type']. \
                                    replace('naiveSyncBN', 'BN')
            else:
                convert_SyncBN(config[item])

def build_backbone_model(config, checkpoint=None, device='cuda:0'):
    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    convert_SyncBN(config.model)
    config.model.train_cfg = None
    model = BEVFusionLidarCam(config.model)
    model.to('cuda').eval()

    checkpoint = torch.load(checkpoint, map_location='cuda')
    dicts = {}
    for key in checkpoint["state_dict"].keys():
        if "pts_middle_encoder" in key:
            dicts[key] = checkpoint["state_dict"][key]
        if "pts_backbone" in key:
            dicts[key] = checkpoint["state_dict"][key]
        if "pts_neck" in key:
            dicts[key] = checkpoint["state_dict"][key]
        if "bbox_head" in key:
            dicts[key] = checkpoint["state_dict"][key]

        if "img_backbone" in key:
            dicts[key] = checkpoint["state_dict"][key]
        if "img_neck" in key:
            dicts[key] = checkpoint["state_dict"][key]
        if "fusion_layer" in key:
            dicts[key] = checkpoint["state_dict"][key]
        if "view_transform" in key:
            dicts[key] = checkpoint["state_dict"][key]

    model.load_state_dict(dicts)

    torch.cuda.set_device(device)
    model.to(device)
    model.eval()
    return model



class ImageAug3D(nn.Module):
    def __init__(self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip,
                 is_train):
        super(ImageAug3D, self).__init__()
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results):
        H, W = results['ori_shape']
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, rotation, translation, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b
        return img, rotation, translation

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data['img']
        new_imgs = []
        transforms = []
        for img in imgs:
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                data)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(np.array(new_img).astype(np.float32))
            transforms.append(transform.numpy())
        data['img'] = new_imgs
        # update the calibration matrices
        data['img_aug_matrix'] = transforms
        return data
    
def load_image(filename, color_type='color', to_float32=True):
    results = {}

    with open(filename, 'rb') as f:
        img = Image.open(f)
        if color_type == 'color':
            img = img.convert('RGB')  
        elif color_type == 'grayscale':
            img = img.convert('L')  
        else:
            raise ValueError(f"Unsupported color_type: {color_type}")

        img_np = np.array(img)

        if color_type == 'grayscale':
            img_np = np.expand_dims(img_np, axis=-1)  # Shape: (H, W, 1)
        else:
            # Ensure channel-last format (H, W, C) for RGB
            if img_np.shape[0] in [1, 3]:  # Handle (C, H, W) case
                img_np = np.transpose(img_np, (1, 2, 0))

        if to_float32:
            img_np = img_np.astype(np.float32)

        results['img'] = [img_np]  # List with single image array
    return img_np


def build_bevfusion_model(config, checkpoint=None, lidar_path=None, image_path=None, device=None):      

    # === Step 1: Load config ===
    config = Config.fromfile(config)
    init_default_scope(config.get('default_scope', 'mmdet3d'))
    register_all_modules(init_default_scope=True)

     # === Step 1: Load config ===    
    backbone_model = build_backbone_model(config, checkpoint, device=device)

    # === Step 2: Load lidar  ===
    points = load_lidar(lidar_path)
    points_np = points.astype(np.float32)

    # === Step 3: Voxelization === 
    voxelize_cfg = dict(
        max_num_points=10,
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        voxel_size=[0.075, 0.075, 0.2],  # Example voxel size, adjust as needed
        max_voxels=[120000, 160000],
        voxelize_reduce=True
    )
    # === Step 4: Load Image & Do Augmentation === 
    data_dict = {'img': None, 'img_shape': None, 'ori_shape': None, 'pad_shape': None}
    image_ = load_image(image_path) # (2168, 3848, 3)
    data_dict['img'] = [image_]
    data_dict['img_shape'] = image_.shape[:2]
    data_dict['ori_shape'] = image_.shape[:2]
    data_dict['pad_shape'] = image_.shape[:2]

    image_aug3d = ImageAug3D(final_dim=[256, 704],
                            resize_lim=[0.48, 0.48],
                            bot_pct_lim=[0.0, 0.0],
                            rot_lim=[0.0, 0.0],
                            rand_flip=False,
                            is_train=False)


    data_info_img_3d = image_aug3d(data_dict)
    image = data_info_img_3d['img'][0] # (256, 704, 3)
    transform = transforms.Compose([transforms.ToTensor(),          # Convert to tensor (normalizes to [0, 1] and transposes to [C, H, W])
                                    ])  
    image_tensor = transform(image).unsqueeze(0).to(device) # (1,3,H,W)

    with torch.autocast('cuda', enabled=False):
        _, _, _, dense_grid = voxelize_point_cloud(
        points_np, voxelize_cfg, batch_idx=0 )  

    torch.onnx.export(
                backbone_model,
                (dense_grid, image_tensor),
                "/var/local/home/thungo/mmdetection3d/projects/BEVFusion/demo/bevfusion_lidar_cam_s.onnx",
                input_names=["dense_grid", "image"],
                output_names=["bboxes_3d", "scores_3d", "labels_3d"],
                opset_version=17,
                verbose=True,
                dynamic_axes={
                            'bboxes_3d': {0: 'num_boxes'},
                            'scores_3d': {0: 'num_boxes'},
                            'labels_3d': {0: 'num_boxes'}
                        }
    )
    print("Exported BEVFusion to bevfusion.onnx")

def load_lidar(lidar_path):
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32), ('ObjTag', np.uint32)]
    points = np.fromfile(lidar_path, dtype=dtype)
    x = points['x']
    y = points['y']
    z = points['z']
    intensity = points['intensity']
    obj_tag = points['ObjTag']

    points_array = np.stack((x, y, z, intensity, obj_tag), axis=-1)
    return points_array

def voxelize_point_cloud(points, voxelize_cfg, batch_idx):
    # Extract configuration
    voxel_size = voxelize_cfg['voxel_size']
    max_num_points = voxelize_cfg['max_num_points']
    max_voxels = voxelize_cfg['max_voxels'][0]  # Use training max_voxels
    point_cloud_range = voxelize_cfg['point_cloud_range']

    # Initialize VoxelGenerator
    voxel_generator = VoxelGenerator(
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_num_points=max_num_points,
        max_voxels=max_voxels,
    )
    
    # Generate voxels
    voxels_np, coordinates_np, num_points_per_voxel_np = voxel_generator.generate(points)
    # Ensure num_points_per_voxel is a NumPy array to avoid scalar error
    if np.isscalar(num_points_per_voxel_np):
        num_points_per_voxel_np = np.array([num_points_per_voxel_np], dtype=np.int32)
    
    # Add batch index to coordinates
    batch_idx_array = np.zeros((coordinates_np.shape[0], 1), dtype=np.int32) + batch_idx
    coords_with_batch = np.concatenate([batch_idx_array, coordinates_np], axis=1)
    
    # Convert to PyTorch tensors with specified dtypes
    feats = torch.from_numpy(voxels_np).float()  # Convert to float32
    coords = torch.from_numpy(coords_with_batch).int()  # Convert to int32
    sizes = torch.from_numpy(num_points_per_voxel_np).int()  # Convert to int32
    batch_size = coords[-1, 0] + 1   # tensor(1, dtype=torch.int32)

    # Compute grid size
    grid_size=[1440, 1440, 41]  # [1440, 1440, 40] for your config
    D, H, W = grid_size[2], grid_size[1], grid_size[0]  # z, y, x order
    
    # Aggregate voxel features (mean pooling over points in each voxel)
    num_points = torch.clamp(sizes, min=1)  # Avoid division by zero
    voxel_features = torch.sum(feats, dim=1) / num_points.view(-1, 1)  # Shape: (M, 5)
    
    # Create dense grid
    dense_grid = torch.zeros((1, 5, D, H, W), dtype=torch.float32)
    batch_idx_t = coords[:, 0]
    z_idx = coords[:, 1]
    y_idx = coords[:, 2]
    x_idx = coords[:, 3]
    dense_grid[batch_idx_t, :, z_idx, y_idx, x_idx] = voxel_features
    
    return feats, coords, batch_size, dense_grid


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('lidar_path', help='Lidar file')
    parser.add_argument('image_path', help='Image file')
    
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    build_bevfusion_model(args.config, args.checkpoint, args.lidar_path, args.image_path, device=args.device)


if __name__ == '__main__':
    main()
