# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.registry import MODELS
from torch import nn

import torch
import torch.nn as nn

class BasicBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Initializes convolutional block
        Args:
            in_channels: int, Number of input channels
            out_channels: int, Number of output channels
            **kwargs: Dict, Extra arguments for nn.Conv2d
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        """
        Applies convolutional block
        Args:
            features: (B, C_in, H, W), Input features
        Returns:
            x: (B, C_out, H, W), Output features
        """
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        return x


@MODELS.register_module()
class BEVFusion2DEncoder(nn.Module):
    r"""Sparse encoder for BEVFusion. The difference between this
    implementation and that of ``SparseEncoder`` is that the shape order of 3D
    conv is (H, W, D) in ``BEVFusionSparseEncoder`` rather than (D, H, W) in
    ``SparseEncoder``. This difference comes from the implementation of
    ``voxelization``.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str], optional): Order of conv module.
            Defaults to ('conv', 'norm', 'act').
        norm_cfg (dict, optional): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int, optional): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int, optional): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]], optional):
            Convolutional channels of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        encoder_paddings (tuple[tuple[int]], optional):
            Paddings of each encode block.
            Defaults to ((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)).
        block_type (str, optional): Type of the block to use.
            Defaults to 'conv_module'.
        return_middle_feats (bool): Whether output middle features.
            Default to False.
    """
    def __init__(self,
                 in_channels,
                 sparse_shape,
                 output_channels=64):
        super().__init__()
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.output_channels = output_channels
        
        self.con2dblock = BasicBlock2D(205, self.output_channels)        
        self.conv_resize = nn.Conv2d(
                in_channels=self.output_channels,        # Input channels
                out_channels=self.output_channels,       # Output channels (same as input to preserve channels)
                kernel_size=3,          # Kernel size
                stride=8,               # Stride to downsample by 8
                padding=0               # No padding
            )
        self.conv3d = nn.Conv3d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1)

    def forward(self, dense_grid):
        output_features = self.conv3d(dense_grid.to(self.conv3d.weight.device)) # nn.Conv3d expects (N, C, D, H, W)
        # print("conv3d out shape:", output_features.shape) # Output: [N, 4, D, H, W]
        N, C, D, H, W = output_features.shape
        bev_features = output_features.view(N, C * D, H, W)
        bev_features = self.con2dblock(bev_features) # torch.Size([4, 64, 1440, 1440])
        bev_features = self.conv_resize(bev_features) # torch.Size([4, 64, 180, 180])
        return bev_features


    def densify_voxels(self,coors, voxel_features, sparse_shape, batch_size=None):
        """
        Converts sparse voxel features into a dense 5D grid.

        Args:
            coors: [N, 4] tensor -> [batch_idx, x, y, z]
            voxel_features: [N, C] tensor
            sparse_shape: tuple or list of (D, H, W)
            batch_size: optional int. If None, will be inferred from coors.

        Returns:
            [B, C, D, H, W] dense tensor
        """
        coors = coors.long()
        C = voxel_features.shape[1]
        D, H, W = sparse_shape

        if batch_size is None:
            batch_size = int(coors[:, 0].max().item()) + 1

        grid = torch.zeros((batch_size, C, D, H, W), device=voxel_features.device)

        b, x, y, z = coors[:, 0], coors[:, 1], coors[:, 2], coors[:, 3]
        valid_mask = (x >= 0) & (x < D) & (y >= 0) & (y < H) & (z >= 0) & (z < W)

        b, x, y, z = b[valid_mask], x[valid_mask], y[valid_mask], z[valid_mask]
        feats = voxel_features[valid_mask]  # [N_valid, C]

        # Flatten all dimensions for correct indexing
        flat_indices = (
            b.repeat_interleave(C),
            torch.arange(C, device=feats.device).repeat(len(b)),
            x.repeat_interleave(C),
            y.repeat_interleave(C),
            z.repeat_interleave(C)
        )
        grid.index_put_(flat_indices, feats.T.flatten(), accumulate=True)

        return grid
