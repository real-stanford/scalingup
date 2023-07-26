from typing import List, Optional, Type

import numpy as np
import torch
from point_transformer_pytorch import PointTransformerLayer
from torch.nn import Conv3d, Linear, ModuleList, Sequential

from scalingup.algo.state_encoder.vision_encoder import PointCloudEncoder


class PointTransformerEncoder(PointCloudEncoder):
    def __init__(
        self,
        hidden_dim: int,
        pos_mlp_hidden_dim: int,
        attn_mlp_hidden_mult: int,
        num_neighbors: int,
        num_layers: int,
        output_dim: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layers = ModuleList(
            [
                PointTransformerLayer(
                    dim=hidden_dim,
                    pos_mlp_hidden_dim=pos_mlp_hidden_dim,
                    attn_mlp_hidden_mult=attn_mlp_hidden_mult,
                    num_neighbors=num_neighbors,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = Linear(self.num_obs_pts * hidden_dim, output_dim)

    def forward(self, input_xyz_pts: torch.Tensor, input_rgb_pts: torch.Tensor):
        feature_pts = super().forward(
            input_xyz_pts=input_xyz_pts, input_rgb_pts=input_rgb_pts
        )
        for layer in self.layers:
            feature_pts = layer(feature_pts, input_xyz_pts)
        return self.out_proj(feature_pts.flatten(start_dim=1))
