from typing import List, Optional, Type

import torch
from scalingup.algo.state_encoder.vision_encoder import VolumeEncoder
from scalingup.algo.unet3d import Encoder, ExtResNetBlock
from torch.nn import Conv3d, Linear, Sequential, Module
import numpy as np


class Conv3DVolumeEncoder(VolumeEncoder):
    def __init__(
        self,
        output_dim: int,
        f_maps: List[int],
        num_groups: int = 8,
        basic_module: Type[Module] = ExtResNetBlock,
        layer_order: str = "gcr",
        flattened_dim: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        encoders: List[Module] = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(
                    self.hidden_dim,
                    out_feature_num,
                    apply_pooling=False,
                    basic_module=basic_module,
                    conv_layer_order=layer_order,
                    num_groups=num_groups,
                )
            else:
                encoder = Encoder(
                    f_maps[i - 1],
                    out_feature_num,
                    basic_module=basic_module,
                    conv_layer_order=layer_order,
                    num_groups=num_groups,
                )
            encoders.append(encoder)

        self.encoders = Sequential(*encoders)
        self.proj = Linear(flattened_dim, output_dim)

    def featurize_volume(self, volume: torch.Tensor):
        volume = self.encoders(volume)
        feature = volume.view(volume.shape[0], -1)
        feature = self.proj(feature)
        return feature
