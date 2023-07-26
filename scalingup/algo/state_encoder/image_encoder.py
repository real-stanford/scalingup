from copy import deepcopy
import logging
from typing import Dict, Optional, Tuple
import typing
import torch
from scalingup.algo.state_encoder.vision_encoder import ImageEncoder
from scalingup.algo.robomimic_nets import SpatialSoftmax
import torchvision


class PerViewImageEncoder(ImageEncoder):
    def __init__(
        self,
        base_model: torch.nn.Module,
        per_view_output_dim: int,
        use_spatial_softmax: bool = False,
        spatial_softmax_input_shape: Optional[Tuple[int, int, int]] = None,
        spatial_softmax_num_kp: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if issubclass(type(base_model), torchvision.models.ResNet):
            resnet_model = typing.cast(torchvision.models.ResNet, base_model)
            resnet_model.fc = torch.nn.Identity()  # type: ignore
            if use_spatial_softmax:
                assert spatial_softmax_input_shape is not None
                resnet_model.avgpool = SpatialSoftmax(  # type: ignore
                    input_shape=spatial_softmax_input_shape, num_kp=spatial_softmax_num_kp
                )
        else:
            logging.warning(
                f"PerViewImageEncoder: {type(base_model).__name__} is not a ResNet. "
                + "Ignoring spatial softmax arguments"
            )

        # TODO try changing
        # ```
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # ```
        # to identity also
        self.nets = torch.nn.ModuleDict(
            {view: deepcopy(base_model) for view in self.views}
        ).to(memory_format=self.memory_format)
        self.proj = torch.nn.Linear(
            len(self.views) * per_view_output_dim,
            self.output_dim,
        )

    def process_views(self, views: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, obs_horizon, channels, height, width = views[self.views[0]].shape
        features = []
        for view_name, view in views.items():
            view = view.reshape(batch_size * obs_horizon, channels, height, width).to(
                memory_format=self.memory_format
            )
            features.append(self.nets[view_name](view).view(batch_size, obs_horizon, -1))
        return self.proj(torch.cat(features, dim=-1))
