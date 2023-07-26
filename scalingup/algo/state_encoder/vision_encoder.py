from __future__ import annotations

from abc import abstractmethod
import logging
from typing import Dict, List, Optional, Tuple, Union
import torch

from scalingup.algo.end_effector_policy_utils import Discretizer
from torchvision.transforms import Resize, InterpolationMode


class VisionEncoder(torch.nn.Module):
    def __init__(self, vision_obs_horizon: int):
        super().__init__()
        self.vision_obs_horizon = vision_obs_horizon


class ImageEncoder(VisionEncoder):
    def __init__(
        self,
        obs_dim: Tuple[int, int],
        vision_obs_horizon: int,
        views: List[str],
        output_dim: int,
        channels_last: bool = True,
        resize_obs_dim: Optional[Tuple[int, int]] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
    ):
        super().__init__(vision_obs_horizon=vision_obs_horizon)
        self.obs_dim = obs_dim
        self.views = views
        self.output_dim = output_dim
        self.memory_format = (
            torch.channels_last if channels_last else torch.contiguous_format
        )
        logging.info(f"Using {self.memory_format} for `ImageEncoder`")
        if transforms is None:
            transforms = []
        if resize_obs_dim is not None:
            self.resize_obs_dim = resize_obs_dim
            if self.resize_obs_dim != self.obs_dim or len(transforms) > 0:
                transforms.append(
                    Resize(
                        size=self.resize_obs_dim,
                        interpolation=InterpolationMode.BILINEAR,
                        antialias=True,  # type: ignore
                    )
                )
        else:
            self.resize_obs_dim = self.obs_dim
        self.transforms = transforms
        logging.info(f"On {self.obs_dim} images, transforming with {transforms}")

    @abstractmethod
    def process_views(self, views: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, views: Dict[str, torch.Tensor]) -> torch.Tensor:
        view = views[self.views[0]]
        assert (
            view.dtype == torch.uint8
        ), "Casted before reaching vision encoder, inefficient PCIe bandwidth usage!"
        batch_size, obs_horizon, channels, height, width = view.shape
        assert height == self.obs_dim[0] and width == self.obs_dim[1]
        assert obs_horizon == self.vision_obs_horizon
        processed_views = {}
        for view_name, view in views.items():
            view = view.reshape(batch_size * obs_horizon, channels, height, width)
            for t in self.transforms:
                view = t(view)
            processed_views[view_name] = (
                view.view(batch_size, obs_horizon, channels, *self.resize_obs_dim) / 255.0
            )
        return self.process_views(views=processed_views)


class PointCloudEncoder(VisionEncoder):
    def __init__(
        self,
        num_obs_pts: int,
        use_rgb: bool,
        point_feature_dim: int,
        hidden_dim: int,
        vision_obs_horizon: int,
    ):
        super().__init__(vision_obs_horizon=vision_obs_horizon)
        self.num_obs_pts = num_obs_pts
        self.use_rgb = use_rgb
        if not self.use_rgb:
            logging.info("Using non-rgb point clouds")
            assert point_feature_dim == 1
        self.point_feature_dim = point_feature_dim
        self.hidden_dim = hidden_dim
        self.point_proj = torch.nn.Linear(self.point_feature_dim + 3, self.hidden_dim)

    def forward(self, input_xyz_pts: torch.Tensor, input_rgb_pts: torch.Tensor):
        return self.point_proj(
            torch.cat(
                (
                    input_rgb_pts
                    if self.use_rgb
                    else torch.ones_like(input_rgb_pts)[..., [0]],
                    input_xyz_pts,
                ),
                dim=-1,
            )
        )


class VolumeEncoder(PointCloudEncoder):
    def __init__(self, discretizer: Discretizer, **kwargs):
        super().__init__(**kwargs)
        self.discretizer = discretizer

    @abstractmethod
    def featurize_volume(self, volume: torch.Tensor):
        pass

    def forward(self, input_xyz_pts: torch.Tensor, input_rgb_pts: torch.Tensor):
        feature_pts = super().forward(
            input_xyz_pts=input_xyz_pts, input_rgb_pts=input_rgb_pts
        )
        volume = self.discretizer.discretize_pointcloud(
            xyz_pts=input_xyz_pts,
            feature_pts=feature_pts,
        )
        assert volume.shape[-3:] == self.discretizer.grid_shape
        volume = self.featurize_volume(volume=volume)
        return volume
