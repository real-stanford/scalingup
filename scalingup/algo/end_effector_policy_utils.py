from typing import Tuple
from scalingup.algo.virtual_grid import Point3D, VirtualGrid
import torch
import numpy as np
import logging


class Discretizer:
    def __init__(
        self,
        min_rot_val: float,
        max_rot_val: float,
        rotation_resolution: float,
        virtual_grid: VirtualGrid,
    ):
        self.virtual_grid = virtual_grid
        assert (max_rot_val - min_rot_val) % rotation_resolution == 0.0
        self.min_rot_val = min_rot_val
        self.max_rot_val = max_rot_val
        self.rotation_resolution = rotation_resolution
        self.rotation_class_boundaries = torch.tensor(
            np.arange(
                self.min_rot_val,
                self.max_rot_val,
                rotation_resolution,
            ).tolist()
            + [self.max_rot_val]
        ).float()
        self.num_rotation_classes = (
            int((self.max_rot_val - self.min_rot_val) // rotation_resolution) + 1
        )
        lower = torch.tensor(self.virtual_grid.lower_corner)
        upper = torch.tensor(self.virtual_grid.upper_corner)
        self.position_resolutions = (upper - lower) / (
            torch.tensor(self.virtual_grid.grid_shape) - 1
        )

    def discretize_pos(self, pos: torch.Tensor):
        pos_onehot_label = self.virtual_grid.scatter_points(
            xyz_pts=pos[..., None, :],
            feature_pts=torch.ones_like(pos[..., None, [0]]),
            reduce_method="max",
        ).bool()
        return pos_onehot_label[:, 0]

    def undiscretize_pos(self, pos_onehot: torch.Tensor):
        if not all(onehot.float().sum() == 1 for onehot in pos_onehot):
            logging.error("pos_onehot is not one-hot")
        grid_points = self.virtual_grid.get_grid_points(include_batch=False).to(
            pos_onehot.device
        )
        return torch.stack([grid_points[one_hot][0] for one_hot in pos_onehot])

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        return self.virtual_grid.grid_shape

    def discretize_rot(self, rot: torch.Tensor):
        return torch.nn.functional.one_hot(
            torch.bucketize(
                input=rot,
                boundaries=self.rotation_class_boundaries.to(
                    rot.device, non_blocking=True
                )
                + (self.rotation_resolution / 2),
            ),
            num_classes=self.num_rotation_classes,
        )

    def undiscretize_rot(self, rot_scores: torch.Tensor):
        assert rot_scores.shape[-2:] == (3, self.num_rotation_classes)
        rot_indices = rot_scores.argmax(dim=-1)
        shape = rot_indices.shape
        euler_rot = self.rotation_class_boundaries.to(rot_indices.device)[
            rot_indices.reshape(-1)
        ].reshape(shape)
        return euler_rot

    def discretize_pointcloud(self, xyz_pts, feature_pts):
        return self.virtual_grid.scatter_points(
            xyz_pts=xyz_pts,
            feature_pts=feature_pts,
        )

    @property
    def scene_bounds(self) -> Tuple[Point3D, Point3D]:
        return self.virtual_grid.scene_bounds

    @property
    def orn_bounds(self) -> Tuple[Point3D, Point3D]:
        return (
            (
                self.min_rot_val,
                self.min_rot_val,
                self.min_rot_val,
            ),
            (
                self.max_rot_val,
                self.max_rot_val,
                self.max_rot_val,
            ),
        )
