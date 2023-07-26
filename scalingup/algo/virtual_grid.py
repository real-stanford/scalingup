import torch
from typing import Optional, Tuple
import numpy as np
from torch_scatter import scatter

Point3D = Tuple[float, float, float]


class VirtualGrid:
    def __init__(
        self,
        scene_bounds: Tuple[Point3D, Point3D],
        grid_shape: Tuple[int, int, int],
        int_dtype: torch.dtype = torch.int64,
        float_dtype: torch.dtype = torch.float32,
        batch_size: int = 8,
        reduce_method: str = "mean",
    ):
        self.scene_bounds = scene_bounds
        self.lower_corner = scene_bounds[0]
        self.upper_corner = scene_bounds[1]
        self.grid_shape = grid_shape
        self.batch_size = int(batch_size)
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype
        self.reduce_method = reduce_method

    @property
    def num_grids(self):
        grid_shape = self.grid_shape
        batch_size = self.batch_size
        return int(np.prod((batch_size,) + grid_shape))

    @property
    def corners(
        self,
    ) -> Tuple[Point3D, Point3D, Point3D, Point3D, Point3D, Point3D, Point3D, Point3D]:
        x_min, y_min, z_min = self.lower_corner
        x_max, y_max, z_max = self.upper_corner
        return (
            (x_min, y_min, z_min),
            (x_min, y_min, z_max),
            (x_min, y_max, z_min),
            (x_min, y_max, z_max),
            (x_max, y_min, z_min),
            (x_max, y_min, z_max),
            (x_max, y_max, z_min),
            (x_max, y_max, z_max),
        )

    def get_grid_idxs(self, include_batch=True):
        batch_size = self.batch_size
        grid_shape = self.grid_shape
        int_dtype = self.int_dtype
        axis_coords = [
            torch.arange(0, x, dtype=int_dtype)
            for x in ((batch_size,) + grid_shape if include_batch else grid_shape)
        ]
        coords_per_axis = torch.meshgrid(*axis_coords, indexing="ij")
        grid_idxs = torch.stack(coords_per_axis, dim=-1)
        return grid_idxs

    def get_grid_points(self, include_batch=True):
        lower_corner = self.lower_corner
        upper_corner = self.upper_corner
        grid_shape = self.grid_shape
        float_dtype = self.float_dtype
        grid_idxs = self.get_grid_idxs(include_batch=include_batch)

        lc = torch.tensor(lower_corner, dtype=float_dtype)
        uc = torch.tensor(upper_corner, dtype=float_dtype)
        idx_scale = torch.tensor(grid_shape, dtype=float_dtype)
        scales = (uc - lc) / idx_scale
        offsets = lc

        grid_idxs_no_batch = grid_idxs
        if include_batch:
            grid_idxs_no_batch = grid_idxs[:, :, :, :, 1:]
        grid_idxs_f = grid_idxs_no_batch.to(float_dtype)
        grid_points = grid_idxs_f * scales + offsets
        return grid_points

    def get_points_grid_idxs(self, points, cast_to_int=True, batch_idx=None):
        device = points.device
        lower_corner = self.lower_corner
        upper_corner = self.upper_corner
        grid_shape = self.grid_shape
        int_dtype = self.int_dtype
        float_dtype = self.float_dtype
        lc = torch.tensor(lower_corner, dtype=float_dtype, device=device)
        uc = torch.tensor(upper_corner, dtype=float_dtype, device=device)
        idx_scale = torch.tensor(grid_shape, dtype=float_dtype, device=device)
        offsets = -lc
        scales = idx_scale / (uc - lc)
        points_idxs_i = (points + offsets) * scales
        if cast_to_int:
            points_idxs_i = (points_idxs_i + 0.5).to(dtype=int_dtype)
        points_idxs = torch.empty_like(points_idxs_i)
        for i in range(3):
            points_idxs[..., i] = torch.clamp(
                points_idxs_i[..., i], min=0, max=grid_shape[i] - 1
            )
        final_points_idxs = points_idxs
        if batch_idx is not None:
            final_points_idxs = torch.cat(
                [
                    batch_idx.view(*points.shape[:-1], 1).to(dtype=points_idxs.dtype),
                    points_idxs,
                ],
                dim=-1,
            )
        return final_points_idxs

    def flatten_idxs(self, idxs, keepdim=False):
        grid_shape = self.grid_shape
        batch_size = self.batch_size

        coord_size = idxs.shape[-1]
        target_shape: Optional[Tuple[int, ...]] = None
        if coord_size == 4:
            # with batch
            target_shape = (batch_size,) + grid_shape
        elif coord_size == 3:
            # without batch
            target_shape = grid_shape
        else:
            raise RuntimeError("Invalid shape {}".format(str(idxs.shape)))
        target_stride = tuple(np.cumprod(np.array(target_shape)[::-1])[::-1])[1:] + (1,)
        flat_idxs = (
            idxs * torch.tensor(target_stride, dtype=idxs.dtype, device=idxs.device)
        ).sum(dim=-1, keepdim=keepdim, dtype=idxs.dtype)
        return flat_idxs

    def unflatten_idxs(self, flat_idxs, include_batch=True):
        grid_shape = self.grid_shape
        batch_size = self.batch_size
        target_shape: Tuple[int, ...] = grid_shape
        if include_batch:
            target_shape = (batch_size,) + grid_shape
        target_stride = tuple(np.cumprod(np.array(target_shape)[::-1])[::-1])[1:] + (1,)

        source_shape = tuple(flat_idxs.shape)
        if source_shape[-1] == 1:
            source_shape = source_shape[:-1]
            flat_idxs = flat_idxs[..., 0]
        source_shape += (4,) if include_batch else (3,)

        idxs = torch.empty(
            size=source_shape, dtype=flat_idxs.dtype, device=flat_idxs.device
        )
        mod = flat_idxs
        for i in range(source_shape[-1]):
            idxs[..., i] = mod / target_stride[i]
            mod = mod % target_stride[i]
        return idxs

    def idxs_to_points(self, idxs):
        lower_corner = self.lower_corner
        upper_corner = self.upper_corner
        grid_shape = self.grid_shape
        float_dtype = self.float_dtype
        device = idxs.device

        source_shape = idxs.shape
        point_idxs = None
        if source_shape[-1] == 4:
            # has batch idx
            point_idxs = idxs[..., 1:]
        elif source_shape[-1] == 3:
            point_idxs = idxs
        else:
            raise RuntimeError("Invalid shape {}".format(tuple(source_shape)))

        lc = torch.tensor(lower_corner, dtype=float_dtype, device=device)
        uc = torch.tensor(upper_corner, dtype=float_dtype, device=device)
        idx_scale = torch.tensor(grid_shape, dtype=float_dtype, device=device)
        offsets = lc
        scales = (uc - lc) / idx_scale

        idxs_points = point_idxs * scales + offsets
        return idxs_points

    def scatter_points(self, xyz_pts, feature_pts, reduce_method=None, **kwargs):
        if reduce_method is None:
            reduce_method = self.reduce_method
        batch_size = feature_pts.shape[0]
        idxs = self.get_points_grid_idxs(xyz_pts)
        flat_idxs = self.flatten_idxs(idxs, keepdim=False)
        vol_features = scatter(
            src=feature_pts,
            index=flat_idxs,
            dim=-2,
            dim_size=int(np.prod(self.grid_shape)),
            reduce=self.reduce_method,
            **kwargs,
        ).view(batch_size, *self.grid_shape, -1)
        return vol_features.permute(0, 4, 1, 2, 3).contiguous()
