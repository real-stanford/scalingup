from __future__ import annotations
from glob import glob
import logging
import re
import pytorch3d as pt3d
import zarr
import typing
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import lmdb
from rich.progress import track
import numpy as np
import torch
from pydantic import dataclasses, validator

from scalingup.data.dataset import (
    ReplayBuffer,
    StateTensor,
    TensorDataClass,
)
from scalingup.utils.generic import AllowArbitraryTypes, limit_threads
from scalingup.utils.core import (
    PartialObservation,
    PointCloud,
    split_state_phrase,
)
from transforms3d import quaternions, affines, euler


@dataclasses.dataclass(frozen=True)
class PolicyWindowRolloutConfig:
    # all units here are in action frequency cycles
    prediction_horizon: int
    proprio_obs_horizon: int
    vision_obs_horizon: int
    action_horizon: int
    #   |v|                             vision_obs_horizon
    # |o|o|                             proprio_obs_horizon
    # | |a|a|a|a|a|a|a|a|               action_horizon
    # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| prediction_horizon

    @validator("vision_obs_horizon")
    @classmethod
    def vision_obs_horizon_is_less_than_proprio(cls, v: int, values: Dict[str, Any]):
        if v > values["proprio_obs_horizon"]:
            raise ValueError(
                f"vision_obs_horizon ({v}) must be less than"
                + f" proprio_obs_horizon ({values['proprio_obs_horizon']})"
            )
        return v


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class StateSequenceTensor(TensorDataClass):
    sequence: List[StateTensor]  # T x B x ...

    @property
    def dtype(self) -> torch.dtype:
        return self.sequence[0].dtype

    @property
    def device(self) -> torch.device:
        return self.sequence[0].device

    @property
    def batch_size(self) -> int:
        return self.sequence[0].batch_size

    @property
    def is_batched(self) -> bool:
        return self.sequence[0].is_batched

    def transform(self, matrix: np.ndarray) -> StateSequenceTensor:
        return StateSequenceTensor(
            sequence=[state.transform(matrix=matrix) for state in self.sequence]
        )

    @classmethod
    def from_obs_sequence(
        cls,
        control_sequence: List[PartialObservation],
        pad_left_count: int,
        pad_right_count: int,
        **kwargs,
    ) -> StateSequenceTensor:
        # control_sequence should end with the obs
        # to currently predict the action for
        state_sequence = StateSequenceTensor(
            sequence=[
                StateTensor.from_obs(partial_obs=partial_obs, **kwargs)
                for partial_obs in control_sequence
            ]
        )
        if pad_left_count > 0:
            state_sequence = state_sequence.pad_left(count=pad_left_count)
        if pad_right_count > 0:
            state_sequence = state_sequence.pad_right(count=pad_right_count)

        return state_sequence

    def pad_left(self, count: int) -> StateSequenceTensor:
        return StateSequenceTensor(sequence=[self.sequence[0]] * count + self.sequence)

    def pad_right(self, count: int) -> StateSequenceTensor:
        return StateSequenceTensor(sequence=self.sequence + [self.sequence[-1]] * count)

    @classmethod
    def collate(cls, batch: Sequence[StateSequenceTensor]) -> StateSequenceTensor:
        assert all(
            len(state_sequence.sequence) == len(batch[0].sequence)
            for state_sequence in batch
        ), "Unequal window sizes"
        sequence_size = len(batch[0].sequence)
        attrs = list(StateTensor.__annotations__.keys())
        return StateSequenceTensor(
            sequence=[
                StateTensor(
                    **torch.utils.data.default_collate(  # type: ignore
                        [
                            {
                                attr: getattr(state_sequence.sequence[i], attr)
                                for attr in attrs
                            }
                            for state_sequence in batch
                        ]
                    )
                )
                for i in range(sequence_size)
            ],
        )

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = True,
    ) -> StateSequenceTensor:
        return StateSequenceTensor(
            sequence=[
                state.to(device=device, dtype=dtype, non_blocking=non_blocking)
                for state in self.sequence
            ],
        )

    def get_proprioception_data(self, proprioception_keys: Set[str]) -> torch.Tensor:
        return torch.stack(
            [
                state.get_proprioception_data(proprioception_keys)
                for state in self.sequence
            ],
            dim=-2,
        )

    @property
    def input_xyz_pts(self) -> torch.Tensor:
        return torch.stack(
            [state.input_xyz_pts for state in self.sequence],
            dim=-3,
        )

    @property
    def input_rgb_pts(self) -> torch.Tensor:
        return torch.stack(
            [state.input_rgb_pts for state in self.sequence],
            dim=-3,
        )

    @property
    def views(self) -> Dict[str, torch.Tensor]:
        return {
            view: torch.stack(
                [state.views[view] for state in self.sequence],
                dim=-4,
            )
            for view in self.sequence[0].views.keys()
        }


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class ControlActionTensor(TensorDataClass):
    value: torch.Tensor

    @property
    def dtype(self) -> torch.dtype:
        return self.value.dtype

    @property
    def device(self) -> torch.device:
        return self.value.device

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = True,
    ) -> ControlActionTensor:
        return typing.cast(
            ControlActionTensor,
            super().to(device=device, dtype=dtype, non_blocking=non_blocking),
        )


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class ControlActionSequenceTensor(TensorDataClass):
    sequence: List[ControlActionTensor]  # T x B x ...

    @property
    def dtype(self) -> torch.dtype:
        return self.sequence[0].dtype

    @property
    def device(self) -> torch.device:
        return self.sequence[0].device

    @property
    def tensor(self) -> torch.Tensor:
        return torch.stack([action.value for action in self.sequence], dim=-2)

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = True,
    ) -> ControlActionSequenceTensor:
        return ControlActionSequenceTensor(
            sequence=[
                action.to(device=device, dtype=dtype, non_blocking=non_blocking)
                for action in self.sequence
            ]
        )

    @classmethod
    def collate(
        cls, batch: Sequence[ControlActionSequenceTensor]
    ) -> ControlActionSequenceTensor:
        assert all(
            len(action_sequence.sequence) == len(batch[0].sequence)
            for action_sequence in batch
        ), "Unequal window sizes"
        sequence_size = len(batch[0].sequence)
        return ControlActionSequenceTensor(
            sequence=[
                ControlActionTensor(
                    value=torch.stack(
                        [action_sequence.sequence[i].value for action_sequence in batch]
                    )
                )
                for i in range(sequence_size)
            ],
        )

    @classmethod
    def from_control_sequence(
        cls,
        control_sequence: torch.Tensor,
        pad_left_count: int,
        pad_right_count: int,
    ) -> ControlActionSequenceTensor:
        # observation_sequence should end with the obs to currently
        # predict for
        action_sequence = ControlActionSequenceTensor(
            sequence=[ControlActionTensor(value=control) for control in control_sequence]
        )
        if pad_left_count > 0:
            action_sequence = action_sequence.pad_left(count=pad_left_count)
        if pad_right_count > 0:
            action_sequence = action_sequence.pad_right(count=pad_right_count)

        return action_sequence

    def pad_left(self, count: int) -> ControlActionSequenceTensor:
        return ControlActionSequenceTensor(
            sequence=[self.sequence[0]] * count + self.sequence
        )

    def pad_right(self, count: int) -> ControlActionSequenceTensor:
        return ControlActionSequenceTensor(
            sequence=self.sequence + [self.sequence[-1]] * count
        )


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class TrajectoryWindowTensor(TensorDataClass):
    state_sequence: StateSequenceTensor
    action_sequence: ControlActionSequenceTensor

    task_metrics: torch.Tensor  # T x B x ... (success, dense_success, etc.)
    task_names: List[str]  # B

    @property
    def dtype(self) -> torch.dtype:
        return self.state_sequence.dtype

    @property
    def device(self) -> torch.device:
        return self.state_sequence.device

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = True,
    ) -> TrajectoryWindowTensor:
        return TrajectoryWindowTensor(
            state_sequence=self.state_sequence.to(
                device=device, dtype=dtype, non_blocking=non_blocking
            ),
            action_sequence=self.action_sequence.to(
                device=device, dtype=dtype, non_blocking=non_blocking
            ),
            task_metrics=self.task_metrics.to(
                device=device, dtype=dtype, non_blocking=non_blocking
            ),
            task_names=self.task_names,
        )

    @classmethod
    def collate(cls, batch: Sequence[TrajectoryWindowTensor]) -> TrajectoryWindowTensor:
        assert all(
            traj_window_tensor.window_size == batch[0].window_size
            for traj_window_tensor in batch
        ), "Unequal window sizes"
        return TrajectoryWindowTensor(
            state_sequence=StateSequenceTensor.collate(
                [traj_window_tensor.state_sequence for traj_window_tensor in batch]
            ),
            action_sequence=ControlActionSequenceTensor.collate(
                [traj_window_tensor.action_sequence for traj_window_tensor in batch]
            ),
            task_metrics=torch.utils.data.default_collate(  # type: ignore
                [traj_step.task_metrics for traj_step in batch]
            ),
            task_names=sum((traj_step.task_names for traj_step in batch), start=[]),
        )

    @property
    def window_size(self) -> int:
        return len(self.state_sequence.sequence)

    def transform(self, matrix: np.ndarray) -> TrajectoryWindowTensor:
        return TrajectoryWindowTensor(
            state_sequence=self.state_sequence.transform(matrix=matrix),
            action_sequence=self.action_sequence,
            task_metrics=self.task_metrics,
            task_names=self.task_names,
        )

    @property
    def batch_size(self) -> int:
        return self.state_sequence.batch_size

    @property
    def is_batched(self) -> bool:
        return self.state_sequence.is_batched


@dataclasses.dataclass(config=AllowArbitraryTypes)
class NormalizationConfig:

    """
    normalizes input data to range [-1,1]
    """

    max_value: torch.Tensor
    min_value: torch.Tensor

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self.min_value = self.min_value.to(x.device, x.dtype)
        self.max_value = self.max_value.to(x.device, x.dtype)
        return (x - self.min_value) / (self.max_value - self.min_value) * 2 - 1

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        self.min_value = self.min_value.to(x.device, x.dtype)
        self.max_value = self.max_value.to(x.device, x.dtype)
        return (x + 1) / 2 * (self.max_value - self.min_value) + self.min_value

    def to(self, device: Union[torch.device, str], dtype: Optional[torch.dtype] = None):
        self.min_value = self.min_value.to(device, dtype)
        self.max_value = self.max_value.to(device, dtype)
        return self


def copy_from_store(
    src: zarr.Group,
    dest: zarr.Group,
    if_exists="replace",
    compressor: Any = -1,
) -> zarr.Group:
    for key in src.keys():
        value = src[key]
        # copy with recompression
        if type(value) == zarr.Array:
            n_copied, n_skipped, n_bytes_copied = zarr.copy(
                source=value,
                dest=dest,
                name=key,
                chunks=value.chunks,
                compressor=value.compressor if compressor == -1 else compressor,
                if_exists=if_exists,
            )
        elif type(value) == zarr.Group:
            copy_from_store(
                src=value,
                dest=dest.create_group(key),
                if_exists=if_exists,
                compressor=compressor,
            )
        else:
            raise NotImplementedError(f"Cannot copy {type(value)}")
    for attr, value in src.attrs.asdict().items():
        dest.attrs[attr] = value
    return dest


class TrajectoryWindowDataset(ReplayBuffer):
    def __init__(self, rollout_config: PolicyWindowRolloutConfig, **kwargs):
        self.rollout_config = rollout_config
        self.min_action_values: Optional[torch.Tensor] = None
        self.max_action_values: Optional[torch.Tensor] = None
        self.path_obs_idx_tuples: List[Tuple[str, int]] = []
        self.control_frequencies: List[float] = []
        self.path_info: Dict[str, Tuple[str, Union[bool, float]]] = {}
        super().__init__(collate_fn=TrajectoryWindowTensor.collate, **kwargs)

    @property
    def control_frequency(self) -> int:
        if self.control_frequencies is None:
            raise ValueError("Action frequency statistics has not been computed")
        frequency = np.mean(self.control_frequencies)
        if not np.allclose(frequency, np.rint(frequency)):
            raise ValueError(f"Action frequency ({frequency}Hz) is not integer")
        return int(np.rint(frequency))

    def reset_index(self):
        super().reset_index()
        self.path_obs_idx_tuples.clear()
        self.control_frequencies.clear()
        self.path_info.clear()
        self.min_action_values = None
        self.max_action_values = None

    @property
    def action_norm_config(self) -> NormalizationConfig:
        if self.min_action_values is None or self.max_action_values is None:
            raise ValueError("Action normalization statistics has not been computed")
        return NormalizationConfig(
            max_value=self.max_action_values, min_value=self.min_action_values
        )

    # @profile
    def __getitem__(self, idx):
        limit_threads(n=1)
        path, obs_end_idx = self.path_obs_idx_tuples[idx]
        """
        NOTE: `readonly=True` to prevent this error?
        ```
        self.db = lmdb.open(path, **kwargs)
        lmdb.InvalidParameterError: mdb_txn_begin: Invalid argument
        ```
        """
        # Use consolidated metadata store to make listing keys more
        # efficient.
        # https://zarr.readthedocs.io/en/stable/api/convenience.html#zarr.convenience.consolidate_metadata

        with zarr.LMDBStore(
            path,
            subdir=False,
            lock=False,
            readonly=True,
            readahead=False,
            map_size=int(2**18),
        ) as store:
            root = zarr.group(store=store)
            length = root.attrs["length"]
            task_desc = root.attrs["task_desc"]
            task_metric = self.task_metrics[idx]
            """
            example:
                pred_start_idx (-2)
                |       obs_end_idx (2)
                |       |
                V       V
                |o|o|o|o|o|                        obs_horizon (5)
                |a|a|a|a|a|a|a|a|a|a|a|a|          prediction_horizon (17)
                    |0|1|2|3|4|5|6|                control_sequence (7)
                |-|-|                              seq_pad_left (2)
                                |-|-|-|          action_pad_right_count (3)
            """

            assert (
                obs_end_idx >= 0
                and obs_end_idx < length + self.rollout_config.proprio_obs_horizon
            )

            pred_start_idx = obs_end_idx - self.rollout_config.proprio_obs_horizon + 1
            pred_end_idx = pred_start_idx + self.rollout_config.prediction_horizon
            clipped_pred_start_idx = max(0, pred_start_idx)
            assert (
                clipped_pred_start_idx < length and clipped_pred_start_idx < pred_end_idx
            )

            transform_matrix = (
                self.transform_augmentation.get_transform(
                    numpy_random=np.random.RandomState(
                        torch.randint(low=0, high=int(2**20), size=(1,)).item()  # type: ignore
                    )
                )
                if self.transform_augmentation is not None
                else None
            )

            proprio_indices = []
            control_indices = []
            for i in range(clipped_pred_start_idx, pred_end_idx):
                if i >= length:
                    break
                if (
                    i
                    < pred_end_idx
                    - self.rollout_config.prediction_horizon
                    + self.rollout_config.proprio_obs_horizon
                ):
                    # beyond this index, won't need observations anyways
                    proprio_indices.append(i)
                control_indices.append(i)
            control_sequence = self.get_action(
                root["control"],
                slice(control_indices[0], control_indices[-1] + 1),
                transform_matrix=transform_matrix,
            )
            proprio_slice = slice(proprio_indices[0], proprio_indices[-1] + 1)
            # vision horizon will always be shorter than proprio horizon
            vision_indices = proprio_indices[-self.rollout_config.vision_obs_horizon :]
            vision_slice = slice(vision_indices[0], vision_indices[-1] + 1)
            assert len(control_sequence) > 0
            state_tensor_sequence: List[StateTensor] = []
            if self.return_point_clouds:
                chunk_num_pts: int = root["state_tensor/input_rgb_pts"].chunks[1]  # type: ignore
                num_pts = int(root["state_tensor/input_rgb_pts"].shape[1])  # type: ignore
                num_chunks: int = num_pts // chunk_num_pts  # type: ignore
                # multiply by 1.5 because some points will likely fall outside
                # the scene bounds
                num_subsample_pts = min(int(self.num_obs_pts * 1.5), num_pts)
                num_subsample_chunks: int = num_subsample_pts // chunk_num_pts  # type: ignore
                chunk_indices = np.random.choice(
                    num_chunks, num_subsample_chunks, replace=False
                ).astype(int)
                pts_indices = (
                    chunk_indices[None, :].repeat(chunk_num_pts, axis=0) * chunk_num_pts
                )
                pts_indices += np.arange(chunk_num_pts)[:, None]
                pts_indices = pts_indices.reshape(-1)
                pts_indices.sort(axis=0)
                assert len(pts_indices) == len(np.unique(pts_indices))
            else:
                pts_indices = slice(0, 1)
            obs_views = {}
            if self.return_images:
                for view_name in self.obs_cameras:
                    images = root[f"state_tensor/views/{view_name}"][vision_slice]
                    obs_views[view_name] = torch.from_numpy(images).permute(
                        0, 3, 1, 2
                    )  # t h w c -> t c h w, where t is trajectory timestep
                    # vision_slice is shorter than proprio slice, so pad left
                    obs_views[view_name] = torch.cat(
                        (
                            torch.zeros(
                                [
                                    len(proprio_indices)
                                    - self.rollout_config.vision_obs_horizon,
                                ]
                                + list(obs_views[view_name].shape[1:]),
                                dtype=obs_views[view_name].dtype,
                            ),
                            obs_views[view_name],
                        ),
                        dim=0,
                    )
                assert all(
                    len(view) == len(proprio_indices) for view in obs_views.values()
                )

            input_rgb_pts_history = root[
                "state_tensor/input_rgb_pts"
            ].get_orthogonal_selection(
                (vision_slice, pts_indices, slice(None))
            )  # type: ignore
            input_xyz_pts_history = root[
                "state_tensor/input_xyz_pts"
            ].get_orthogonal_selection(
                (vision_slice, pts_indices, slice(None))
            )  # type: ignore

            if self.return_occupancies:
                assert (
                    transform_matrix is None
                ), "Occupancy volume not supported with transform augmentation"
                occupancy_vol_history = root["state_tensor/occupancy_vol"][
                    vision_slice
                ]  # type: ignore
            else:
                # fake occupancy volume
                occupancy_vol_history = np.zeros(
                    (len(input_xyz_pts_history), 1), dtype=bool
                )

            # vision_slice is shorter than proprio slice, so pad left
            input_rgb_pts_history = np.concatenate(
                (
                    np.zeros(
                        [len(proprio_indices) - self.rollout_config.vision_obs_horizon]
                        + list(input_rgb_pts_history[0].shape),
                        dtype=input_rgb_pts_history.dtype,
                    ),
                    input_rgb_pts_history,
                ),
                axis=0,
            )
            input_xyz_pts_history = np.concatenate(
                (
                    np.zeros(
                        [len(proprio_indices) - self.rollout_config.vision_obs_horizon]
                        + list(input_xyz_pts_history[0].shape),
                        dtype=input_xyz_pts_history.dtype,
                    ),
                    input_xyz_pts_history,
                ),
                axis=0,
            )
            occupancy_vol_history = np.concatenate(
                (
                    np.zeros(
                        [len(proprio_indices) - self.rollout_config.vision_obs_horizon]
                        + list(occupancy_vol_history[0].shape),
                        dtype=occupancy_vol_history.dtype,
                    ),
                    occupancy_vol_history,
                ),
                axis=0,
            )
            assert len(input_rgb_pts_history) == len(input_xyz_pts_history) and len(
                input_rgb_pts_history
            ) == len(proprio_indices)
            for i, (
                end_effector_rot_mat,
                end_effector_position,
                gripper_command,
                input_rgb_pts,
                input_xyz_pts,
                occupancy_vol,
                t,
            ) in enumerate(
                zip(
                    root["state_tensor/end_effector_orientation"][proprio_slice],
                    root["state_tensor/end_effector_position"][proprio_slice],
                    root["state_tensor/gripper_command"][proprio_slice],
                    input_rgb_pts_history,
                    input_xyz_pts_history,
                    occupancy_vol_history,
                    root["state_tensor/time"][proprio_slice],
                )
            ):
                rgb_pts = input_rgb_pts
                xyz_pts = input_xyz_pts
                if self.pos_bounds is not None:
                    point_cloud = (
                        PointCloud(
                            xyz_pts=xyz_pts,
                            rgb_pts=rgb_pts,
                            segmentation_pts={},
                        )
                        .filter_bounds(bounds=self.pos_bounds)
                        .subsample(
                            num_pts=self.num_obs_pts,
                            numpy_random=np.random.RandomState(
                                torch.randint(
                                    low=0, high=int(2**20), size=(1,)
                                ).item()  # type: ignore
                            ),
                        )
                    )
                    rgb_pts = point_cloud.rgb_pts
                    xyz_pts = point_cloud.xyz_pts

                # use half precision to save memory
                state_tensor_sequence.append(
                    StateTensor(
                        end_effector_orientation=torch.from_numpy(
                            end_effector_rot_mat.reshape(
                                -1
                            )  # TODO support other rotation representations
                        ).half(),
                        end_effector_position=torch.from_numpy(
                            end_effector_position
                        ).half(),
                        gripper_command=torch.from_numpy(gripper_command).bool(),
                        input_rgb_pts=torch.from_numpy(rgb_pts).half() / 255.0,
                        input_xyz_pts=torch.from_numpy(xyz_pts).half(),
                        time=torch.from_numpy(t).half(),
                        occupancy_vol=torch.from_numpy(occupancy_vol),
                        views={
                            view_name: view[i] for view_name, view in obs_views.items()
                        }
                        if self.return_images
                        else {},
                    )
                )

            # to handle cases where obs_horizon extends beyond the start of the trajectory
            seq_pad_left = clipped_pred_start_idx - pred_start_idx
            obs_end_idx = -seq_pad_left + self.rollout_config.proprio_obs_horizon

            state_sequence = StateSequenceTensor(sequence=state_tensor_sequence).pad_left(
                seq_pad_left
            )
            if transform_matrix is not None:
                state_sequence = state_sequence.transform(transform_matrix)

            # NOTE, currently not filtering position bounds or subsampling points

            assert len(state_sequence.sequence) == self.rollout_config.proprio_obs_horizon
            action_pad_right_count = max(
                self.rollout_config.prediction_horizon
                - (len(control_sequence) + seq_pad_left),
                0,
            )
            action_sequence = ControlActionSequenceTensor.from_control_sequence(
                control_sequence=control_sequence,
                pad_left_count=seq_pad_left,
                pad_right_count=action_pad_right_count,
            )
            assert (
                len(action_sequence.sequence) == self.rollout_config.prediction_horizon
            ), (
                f"expected {self.rollout_config.prediction_horizon} actions,"
                + f" got {len(action_sequence.sequence)}"
            )
            return TrajectoryWindowTensor(
                state_sequence=state_sequence,
                action_sequence=action_sequence,
                task_metrics=torch.tensor([task_metric]),
                task_names=[task_desc],
            )

    # @profile
    def get_action(
        self,
        control_root,
        selector: Optional[slice] = None,
        transform_matrix: Optional[np.ndarray] = None,
    ):
        if selector is None:
            selector = slice(None)
        if self.control_representation == "joint":
            control_actions = torch.tensor(control_root["joint_pos"][selector])
        elif self.control_representation == "ee":
            ee_pos = np.array(control_root["ee_pos"][selector])
            ee_rotmat = np.array(control_root["ee_rotmat"][selector])
            if transform_matrix is not None:
                ee_poss = []
                ee_rotmats = []
                for pos, rot_mat in zip(ee_pos, ee_rotmat):
                    pos, rot_mat = affines.decompose(
                        transform_matrix @ affines.compose(T=pos, R=rot_mat, Z=np.ones(3))
                    )[:2]

                    ee_poss.append(pos)
                    ee_rotmats.append(rot_mat)
                ee_pos = np.stack(ee_poss)
                ee_rotmat = np.stack(ee_rotmats)
            if self.rotation_representation == "quat":
                ee_rot = np.stack(
                    [quaternions.mat2quat(rotmat).reshape(-1) for rotmat in ee_rotmat]
                )
            elif self.rotation_representation == "mat":
                ee_rot = ee_rotmat
            elif self.rotation_representation == "upper_mat":
                ee_rot = ee_rotmat[:, :2, :].reshape(len(ee_rotmat), 6)
            else:
                raise ValueError(
                    f"Unknown rotation representation {self.rotation_representation}"
                )
            control_actions = torch.tensor(
                np.concatenate(
                    [
                        ee_pos,
                        ee_rot.reshape(len(ee_rot), -1),
                        np.array(control_root["ee_gripper_comm"][selector][:, None]),
                    ],
                    axis=-1,
                )
            )
        else:
            raise ValueError(
                f"Unknown control representation {self.control_representation}"
            )
        return control_actions

    def reindex(self, pbar: bool = False, remote: bool = True) -> int:
        mdb_pattern = re.compile(r"\.mdb$")
        self.reset_index()
        keep_indicator = []
        paths = list(
            filter(
                mdb_pattern.findall,
                map(str, sorted(glob(self.rootdir + "/**/*", recursive=True))),
            )
        )
        if self.max_num_datapoints is not None and self.max_num_datapoints < len(paths):
            logging.info(
                f"Subsampling {len(paths)} trajectories to {self.max_num_datapoints}"
            )
            self.numpy_random.shuffle(paths)
            paths = paths[: self.max_num_datapoints]
        total = len(paths)
        if total == 0:
            logging.info(f'{self.rootdir} does not contain any ".mdb" files')
            return 0
        if pbar:
            paths = track(paths, description=f"Reindexing {self.rootdir}")
        for path in paths:
            with zarr.LMDBStore(
                path, subdir=False, lock=False, readonly=False, map_size=int(2**18)
            ) as store:
                try:
                    root = zarr.group(store=store)
                    before = len(self)
                    # as the last element to get dumped, access state tensor just to make sure
                    # file is complete
                    if any(
                        f"state_tensor/views/{view}" not in root
                        for view in self.obs_cameras
                    ):
                        continue
                    control_actions = self.get_action(root["control"])
                    min_action_value = torch.min(control_actions, dim=0).values
                    max_action_value = torch.max(control_actions, dim=0).values
                    self.reindex_helper(
                        data=(
                            root.attrs["success"],
                            root.attrs["perfect"],
                            root.attrs["subtrajectory_success_rate"],
                            root.attrs["is_inferred_task"],
                            root.attrs["control_frequency"],
                            root.attrs["length"],
                            root.attrs["task_desc"],
                            min_action_value,
                            max_action_value,
                        ),
                        path=path,
                    )
                    after = len(self)
                    keep_indicator.append(after > before)
                except lmdb.CorruptedError:
                    logging.error(path + " corrupted")
                except Exception as e:  # noqa: B902
                    logging.error(e)
        logging.info(
            f"Using {len(self)} points, from {sum(keep_indicator)} trajectories"
            + f" out of {total} ({(sum(keep_indicator)/total)*100:.01f}%)"
        )
        self.summarize()
        freq = self.control_frequencies
        if len(freq) > 0:
            logging.info(f"Control frequency: {np.mean(freq):.02f}±{np.std(freq):.02f}Hz")
        return len(self)

    def reindex_helper(
        self,
        data: Tuple[bool, bool, float, bool, int, int, str, torch.Tensor, torch.Tensor],
        path: str,
    ):
        (
            is_successful,
            is_perfect,
            subtrajectory_success_rate,
            is_inferred_task,
            control_frequency,
            length,
            task_desc,
            min_action_value,
            max_action_value,
        ) = data
        task_metric: float = 0.0
        if self.task_metric == "success":
            task_metric = float(is_successful)
        elif self.task_metric == "dense_success":
            # subtrajectory success doesn't include root task
            # this means at test time, as long as we condition
            # on a dense success that is higher than 0.5 we should
            # get robust behavior that eventually succeed
            task_metric = (
                float(subtrajectory_success_rate) * 0.5 + float(is_successful) * 0.5
            )
        elif self.task_metric == "perfect":
            task_metric = float(is_perfect and is_successful)
        else:
            raise ValueError(f"Unknown task metric {self.task_metric}")

        if self.filter_negatives and task_metric == 0.0:
            logging.debug(f"Reject {path}: not success")
            return
        elif self.filter_manually_designed_trajectories and not is_inferred_task:
            logging.debug(f"Reject {path}: not inferred task")
            return
        task_desc = (
            task_desc
            if not self.remove_with_statement
            else split_state_phrase(task_desc)[1]
        )
        if self.train_tasks is not None and task_desc not in self.train_tasks:
            logging.debug(f"Reject {path}: not in train tasks")
            return
        """
        TODO write unit test for this

        Diffusion Policy is a behavior cloning framework, which assumes
        all trajectories given to it are successful. It indexes its
        trajectories start from this
          |-|-|-|*|
                |o|o|o|o|o|o|o|o|o|o|  observation_sequence

        to this
                            |-|-|-|*|
                |o|o|o|o|o|o|o|o|o|o|  observation_sequence

        where |-|-|-|*| has length self.rollout_config.proprio_obs_horizon
        and * is the current observation.

        In the latter case, the observation sequence is then right padded
        to encourage the policy to stop at the end of the trajectory. This is done
        by repeating the last observation and action in the sequence
                            |-|-|-|*|
                |o|o|o|o|o|o|o|o|o|o|           observation_sequence
                |a|a|a|a|a|a|a|a|a|a|           action_sequence
                                   ^ this action will be repeated
        However, in our case, we have trajectories that are not successful. We do
        not want the policy to pause if it didn't finish the task. Therefore, we
        will index the trajectory up until the prediction horizon hits the last
        action. For positive trajectories, we handle identically to diffusion policy.
        For negative trajectories, we index from this
          |-|-|-|*|
                |o|o|o|o|o|o|o|o|o|o|  observation_sequence
        to this
                      |-|-|-|*|!|!|!|  < this has length of `prediction_horizon`
                |o|o|o|o|o|o|o|o|o|o|  observation_sequence
        where all ! marks future observations the policy has to predict the actions
        for, including the *.
        """
        self.control_frequencies.append(control_frequency)
        if task_metric == 1.0:
            # NOTE: even for dense_success, we'll handle the same way. We'll drop
            # a few data points to prioritize correctness
            self.path_obs_idx_tuples.extend((path, i) for i in range(length))
            self.task_metrics.extend([task_metric] * length)
            self.task_names.extend([task_desc] * length)
        else:
            if length < self.rollout_config.prediction_horizon:
                # this is a failed trajectory that's too short, and will get
                # padded anyways. We'll just drop it.
                return
            # these are failed trajectories with long enough length.
            length -= self.rollout_config.prediction_horizon
            self.path_obs_idx_tuples.extend((path, i) for i in range(length))
            self.task_metrics.extend([task_metric] * length)
            self.task_names.extend([task_desc] * length)

        if self.min_action_values is None or self.max_action_values is None:
            self.min_action_values = min_action_value
            self.max_action_values = max_action_value
        else:
            self.min_action_values = torch.min(
                torch.cat(
                    [self.min_action_values[None, :], min_action_value[None, :]],
                    dim=0,
                ),
                dim=0,
            ).values
            self.max_action_values = torch.max(
                torch.cat(
                    [self.max_action_values[None, :], max_action_value[None, :]],
                    dim=0,
                ),
                dim=0,
            ).values
        self.path_info[path] = (task_desc, task_metric)
