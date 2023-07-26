from __future__ import annotations
from abc import abstractmethod, abstractproperty
import zipfile

import logging
import re
import typing
from copy import deepcopy
from dataclasses import fields
from glob import glob
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union, cast
from matplotlib import pyplot as plt
import ray
from rich.progress import track
import numpy as np
import torch
from pydantic import dataclasses, validator
from transforms3d import affines, euler, quaternions
from scalingup.algo.tsdf import obs_to_tsdf_vol
from scalingup.algo.virtual_grid import Point3D, VirtualGrid
from scalingup.utils.core import Action, PartialObservation, split_state_phrase
from scalingup.utils.generic import AllowArbitraryTypes, limit_threads
from scalingup.utils.core import (
    EndEffectorAction,
    Observation,
    PointCloud,
    Pose,
    Trajectory,
)
import torchvision


def transform_position(position, matrix):
    assert matrix.shape == (
        4,
        4,
    ), f"expected 4x4 transformation matrix but got {matrix.shape}"
    assert type(position) == torch.Tensor
    dtype = position.dtype
    # float16 not fully supported, so do transform in float32
    matrix = torch.tensor(matrix).to(device=position.device, dtype=torch.float32)
    position = position.to(dtype=torch.float32)
    position = torch.cat((position, torch.ones_like(position[..., [0]])), dim=-1)
    if len(position.shape) > 1:
        transformed_position = (matrix @ position.mT).mT[..., :3]
    else:
        transformed_position = (matrix @ position)[..., :3]
    return transformed_position.to(dtype=dtype)


def transform_orientation(orientation, matrix):
    assert matrix.shape == (
        4,
        4,
    ), f"expected 4x4 transformation matrix but got {matrix.shape}"
    assert orientation.shape[-1] == 9, "Expected orientation in flattened 3x3 matrix"
    if len(orientation.shape) == 2:
        return torch.stack([transform_orientation(o, matrix) for o in orientation])
    else:
        rotmat = affines.decompose(matrix)[1]
        transformed_mat = orientation.cpu().numpy().reshape(3, 3) @ rotmat
        return torch.from_numpy(transformed_mat.reshape(-1)).to(
            orientation.device, orientation.dtype
        )


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class TensorDataClass:
    @abstractproperty
    def dtype(self) -> torch.dtype:
        pass

    @abstractproperty
    def device(self) -> torch.device:
        pass

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = True,
    ):
        if device is None:
            device = torch.device("cuda")
        attrs = list(self.__class__.__annotations__.keys())
        kwargs = {}
        for attr in attrs:
            if hasattr(getattr(self, attr), "to"):
                kwargs[attr] = getattr(self, attr).to(
                    device=device, dtype=dtype, non_blocking=non_blocking
                )
            elif type(getattr(self, attr)) == dict and (
                len(getattr(self, attr)) == 0
                or hasattr(next(iter(getattr(self, attr).values())), "to")
            ):
                assert attr == "views"
                kwargs[attr] = {
                    k: v.to(
                        device=device,
                        dtype=None,  # Don't cast images
                        non_blocking=non_blocking,
                    )
                    for k, v in getattr(self, attr).items()
                }
            else:
                kwargs[attr] = getattr(self, attr)
        return self.__class__(**kwargs)


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class StateTensor(TensorDataClass):
    end_effector_position: torch.Tensor
    end_effector_orientation: torch.Tensor
    gripper_command: torch.Tensor
    input_xyz_pts: torch.Tensor
    input_rgb_pts: torch.Tensor
    occupancy_vol: torch.Tensor
    time: torch.Tensor
    views: Dict[str, torch.Tensor]

    @property
    def dtype(self) -> torch.dtype:
        return self.end_effector_position.dtype

    @property
    def device(self) -> torch.device:
        return self.end_effector_position.device

    @property
    def batch_size(self) -> int:
        if len(self.end_effector_position.shape) != 2:
            raise ValueError("Tensor Data is not batched")
        return self.end_effector_position.shape[0]

    @property
    def is_batched(self) -> bool:
        return len(self.end_effector_position.shape) == 2

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = True,
    ) -> StateTensor:
        return typing.cast(
            StateTensor, super().to(device=device, dtype=dtype, non_blocking=non_blocking)
        )

    @validator("end_effector_position")
    @classmethod
    def pos_is_three_dimensional(cls, v: torch.Tensor):
        if v.shape[-1] != 3:
            raise ValueError("end_effector_position must be a point in 3D")
        return v

    @classmethod
    def from_obs(
        cls,
        partial_obs: PartialObservation,
        numpy_random: np.random.RandomState,
        obs_cameras: List[str],
        pos_bounds: Optional[Tuple[Point3D, Point3D]] = None,
        num_obs_pts: Optional[int] = None,
        transform_matrix: Optional[np.ndarray] = None,
        voxel_dim: Optional[float] = None,
        occupancy_dim: Optional[Tuple[int, int, int]] = None,
    ) -> StateTensor:
        point_clouds = [
            partial_obs.images[camera_name].point_cloud for camera_name in obs_cameras
        ]
        point_cloud = sum(point_clouds[1:], point_clouds[0])
        occupancy_vol = np.empty((0, 0), dtype=bool)
        if pos_bounds is not None:
            point_cloud = point_cloud.filter_bounds(bounds=pos_bounds)
        if occupancy_dim is not None:
            assert pos_bounds is not None
            tsdf_vol = obs_to_tsdf_vol(
                obs=partial_obs,
                virtual_grid=VirtualGrid(
                    scene_bounds=pos_bounds,
                    grid_shape=occupancy_dim,
                    batch_size=1,
                ),
                use_gpu=False,
            )
            tsdf, _ = tsdf_vol.get_volume()
            occupancy_vol = tsdf < 0
        if voxel_dim is not None:
            point_cloud = point_cloud.voxel_downsample(
                voxel_dim=voxel_dim, skip_segmentation=True
            )
        if num_obs_pts is not None:
            point_cloud = point_cloud.subsample(
                num_pts=num_obs_pts, numpy_random=numpy_random
            )
        ee_pose = deepcopy(partial_obs.state.end_effector_pose)
        if transform_matrix is not None:
            ee_pose = ee_pose.transform(transform_matrix=transform_matrix)
        return StateTensor(
            gripper_command=torch.tensor([partial_obs.state.gripper_command]),
            end_effector_position=torch.from_numpy(ee_pose.position),
            end_effector_orientation=torch.tensor(
                quaternions.quat2mat(ee_pose.orientation).reshape(-1)
            ),
            input_xyz_pts=transform_position(
                torch.from_numpy(point_cloud.xyz_pts), matrix=transform_matrix
            )
            if transform_matrix is not None
            else torch.from_numpy(point_cloud.xyz_pts.copy()),
            input_rgb_pts=torch.from_numpy(point_cloud.rgb_pts.copy()) / 255,
            occupancy_vol=torch.from_numpy(occupancy_vol),
            time=torch.from_numpy(np.array([partial_obs.time])),
            views={
                camera_name: torch.from_numpy(
                    partial_obs.images[camera_name].rgb.copy()
                ).permute(
                    2, 0, 1
                )  # HWC to CHW
                for camera_name in obs_cameras
            },
        )

    def get_proprioception_data(self, proprioception_keys: Set[str]) -> torch.Tensor:
        data: List[torch.Tensor] = []
        assert len(proprioception_keys) > 0, "proprioception can't be empty"
        for key in sorted(proprioception_keys):
            data.append(getattr(self, key))
        return torch.cat(data, dim=-1)

    def transform(self, matrix: np.ndarray) -> StateTensor:
        assert not self.occupancy_vol.any(), "transforming occupancy_vol not supported"
        return StateTensor(
            end_effector_position=transform_position(
                position=self.end_effector_position,
                matrix=matrix,
            ),
            end_effector_orientation=transform_orientation(
                orientation=self.end_effector_orientation, matrix=matrix
            ),
            gripper_command=self.gripper_command,
            input_xyz_pts=transform_position(position=self.input_xyz_pts, matrix=matrix),
            input_rgb_pts=self.input_rgb_pts,
            occupancy_vol=self.occupancy_vol,  # NOTE: not supported
            time=self.time,
            views=self.views,
        )

    def show(self):
        fig, axes = plt.subplots(len(self.views))
        if len(self.views) == 1:
            axes = np.array([axes])
        for i, (camera_name, view) in enumerate(self.views.items()):
            axes[i].imshow(view.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            axes[i].set_title(camera_name)
            axes[i].axis("off")
        plt.show()
        # TODO show point clouds as well


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class EndEffectorActionTensor(TensorDataClass):
    gripper_command: torch.Tensor
    end_effector_position: torch.Tensor
    end_effector_orientation_euler: torch.Tensor
    allow_contact: torch.Tensor

    @property
    def dtype(self) -> torch.dtype:
        return self.end_effector_position.dtype

    @property
    def device(self) -> torch.device:
        return self.end_effector_position.device

    def __eq__(self, other):
        return all(
            torch.allclose(getattr(self, field.name), getattr(other, field.name))
            for field in fields(EndEffectorActionTensor)
        )

    @validator(
        "end_effector_position",
    )
    @classmethod
    def pos_is_three_dimensional(cls, v: torch.Tensor):
        if v.shape[-1] != 3:
            raise ValueError("end_effector_position must be a point in 3D")
        return v

    @validator(
        "end_effector_orientation_euler",
    )
    @classmethod
    def orn_is_three_dimensional(cls, v: torch.Tensor):
        if v.shape[-1] != 3:
            raise ValueError("end_effector_orientation_euler must be an 3D euler")
        return v

    def transform(self, matrix: np.ndarray) -> EndEffectorActionTensor:
        return EndEffectorActionTensor(
            gripper_command=self.gripper_command,
            allow_contact=self.allow_contact,
            end_effector_position=transform_position(
                position=self.end_effector_position,
                matrix=matrix,
            ),
            end_effector_orientation_euler=transform_orientation(
                orientation=self.end_effector_orientation_euler, matrix=matrix
            ),
        )

    @classmethod
    def from_ee_action(cls, ee_action: EndEffectorAction) -> EndEffectorActionTensor:
        return EndEffectorActionTensor(
            gripper_command=torch.tensor([ee_action.gripper_command]),
            allow_contact=torch.tensor([ee_action.allow_contact]),
            end_effector_position=torch.tensor(ee_action.end_effector_position),
            end_effector_orientation_euler=torch.tensor(
                euler.quat2euler(ee_action.end_effector_orientation)
            )
            * 180
            / np.pi,
        )


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class TrajectoryStepTensor(TensorDataClass):
    state: StateTensor
    action: EndEffectorActionTensor
    task_metrics: torch.Tensor
    task_names: List[str]

    @property
    def dtype(self) -> torch.dtype:
        return self.state.dtype

    @property
    def device(self) -> torch.device:
        return self.state.device

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = True,
    ) -> TrajectoryStepTensor:
        return typing.cast(
            TrajectoryStepTensor,
            super().to(device=device, dtype=dtype, non_blocking=non_blocking),
        )

    def __eq__(self, other):
        return (
            self.state == other.state
            and self.action == other.action
            and (self.task_metrics == other.task_metrics).all()
            and all(t1 == t2 for t1, t2 in zip(self.task_names, other.task_names))
        )

    @property
    def batched(self):
        return len(self.state.end_effector_position.shape) > 1

    def transform(self, matrix: np.ndarray) -> TrajectoryStepTensor:
        return TrajectoryStepTensor(
            state=self.state.transform(matrix=matrix),
            action=self.action.transform(matrix=matrix),
            task_metrics=self.task_metrics,
            task_names=self.task_names,
        )

    @property
    def visualization(self) -> PointCloud:
        assert (
            not self.batched
        ), "visualization only supported for unbatched `TrajectoryStepTensors`"
        obs = PointCloud(
            xyz_pts=self.state.input_xyz_pts.cpu().numpy(),
            rgb_pts=(self.state.input_rgb_pts.cpu().numpy() * 255).astype(np.uint8),
            segmentation_pts={},
        )
        action = PointCloud(
            xyz_pts=torch.stack(
                [
                    # self.next_state.end_effector_position,
                    self.state.end_effector_position,
                ]
            )
            .cpu()
            .numpy(),
            rgb_pts=np.array([[255, 0, 0], [0, 255, 0]]).astype(np.uint8),
            segmentation_pts={},
        )
        return obs + action

    @classmethod
    def collate(cls, batch: Sequence[TrajectoryStepTensor]) -> TrajectoryStepTensor:
        # NOTE: this is a hacky and non pythonic way to go around
        # dataclasses's deepcopy when doing a asdict
        attrs = list(StateTensor.__annotations__.keys())
        state = StateTensor(
            **torch.utils.data.default_collate(  # type: ignore
                [
                    {attr: getattr(traj_step.state, attr) for attr in attrs}
                    for traj_step in batch
                ]
            )
        )  # type: ignore
        attrs = list(EndEffectorActionTensor.__annotations__.keys())
        action = EndEffectorActionTensor(
            **torch.utils.data.default_collate(  # type: ignore
                [  # type: ignore
                    {attr: getattr(traj_step.action, attr) for attr in attrs}
                    for traj_step in batch
                ]
            )
        )
        task_metrics = torch.utils.data.default_collate(  # type: ignore
            [traj_step.task_metrics for traj_step in batch]
        )
        task_names: List[str] = sum(
            (traj_step.task_names for traj_step in batch), start=[]
        )
        return TrajectoryStepTensor(
            state=state,
            action=action,
            # next_state=next_state,
            task_metrics=task_metrics,
            task_names=task_names,
        )

    @classmethod
    def from_traj_step(
        cls,
        obs: Observation,
        action: Action,
        task_name: str,
        task_metric: Union[bool, float],
        pos_bounds: Optional[Tuple[Point3D, Point3D]],
        num_obs_pts: int,
        numpy_random: np.random.RandomState,
        obs_cameras: List[str],
        transform_matrix: Optional[np.ndarray] = None,
    ) -> TrajectoryStepTensor:
        ee_action = cast(EndEffectorAction, action)
        ee_pose = Pose(
            position=ee_action.end_effector_position,
            orientation=ee_action.end_effector_orientation,
        )
        if transform_matrix is not None:
            ee_pose = ee_pose.transform(transform_matrix=transform_matrix)
        return TrajectoryStepTensor(
            task_names=[task_name],
            state=StateTensor.from_obs(
                partial_obs=PartialObservation.from_obs(obs=obs),
                pos_bounds=pos_bounds,
                num_obs_pts=num_obs_pts,
                numpy_random=numpy_random,
                obs_cameras=obs_cameras,
                transform_matrix=transform_matrix,
            ),
            action=EndEffectorActionTensor(
                gripper_command=torch.tensor([ee_action.gripper_command]),
                allow_contact=torch.tensor([ee_action.allow_contact]),
                end_effector_position=torch.tensor(ee_pose.position),
                end_effector_orientation_euler=torch.tensor(
                    euler.quat2euler(ee_pose.orientation)
                )
                * 180
                / np.pi,
            ),
            task_metrics=torch.tensor([task_metric]).bool(),
        )


@dataclasses.dataclass(frozen=True)
class TransformAugmentation:
    position_magnitude: Point3D
    orientation_magnitude: Point3D  # radians
    scale_magnitude: Point3D

    def get_transform(self, numpy_random: np.random.RandomState):
        position_magnitude_arr = np.array(self.position_magnitude)
        orientation_magnitude_arr = np.array(self.orientation_magnitude)
        scale_magnitude_arr = np.array(self.scale_magnitude)
        return affines.compose(
            T=numpy_random.uniform(
                low=-position_magnitude_arr / 2, high=position_magnitude_arr / 2
            ),
            R=euler.euler2mat(
                *numpy_random.uniform(
                    low=-orientation_magnitude_arr / 2,
                    high=orientation_magnitude_arr / 2,
                )
            ),
            Z=np.ones(3)
            + numpy_random.uniform(
                low=-scale_magnitude_arr / 2, high=scale_magnitude_arr / 2
            ),
        )


@ray.remote
def check_path_trajectory(
    path: str,
    pos_bounds: Optional[Tuple[Point3D, Point3D]],
    orn_bounds: Optional[Tuple[Point3D, Point3D]],
) -> Tuple[bool, bool, float, List[int], str]:
    traj = Trajectory.load(path).flatten()
    is_successful = traj.is_successful
    in_bounds = True
    if pos_bounds is not None and orn_bounds is not None:
        in_bounds = traj.all_ee_in_bounds(
            pos_bounds=pos_bounds,
            orn_bounds=orn_bounds,
        )
    control_states = [len(traj_step.control_states) + 1 for traj_step in traj.episode]
    return in_bounds, is_successful, traj.duration, control_states, path


class ReplayBuffer(torch.utils.data.Dataset):  # type: ignore
    def __init__(
        self,
        rootdir: str,
        num_obs_pts: int,
        obs_cameras: List[str],
        pos_bounds: Optional[Tuple[Point3D, Point3D]],
        proprioception_keys: Set[str],
        filter_negatives: bool,
        # NOTE: grasp object is also currently a manually designed task
        filter_manually_designed_trajectories: bool,
        seed: int,
        batch_size: int,
        num_steps_per_update: int,
        collate_fn: Callable,
        balance_positive_negative: bool,
        task_metric: str,
        balance_tasks: bool,
        remove_with_statement: bool,
        control_representation: str,  # joint v.s. ee
        rotation_representation: str,  # quat v.s. upper_rot_mat v.s. mat
        obs_dim: Tuple[int, int],
        return_point_clouds: bool = False,
        return_images: bool = False,
        return_occupancies: bool = False,
        transform_augmentation: Optional[TransformAugmentation] = None,
        max_num_datapoints: Optional[int] = None,
        train_tasks: Optional[List[str]] = None,
        num_workers: int = 32,
        should_reindex: bool = False,
    ):
        limit_threads(n=1)  # prevent decompressors from using too many threads
        if transform_augmentation is not None and control_representation == "joint":
            raise ValueError(
                "Transform augmentation is not supported for joint control representation"
            )
        self.rootdir = rootdir
        self.pos_bounds = pos_bounds
        self.train_tasks = train_tasks
        self.max_num_datapoints = max_num_datapoints
        self.collate_fn = collate_fn
        self.proprioception_keys = proprioception_keys
        self.num_obs_pts = num_obs_pts
        self.batch_size = batch_size
        self.num_steps_per_update = num_steps_per_update
        self.num_workers = num_workers
        self.obs_cameras = obs_cameras
        assert len(self.obs_cameras) > 0

        self.filter_negatives = filter_negatives
        self.filter_manually_designed_trajectories = filter_manually_designed_trajectories
        self.numpy_random = np.random.RandomState(seed=seed)
        self.balance_positive_negative = balance_positive_negative
        self.return_point_clouds = return_point_clouds
        self.return_images = return_images
        self.return_occupancies = return_occupancies
        self.task_metric = task_metric
        self.balance_tasks = balance_tasks
        self.control_representation = control_representation
        self.rotation_representation = rotation_representation
        self.transform_augmentation = transform_augmentation
        self.obs_dim = obs_dim
        logging.info(
            f"""filter_negatives: {self.filter_negatives}
filter_manually_designed_trajectories: {self.filter_manually_designed_trajectories}
balance_positive_negative: {self.balance_positive_negative}
task_metric: {self.task_metric}
balance_tasks: {self.balance_tasks}
control_representation: {self.control_representation}
rotation_representation: {self.rotation_representation}
return_point_clouds: {self.return_point_clouds}
return_images: {self.return_images}
return_occupancies: {self.return_occupancies}
transform_augmentation: {self.transform_augmentation}
obs_cameras: {self.obs_cameras}
"""
        )
        if self.task_metric == "dense_success" and self.balance_positive_negative:
            raise ValueError(
                "Cannot balance positive and negative trajectories when using float dense_success"
            )
        self.remove_with_statement = remove_with_statement
        self.task_metrics: List[Union[bool, float]] = []
        self.task_names: List[str] = []
        if self.return_point_clouds:
            assert self.pos_bounds is not None
        if should_reindex:
            self.reindex(pbar=True)

    def reset_index(self):
        self.task_metrics.clear()
        self.task_names.clear()

    @abstractmethod
    def reindex_helper(self, data: Any, path: str):
        pass

    @property
    def file_pattern(self) -> re.Pattern:
        return re.compile(r"traj-\d{5}-\d{5}")

    @property
    def file_loader(self) -> Callable[[str], Any]:
        return lambda path: Trajectory.load(path=path).flatten()

    def reindex(self, pbar: bool = False, remote: bool = True) -> int:
        # TODO efficiently handle large and dynamically growing replay buffer
        paths = list(
            filter(
                self.file_pattern.findall,
                map(str, sorted(glob(self.rootdir + "/**/*", recursive=True))),
            )
        )
        if len(paths) == 0:
            logging.warning(f"No trajectories found under path `{self.rootdir}`")
            return 0
        self.reset_index()
        keep_indicator = []
        if remote:
            fn = ray.remote(self.file_loader)
            tasks = [fn.remote(path=path) for path in paths]  # type: ignore
            tasks_iter = tasks
            if pbar:
                tasks_iter = track(tasks, description=f"Reindexing {self.rootdir}")
            for task, path in zip(tasks_iter, paths):
                ray.wait(tasks, timeout=1e-6)
                # wait on all tasks to make sure all tasks progress
                try:
                    before = len(self)
                    ready, not_ready = ray.wait([task], timeout=10)
                    if len(ready) > 0:
                        self.reindex_helper(data=ray.get(task), path=path)
                    else:
                        logging.error("failed to get task after timeout...")
                        keep_indicator.append(False)
                        continue
                    after = len(self)
                    keep_indicator.append(after > before)
                except zipfile.BadZipFile:
                    continue
                except Exception as e:  # noqa: B902
                    logging.error(e)
                    continue
        else:
            for path in paths:
                before = len(self)
                self.reindex_helper(data=self.file_loader(path), path=path)
                after = len(self)
                keep_indicator.append(after > before)
        logging.info(
            f"Using {len(self)} points, from {sum(keep_indicator)} trajectories"
            + f" out of {len(paths)} ({(sum(keep_indicator)/len(paths))*100:.01f}%)"
        )
        self.summarize()
        return len(self)

    @property
    def summary(self) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        metrics_by_tasks: Dict[str, List[float]] = {
            task_name: [] for task_name in set(self.task_names)
        }
        for task_name, task_metric in zip(self.task_names, self.task_metrics):
            metrics_by_tasks[task_name].append(float(task_metric))
        for task_name, task_metrics in metrics_by_tasks.items():
            summary[f"{task_name}/mean"] = float(np.mean(task_metrics))
            summary[f"{task_name}/std"] = float(np.std(task_metrics))
            summary[f"{task_name}/count"] = float(len(task_metrics))
        summary["num_datapoints"] = float(len(self))
        return summary

    def summarize(self):
        # count up average task metrics for each task
        for k, v in self.summary.items():
            logging.info(f"{k!r}: {v:.02f}")

    def __len__(self) -> int:
        return len(self.task_metrics)

    def get_sampler_weights(self) -> torch.Tensor:
        weights = np.ones(len(self), dtype=np.float64)
        positive_mask = np.array(self.task_metrics).astype(bool)
        if not self.balance_tasks:
            if self.balance_positive_negative:
                num_success = sum(self.task_metrics)
                if sum(self.task_metrics) < len(self) and num_success > 0:
                    weights[positive_mask] = 1 / num_success
                    weights[~positive_mask] = 1 / (len(self) - num_success)
        elif self.balance_tasks:
            task_names = sorted(set(self.task_names))
            num_tasks = len(task_names)
            for task_name in task_names:
                task_mask = np.array(self.task_names) == task_name
                num_task_trajs = sum(task_mask)
                if self.balance_positive_negative:
                    task_positive_mask = positive_mask[task_mask]
                    num_success = sum(task_positive_mask)
                    if num_success > 0 and num_success < num_task_trajs:
                        weights[task_mask][task_positive_mask] = 1 / num_success
                        weights[task_mask][~task_positive_mask] = 1 / (
                            num_task_trajs - num_success
                        )
                # sum up to 1/num_tasks
                weights[task_mask] /= weights[task_mask].sum() * num_tasks
        return weights / weights.sum()

    def get_loader(
        self,
        num_steps_per_update: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        shuffle: bool = True,
        repeat: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        if len(self) == 0:
            logging.error(f"`ReplayBuffer` at {self.rootdir} is empty!")
            exit(1)
        if batch_size is None:
            batch_size = self.batch_size
        if num_steps_per_update is None:
            num_steps_per_update = self.num_steps_per_update
        weights = self.get_sampler_weights()
        sampler = (
            torch.utils.data.WeightedRandomSampler(  # type: ignore
                weights=weights,
                replacement=True,
                num_samples=int(num_steps_per_update * batch_size),
            )
            if repeat
            else None
        )
        if num_workers is None:
            num_workers = self.num_workers
        return torch.utils.data.DataLoader(  # type: ignore
            self,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=persistent_workers and num_workers > 0,
        )


class TrajectoryStepReplayBuffer(ReplayBuffer):
    def __init__(self, orn_bounds: Optional[Tuple[Point3D, Point3D]], **kwargs):
        super().__init__(collate_fn=TrajectoryStepTensor.collate, **kwargs)
        self.orn_bounds = orn_bounds
        self.path_step_tuples: List[Tuple[str, int, int]] = []

    def reset_index(self):
        super().reset_index()
        self.path_step_tuples.clear()

    def reindex_helper(self, data: Any, path: str):
        traj = typing.cast(Trajectory, data)
        traj = traj.flatten()
        task_metric: Union[bool, float] = 0.0
        if self.task_metric == "success":
            task_metric = traj.is_successful
        elif self.task_metric == "dense_success":
            task_metric = traj.compute_subtrajectory_success_rate()
        elif self.task_metric == "perfect":
            task_metric = traj.compute_subtrajectory_success_rate() == 1.0
        else:
            raise ValueError(f"Unknown task metric {self.task_metric}")
        in_bounds = True
        if self.pos_bounds is not None and self.orn_bounds is not None:
            in_bounds = traj.all_ee_in_bounds(
                pos_bounds=self.pos_bounds,
                orn_bounds=self.orn_bounds,
            )
        if in_bounds and (not self.filter_negatives or task_metric):
            control_states_len = [
                len(traj_step.control_states) + 1 for traj_step in traj.episode
            ]
            for i in range(len(traj)):
                for j in range(control_states_len[i]):
                    self.path_step_tuples.append((path, i, j))
            self.task_metrics.extend([task_metric] * sum(control_states_len))
            self.task_names.extend(
                [
                    traj.task.desc
                    if not self.remove_with_statement
                    else split_state_phrase(traj.task.desc)[1]
                ]
                * sum(control_states_len)
            )
        else:
            logging.debug(path + " rejected")

    def __getitem__(self, idx):
        path, episode_step, control_state_idx = self.path_step_tuples[idx]
        traj = Trajectory.load(path).flatten()
        traj_step = traj.episode[episode_step]
        if control_state_idx == 0:
            obs = traj_step.obs
        else:
            obs = traj_step.control_state[control_state_idx - 1]
        task_metric = self.task_metrics[idx]
        return TrajectoryStepTensor.from_traj_step(
            obs=obs,
            action=traj_step.action,
            pos_bounds=self.pos_bounds,
            task_name=traj.task.desc,
            task_metric=task_metric,
            num_obs_pts=self.num_obs_pts,
            numpy_random=self.numpy_random,
            obs_cameras=self.obs_cameras,
            transform_matrix=self.transform_augmentation.get_transform(
                numpy_random=self.numpy_random
            )
            if self.transform_augmentation is not None
            else None,
        )


class TrajectoryReplayBuffer(ReplayBuffer):
    def __init__(self, orn_bounds: Optional[Tuple[Point3D, Point3D]], **kwargs):
        super().__init__(collate_fn=lambda x: x, **kwargs)
        self.orn_bounds = orn_bounds
        self.paths: List[str] = []

    def reset_index(self):
        super().reset_index()
        self.paths.clear()

    def __getitem__(self, idx: int):
        return Trajectory.load(self.paths[idx][0])

    def get_loader(self, **kwargs):
        kwargs = {**kwargs, "batch_size": 1}
        return super().get_loader(**kwargs)

    def reindex_helper(self, data: Any, path: str):
        traj = typing.cast(Trajectory, data)
        traj = traj.flatten()
        is_successful = traj.is_successful
        in_bounds = True
        if self.pos_bounds is not None and self.orn_bounds is not None:
            in_bounds = traj.all_ee_in_bounds(
                pos_bounds=self.pos_bounds,
                orn_bounds=self.orn_bounds,
            )
        if in_bounds and (not self.filter_negatives or is_successful):
            self.paths.append(path)
            self.task_metrics.append(is_successful)
            self.task_names.append(
                traj.task.desc
                if not self.remove_with_statement
                else split_state_phrase(traj.task.desc)[1]
            )
        else:
            logging.debug(path + " rejected")
