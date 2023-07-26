from __future__ import annotations
import gc
import io
import itertools
import logging
import lzma
import os
import pickle
import random
import textwrap
import typing
from abc import ABC, abstractmethod, abstractproperty
from copy import copy
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import brotli
import cv2
import dill
import imageio
import matplotlib.pyplot as plt
import mgzip
import numpy as np
import open3d as o3d
import pandas as pd
import ray
import torch
import zarr
from imagecodecs.numcodecs import Jpeg2k
from matplotlib.patches import Patch
from numcodecs import Blosc, register_codec, VLenUTF8
from PIL import Image
from pydantic import dataclasses, validator, ValidationError
from rich.console import Console
from rich.syntax import Syntax
from rich.tree import Tree
from scipy.spatial.transform import Rotation, Slerp
from transforms3d import affines, euler, quaternions
from wandb import Html  # type: ignore
from wandb import Video  # type: ignore
from wandb import Image as WandbImage  # type: ignore
import pytorch3d.transforms as pt3d
from scalingup.algo.end_effector_policy_utils import Discretizer
from scalingup.algo.virtual_grid import Point3D
from scalingup.utils.constants import LINK_SEPARATOR_TOKEN, MJCF_NEST_TOKEN
from scalingup.utils.generic import AllowArbitraryTypes, add_text_to_image, exec_safe
import dm_control

register_codec(Jpeg2k)

Pixel = Tuple[int, int]
RGB = Tuple[float, float, float]

# is gripper symmetric about the z-axis
# by 180 degrees?
SYMMETRIC_GRIPPER = True

"""
this can be changed in mujoco's xml
<visual>
  <global offwidth="my_width"/>
</visual>
"""
MAX_OFFSCREEN_HEIGHT = 640

VISUAL_GEOM_GROUP = 2
PCD_CHUNK_SIZE: int = 128


def split_state_phrase(sentence: str) -> Tuple[str, str]:
    # if the first phrase of the sentence contains a with statement, then remove it
    # TODO use rephrasing LLM
    if not sentence.lower().startswith("with"):
        return "", sentence
    if len(sentence.split(", ")) > 2:
        logging.debug(f"potentially ambiguous sentence split: {sentence.split(',')}")
    # e.g., with the {phrase 1}, the {phrase 2}, and the {phrase 3},
    # {verb} {phrase 4}, {phrase 5} etc.
    idx = 0
    phrases = sentence.split(", ")
    for phrase in phrases:
        if any(
            phrase.startswith(keyword.strip().lower())
            for keyword in ["with", "the", "and"]
        ):
            idx += 1
            continue
    state_phrases = phrases[:idx]

    action_phrases = phrases[idx:]
    logging.debug(f'{", ".join(state_phrases)!r} | {", ".join(action_phrases)!r}')
    return ", ".join(state_phrases) + ", ", ", ".join(action_phrases)


class Picklable:
    def dump(
        self,
        path: str,
        protocol: Any = mgzip.open,
        pickled_data_compressor: Any = None,
        protocol_kwargs: Optional[Dict[str, Any]] = None,
        compressor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if protocol_kwargs is None:
            protocol_kwargs = {}
            if protocol == mgzip.open:
                protocol_kwargs["thread"] = 8
        if pickled_data_compressor is not None:
            if compressor_kwargs is None:
                compressor_kwargs = {}
            with open(path, "wb", **protocol_kwargs) as f:
                pickled_data = dill.dumps(self)
                compressed_pickled = pickled_data_compressor(
                    pickled_data, **compressor_kwargs
                )
                f.write(compressed_pickled)
        else:
            with protocol(path, "wb", **protocol_kwargs) as f:
                dill.dump(self, f)

    @classmethod
    def load(
        cls,
        path: str,
        protocol: Any = mgzip.open,
        pickled_data_decompressor: Any = None,
        protocol_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Picklable:
        if protocol_kwargs is None:
            protocol_kwargs = {}
            if protocol == mgzip.open:
                protocol_kwargs["thread"] = 8
        if pickled_data_decompressor is not None:
            with open(path, "rb", **protocol_kwargs) as f:
                compressed_pickle = f.read()
                try:
                    decompressed_pickle = pickled_data_decompressor(compressed_pickle)
                except brotli.Error as e:
                    raise ValueError("Invalid encoder format") from e
                obj = pickle.loads(decompressed_pickle)
        else:
            with protocol(path, "rb", **protocol_kwargs) as f:
                try:
                    obj = pickle.load(f)
                except lzma.LZMAError as e:
                    raise ValueError("Invalid encoder format") from e
        if type(obj) != cls:
            raise ValueError(f"Pickle file {path} is not a `{cls.__name__}`")
        return obj


@dataclasses.dataclass(frozen=True)
class DegreeOfFreedomRange:
    upper: float
    lower: float


QPosRange = List[DegreeOfFreedomRange]


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class Velocity(Picklable):
    linear_velocity: np.ndarray  # shape: (3, )
    angular_velocity: np.ndarray  # shape: (3, )

    @validator("linear_velocity")
    @classmethod
    def linear_velocity_shape(cls, v: np.ndarray):
        if v.shape != (3,):
            raise ValueError("linear_velocity must be 3D")
        return v

    @validator("angular_velocity")
    @classmethod
    def angular_velocity_shape(cls, v: np.ndarray):
        if v.shape != (3,):
            raise ValueError("angular_velocity must be 3D")
        return v

    @property
    def flattened(self) -> List[float]:
        return list(self.linear_velocity) + list(self.angular_velocity)

    def __hash__(self) -> int:
        return hash(
            (
                *self.linear_velocity.tolist(),
                *self.angular_velocity.tolist(),
            )
        )


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class Pose(Picklable):
    position: np.ndarray  # shape: (3, )
    orientation: np.ndarray  # shape: (4, ), quaternion

    def __hash__(self) -> int:
        return hash((*self.position.tolist(), *self.orientation.tolist()))

    @validator("position")
    @classmethod
    def position_shape(cls, v: np.ndarray):
        if v.shape != (3,):
            raise ValueError("position must be 3D")
        return v

    @validator("orientation")
    @classmethod
    def orientation_shape(cls, v: np.ndarray):
        if v.shape != (4,):
            raise ValueError("orientation must be a 4D quaternion")
        return v

    @property
    def flattened(self) -> List[float]:
        return list(self.position) + list(self.orientation)

    def __eq__(self, other) -> bool:
        return bool(
            np.allclose(self.position, other.position)
            and np.allclose(self.orientation, other.orientation)
        )

    @property
    def matrix(self) -> np.ndarray:
        return affines.compose(
            T=self.position, R=quaternions.quat2mat(self.orientation), Z=np.ones(3)
        )

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> Pose:
        T, R = affines.decompose(matrix)[:2]
        return Pose(position=T.copy(), orientation=quaternions.mat2quat(R.copy()))

    def transform(self, transform_matrix: np.ndarray) -> Pose:
        assert transform_matrix.shape == (
            4,
            4,
        ), f"expected 4x4 transformation matrix but got {transform_matrix.shape}"
        T, R, _, _ = affines.decompose(transform_matrix @ self.matrix)
        return Pose(position=T, orientation=quaternions.mat2quat(R))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        pos_str = ",".join(f"{x:.05f}" for x in self.position)
        rot_str = ",".join(f"{x:.05f}" for x in euler.quat2euler(self.orientation))
        return f"Pose(pos=({pos_str}),rot=({rot_str}))"

    @staticmethod
    def orientation_distance(q1: np.ndarray, q2: np.ndarray) -> float:
        diff = (
            Rotation.from_quat(q1[[1, 2, 3, 0]])
            * Rotation.from_quat(q2[[1, 2, 3, 0]]).inv()
        )
        dist = diff.magnitude()
        assert type(dist) == float
        return dist

    def distance(self, other: Pose, orientation_factor: float = 0.05) -> float:
        position_distance = float(np.linalg.norm(self.position - other.position))
        orientation_distance = Pose.orientation_distance(
            self.orientation, other.orientation
        )
        dist = position_distance + orientation_factor * orientation_distance
        return dist


def get_best_orn_for_gripper(reference_orn: np.ndarray, query_orn: np.ndarray):
    if not SYMMETRIC_GRIPPER:
        return query_orn
    # rotate gripper about z-axis, choose the closer one
    other_orn = quaternions.qmult(
        euler.euler2quat(0, 0, np.pi),
        query_orn,
    )
    if Pose.orientation_distance(reference_orn, other_orn) < Pose.orientation_distance(
        reference_orn, query_orn
    ):
        return other_orn
    return query_orn


@dataclasses.dataclass
class AABB:
    center: Point3D
    size: Point3D
    pose: Pose = Pose(position=np.array((0, 0, 0)), orientation=np.array((1, 0, 0, 0)))

    @classmethod
    def union(cls, aabbs: List[AABB]) -> AABB:
        """
        computes the union AABB of `aabbs`
        """
        assert len(aabbs) > 0, "cannot compute union of empty list"
        # compute extents
        min_extent = np.array([np.inf, np.inf, np.inf])
        max_extent = np.array([-np.inf, -np.inf, -np.inf])
        for aabb in aabbs:
            for point in aabb.corners:
                min_extent = np.minimum(min_extent, point)
                max_extent = np.maximum(max_extent, point)
        # compute center and size
        center = (min_extent + max_extent) / 2
        size = max_extent - min_extent
        return AABB(
            center=(center[0], center[1], center[2]), size=(size[0], size[1], size[2])
        )

    @classmethod
    def total_volume(cls, aabbs: List[AABB]) -> float:
        """
        returns total volumes of all aabbs
        TODO compute only non-overlapping regions
        """
        return sum(aabb.volume for aabb in aabbs)

    @property
    def volume(self) -> float:
        return float(np.prod(self.size))

    @property
    def corners(self) -> List[Point3D]:
        """
        returns the corner points of the AABB,
        accounting for the point's transforms
        """
        normalized_corners = np.array(list(itertools.product([0, 1], repeat=3))) - 0.5
        canonical_corners = normalized_corners * self.size + self.center
        mat = self.pose.matrix
        return [
            tuple(mat.dot(np.append(corner, 1))[:3])  # type: ignore
            for corner in canonical_corners
        ]

    @property
    def extents(self) -> Tuple[Point3D, Point3D]:
        # lower and upper extents respectively
        return (
            tuple(np.array(self.center) - np.array(self.size) / 2),
            tuple(np.array(self.center) + np.array(self.size) / 2),
        )  # type: ignore

    def intersection(self, other: AABB) -> float:
        """
        returns the intersection volume of self and other
        """
        # pose must be identity
        assert (self.pose.position == np.array((0, 0, 0))).all() and (
            self.pose.orientation == euler.euler2quat(0, 0, 0)
        ).all()
        assert (other.pose.position == np.array((0, 0, 0))).all() and (
            other.pose.orientation == euler.euler2quat(0, 0, 0)
        ).all()
        if self.center == other.center and self.size == other.size:
            return self.volume
        min_extent = np.maximum(
            np.array(self.center) - np.array(self.size) / 2,
            np.array(other.center) - np.array(other.size) / 2,
        )
        max_extent = np.minimum(
            np.array(self.center) + np.array(self.size) / 2,
            np.array(other.center) + np.array(other.size) / 2,
        )
        intersection_size = np.maximum(max_extent - min_extent, 0)
        intersection_volume = float(np.prod(intersection_size))
        return intersection_volume


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class Contact(Picklable):
    other_link: str
    other_name: str
    self_link: str
    position: Point3D
    normal: Point3D

    @validator("normal")
    @classmethod
    def contact_normal_normalized(cls, v: Point3D):
        norm = np.linalg.norm(v)
        if norm == 0:
            raise ValueError("normal can't be zero")
        return tuple(np.array(v) / norm)

    def __hash__(self) -> int:
        return hash(
            (self.other_link, self.other_name, self.self_link, self.position, self.normal)
        )

    def __eq__(self, other) -> bool:
        return all(
            [
                self.other_link == other.other_link,
                self.other_name == other.other_name,
                self.self_link == other.self_link,
                self.position == other.position,
                self.normal == other.normal,
            ]
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return (
            "Contact("
            + f"other_link={self.other_link}, "
            + f"other_name={self.other_name}, "
            + f"self_link={self.self_link})"
        )


class JointType(IntEnum):
    REVOLUTE = 0
    PRISMATIC = 1
    FREE = 2


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class JointState(Picklable):
    name: str
    joint_type: JointType
    min_value: float
    max_value: float
    current_value: float  # DOF value
    axis: Tuple[float, float, float]  # axis direction in world frame
    position: Point3D  # origin of joint in world frame
    orientation: np.ndarray
    parent_link: str
    child_link: str


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class LinkState(Picklable):
    link_path: str
    obj_name: str
    pose: Pose
    velocity: Velocity
    contacts: Set[Contact]
    aabbs: List[AABB]

    @property
    def flattened(self) -> List[float]:
        return self.pose.flattened + self.velocity.flattened

    def __hash__(self) -> int:
        return hash(
            (self.link_path, self.obj_name, self.pose, self.velocity, *self.contacts)
        )

    def get_contacts_with(self, other: LinkState) -> Set[Contact]:
        return set(
            filter(
                lambda c: c.other_link == other.link_path
                and c.other_name == other.obj_name,
                self.contacts,
            )
        )


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class ObjectState(Picklable):
    link_states: Dict[str, LinkState]
    joint_states: Dict[str, JointState]

    @property
    def contacts(self) -> Set[Contact]:
        link_contacts: Set[Contact] = set()
        for link_state in self.link_states.values():
            link_contacts = link_contacts.union(link_state.contacts)
        return link_contacts

    @property
    def flattened(self) -> List[float]:
        return sum((body_state.flattened for body_state in self.link_states.values()), [])

    def __hash__(self) -> int:
        return hash((*self.link_states.keys(), *self.link_states.values()))


@dataclasses.dataclass(frozen=True)
class EnvState(Picklable):
    object_states: Dict[str, ObjectState]
    end_effector_pose: Pose
    gripper_command: bool
    robot_name: str
    robot_gripper_links: List[str]
    grasped_objects: FrozenSet[str]
    robot_joint_velocities: List[float]

    @property
    def end_effector_contacts(self) -> Set[Contact]:
        contacts: Set[Contact] = set()
        for link_path in self.robot_gripper_links:
            contacts = contacts.union(
                self.object_states[self.robot_name].link_states[link_path].contacts
            )
        return contacts

    @property
    def flattened(self) -> List[float]:
        return sum((obj_state.flattened for obj_state in self.object_states.values()), [])

    def __hash__(self) -> int:
        sorted_keys = sorted(self.object_states.keys())
        return hash(
            (
                *sorted_keys,
                *[self.object_states[k] for k in sorted_keys],
                self.end_effector_pose,
                self.gripper_command,
            )
        )

    def __eq__(self, other) -> bool:
        if (
            self.end_effector_pose == other.end_effector_pose
            and self.gripper_command == other.gripper_command
            and all(
                (self_obj_state == other_obj_state).all()
                if type(self_obj_state) == np.ndarray
                else (self_obj_state == other_obj_state)
                for self_obj_state, other_obj_state in zip(
                    self.object_states, other.object_states
                )
            )
        ):
            return True
        else:
            return False

    @property
    def robot_state(self):
        return self.object_states[self.robot_name]

    def get_pose(self, key: str) -> Pose:
        if LINK_SEPARATOR_TOKEN in key:
            # TODO: migrate entire system to this format
            obj_name = key.split(LINK_SEPARATOR_TOKEN)[0]
            if key not in self.object_states[obj_name].link_states.keys():
                raise KeyError(
                    f"{key} not in {self.object_states[obj_name].link_states.keys()}"
                )
            return self.object_states[obj_name].link_states[key].pose
        elif key in self.object_states:
            return self.object_states[key].link_states[key].pose
        raise KeyError(f"{key} not in {self.object_states.keys()}")

    def get_state(self, key: str) -> Union[ObjectState, LinkState]:
        if key in self.object_states:
            return self.object_states[key]
        else:
            for obj_state in self.object_states.values():
                if key in obj_state.link_states:
                    return obj_state.link_states[key]
            raise KeyError(f"{key} not in {self.object_states.keys()}")

    def get_link_joint_type(self, link_path):
        obj_name = link_path.split(LINK_SEPARATOR_TOKEN)[0]
        obj_state = self.object_states[obj_name]
        joints = [
            joint_state
            for joint_state in obj_state.joint_states.values()
            if joint_state.child_link == link_path
        ]
        if len(joints) == 0:
            # free joint
            return JointType.FREE
        elif len(joints) == 1:
            return joints[0].joint_type
        else:
            raise Exception(f"link {link_path} has {len(joints)} joints")


def plot_to_png(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = np.array(Image.open(buf)).astype(RGB_DTYPE)
    return img


def set_view_and_save_img(fig, ax, views):
    for elev, azim in views:
        ax.view_init(elev=elev, azim=azim)
        yield plot_to_png(fig)


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class PointCloud:
    rgb_pts: np.ndarray
    segmentation_pts: Dict[str, np.ndarray]
    xyz_pts: np.ndarray

    @validator("rgb_pts")
    @classmethod
    def rgb_dtype(cls, rgb_pts: np.ndarray):
        if (
            (rgb_pts.dtype in {np.float32, np.float64})
            and rgb_pts.max() < 1.0
            and rgb_pts.min() > 0.0
        ):
            rgb_pts = rgb_pts * 255
            return rgb_pts.astype(RGB_DTYPE)
        elif rgb_pts.dtype == RGB_DTYPE:
            return rgb_pts
        else:
            raise ValueError(f"`rgb_pts` in unexpected format: dtype {rgb_pts.dtype}")

    @validator("segmentation_pts")
    @classmethod
    def segmentation_pts_shape(cls, v: Dict[str, np.ndarray]):
        for pts in v.values():
            if len(pts.shape) > 2:
                raise ValueError(f"points.shape should N, but got {pts.shape}")
        return v

    @validator("xyz_pts")
    @classmethod
    def xyz_pts_shape(cls, v: np.ndarray):
        if len(v.shape) != 2 or v.shape[1] != 3:
            raise ValueError("points should be Nx3")
        return v

    @validator("xyz_pts")
    @classmethod
    def same_len(cls, v: np.ndarray, values):
        if "rgb_pts" in values and len(values["rgb_pts"]) != len(v):
            raise ValueError("`len(rgb_pts) != len(xyz_pts)`")
        if "segmentation_pts" in values and not all(
            len(pts) == len(v) for pts in values["segmentation_pts"].values()
        ):
            raise ValueError("`len(segmentation_pts) != len(xyz_pts)`")
        return v

    def __len__(self):
        return len(self.xyz_pts)

    def __add__(self, other: PointCloud):
        return PointCloud(
            xyz_pts=np.concatenate((self.xyz_pts, other.xyz_pts), axis=0),
            rgb_pts=np.concatenate((self.rgb_pts, other.rgb_pts), axis=0),
            segmentation_pts={
                k: np.concatenate(
                    (self.segmentation_pts[k], other.segmentation_pts[k]), axis=0
                )
                for k in self.segmentation_pts.keys()
            },
        )

    def to_open3d(self, use_segmentation_pts: bool = False) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz_pts)
        if use_segmentation_pts:
            pcd.colors = o3d.utility.Vector3dVector(
                np.stack(
                    [
                        np.stack(list(self.segmentation_pts.values()), axis=1).argmax(
                            axis=1
                        )
                    ]
                    * 3,
                    axis=1,
                )
            )
        else:
            pcd.colors = o3d.utility.Vector3dVector(self.rgb_pts / 255)
        return pcd

    def voxel_downsample(
        self, voxel_dim: float = 0.015, skip_segmentation: bool = False
    ) -> PointCloud:
        pcd = self.to_open3d()
        pcd = pcd.voxel_down_sample(voxel_dim)
        xyz_pts = np.array(pcd.points).astype(DEPTH_DTYPE)
        rgb_pts = (np.array(pcd.colors) * 255).astype(RGB_DTYPE)
        if skip_segmentation:
            return PointCloud(
                xyz_pts=xyz_pts,
                rgb_pts=rgb_pts,
                segmentation_pts={},
            )
        assert (
            len(self.xyz_pts) < 20000
        ), f"{len(self.xyz_pts)} is too many points, will consume too much RAM"
        distances = ((xyz_pts[None, ...] - self.xyz_pts[:, None, :]) ** 2).sum(axis=2)
        indices = distances.argmin(axis=0)

        segmentation_pts = {k: v[indices] for k, v in self.segmentation_pts.items()}
        return PointCloud(
            xyz_pts=xyz_pts,
            rgb_pts=rgb_pts,
            segmentation_pts=segmentation_pts,
        )

    @property
    def normals(self) -> np.ndarray:
        pcd = self.to_open3d()
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30)
        )
        # visualize
        return np.asarray(pcd.normals)

    def filter_bounds(self, bounds: Tuple[Point3D, Point3D]):
        in_bounds_mask = np.logical_and(
            (self.xyz_pts > np.array(bounds[0])).all(axis=1),
            (self.xyz_pts < np.array(bounds[1])).all(axis=1),
        )
        return PointCloud(
            xyz_pts=self.xyz_pts[in_bounds_mask],
            rgb_pts=self.rgb_pts[in_bounds_mask],
            segmentation_pts={
                k: self.segmentation_pts[k][in_bounds_mask]
                for k in self.segmentation_pts.keys()
            },
        )

    def subsample(self, num_pts: int, numpy_random: np.random.RandomState) -> PointCloud:
        indices = numpy_random.choice(
            len(self), size=num_pts, replace=num_pts > len(self)
        )
        return PointCloud(
            xyz_pts=self.xyz_pts[indices],
            rgb_pts=self.rgb_pts[indices],
            segmentation_pts={k: v[indices] for k, v in self.segmentation_pts.items()},
        )

    def __getitem__(self, key: str) -> PointCloud:
        assert key in self.segmentation_pts

        seg_mask = self.segmentation_pts[key]
        if not seg_mask.any():
            return PointCloud(
                xyz_pts=np.empty((0, 3), dtype=DEPTH_DTYPE),
                rgb_pts=np.empty((0, 3), dtype=RGB_DTYPE),
                segmentation_pts={key: np.ones(seg_mask.sum(), dtype=bool)},
            )
        link_point_cloud = PointCloud(
            xyz_pts=self.xyz_pts[seg_mask],
            rgb_pts=self.rgb_pts[seg_mask],
            segmentation_pts={key: np.ones(seg_mask.sum(), dtype=bool)},
        )
        if len(link_point_cloud) == 0:
            return link_point_cloud

        # help remove outliers due to noisy segmentations
        # this is actually quite expensive, and can be improved
        _, ind = link_point_cloud.to_open3d().remove_radius_outlier(
            nb_points=32, radius=0.02
        )
        return PointCloud(
            xyz_pts=link_point_cloud.xyz_pts[ind],
            rgb_pts=link_point_cloud.rgb_pts[ind],
            segmentation_pts={key: np.ones(len(ind), dtype=bool)},
        )

    def show(
        self: PointCloud,
        background_color: Point3D = (0.1, 0.1, 0.1),
        views: Optional[Sequence[Tuple[float, float]]] = None,
        pts_size: float = 3,
        bounds: Optional[Tuple[Point3D, Point3D]] = None,
        show: bool = True,
        show_segmentation: bool = False,
    ):
        if views is None:
            views = [(45, 135)]
        fig = plt.figure(figsize=(6, 6), dpi=160)
        ax = fig.add_subplot(111, projection="3d")  # type: ignore
        point_cloud = self
        if bounds is not None:
            point_cloud = point_cloud.filter_bounds(bounds=bounds)
        else:
            bounds = (
                (
                    float(point_cloud.xyz_pts[:, 0].min()),
                    float(point_cloud.xyz_pts[:, 1].min()),
                    float(point_cloud.xyz_pts[:, 2].min()),
                ),
                (
                    float(point_cloud.xyz_pts[:, 0].max()),
                    float(point_cloud.xyz_pts[:, 1].max()),
                    float(point_cloud.xyz_pts[:, 2].max()),
                ),
            )
        x, y, z = (
            point_cloud.xyz_pts[:, 0],
            point_cloud.xyz_pts[:, 1],
            point_cloud.xyz_pts[:, 2],
        )
        ax.set_facecolor(background_color)
        ax.w_xaxis.set_pane_color(background_color)  # type: ignore
        ax.w_yaxis.set_pane_color(background_color)  # type: ignore
        ax.w_zaxis.set_pane_color(background_color)  # type: ignore
        if show_segmentation:
            object_labels = [
                k.split(LINK_SEPARATOR_TOKEN)[-1]
                for k in point_cloud.segmentation_pts.keys()
            ]
            stacked_points = np.stack(list(point_cloud.segmentation_pts.values()), axis=1)
            points = stacked_points.argmax(axis=1)
            # repack object ids
            repacked_obj_ids = np.zeros(points.shape).astype(np.uint32)
            for i, j in enumerate(np.unique(points)):
                repacked_obj_ids[points == j] = i
            points = repacked_obj_ids

            object_ids = list(np.unique(points))
            colors: np.ndarray = np.zeros((len(points), 4)).astype(RGB_DTYPE)
            if len(object_ids) > 20:
                object_colors = (
                    255
                    * plt.get_cmap("gist_rainbow")(np.array(object_ids) / len(object_ids))
                ).astype(RGB_DTYPE)
            else:
                object_colors = (
                    255 * plt.get_cmap("tab20")(np.array(object_ids))
                ).astype(RGB_DTYPE)
            for obj_id in np.unique(points):
                colors[points == obj_id, :] = object_colors[obj_id]
            colors = colors.astype(float) / 255.0
            object_colors = object_colors.astype(float) / 255
            handles = [
                Patch(facecolor=c, edgecolor="grey", label=label)
                for label, c in zip(object_labels, object_colors)
            ]

            legend = ax.legend(
                handles=handles,
                labels=object_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0),
                ncol=4,
                facecolor=(0, 0, 0, 0.1),
                fontsize=7,
                framealpha=0,
            )
            plt.setp(legend.get_texts(), color=(0.8, 0.8, 0.8))
            ax.scatter(
                x,
                y,
                z,
                c=colors,
                s=pts_size,  # type: ignore
            )
        else:
            ax.scatter(
                x,
                y,
                z,
                c=point_cloud.rgb_pts.astype(float) / 255.0,
                s=pts_size,  # type: ignore
            )

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])  # type: ignore
        if bounds is not None:
            ax.axes.set_xlim3d(left=bounds[0][0], right=bounds[1][0])  # type: ignore
            ax.axes.set_ylim3d(bottom=bounds[0][1], top=bounds[1][1])  # type: ignore
            ax.axes.set_zlim3d(bottom=bounds[0][2], top=bounds[1][2])  # type: ignore
        plt.tight_layout(pad=0)
        imgs = list(set_view_and_save_img(fig, ax, views))
        if show:
            plt.show()
        plt.close(fig)
        return imgs


VISION_SENSOR_OUTPUT_COMPRESSOR = Blosc(cname="zstd", clevel=7, shuffle=Blosc.NOSHUFFLE)
RGB_DTYPE = np.uint8
DEPTH_DTYPE = np.float32
SEGMENTATION_DTYPE = bool


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class VisionSensorOutput(Picklable):
    obs_dim: Tuple[int, int]
    compressed_rgb: Union[np.ndarray, bytes]
    compressed_depth: Union[np.ndarray, bytes]
    compressed_segmentation: Dict[str, Union[np.ndarray, bytes]]
    pos: Point3D
    rot_mat: np.ndarray
    fovy: float

    @property
    def rgb(self) -> np.ndarray:
        decompressed_bytes = VISION_SENSOR_OUTPUT_COMPRESSOR.decode(self.compressed_rgb)
        return np.frombuffer(decompressed_bytes, dtype=RGB_DTYPE).reshape(
            *self.obs_dim, 3
        )

    @property
    def depth(self) -> np.ndarray:
        decompressed_bytes = VISION_SENSOR_OUTPUT_COMPRESSOR.decode(self.compressed_depth)
        return np.frombuffer(decompressed_bytes, dtype=DEPTH_DTYPE).reshape(*self.obs_dim)

    def get_segmentation(self, key: str) -> np.ndarray:
        decompressed_bytes = VISION_SENSOR_OUTPUT_COMPRESSOR.decode(
            self.compressed_segmentation[key]
        )
        return np.frombuffer(decompressed_bytes, dtype=SEGMENTATION_DTYPE).reshape(
            *self.obs_dim
        )

    @validator("compressed_depth")
    @classmethod
    def depth_shape(cls, v: Union[np.ndarray, bytes]):
        if type(v) == bytes:
            return v
        assert type(v) == np.ndarray
        # this is hacky, consider factory functions?
        if v is not None and len(v.shape) != 2:
            raise ValueError("depth images should be HxW")
        depth = v.astype(DEPTH_DTYPE)
        return VISION_SENSOR_OUTPUT_COMPRESSOR.encode(np.ascontiguousarray(depth))

    @validator("compressed_rgb")
    @classmethod
    def rgb_shape(cls, v: Union[np.ndarray, bytes]):
        if type(v) == bytes:
            return v
        assert type(v) == np.ndarray
        if v.dtype == np.float32:
            if v.min() < 0.0 or v.max() > 1.0:
                raise ValueError(f"rgb values out of expected range: {v.min()},{v.max()}")
            v = (v * 255).astype(RGB_DTYPE)
        if v.shape[-1] != 3:
            raise ValueError("rgb images should be HxWx3")
        return VISION_SENSOR_OUTPUT_COMPRESSOR.encode(np.ascontiguousarray(v))

    @validator("compressed_segmentation")
    @classmethod
    def segmentation_shape(cls, segmentation: Dict[str, Union[bytes, np.ndarray]]):
        if len(segmentation) == 0:
            return {}
        if type(list(segmentation.values())[0]) == bytes:
            return segmentation
        return {
            k: VISION_SENSOR_OUTPUT_COMPRESSOR.encode(np.ascontiguousarray(v))
            for k, v in segmentation.items()
        }

    @validator("rot_mat")
    @classmethod
    def rot_mat_dtype(cls, v: np.ndarray):
        return v.astype(np.float16)

    @validator("rot_mat")
    @classmethod
    def check_rot_mat(cls, v: np.ndarray):
        if v.shape != (3, 3):
            raise ValueError("rot_mat matrix has incorrect shape")
        return v

    @property
    def valid_points_mask(self):
        assert (self.depth != 0).reshape(-1).all(), "not all depth values are valid"
        return np.ones_like((self.depth != 0).reshape(-1))

    @property
    def height(self) -> int:
        return self.depth.shape[0]

    @property
    def width(self) -> int:
        return self.depth.shape[1]

    @property
    def intrinsic(self) -> np.ndarray:
        # in a perfect pinhole camera, the vertical focal length
        # is the same as the horizon focal length
        focal_length = (self.height / 2) / np.tan(np.deg2rad(self.fovy) / 2)
        return np.array(
            [
                [focal_length, 0, self.width / 2],
                [0, focal_length, self.height / 2],
                [0, 0, 1],
            ],
            dtype=DEPTH_DTYPE,
        )

    @property
    def forward(self) -> np.ndarray:
        forward = np.matmul(self.rot_mat, np.array([0, 0, -1]))
        return forward / np.linalg.norm(forward)

    @property
    def up(self) -> np.ndarray:
        up = np.matmul(self.rot_mat, np.array([0, 1, 0]))
        return up / np.linalg.norm(up)

    @property
    def pose_matrix(self) -> np.ndarray:
        pos = self.pos
        forward = self.forward.copy()
        u = self.up.copy()
        s = np.cross(forward, u)
        s = s / np.linalg.norm(s)
        u = np.cross(s, forward)
        view_matrix = np.array(
            [
                s[0],
                u[0],
                -forward[0],
                0,
                s[1],
                u[1],
                -forward[1],
                0,
                s[2],
                u[2],
                -forward[2],
                0,
                -np.dot(s, pos),
                -np.dot(u, pos),
                np.dot(forward, pos),
                1,
            ]
        )
        view_matrix = view_matrix.reshape(4, 4).T
        pose_matrix = np.linalg.inv(view_matrix)
        pose_matrix[:, 1:3] = -pose_matrix[:, 1:3]
        return pose_matrix.astype(DEPTH_DTYPE)

    @property
    def point_cloud(self) -> PointCloud:
        img_h = self.depth.shape[0]
        img_w = self.depth.shape[1]

        # Project depth into 3D pointcloud in camera coordinates
        pixel_x, pixel_y = np.meshgrid(
            np.linspace(0, img_w - 1, img_w), np.linspace(0, img_h - 1, img_h)
        )
        cam_pts_x = np.multiply(
            pixel_x - self.intrinsic[0, 2], self.depth / self.intrinsic[0, 0]
        ).astype(DEPTH_DTYPE)
        cam_pts_y = np.multiply(
            pixel_y - self.intrinsic[1, 2], self.depth / self.intrinsic[1, 1]
        ).astype(DEPTH_DTYPE)
        cam_pts_z = self.depth
        cam_pts = (
            np.array([cam_pts_x, cam_pts_y, cam_pts_z]).transpose(1, 2, 0).reshape(-1, 3)
        )
        world_pts = np.matmul(
            self.pose_matrix,
            np.concatenate((cam_pts, np.ones_like(cam_pts[:, [0]])), axis=1).T,
        ).T[:, :3]
        return PointCloud(
            xyz_pts=world_pts.astype(DEPTH_DTYPE),
            rgb_pts=self.rgb[:, :, :3].reshape(-1, 3),
            segmentation_pts={
                k: self.get_segmentation(key=k).reshape(-1)
                for k in self.compressed_segmentation.keys()
            },
        )

    @property
    def camera_matrix(self):
        # TODO add mujoco tutorial code acknowledgement
        # Translation matrix (4x4).
        translation = np.eye(4)
        translation[0:3, 3] = -np.array(self.pos)

        # Rotation matrix (4x4).
        rotation = np.eye(4)
        rotation[0:3, 0:3] = self.rot_mat.T

        # Focal transformation matrix (3x4).
        focal_scaling = (1.0 / np.tan(np.deg2rad(self.fovy) / 2)) * self.height / 2.0
        focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

        # Image matrix (3x3).
        image = np.eye(3)
        image[0, 2] = (self.width - 1) / 2.0
        image[1, 2] = (self.height - 1) / 2.0
        return (image @ focal @ rotation @ translation).astype(np.float16)

    def global_world_to_pixel(self, xyz: Point3D) -> Pixel:
        # Project world coordinates into pixel space. See:
        # https://en.wikipedia.org/wiki/3D_projection#Mathematical_formula
        homogeneous_coord = np.ones((4,), dtype=float)
        homogeneous_coord[:3] = xyz
        xs, ys, s = self.camera_matrix @ homogeneous_coord
        # x and y are in the pixel coordinate system.
        x = xs / s
        y = ys / s
        return (int(x), int(y))

    def show(self):
        fig, axes = plt.subplots(1, 3)
        for ax in axes:
            ax.axis("off")
        axes[0].imshow(self.rgb)
        axes[1].imshow(self.depth)
        axes[2].imshow(
            np.stack(
                [
                    self.get_segmentation(key=k)
                    for k in self.compressed_segmentation.keys()
                ],
                axis=1,
            ).argmax(axis=1),
            cmap="gist_rainbow",
        )
        plt.tight_layout()
        plt.show()


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class ControlValue:
    end_effector_pos: np.ndarray
    end_effector_quat: np.ndarray

    gripper_pos: float

    joint_pos: np.ndarray

    def __eq__(self, __value: object) -> bool:
        if type(__value) is not ControlValue:
            return False
        other_control_value = typing.cast(ControlValue, __value)
        return (
            self.end_effector_pos.all() == other_control_value.end_effector_pos.all()
            and self.end_effector_quat.all()
            == other_control_value.end_effector_quat.all()
            and self.gripper_pos == other_control_value.gripper_pos
            and self.joint_pos.all() == other_control_value.joint_pos.all()
        )


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class Observation(Picklable):
    state: EnvState
    images: Dict[str, VisionSensorOutput]
    control: Dict[str, Union[np.ndarray, bool]]
    episode_id: int
    time: float

    def __eq__(self, __value: object) -> bool:
        if type(__value) is not Observation:
            return False
        return (
            self.state == __value.state
            and self.episode_id == __value.episode_id
            and self.time == __value.time
            and self.control == __value.control
        )

    def __hash__(self) -> int:
        return hash((self.state, self.episode_id, self.time))


@dataclasses.dataclass(frozen=True)
class Action(Picklable):
    class InfeasibleAction(Exception):
        def __init__(
            self, action_class_name: str, message: str, stop_episode: bool = True
        ):
            super().__init__(f"[{action_class_name}] {message}")
            self.stop_episode = stop_episode

    class FailedExecution(Exception):
        def __init__(self, message: str):
            super().__init__(f"{message}")

    class DroppedGraspedObj(FailedExecution):
        def __init__(self, drop_time: float, num_ctrl_cycles_left: int):
            super().__init__(
                f"Dropped grasped object at {drop_time:02f}s"
                f" with {num_ctrl_cycles_left} control cycles left"
            )

    def get_info(self) -> Dict[str, Any]:
        return {}

    def __call__(
        self,
        env: Env,
        active_task: Task,
        sampler_config: EnvSamplerConfig,
        info: Dict[str, Any],
    ) -> bool:
        self.check_feasibility(env=env)
        return self.execute(
            env=env, active_task=active_task, sampler_config=sampler_config, info=info
        )

    @abstractmethod
    def check_feasibility(self, env: Env):
        pass

    @abstractmethod
    def execute(
        self,
        env: Env,
        active_task: Task,
        sampler_config: EnvSamplerConfig,
        info: Dict[str, Any],
    ) -> bool:
        """
        All actions must implement how it wants to interact with the `Env`
        returns whether the episode should end
        """
        pass


@dataclasses.dataclass(frozen=True)
class DoNothingAction(Action):
    end_subtrajectory: bool = False
    end_episode: bool = False

    def execute(
        self,
        env: Env,
        active_task: Task,
        sampler_config: EnvSamplerConfig,
        info: Dict[str, Any],
    ) -> bool:
        if self.end_episode:
            env.done = True
        return self.end_subtrajectory

    def check_feasibility(self, env: Env):
        pass


class Task:
    """
    `Task` defines things the robot should do, like `GraspObj`.

    It is structured with `reward_fn` and `success_fn` to allow for easy
    integration with Language-Model inferred tasks, by using `fn_str_to_fn` to
    compile code strings dynamically.
    """

    def __init__(
        self,
        desc: str,
        info: Optional[Dict[str, Any]] = None,  # just used to store extra info about task
    ):
        self.desc = desc
        self.info = info if info is not None else {}

    def check_success(self, traj: Trajectory) -> bool:
        return False

    def get_reward(self, state: EnvState) -> float:
        return 0.0

    def get_returns(self, traj: Trajectory) -> float:
        return sum(
            self.get_reward(state=traj_step.obs.state) for traj_step in traj.episode
        )

    def __str__(self) -> str:
        return f"Task(desc={self.desc!r})"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other) -> bool:
        # NOTE: tasks can be functionally equivalent, but has different
        # success condition source codes and descriptions.
        # since there are no good ways for comparison without a dataset
        # of trajectories, always return True
        return True

    @property
    def name(self) -> str:
        return self.desc

    def __hash__(self) -> int:
        return hash(
            (
                self.desc,
                self.__class__.__name__,
                *sorted(self.info.keys()),
                *self.info.values(),
            )
        )


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class TrajectoryStep(Picklable):
    info: Dict[str, Any]
    obs: Observation
    action: Action
    done: bool
    next_obs: Observation
    compressed_renders: List[Union[bytes, np.ndarray]]
    visualization_dim: Tuple[int, int]
    # dense lower level states and control values between obs and next_obs
    control_states: List[Observation]

    @property
    def renders(self) -> List[np.ndarray]:
        return [
            np.frombuffer(
                VISION_SENSOR_OUTPUT_COMPRESSOR.decode(compressed_render), dtype=RGB_DTYPE
            ).reshape(*self.visualization_dim, 3)
            for compressed_render in self.compressed_renders
        ]

    def __eq__(self, other: object) -> bool:
        if type(other) is not TrajectoryStep:
            return False
        return (
            self.obs == other.obs
            and self.action == other.action
            and self.done == other.done
            and self.next_obs == other.next_obs
            and self.info == other.info
            and self.control_states == other.control_states
        )

    def __hash__(self) -> int:
        assert len(self.subtrajectories) == 0
        return hash(
            (
                self.obs,
                self.action,
                self.done,
                self.next_obs,
            )
        )

    @property
    def subtrajectories(self) -> List[Trajectory]:
        return self.info["subtrajectories"] if "subtrajectories" in self.info else []

    @validator("action")
    @classmethod
    def action_subtrajectories(cls, v, values, **kwargs):
        # besides a few basic action classes, all others should
        # have subtrajectories
        is_subtrajectory_action = issubclass(type(v), SubtrajectoryAction)
        is_highlevel_link_action = issubclass(type(v), HighLevelLinkAction)
        has_subtrajectories = (
            "subtrajectories" in values["info"]
            and len(values["info"]["subtrajectories"]) > 0
        )
        if is_subtrajectory_action and not has_subtrajectories:
            if is_highlevel_link_action:
                # high level link action failed to compile to low level link action
                # since all candidates failed
                return v
            else:
                error_message = f"{type(v).__name__!r} should have subtrajectories"
                logging.error(error_message)
                # TODO decide whether this should cause exceptions or not
                # raise ValueError(error_message)
        return v

    @validator("control_states")
    @classmethod
    def control_states_time_order(cls, v, values, **kwargs):
        if len(v) > 0:
            last_time: float = values["obs"].time
            for i, control_states in enumerate(typing.cast(List[Observation], v)):
                if control_states.time < last_time:
                    raise ValueError(
                        f" {i}th `control_states.time` "
                        + f"({control_states.time}) is decreasing"
                    )
                else:
                    last_time = control_states.time
            if values["next_obs"].time < last_time:
                raise ValueError(
                    f" `next_obs.time` ({values['next_obs'].time}) is decreasing"
                )
        return v

    @validator("compressed_renders", each_item=True)
    @classmethod
    def int8_images(cls, v: Union[np.ndarray, bytes]):
        if type(v) == bytes:
            return v
        assert type(v) == np.ndarray
        if v.dtype == np.float32:
            if v.min() < 0.0 or v.max() > 1.0:
                raise ValueError(
                    "Render frame values out of range:" + f" [{v.min()},{v.max()}]"
                )
            v *= 255
        return VISION_SENSOR_OUTPUT_COMPRESSOR.encode(
            np.ascontiguousarray(v.astype(RGB_DTYPE))
        )

    def is_ee_in_bounds(
        self,
        pos_bounds: Tuple[Point3D, Point3D],
        orn_bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    ) -> bool:
        if type(self.action) == EndEffectorAction:
            ee_action = cast(EndEffectorAction, self.action)
            ee_pos = ee_action.end_effector_position
            ee_orn_euler = (
                np.array(euler.quat2euler(ee_action.end_effector_orientation))
                * 180.0
                / np.pi
            )
            ee_pos_in_bounds = typing.cast(
                bool,
                (ee_pos >= np.array(pos_bounds[0])).all()
                and (ee_pos <= np.array(pos_bounds[1])).all(),
            )
            if not ee_pos_in_bounds:
                logging.debug(f"ee position {ee_pos} is out of bounds {pos_bounds}")

            ee_orn_in_bounds = typing.cast(
                bool,
                (ee_orn_euler > np.array(orn_bounds[0])).all()
                and (ee_orn_euler <= np.array(orn_bounds[1])).all(),
            )

            if not ee_orn_in_bounds:
                logging.debug(f"ee orn {ee_orn_euler} is out of bounds {orn_bounds}")
            return ee_pos_in_bounds and ee_orn_in_bounds
        return True

    def get_visualization_frames(
        self, texts: Optional[List[str]] = None, add_text: bool = True
    ) -> List[np.ndarray]:
        if texts is None:
            texts = []
        texts.append(" " * 4 + str(self.action))
        fontsize = 12
        renders = self.renders
        if len(renders) > 0:
            h, w = renders[0].shape[:2]
            img_dim = min(h, w)
            fontsize = int(img_dim * 0.03)
        images = renders
        if add_text:
            wrapper = textwrap.TextWrapper(width=75)
            accumulated_text = "\n".join(
                "\n".join(wrapper.wrap(line))
                for line in texts
                # + [" " * 4 + f'LOG: {self.info["log"]!r}']
            )
            images = [
                add_text_to_image(
                    image=img,
                    texts=[accumulated_text],
                    positions=[(0, 0)],
                    fontsize=fontsize,
                )
                for img in images
            ]
        for subtraj in self.subtrajectories:
            images += subtraj.get_visualization_frames(texts=texts, add_text=add_text)
        return images

    def strip_renders(self) -> TrajectoryStep:
        return TrajectoryStep(
            info={
                **self.info,
                "subtrajectories": [
                    subtraj.strip_renders() for subtraj in self.subtrajectories
                ],
            },
            obs=self.obs,
            action=self.action,
            done=self.done,
            next_obs=self.next_obs,
            compressed_renders=[],
            visualization_dim=self.visualization_dim,
            control_states=self.control_states,
        )


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True, eq=True)
class Trajectory(Picklable):
    episode_id: int
    episode: Tuple[TrajectoryStep, ...]  # tuple > list to ensure immutability
    task: Task
    policy_id: str

    def __hash__(self):
        flattened_self = self.flatten()
        return hash((self.policy_id, self.episode_id, self.task, *flattened_self.episode))

    def __eq__(self, other: object) -> bool:
        if type(other) is Trajectory:
            other_traj = typing.cast(Trajectory, other)
            return (
                other_traj.episode_id == self.episode_id
                and other_traj.task == self.task
                and other_traj.policy_id == self.policy_id
                and other_traj.flatten().episode == self.flatten().episode
            )
        return False

    @validator("episode")
    @classmethod
    def trajectory_not_empty(cls, v: Tuple[TrajectoryStep]):
        if len(v) == 0:
            raise ValueError("Trajectory can't empty")
        return v

    @validator("episode")
    @classmethod
    def episode_has_ascending_times(cls, v: Tuple[TrajectoryStep]):
        times = np.array([step.obs.time for step in v])
        # check that times are ascending
        diff = np.diff(np.stack([times[:-1], times[1:]]), axis=0)
        # allow non-strictly increasing because actions can fail, resulting in
        # no time difference between steps
        increasing = (diff >= 0).all()
        if not increasing:
            raise ValueError("episode is not increasing")
        return v

    def __len__(self):
        return len(self.flatten().episode)

    @property
    def final_state(self) -> EnvState:
        return self.episode[-1].next_obs.state

    @property
    def init_state(self) -> EnvState:
        return self.episode[0].obs.state

    @property
    def is_leaf_trajectory(self) -> bool:
        return not any(len(traj_step.subtrajectories) > 0 for traj_step in self.episode)

    def flatten(self) -> Trajectory:
        # count episode step properly
        flattened_episode: List[TrajectoryStep] = []
        for traj_step in self.episode:
            subtrajectories = traj_step.subtrajectories
            assert not (
                type(traj_step.action) in {EndEffectorAction, GripperAction}
                and len(subtrajectories) > 0
            )
            if len(subtrajectories) > 0:
                for subtraj in subtrajectories:
                    flattened_episode.extend(subtraj.flatten().episode)
            elif type(traj_step.action) == GripperAction:
                flattened_episode.append(
                    TrajectoryStep(
                        obs=traj_step.obs,
                        action=EndEffectorAction(
                            end_effector_position=traj_step.obs.state.end_effector_pose.position,
                            end_effector_orientation=traj_step.obs.state.end_effector_pose.orientation,
                            gripper_command=cast(GripperAction, traj_step.action).command,
                            allow_contact=True,
                        ),
                        next_obs=traj_step.next_obs,
                        done=traj_step.done,
                        info=traj_step.info,
                        compressed_renders=traj_step.compressed_renders,
                        visualization_dim=traj_step.visualization_dim,
                        control_states=traj_step.control_states,
                    )
                )
            elif type(traj_step.action) == DoNothingAction or issubclass(
                type(traj_step.action), HighLevelLinkAction
            ):
                # if high level link action has not subtrajectories, then all
                # low level link action candidates failed feasibility checks
                flattened_episode.append(
                    TrajectoryStep(
                        obs=traj_step.obs,
                        action=EndEffectorAction(
                            end_effector_position=traj_step.obs.state.end_effector_pose.position,
                            end_effector_orientation=traj_step.obs.state.end_effector_pose.orientation,
                            gripper_command=False,
                            allow_contact=True,
                        ),
                        next_obs=traj_step.next_obs,
                        done=traj_step.done,
                        info=traj_step.info,
                        compressed_renders=traj_step.compressed_renders,
                        visualization_dim=traj_step.visualization_dim,
                        control_states=traj_step.control_states,
                    )
                )
            else:
                # if is a basic action, then just add it
                if type(traj_step.action) in {
                    EndEffectorAction,
                    ControlAction,
                }:
                    flattened_episode.append(traj_step)

        for traj_step in flattened_episode:
            assert type(traj_step.action) in {
                EndEffectorAction,
                ControlAction,
            }
        return Trajectory(
            episode_id=self.episode_id,
            episode=tuple(flattened_episode),
            task=self.task,
            policy_id=self.policy_id,
        )

    def __getitem__(self, val: slice) -> Trajectory:
        if type(val) == slice:
            return Trajectory(
                episode_id=self.episode_id,
                episode=tuple(
                    TrajectoryStep(
                        obs=sub_traj_step.obs,
                        action=sub_traj_step.action,
                        next_obs=sub_traj_step.next_obs,
                        done=sub_traj_step.done,
                        info=sub_traj_step.info,
                        compressed_renders=sub_traj_step.compressed_renders,
                        visualization_dim=sub_traj_step.visualization_dim,
                        control_states=sub_traj_step.control_states,
                    )
                    for offset, sub_traj_step in enumerate(self.episode[val])
                ),
                task=self.task,
                policy_id=self.policy_id,
            )
        else:
            raise ValueError(f"Inappropriate indexing type: {type(val)}")

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def all_ee_in_bounds(
        self,
        pos_bounds: Tuple[Point3D, Point3D],
        orn_bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    ) -> bool:
        return all(
            traj_step.is_ee_in_bounds(pos_bounds=pos_bounds, orn_bounds=orn_bounds)
            for traj_step in self.episode
        )

    def get_visualization_frames(
        self, texts: Optional[List[str]] = None, add_text: bool = True
    ) -> List[np.ndarray]:
        # TODO use obs.time to compute exact time speedup
        images: List[np.ndarray] = []
        if texts is None:
            texts = []
        for episode_step, traj_step in enumerate(self.episode):
            images += traj_step.get_visualization_frames(
                texts=texts
                + [
                    f"[{episode_step+1}/{len(self.episode)}] {self.task.desc}: "
                    + ("Success" if self.is_successful else "Fail")
                ],
                add_text=add_text,
            )
        return images

    @property
    def subtrajectories(self) -> List[Trajectory]:
        """
        Recurse all subtrajectories. Includes itself.
        """
        subtrajectories: List[Trajectory] = []
        for traj_step in self.episode:
            for subtraj in traj_step.subtrajectories:
                subtrajectories += subtraj.subtrajectories
        return subtrajectories + [self]

    def print_tree(self, parent_node: Optional[Tree] = None) -> Tree:
        task_str = f"[blue]{self.task.desc}[/blue]"
        label = f"""{task_str}: EMPTY"""
        if len(self) > 0:
            policy_str = f"[yellow italic]{self.policy_id}[/yellow italic]"
            success = self.is_successful
            success_str = (
                "[green bold] Success [/green bold]"
                if success
                else "[red bold] Fail [/red bold]"
            )
            subtraj_success = self.compute_subtrajectory_success_rate()
            success_str += (
                f"[green]({subtraj_success*100:.01f})[/green]"
                if subtraj_success == 1.0
                else f"[red]({subtraj_success*100:.01f})[/red]"
            )
            label = f"""{task_str}: {success_str} ({policy_str})"""
        if parent_node is not None:
            node = parent_node.add(label)
        else:
            node = Tree(label)

        for traj_step in self.episode:
            for subtraj in traj_step.subtrajectories:
                subtraj.print_tree(parent_node=node)
        return node

    def dump_video(
        self,
        output_path: str,
        images: Optional[List[np.ndarray]] = None,
        repeat_last_frame: int = 48,
        add_text: bool = True,
    ):
        if images is None:
            images = self.get_visualization_frames(add_text=add_text)
        if len(images) != 0:
            with imageio.get_writer(output_path, fps=24) as writer:
                for img in images + [images[-1]] * repeat_last_frame:
                    writer.append_data(img)  # type: ignore
            logging.debug(str(self) + " dumped to " + output_path)

    def dump(
        self,
        path: str,
        protocol: Any = open,
        pickled_data_compressor: Any = brotli.compress,
        protocol_kwargs: Optional[Dict[str, Any]] = None,
        compressor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if compressor_kwargs is None:
            compressor_kwargs = {"quality": 9}
        return super().dump(
            path=path,
            protocol=protocol,
            pickled_data_compressor=pickled_data_compressor,
            protocol_kwargs=protocol_kwargs,
            compressor_kwargs=compressor_kwargs,
        )

    @classmethod
    def load(
        cls,
        path: str,
        protocol: Any = open,
        pickled_data_decompressor: Any = brotli.decompress,
        protocol_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Trajectory:
        return typing.cast(
            Trajectory,
            super().load(
                path=path,
                protocol=protocol,
                pickled_data_decompressor=pickled_data_decompressor,
                protocol_kwargs=protocol_kwargs,
            ),
        )

    def strip_renders(self) -> Trajectory:
        # remove render frames to save space
        return Trajectory(
            episode_id=self.episode_id,
            episode=tuple(traj_step.strip_renders() for traj_step in self.episode),
            task=self.task,
            policy_id=self.policy_id,
        )

    @property
    def control_states(self) -> List[Observation]:
        return sum([traj_step.control_states for traj_step in self.episode], [])

    @property
    def duration(self) -> float:
        return self.episode[-1].next_obs.time - self.episode[0].obs.time

    def compute_subtrajectory_success_rates(self) -> List[float]:
        unique_subtrajectories: Dict[int, Trajectory] = {}
        all_subtrajectories = set(self.subtrajectories)
        # allow skipping of functionally equivalent trajectories
        for traj in all_subtrajectories:
            flat_traj = traj.flatten()
            # hash based on actual sensory observations
            # and what the trajectory is trying to do
            hash_val = hash((flat_traj.episode, traj.task.desc))
            if hash_val not in unique_subtrajectories:
                unique_subtrajectories[hash_val] = traj
        successes: List = []
        for subtraj in unique_subtrajectories.values():
            successes.append(subtraj.is_successful)
        return successes

    def compute_subtrajectory_success_rate(self) -> float:
        successes = self.compute_subtrajectory_success_rates()
        return float(np.mean(successes))

    @property
    def is_perfect(self) -> bool:
        # a perfect trajectory is one where all subtasks
        # are successful and the overall task is successful
        return self.compute_subtrajectory_success_rate() == 1.0

    @property
    def is_successful(self) -> bool:
        try:
            return self.task.check_success(traj=self)
        except Exception as e:  # noqa: B902
            if "inferred_success_fn_code_str" in self.task.info:
                raise RuntimeError(
                    f"inference success fn for task {self.task.desc!r} has a bug: "
                    + self.task.info["inferred_success_fn_code_str"]
                ) from e
            else:
                raise e


RewardFn = Callable[[EnvState], float]
SuccessFn = Callable[[Trajectory], bool]


class Policy(ABC):
    def __init__(self, supported_tasks: Optional[FrozenSet[str]] = None):
        self._seed = 0
        self.supported_tasks = supported_tasks

    def is_task_supported(self, task_desc: str) -> bool:
        if self.supported_tasks is None:
            return True
        return task_desc in self.supported_tasks

    def get_supported_tasks(self) -> FrozenSet[str]:
        return self.supported_tasks if self.supported_tasks is not None else frozenset()

    def set_seed(self, seed: int):
        self._seed = seed

    def __call__(self, obs: Observation, task: Task, seed: int) -> Action:
        self.set_seed(seed)
        return self._get_action(obs=obs, task=task)

    @property
    def numpy_random(self):
        return np.random.RandomState(self._seed)

    def get_torch_random_generator(self, device: torch.device):
        generator = torch.Generator(device=device)
        generator.manual_seed(self._seed)
        return generator

    @abstractmethod
    def _get_action(self, obs: Observation, task: Task) -> Action:
        pass

    def update(  # noqa: B027
        self,
        traj: Trajectory,
    ):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}"  # type: ignore

    def __repr__(self):
        return str(self)


class RayPolicy(Policy):
    def __init__(self, policy):
        self.policy = policy
        assert type(self.policy) is not ActionListPolicy

    def __call__(self, obs: Observation, task: Task, seed: int) -> Action:
        if issubclass(type(self.policy), Policy):
            return self.policy(obs=obs, task=task, seed=seed)
        return ray.get(
            self.policy.__call__.remote(obs=obs, task=task, seed=seed)
        )  # type: ignore

    def update(self, traj: Trajectory):
        if issubclass(type(self.policy), Policy):
            return self.policy.update(traj=traj)
        return ray.get(self.policy.update.remote(traj=traj))  # type: ignore

    def is_task_supported(self, task_desc: str) -> bool:
        if issubclass(type(self.policy), Policy):
            return self.policy.is_task_supported(task_desc=task_desc)
        return ray.get(self.policy.is_task_supported.remote(task_desc=task_desc))

    def get_supported_tasks(self) -> FrozenSet[str]:
        if issubclass(type(self.policy), Policy):
            return self.policy.get_supported_tasks()
        return ray.get(self.policy.get_supported_tasks.remote())

    def _get_action(self, obs: Observation, task: Task) -> Action:
        if issubclass(type(self.policy), Policy):
            return self.policy._get_action(obs=obs, task=task)
        return ray.get(self.policy._get_action.remote(obs=obs, task=task))  # type: ignore

    def __str__(self) -> str:
        if issubclass(type(self.policy), Policy):
            return str(self.policy)
        return str(ray.get(self.policy.__str__.remote()))


@dataclasses.dataclass
class ActionListPolicy(Policy):
    actions: Sequence[Action]

    def __hash__(self) -> int:
        return hash(tuple(self.actions))

    def _get_action(self, obs: Observation, task: Task) -> Action:
        raise NotImplementedError("this should be handled by `EnvSampler`")

    def __str__(self):
        output = "{"
        for i, action in enumerate(self.actions):
            output += f"{i}. {str(action)}"
            if i < len(self.actions) - 1:
                output += ", "
        output += "}"
        return output


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class SubtrajectoryAction(Action):
    def execute(
        self,
        env: Env,
        active_task: Task,
        sampler_config: EnvSamplerConfig,
        info: Dict[str, Any],
    ) -> bool:
        subtrajs: List[Trajectory] = []
        try:
            subtrajs.extend(
                self.execute_subtrajectory(
                    env=env, active_task=active_task, sampler_config=sampler_config
                )
            )
        except ValidationError as e:
            # trajectory probably empty because all actions failed
            logging.error(e)
            pass
        except Exception as e:  # noqa: B902
            raise e
            info["log"] += str(e) + "|"
            logging.warning(f"{str(type(e))}: {e}")
            env.done = True
        info["subtrajectories"].extend(subtrajs)
        return False

    @abstractmethod
    def execute_subtrajectory(
        self, env: Env, active_task: Task, sampler_config: EnvSamplerConfig
    ) -> List[Trajectory]:
        pass


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class SubpolicyAction(SubtrajectoryAction):
    @abstractmethod
    def get_subtrajectory_policy(self, env: Env) -> Policy:
        pass

    def execute_subtrajectory(
        self, env: Env, active_task: Task, sampler_config: EnvSamplerConfig
    ) -> List[Trajectory]:
        sampler = EnvSampler(
            env=env,
            task_sampler=TaskSampler(tasks=[active_task]),
            do_reset=False,
        )
        subtraj_policy = self.get_subtrajectory_policy(env=env)
        # see `tests/test_memory_leak.py``, this needs to be here,
        # otherwise, a few MBs will leak each time
        gc.collect()
        traj = sampler.sample(
            policy=subtraj_policy,
            episode_id=env.episode_id,
            task=active_task,
            return_trajectory=True,
            config=sampler_config,
        ).trajectory
        assert traj is not None
        return [traj]


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class CompositeEndEffectorAction(SubpolicyAction):
    def get_subtrajectory_policy(self, env: Env) -> Policy:
        return ActionListPolicy(
            actions=self.to_end_effector_actions(
                obs=env.obs,
                numpy_random=env.policy_numpy_random,
                env=env,
            )
        )

    def check_feasibility(self, env: Env):
        for i, ee_action in enumerate(
            self.to_end_effector_actions(
                obs=env.obs,
                numpy_random=env.policy_numpy_random,
                env=env,
            )
        ):
            try:
                ee_action.check_feasibility(env=env)
            except Action.InfeasibleAction as e:
                raise Action.InfeasibleAction(
                    action_class_name=type(self).__name__,
                    message=f"{i}th action failed: {e}",
                ) from e

    @abstractmethod
    def to_end_effector_actions(
        self, obs: Observation, numpy_random: np.random.RandomState, env: Env
    ) -> List[EndEffectorAction]:
        pass


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class PolicyTaskAction(SubtrajectoryAction):
    # uses policy to do task until task
    # succeeds
    policy: Policy
    task: Task
    retry_until_success: bool
    MAX_RETRIES = 100

    def check_feasibility(self, env: Env):
        if type(self.policy) == ActionListPolicy:
            action_list_policy = cast(ActionListPolicy, self.policy)
            action_list_policy.actions[0].check_feasibility(env=env)
        # TODO design efficient way to check action feasibility, then
        # percolate feasibility proof in env sample recursion

    def __str__(self) -> str:
        return f"PolicyTaskAction(policy={type(self.policy).__name__}, task={self.task.name!r})"

    def __hash__(self) -> int:
        return hash((self.task, str(self.policy)))

    def execute_subtrajectory(
        self,
        env: Env,
        active_task: Task,
        sampler_config: EnvSamplerConfig,
    ) -> List[Trajectory]:
        subtask = self.task
        subtask_policy = self.policy
        # NOTE this assumes ground truth access to when a sub-task finishes
        # Ideally each policy predicts this as well
        sampler = EnvSampler(
            env=env,
            task_sampler=TaskSampler(tasks=[subtask]),
            do_reset=False,
        )
        # retry subtask until successful or out of timesteps
        reset_task = ResetTask()
        subtrajs: List[Trajectory] = []
        for _ in range(self.MAX_RETRIES):
            subtraj = sampler.sample(
                policy=subtask_policy,
                task=subtask,
                episode_id=env.episode_id,
                return_trajectory=True,
                config=sampler_config,
            ).trajectory
            assert subtraj is not None
            if len(subtraj) > 0:
                subtrajs.append(subtraj)
            out_of_time = env.time >= sampler_config.max_time
            success = subtraj.is_successful
            if success or out_of_time or env.done or not self.retry_until_success:
                break
            # if already not touching anything then proceed
            if reset_task.check_success(subtraj):
                continue
            # TODO compute average contact normal, then move in that direction
            # see if a few different directions work, before trying reset action
            # ee_pose = env.get_state().end_effector_pose
            # TODO refactor into an official reset action
            reset_actions = [
                GripperAction(command=False)
                if reset_task.is_robot_collision_free(
                    # if this is true, then only gripper
                    # is in collision with itself
                    state=env.get_state(),
                    allow_gripper_self_collision=True,
                )
                else RESET_ACTION,
            ]
            subtraj = sampler.sample(
                task=reset_task,
                policy=ActionListPolicy(actions=reset_actions),
                episode_id=env.episode_id,
                return_trajectory=True,
                config=sampler_config,
            ).trajectory
            assert subtraj is not None
            subtrajs.append(subtraj)
            out_of_time = env.time >= sampler_config.max_time
            success = active_task.check_success(traj=subtraj)
            if success or out_of_time or env.done:
                break
        return subtrajs


@dataclasses.dataclass(frozen=True)
class GripperAction(Action):
    command: bool
    wait_time: Optional[float] = None

    def execute(
        self,
        env: Env,
        active_task: Task,
        sampler_config: EnvSamplerConfig,
        info: Dict[str, Any],
    ) -> bool:
        env.move_end_effector(
            target_pose=env.robot.end_effector_pose,
            gripper_command=self.command,
            allow_contact=True,
            info=info,
            wait_time=self.wait_time
            if self.wait_time is not None
            else 1 / env.config.ctrl.frequency,  # wait for one ctrl cycle
            use_early_stop_checker=False,
        )
        return False

    def check_feasibility(self, env: Env):
        pass


@dataclasses.dataclass(frozen=True)
class CodeAction(SubtrajectoryAction):
    code_str: str
    query_str: str = ""
    prompt_str: str = ""
    logprob: float = 0.0

    def __str__(self):
        if len(self.code_str) < 30:
            return f"CodeAction(code_str={self.code_str!r})"
        return f'CodeAction(code_str="{self.code_str[:27]}...")'

    def check_feasibility(self, env: Env):
        pass

    def get_info(self) -> Dict[str, Any]:
        console = Console(record=True)
        with console.capture():
            console.print(
                Syntax(
                    self.code_str,
                    "python",
                    theme="one-dark",
                )
            )
        code_action_html = console.export_html(clear=True)
        console = Console(record=True)
        with console.capture():
            console.print(
                Syntax(
                    self.query_str,
                    "python",
                    theme="one-dark",
                )
            )
        code_query_html = console.export_html(clear=True)
        console = Console(record=True)
        with console.capture():
            console.print(
                Syntax(
                    self.prompt_str,
                    "python",
                    theme="one-dark",
                )
            )
        code_prompt_html = console.export_html(clear=True)
        return {
            "code_action": Html(code_action_html),
            "code_query": Html(code_query_html),
            "logprob": self.logprob,
            "code_prompt": Html(code_prompt_html),
        }

    def execute_subtrajectory(
        self, env: Env, active_task: Task, sampler_config: EnvSamplerConfig
    ) -> List[Trajectory]:
        code_str = self.code_str.strip().replace("np.random", "numpy_random")
        global_vars, local_vars = env.get_exec_vars(
            active_task=active_task,
            sampler_config=sampler_config,
        )
        code_action_seed = env.policy_numpy_random.randint(int(2**20))
        global_vars["numpy_random"] = np.random.RandomState(seed=code_action_seed)
        exec_safe(code_str, global_vars=global_vars, local_vars=local_vars)
        return [
            Trajectory(
                episode=tuple(global_vars["subtrajectory_steps"]),
                episode_id=env.episode_id,
                task=active_task,
                policy_id=code_str,
            )
        ]


class ControlType(IntEnum):
    JOINT = 0
    END_EFFECTOR = 1


class RotationType(IntEnum):
    QUATERNION = 0
    ROT_MAT = 1
    UPPER_ROT_MAT = 2


@dataclasses.dataclass(frozen=True)
class ControlConfig:
    frequency: int
    dof: int = 13
    control_type: ControlType = ControlType.END_EFFECTOR
    rotation_type: RotationType = RotationType.ROT_MAT
    t_lookahead: float = 0.0

    def quat2ctrl_orn(self, quat: np.ndarray) -> np.ndarray:
        assert len(quat) == 4
        if self.rotation_type == RotationType.QUATERNION:
            return quat
        elif self.rotation_type == RotationType.ROT_MAT:
            return quaternions.quat2mat(quat).reshape(-1)
        elif self.rotation_type == RotationType.UPPER_ROT_MAT:
            return quaternions.quat2mat(quat)[:2, :].reshape(-1)
        else:
            raise NotImplementedError()

    def ctrlorn2mat(self, orn: np.ndarray) -> np.ndarray:
        if self.rotation_type == RotationType.QUATERNION:
            return quaternions.quat2mat(orn)
        elif self.rotation_type == RotationType.ROT_MAT:
            return orn
        elif self.rotation_type == RotationType.UPPER_ROT_MAT:
            return pt3d.rotation_6d_to_matrix(torch.from_numpy(orn)).numpy().reshape(3, 3)
        else:
            raise NotImplementedError()

    def annotate_ctrl(self, ctrl: np.ndarray) -> Dict[str, Union[np.ndarray, bool]]:
        assert ctrl.shape[-1] == self.dof
        if self.control_type == ControlType.JOINT:
            return {"joints": ctrl}
        else:
            return {
                "ee_pos": ctrl[..., :3],
                "ee_rotmat": self.ctrlorn2mat(ctrl[..., 3:-1]),
                "ee_gripper_comm": ctrl[..., -1],
            }


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class EndEffectorAction(Action):
    gripper_command: bool
    end_effector_position: np.ndarray
    end_effector_orientation: np.ndarray
    allow_contact: bool
    use_early_stop_checker: bool = True
    wait_time: Optional[float] = None
    visualization: Optional[Dict[str, Any]] = None

    """
    When `end_effector_orientation == euler(0,0,0)`, then gripper
    should be pointing down.
    """

    def __eq__(self, other: object) -> bool:
        if type(other) is not EndEffectorAction:
            return False
        return (
            self.gripper_command == other.gripper_command
            and np.allclose(self.end_effector_position, other.end_effector_position)
            and np.allclose(self.end_effector_orientation, other.end_effector_orientation)
            and self.allow_contact == other.allow_contact
        )

    @validator("end_effector_position")
    @classmethod
    def position_shape(cls, v: np.ndarray):
        if v.shape != (3,):
            raise ValueError("end_effector_position.shape should be (3,)")
        return v

    @validator("end_effector_orientation")
    @classmethod
    def orientation_shape(cls, v: np.ndarray):
        if v.shape != (4,):
            raise ValueError("end_effector_orientation.shape should be (4,)")
        return v

    def get_info(self) -> Dict[str, Any]:
        if self.visualization is not None:
            return {k: v for k, v in self.visualization.items() if k[:2] != "__"}
        return {}

    def __str__(self) -> str:
        return (
            "EndEffectorAction("
            + str(
                Pose(
                    position=self.end_effector_position,
                    orientation=self.end_effector_orientation,
                )
            )
            + f", grip={self.gripper_command}, contact={self.allow_contact})"
        )

    def __hash__(self) -> int:
        return hash(
            (
                str(self),
                self.allow_contact,
                self.gripper_command,
                *list(self.end_effector_orientation),
                *list(self.end_effector_position),
            )
        )

    def check_feasibility(self, env: Env):
        # first check for kinematic reachability
        result = env.robot.inverse_kinematics(
            Pose(
                position=self.end_effector_position,
                orientation=self.end_effector_orientation,
            )
        )
        if result is None:
            raise Action.InfeasibleAction(
                action_class_name=type(self).__name__, message="IK Failed"
            )
        # next, check for collision for motion planning validity
        if self.allow_contact:
            # allow collision, so no motion planner needed
            return
        if env.config.fallback_on_rrt_fail:
            # even when RRT fails, the policy can fall back to
            # no motion planning if needed, so don't waste effort
            # checking for collision
            return

        from scalingup.environment.mujoco.mujocoEnv import MujocoEnv

        assert issubclass(type(env), MujocoEnv)
        mj_env = typing.cast(MujocoEnv, env)
        physics = mj_env.mj_physics.copy(share_model=True)
        mujoco_robot = mj_env.mujoco_robot
        # simulate the gripper open/close

        ctrl = physics.control()
        ctrl[-1] = (
            env.robot.gripper_close_ctrl_val
            if self.gripper_command
            else env.robot.gripper_open_ctrl_val
        )
        physics.set_control(ctrl)
        for _ in range(env.config.ee_action_num_grip_steps):
            physics.step()

        grasp_obj_id = -1
        grasp_pose = None
        if self.gripper_command and mujoco_robot.binary_gripper_command:
            # currently grasping object, and will need motion planning to
            # get to the desired ee pose
            grasp_obj_id = mujoco_robot.get_grasped_obj_id(physics=physics)
            grasp_pose = mujoco_robot.get_grasp_pose(physics=physics)
        in_collision = mujoco_robot.check_collision(
            joints=result,
            physics=physics,
            grasp_obj_id=grasp_obj_id,
            grasp_pose=grasp_pose,
            detect_grasp=False,
        )
        if in_collision:
            relevant_link_names = [mujoco_robot.body_model.name]
            if grasp_obj_id != -1:
                relevant_link_names.append(physics.model.body(grasp_obj_id).name)
            collided_pair_names = [
                (link1, link2)
                for link1, link2 in mujoco_robot.get_unfiltered_collided_pairs_names(
                    joints=result,
                    physics=physics,
                    grasp_obj_id=grasp_obj_id,
                    grasp_pose=grasp_pose,
                    detect_grasp=False,
                )
                if any(
                    link_name in link1 or link_name in link2
                    for link_name in relevant_link_names
                )
            ]
            logging.debug(
                f"{self} collision: "
                + "\n-".join(
                    f"{link1} and {link2}" for link1, link2 in collided_pair_names
                )
            )
            raise Action.InfeasibleAction(
                action_class_name=type(self).__name__,
                message=f"{self} collision: "
                + ", ".join(
                    f"{link1} and {link2}" for link1, link2 in collided_pair_names
                ),
            )

    def execute(
        self,
        env: Env,
        active_task: Task,
        sampler_config: EnvSamplerConfig,
        info: Dict[str, Any],
    ) -> bool:
        ee_action = self
        if env.discretizer is not None:
            logging.info(f"discretizing actions into {env.discretizer.grid_shape}")
            new_ee_pos = self.end_effector_position.copy()
            new_ee_orn = self.end_effector_orientation.copy()
            new_ee_pos, new_ee_orn = env.discretize_action(pos=new_ee_pos, orn=new_ee_orn)
            pos_err = np.linalg.norm(new_ee_pos - self.end_effector_position)
            orn_err = (
                Rotation.from_quat(new_ee_orn[[1, 2, 3, 0]])
                * Rotation.from_quat(self.end_effector_orientation[[1, 2, 3, 0]]).inv()
            ).magnitude()
            logging.info(
                f"discretized action: pos_err={pos_err:.5f}, orn_err={orn_err:.3f}"
            )
            ee_action = EndEffectorAction(
                allow_contact=self.allow_contact,
                gripper_command=self.gripper_command,
                end_effector_position=new_ee_pos,
                end_effector_orientation=new_ee_orn,
            )
        env.move_end_effector(
            target_pose=Pose(
                position=ee_action.end_effector_position,
                orientation=ee_action.end_effector_orientation,
            ),
            gripper_command=ee_action.gripper_command,
            allow_contact=ee_action.allow_contact,
            info=info,
            wait_time=self.wait_time,
            use_early_stop_checker=self.use_early_stop_checker,
        )
        return False


RESET_ACTION = EndEffectorAction(
    end_effector_position=np.array([0.2, 0.0, 0.3]),
    end_effector_orientation=euler.euler2quat(0, 0, 0),
    gripper_command=False,
    allow_contact=True,
)


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class ControlAction(Action):
    """
    to be directly passed to data.ctrl, or a list of such items
    """

    value: np.ndarray
    timestamps: np.ndarray  # in position control, this is the time to reach the position
    config: ControlConfig
    use_early_stop_checker: bool = True

    target_ee_actions: Optional[
        List[Optional[EndEffectorAction]]
    ] = None  # for keypoint extraction

    def __str__(self) -> str:
        return (
            f"ControlAction(value.shape={self.value.shape},"
            + f" frequency={self.config.frequency})"
        )

    @validator("timestamps")
    @classmethod
    def timestamps_are_strictly_monotonic(cls, v: np.ndarray):
        diffs = np.diff(v)
        if (diffs <= 0).any():
            raise ValueError("Control timestamps is not monotonic")
        return v

    @validator("timestamps")
    @classmethod
    def timestamps_and_ctrls_match(cls, v: np.ndarray, values):
        if len(v) != len(values["value"]):
            raise ValueError("timestamps don't match with ctrls")
        return v

    def check_feasibility(self, env: Env):
        # always feasible
        pass

    @property
    def ee_pos(self) -> np.ndarray:
        assert self.config.control_type == ControlType.END_EFFECTOR
        return self.value[:, :3]

    @property
    def ee_rotmat(self) -> np.ndarray:
        assert self.config.control_type == ControlType.END_EFFECTOR
        ee_orientations = self.value[:, 3:-1]
        if self.config.rotation_type == RotationType.QUATERNION:
            return np.stack(
                [quaternions.quat2mat(ee_quat) for ee_quat in ee_orientations]
            )
        elif self.config.rotation_type == RotationType.ROT_MAT:
            return ee_orientations.reshape(-1, 3, 3)
        elif self.config.rotation_type == RotationType.UPPER_ROT_MAT:
            return (
                pt3d.rotation_6d_to_matrix(torch.from_numpy(ee_orientations))
                .numpy()
                .reshape(-1, 3, 3)
            )
        else:
            raise NotImplementedError(
                "Unknown rotation type: {}".format(self.config.rotation_type)
            )

    @property
    def ee_gripper_comm(self) -> np.ndarray:
        assert self.config.control_type == ControlType.END_EFFECTOR
        return self.value[:, -1]

    def get_prev_ctrl(self, t: float) -> Tuple[float, np.ndarray]:
        next_idx = ((t - self.timestamps) > 0).sum()
        idx = min(max(next_idx - 1, 0), len(self.timestamps) - 1)
        return self.timestamps[idx], self.value[idx]

    def get_target_ctrl(
        self, t: float
    ) -> Tuple[float, np.ndarray, Optional[EndEffectorAction]]:
        next_idx = ((t - self.timestamps) > 0).sum()
        idx = min(max(next_idx, 0), len(self.timestamps) - 1)
        return (
            self.timestamps[idx],
            self.value[idx],
            self.target_ee_actions[idx] if self.target_ee_actions is not None else None,
        )

    def get_interpolated_joint_ctrl(
        self, t: float, robot: Robot, env_config: EnvConfig
    ) -> np.ndarray:
        """
        time
        --------------------------------------->
        x              x        ^     x        waypoints
        <--------------><------>|
           ctrl cycle    alpha  |
                               `t` (with lookahead)
        """
        t += self.config.t_lookahead
        prev_time, prev_ctrl = self.get_prev_ctrl(t=t)
        target_time, target_ctrl, target_ee_action = self.get_target_ctrl(t=t)
        ctrl_interval = target_time - prev_time
        assert ctrl_interval >= 0, "interpolated ctrl timestamps is negative"
        if ctrl_interval == 0.0:
            alpha = 0.0
        else:
            alpha = (t - prev_time) / ctrl_interval
        if self.config.control_type == ControlType.JOINT:
            return target_ctrl * alpha + prev_ctrl * (1 - alpha)
        else:
            ee_pos = target_ctrl[:3] * alpha + prev_ctrl[:3] * (1 - alpha)
            ee_grip = target_ctrl[-1] * alpha + prev_ctrl[-1] * (1 - alpha)
            if self.config.rotation_type == RotationType.QUATERNION:
                # convert between quaternion conventions
                prev_ee_rot = Rotation.from_quat(prev_ctrl[3:-1][[1, 2, 3, 0]])
                target_ee_rot = Rotation.from_quat(target_ctrl[3:-1][[1, 2, 3, 0]])
            elif self.config.rotation_type == RotationType.ROT_MAT:
                prev_ee_rot = Rotation.from_matrix(prev_ctrl[3:-1].reshape(3, 3))
                target_ee_rot = Rotation.from_matrix(target_ctrl[3:-1].reshape(3, 3))
            elif self.config.rotation_type == RotationType.UPPER_ROT_MAT:
                prev_ee_rot = Rotation.from_matrix(
                    pt3d.rotation_6d_to_matrix(torch.from_numpy(prev_ctrl[3:-1])).numpy()
                )
                target_ee_rot = Rotation.from_matrix(
                    pt3d.rotation_6d_to_matrix(
                        torch.from_numpy(target_ctrl[3:-1])
                    ).numpy()
                )
            else:
                raise NotImplementedError(
                    "Unknown rotation type: {}".format(self.config.rotation_type)
                )

            orientation_slerp = Slerp(
                times=[0, 1],
                rotations=Rotation.concatenate([prev_ee_rot, target_ee_rot]),
            )
            ee_rot = orientation_slerp([alpha])[0]
            from scalingup.environment.mujoco.mujocoRobot import MujocoRobot

            # conversion
            if env_config.solve_ee_inplace and issubclass(type(robot), MujocoRobot):
                mj_robot = typing.cast(MujocoRobot, robot)
                joint_pos = mj_robot.inverse_kinematics(
                    pose=Pose(
                        position=ee_pos,
                        orientation=ee_rot.as_quat()[[3, 0, 1, 2]],
                    ),
                    inplace=True,
                )
            else:
                # TODO IK is extremely expensive. Switch to OSC
                joint_pos = robot.inverse_kinematics(
                    pose=Pose(
                        position=ee_pos,
                        orientation=ee_rot.as_quat()[[3, 0, 1, 2]],
                    )
                )
            if joint_pos is None:
                logging.warning(
                    "EE control IK failed. Falling back to current joint config"
                )
                # fail quietly
                joint_pos = robot.joint_config.copy()
            return np.concatenate(
                [joint_pos, [ee_grip]],
                axis=0,
            )

    def execute(
        self,
        env: Env,
        active_task: Task,
        sampler_config: EnvSamplerConfig,
        info: Dict[str, Any],
    ) -> bool:
        assert self.config == env.config.ctrl
        early_stop_checker = None
        if self.use_early_stop_checker:
            goal_gripper_command = env.robot.gripper_ctrl_to_binary_gripper_command(
                gripper_ctrl=self.value[-1, -1]
            )
            early_stop_checker = env.get_dropped_grasp_early_stop_checker(
                goal_gripper_command=goal_gripper_command
            )
        env.execute_ctrl(
            ctrl=self,
            early_stop_checker=early_stop_checker,
        )
        return False

    def __hash__(self):
        return hash((*self.value.astype(float).reshape(-1).tolist(), self.config))

    @property
    def start_time(self) -> float:
        return self.timestamps[0]

    @property
    def end_time(self) -> float:
        return self.timestamps[-1]

    def combine(self, new_ctrl: ControlAction) -> ControlAction:
        # insert new ctrl values into self, accounting for start time differences
        ctrl_times = {
            ctrl_time: (ctrl_val, target_ee_action)
            for ctrl_val, ctrl_time, target_ee_action in zip(
                self.value,
                self.timestamps,
                self.target_ee_actions
                if self.target_ee_actions is not None
                else [None] * len(self.value),
            )
        }
        # overwrite old values
        for ctrl_val, ctrl_time, target_ee_action in zip(
            new_ctrl.value,
            new_ctrl.timestamps,
            new_ctrl.target_ee_actions
            if new_ctrl.target_ee_actions is not None
            else [None] * len(new_ctrl.value),
        ):
            ctrl_times[ctrl_time] = (ctrl_val, target_ee_action)
        timestamps = sorted(ctrl_times.keys())
        return ControlAction(
            value=np.stack([ctrl_times[t][0] for t in timestamps]),
            config=self.config,
            timestamps=np.array(timestamps),
            target_ee_actions=[ctrl_times[t][1] for t in timestamps],
        )


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class PlanarPick(CompositeEndEffectorAction):
    pixel: Pixel
    z_angle: float
    vision_sensor_output: VisionSensorOutput
    visualization: Optional[Dict[str, np.ndarray]] = None
    prepick_height: float = 0.1

    def __hash__(self) -> int:
        return hash(
            (
                str(self),
                self.pixel,
                self.z_angle,
                *list(self.vision_sensor_output.rgb.reshape(-1)),
            )
        )

    class InvalidPickPixel(Exception):
        def __init__(self):
            super().__init__("Invalid pick pixel")

    @validator("visualization")
    @classmethod
    def visualization_items(cls, v: Dict[str, np.ndarray], values):
        if v is None:
            v = {}
        vision_sensor_output: VisionSensorOutput = values["vision_sensor_output"]
        pixel: Pixel = values["pixel"]
        h, w = vision_sensor_output.rgb.shape[:2]
        thickness = int(np.ceil(min(h, w) * 0.01))
        rad = values["z_angle"] * np.pi / 180
        left_finger_pixel = np.array(pixel) + (
            np.array(
                [
                    np.sin(rad),
                    np.cos(rad),
                ]
            )
            * thickness
            * 5.0
        ).astype(int)
        right_finger_pixel = np.array(pixel) - (
            np.array(
                [
                    np.sin(rad),
                    np.cos(rad),
                ]
            )
            * thickness
            * 5.0
        ).astype(int)
        v["obs_overlay"] = cv2.line(
            img=vision_sensor_output.rgb,
            pt1=tuple(right_finger_pixel),
            pt2=tuple(left_finger_pixel),
            color=(0, 255, 0, 255),
            thickness=thickness,
        )
        return v

    def get_info(self) -> Dict[str, Any]:
        if self.visualization is not None:
            return {
                "top_down_pick": WandbImage(
                    data_or_path=self.visualization["obs_overlay"]
                )
            }
        return {}

    @property
    def pick_point_index(self) -> int:
        """
        Get index for chosen pixel in point cloud
        after vision sensor output is projected into
        a point cloud.
        """
        # TODO when move to real world
        # if not valid_points_mask[point_idx]:
        #     # pixel is at an invalid position
        #     raise self.InvalidPickPixel()
        # account for filtered point cloud
        # count all valid pixels before this pixel
        # point_idx = sum(valid_points_mask[:point_idx].astype(int))
        return self.pixel[1] * self.vision_sensor_output.rgb.shape[1] + self.pixel[0]

    @property
    def pick_position(self) -> Point3D:
        # convert pixel to 3D position
        point_cloud = self.vision_sensor_output.point_cloud
        return point_cloud.xyz_pts[self.pick_point_index]

    def to_end_effector_actions(
        self, obs: Observation, numpy_random: np.random.RandomState, env: Env
    ) -> List[EndEffectorAction]:
        pick_position = np.array(self.pick_position)
        z_angle_rad = self.z_angle / 180 * np.pi
        orientation = euler.euler2quat(0, 0, z_angle_rad)
        return [
            EndEffectorAction(
                allow_contact=False,
                gripper_command=False,
                end_effector_position=pick_position + [0, 0, self.prepick_height],
                end_effector_orientation=orientation,
            ),
            EndEffectorAction(
                allow_contact=False,
                gripper_command=False,
                end_effector_position=pick_position,
                end_effector_orientation=orientation,
            ),
            EndEffectorAction(
                allow_contact=False,
                gripper_command=True,
                end_effector_position=pick_position,
                end_effector_orientation=orientation,
            ),
            EndEffectorAction(
                allow_contact=False,
                gripper_command=True,
                end_effector_position=pick_position + [0, 0, self.prepick_height],
                end_effector_orientation=orientation,
            ),
        ]


class GraspObj(Task):
    def __init__(
        self,
        link_path: str,
        desc_template: str = "grasp the {link_path}",
    ):
        self.link_path = link_path
        super().__init__(
            desc=desc_template.format(link_path=self.link_path),
        )

    def check_success(self, traj: Trajectory) -> bool:
        # only successful if, in the last step of the episode,
        # the object is colliding with the robot gripper, and
        # not colliding with any other object
        # TODO: implement with contact force
        # TODO: unify all grasp information
        # return self.link_path in traj.final_state.grasped_objects
        return GraspObj.is_obj_only_touching_end_effector(
            state=traj.final_state,
            link_path=self.link_path,
        )

    def get_reward(self, state: EnvState) -> float:
        ee_pos = state.end_effector_pose.position
        obj_pos = state.get_pose(key=self.link_path).position
        dist_between_grip_and_obj = np.linalg.norm(obj_pos - ee_pos)
        return np.exp(-dist_between_grip_and_obj * 5) + float(
            GraspObj.is_obj_only_touching_end_effector(
                link_path=self.link_path, state=state
            )
        )

    @staticmethod
    def is_obj_only_touching_end_effector(link_path: str, state: EnvState) -> bool:
        obj_name = link_path.split(LINK_SEPARATOR_TOKEN)[0]
        link_state = state.object_states[obj_name].link_states[link_path]
        contacts = set(
            filter(
                lambda c: c.other_link == link_path and c.other_name == obj_name,
                state.end_effector_contacts,
            )
        )
        end_effector_touching = len(contacts) > 0
        only_touching_robot = all(
            contact.other_name == state.robot_name for contact in link_state.contacts
        )
        return end_effector_touching and only_touching_robot

    @property
    def name(self) -> str:
        return f"grasp {self.link_path}"


def closest_point_on_line(line_origin, line_vector, point):
    # https://math.stackexchange.com/questions/62633/orthogonal-projection-of-a-point-onto-a-line
    line_vector /= np.linalg.norm(line_vector)
    proj = np.outer(line_vector, line_vector) / np.inner(line_vector, line_vector)
    return proj @ point + (np.identity(3) - proj) @ np.array(line_origin)


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class RevoluteJointAction(SubpolicyAction):
    link_path: str
    towards_max: bool = True
    rotate_gripper: Optional[bool] = None

    def get_contacts(self, state: EnvState) -> List[Contact]:
        obj_name = self.link_path.split(LINK_SEPARATOR_TOKEN)[0]
        contacts = sorted(
            filter(
                lambda c: c.other_name == obj_name,
                state.end_effector_contacts,
            ),
            key=lambda c: c.position,
        )
        return contacts

    def get_subtrajectory_policy(self, env: Env) -> Policy:
        state = env.obs.state
        obj_name = self.link_path.split(LINK_SEPARATOR_TOKEN)[0]
        obj_state = state.object_states[obj_name]
        link_state = obj_state.link_states[self.link_path]
        contacts = self.get_contacts(state=state)
        # grasp a random point on surface
        grasp_point: Contact = contacts[0]
        grasp_position = np.array(grasp_point.position)
        joint_state: JointState = [
            joint_state
            for joint_state in obj_state.joint_states.values()
            if joint_state.child_link == link_state.link_path
        ][0]
        center = closest_point_on_line(
            line_origin=np.array(joint_state.position),
            line_vector=quaternions.quat2mat(joint_state.orientation)
            @ np.array(joint_state.axis),
            point=grasp_position,
        )
        link_state = obj_state.link_states[self.link_path]
        link2world = affines.compose(
            T=center,
            R=np.identity(3),
            Z=np.ones(3),
        )
        grasp2link = affines.compose(
            T=grasp_position - center,
            R=np.identity(3),
            Z=np.ones(3),
        )
        ee_actions = []
        start_angle = 0
        end_angle = joint_state.max_value
        if not self.towards_max:
            end_angle = -joint_state.max_value

        revolute_radius = np.linalg.norm(grasp_position - center)
        if self.rotate_gripper is None:
            rotate_gripper = self.rotate_gripper
        else:
            rotate_gripper = revolute_radius > env.config.rotate_gripper_threshold
        # NOTE: heuristic for how many steps to discretize the revolute joint angles into
        num_steps = int(
            np.ceil(
                (np.abs(start_angle - end_angle) * revolute_radius)
                * env.config.num_steps_multiplier
            )
        )
        num_steps = max(env.config.min_steps, num_steps)
        step_angle = (end_angle - start_angle) / num_steps
        initial_rotation = affines.decompose(
            link2world
            @ affines.compose(
                T=np.zeros(3),
                R=euler.euler2mat(
                    *(
                        quaternions.quat2mat(joint_state.orientation)
                        @ (np.array(joint_state.axis) * start_angle)
                    )
                ),
                Z=np.ones(3),
            )
            @ grasp2link
        )[1]
        # account for base orientation after grasping
        base_orn = quaternions.qmult(
            quaternions.mat2quat(initial_rotation.T),
            state.end_effector_pose.orientation,
        )

        for angle in np.arange(start=start_angle, stop=end_angle, step=step_angle):
            euler_rotation = quaternions.quat2mat(joint_state.orientation) @ (
                np.array(joint_state.axis) * angle
            )
            revolute_transform = affines.compose(
                T=np.zeros(3), R=euler.euler2mat(*euler_rotation), Z=np.ones(3)
            )
            grasp2world = link2world @ revolute_transform @ grasp2link
            T, R = affines.decompose(grasp2world)[:2]
            ee_action = EndEffectorAction(
                end_effector_position=T,
                end_effector_orientation=quaternions.qmult(
                    quaternions.mat2quat(R), base_orn
                )
                if rotate_gripper
                else base_orn,
                allow_contact=True,
                gripper_command=True,
            )
            try:
                ee_action.check_feasibility(env=env)
                ee_actions.append(ee_action)
            except Action.InfeasibleAction as e:
                logging.error(f"Infeasible revolution action: {e}")
                break
        if len(ee_actions) == 0:
            # all revolute joint actions failed
            raise Action.FailedExecution(
                f"Failed to find any feasible revolute joint actions for {self.link_path}"
            )
        if env.config.ctrl.control_type != ControlType.END_EFFECTOR:
            raise NotImplementedError(
                f"Control type {env.config.ctrl.control_type} not supported for revolute joints"
            )
        from scalingup.environment.mujoco.mujocoEnv import MujocoEnv
        from scalingup.environment.mujoco.ur5 import UR5

        assert issubclass(type(env.robot), UR5)
        ur5_robot = typing.cast(UR5, env.robot)
        control_action = ControlAction(
            value=np.array(
                [
                    [
                        *ee_action.end_effector_position.tolist(),
                        *env.config.ctrl.quat2ctrl_orn(
                            quat=ee_action.end_effector_orientation
                        ).tolist(),
                        ur5_robot.gripper_close_ctrl_val
                        if ee_action.gripper_command
                        else ur5_robot.gripper_open_ctrl_val,
                    ]
                    for ee_action in ee_actions
                ]  # type: ignore
            ),
            config=env.config.ctrl,
            timestamps=np.linspace(
                env.time,
                env.time + len(ee_actions) / env.config.ctrl.frequency,
                len(ee_actions),
                endpoint=False,
            )
            + 1 / env.config.ctrl.frequency,
            target_ee_actions=ee_actions,  # type: ignore
        )
        return ActionListPolicy(
            actions=[
                control_action,
                EndEffectorAction(
                    end_effector_position=ee_actions[-1].end_effector_position,
                    end_effector_orientation=ee_actions[-1].end_effector_orientation,
                    allow_contact=True,
                    gripper_command=False,
                ),
            ]
        )

    def check_feasibility(self, env: Env):
        if len(self.get_contacts(state=env.get_state())) == 0:
            raise Action.InfeasibleAction(
                action_class_name=self.__class__.__name__,
                message="No contact with {}".format(self.link_path),
                stop_episode=False,
            )


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class PrismaticJointAction(CompositeEndEffectorAction):
    link_path: str
    towards_max: bool = True

    def to_end_effector_actions(
        self, obs: Observation, numpy_random: np.random.RandomState, env: Env
    ) -> List[EndEffectorAction]:
        state = obs.state
        obj_name = self.link_path.split(LINK_SEPARATOR_TOKEN)[0]
        obj_state = obs.state.object_states[obj_name]
        link_state = obj_state.link_states[self.link_path]
        joint_states: List[JointState] = [
            joint_state
            for joint_state in obj_state.joint_states.values()
            if joint_state.child_link == link_state.link_path
        ]
        assert len(joint_states) > 0
        for joint_state in joint_states:
            start_pose = state.end_effector_pose
            world_axis_direction = -np.array(joint_state.axis)
            world_axis_direction = quaternions.rotate_vector(
                world_axis_direction, joint_state.orientation
            )

            if self.towards_max:
                final_pos = start_pose.position + world_axis_direction * (
                    joint_state.current_value - joint_state.max_value
                )
            else:
                final_pos = start_pose.position + world_axis_direction * (
                    joint_state.current_value - joint_state.min_value
                )
            return [
                EndEffectorAction(
                    allow_contact=True,
                    gripper_command=True,
                    end_effector_position=final_pos,
                    end_effector_orientation=start_pose.orientation,
                ),
                EndEffectorAction(
                    allow_contact=True,
                    gripper_command=False,  # release after done
                    end_effector_position=final_pos,
                    end_effector_orientation=start_pose.orientation,
                ),
            ]
        return []


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class LinkAction(SubpolicyAction):
    link_path: str

    @property
    def short_link_path(self) -> str:
        return [
            linkname
            for linkname in self.link_path.split(LINK_SEPARATOR_TOKEN)[-1].split(
                MJCF_NEST_TOKEN
            )
            if len(linkname) > 0
        ][-1]

    def __str__(self) -> str:
        output = f"{type(self).__name__}(link={self.short_link_path!r}"
        for k, v in self.str_dict.items():
            output += f", {k}={str(v)}"
        output += ")"
        return output

    @property
    def str_dict(self) -> Dict[str, Any]:
        return {}


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class LinkPoseAction(LinkAction, CompositeEndEffectorAction):
    pose: Pose

    @property
    def str_dict(self) -> Dict[str, Any]:
        return {"pose": self.pose}


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class GraspLinkPoseAction(LinkPoseAction):
    pose: Pose
    with_backup: bool = True
    backup_distance: float = 0.1

    @property
    def str_dict(self) -> Dict[str, Any]:
        return {
            **super().str_dict,
            "dist": f"{self.backup_distance:.2f}",
            "with_backup": self.with_backup,
        }

    def to_end_effector_actions(
        self, obs: Observation, numpy_random: np.random.RandomState, env: Env
    ) -> List[EndEffectorAction]:
        link_pose = obs.state.get_pose(key=self.link_path)
        grasp_pose = self.pose.transform(link_pose.matrix)
        rotmat = quaternions.quat2mat(grasp_pose.orientation)
        grasp_direction = rotmat @ np.array([0, 0, 1])
        grasp_backup_pose = Pose(
            position=grasp_pose.position + grasp_direction * self.backup_distance,
            orientation=grasp_pose.orientation,
        )
        ee_commands_sequence = [
            EndEffectorAction(
                allow_contact=False,  # Don't allow contact before activating gripper
                gripper_command=False,
                end_effector_position=grasp_pose.position,
                end_effector_orientation=get_best_orn_for_gripper(
                    reference_orn=euler.euler2quat(0, 0, 0),
                    query_orn=grasp_pose.orientation,
                ),
            ),
            EndEffectorAction(
                allow_contact=True,  # now allow contact
                gripper_command=True,  # and close gripper
                end_effector_position=grasp_pose.position,
                end_effector_orientation=get_best_orn_for_gripper(
                    reference_orn=euler.euler2quat(0, 0, 0),
                    query_orn=grasp_pose.orientation,
                ),
            ),
        ]
        if self.with_backup:
            ee_commands_sequence.append(
                EndEffectorAction(
                    allow_contact=True,
                    gripper_command=True,
                    end_effector_position=grasp_backup_pose.position,
                    end_effector_orientation=get_best_orn_for_gripper(
                        reference_orn=euler.euler2quat(0, 0, 0),
                        query_orn=grasp_backup_pose.orientation,
                    ),
                )
            )
        return ee_commands_sequence

    def check_feasibility(self, env: Env):
        super().check_feasibility(env)
        """
        Good grasps must satisfy the following:
        1. the grasp position is at least past the surface of the sampled point (minus distance)
        2. At the grasp pose, when gripper is opened, the robot is collision free
        3. At the grasp pose, when gripper is closed, the robot's ee is colliding with object

        1) is satisfied by setting a good `pushin_distance` (see below).
        2) is checked in super().check_feasibility(env) because `allow_contact=False`
        in the first action.
        3) will be checked here.
        """
        ee_actions = self.to_end_effector_actions(
            obs=env.obs, numpy_random=env.policy_numpy_random, env=env
        )
        grasp_action = ee_actions[1]
        result = env.robot.inverse_kinematics(
            Pose(
                position=grasp_action.end_effector_position,
                orientation=grasp_action.end_effector_orientation,
            )
        )
        from scalingup.environment.mujoco.mujocoEnv import MujocoEnv
        from scalingup.environment.mujoco.utils import get_part_path

        assert issubclass(type(env), MujocoEnv)
        mj_env = typing.cast(MujocoEnv, env)
        physics = mj_env.mj_physics.copy(share_model=True)
        assert result is not None
        # IK has been checked in super().check_feasibility(env) already
        mj_env.mujoco_robot.set_joint_config(
            joints=result,
            physics=physics,
        )
        # NOTE now have to close gripper and step simulation a bit
        # to simulate a grasp
        # TODO add better API for this
        last_joint_command = mj_env.mujoco_robot.last_joints
        mj_env.mujoco_robot.move_to_joints(target_joints=result)
        physics.set_control(mj_env.mujoco_robot.get_joint_target_control())

        ctrl = physics.control()
        ctrl[-1] = env.robot.gripper_close_ctrl_val
        physics.set_control(ctrl)
        for _ in range(env.config.ee_action_num_grip_steps):
            physics.step()

        gripper_touching_target_obj = False
        for _ in range(env.config.ee_action_num_grip_steps):
            # step for a bit to simulate grasp
            physics.step()
            (
                collided_body1,
                collided_body2,
            ) = mj_env.mujoco_robot.get_unfiltered_collided_pairs(
                physics=physics,
            )
            collided_pairs = [
                {
                    get_part_path(model=physics.model, body=physics.model.body(link1)),
                    get_part_path(model=physics.model, body=physics.model.body(link2)),
                }
                for link1, link2 in zip(collided_body1, collided_body2)
            ]
            gripper_touching_target_obj = (
                len(
                    [
                        1
                        for pair in collided_pairs
                        if self.link_path in pair
                        and any(
                            ee_path in pair
                            for ee_path in env.obs.state.robot_gripper_links
                        )
                    ]
                )
                > 0
            )
            if gripper_touching_target_obj:
                # success!
                break
            gripper_touching_other_obj = (
                len(
                    [
                        1
                        for pair in collided_pairs
                        if self.link_path not in pair
                        and any(
                            ee_path in pair
                            for ee_path in env.obs.state.robot_gripper_links
                        )
                    ]
                )
                > 0
            )
            if gripper_touching_other_obj:
                raise Action.InfeasibleAction(
                    action_class_name=type(self).__name__,
                    message=f"Grasp {str(grasp_action)}  touch other object",
                )
        mj_env.mujoco_robot.last_joints = last_joint_command

        if not gripper_touching_target_obj:
            raise Action.InfeasibleAction(
                action_class_name=type(self).__name__,
                message=f"Grasp {str(grasp_action)} doesn't " + f"touch {self.link_path}",
            )


def normal_to_forward_quat(normal: np.ndarray) -> np.ndarray:
    pos = np.array([0, 0, 0])
    forward = pos + normal
    u = np.array([0, 0, 1])
    if np.allclose(normal, [0, 0, 1]) or np.allclose(normal, [0, 0, -1]):
        u = np.array([0, 1, 0])
    s = np.cross(forward, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, forward)
    view_matrix = np.array(
        [
            s[0],
            u[0],
            -forward[0],
            0,
            s[1],
            u[1],
            -forward[1],
            0,
            s[2],
            u[2],
            -forward[2],
            0,
            -np.dot(s, pos),
            -np.dot(u, pos),
            np.dot(forward, pos),
            1,
        ]
    )
    view_matrix = view_matrix.reshape(4, 4).T
    pose_matrix = np.linalg.inv(view_matrix)
    return quaternions.mat2quat(affines.decompose(pose_matrix)[1])


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class HighLevelLinkAction(LinkAction):
    """
    This action should be able to "compile" down to a lower-level
    more precise action class. This class exists to enable a LLM-friendly
    interface to lower-level actions
    """

    def get_subtrajectory_policy(self, env: Env) -> Policy:
        # use maximum resolution (MAX_OFFSCREEN_HEIGHT) at the same
        # aspect ratio to get the best point cloud
        res = (
            (
                np.array(env.config.obs_dim)
                / max(env.config.obs_dim)
                * MAX_OFFSCREEN_HEIGHT
            )
            .astype(int)
            .tolist()
        )
        images = env.render(obs_dim=(res[0], res[1]))

        # combine all point clouds and get out the relevant link
        link_point_clouds = [
            sensor_output.point_cloud[self.link_path] for sensor_output in images.values()
        ]
        link_point_cloud: PointCloud = sum(
            link_point_clouds[1:], start=link_point_clouds[0]
        )
        if len(link_point_cloud) == 0:
            raise Action.InfeasibleAction(
                action_class_name=type(self).__name__,
                message=f"Link {self.link_path!r} is not visible",
            )
        candidate = self.get_feasible_candidate(
            link_point_cloud=link_point_cloud,
            env=env,
        )
        return candidate

    @abstractmethod
    def get_feasible_candidate(self, link_point_cloud: PointCloud, env: Env) -> Policy:
        pass

    def check_feasibility(self, env: Env):
        """
        if `env.config.num_action_candidates` is large enough, then some action should always
        pass kinematic feasibility checks. When all candidates fail,
        `get_feasible_candidate` should be responsible for raising
        Action.InfeasibleAction exceptions.
        """
        pass


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class GraspLinkAction(HighLevelLinkAction):
    with_backup: bool = True
    top_down_grasp_bias: float = 1e-4
    pushin_more: bool = True
    action_primitive_mode: bool = False

    def handle_action_primitive_mode(
        self, link_point_cloud: PointCloud, state: EnvState, env: Env
    ) -> Policy:
        xyz_pts = link_point_cloud.xyz_pts
        # get all points within a planar radius of the center of the obj
        center = np.median(xyz_pts, axis=0)
        assert center.shape == (3,)
        dists = np.linalg.norm(xyz_pts[:, :2] - center[:2], axis=1)
        radius = np.percentile(dists, 1)
        region_pts = xyz_pts[dists < radius]
        # get the highest point in that region
        highest_pt_idx = np.argmax(region_pts[:, 2])
        position = region_pts[highest_pt_idx]
        link_pose = state.get_pose(key=self.link_path)
        position[
            2
        ] -= env.config.grasp_primitive_z_pushin  # grasp slightly past the surface
        grasp_pose = Pose(
            position=position,
            orientation=euler.euler2quat(0, 0, np.pi / 2),
        ).transform(np.linalg.inv(link_pose.matrix))
        return ActionListPolicy(
            actions=[
                GraspLinkPoseAction(
                    link_path=self.link_path,
                    pose=grasp_pose,
                    with_backup=self.with_backup,
                    backup_distance=env.config.grasp_primitive_z_backup,
                )
            ]
        )

    def get_feasible_candidate(self, link_point_cloud: PointCloud, env: Env) -> Policy:
        obs = env.obs
        state = obs.state
        numpy_random = env.policy_numpy_random
        grasp_to_ee = state.end_effector_pose.position - link_point_cloud.xyz_pts
        grasp_to_ee /= np.linalg.norm(grasp_to_ee, axis=1)[:, None]
        normals = link_point_cloud.normals
        pointing_down = normals[:, 2] < 0.0
        if self.action_primitive_mode:
            return self.handle_action_primitive_mode(
                state=state, link_point_cloud=link_point_cloud, env=env
            )

        p = np.ones(shape=len(link_point_cloud), dtype=np.float64)
        if not pointing_down.all():
            # sample the ones point up more often
            p = np.exp(normals[:, 2] * self.top_down_grasp_bias)
        p /= p.sum()

        candidate: Optional[Action] = None
        errors = []

        candidate_indices = numpy_random.choice(
            len(link_point_cloud),
            size=min(env.config.num_action_candidates, len(link_point_cloud)),
            p=p,
            replace=True,
        )
        for i, idx in enumerate(candidate_indices):
            position = link_point_cloud.xyz_pts[idx].copy()
            grasp_to_ee = state.end_effector_pose.position - position
            grasp_to_ee /= np.linalg.norm(grasp_to_ee)
            if pointing_down.all():
                # disallow bottom up grasps, so using point to ee
                # as normal, with some random exploration
                grasp_to_ee += numpy_random.randn(3) * 0.1
                grasp_to_ee /= np.linalg.norm(grasp_to_ee)
                normal = grasp_to_ee
            else:
                normal = normals[idx]

            # orient normal towards end effector
            if np.dot(grasp_to_ee, normal) < 0:
                normal *= -1.0
            # compute base orientation, randomize along Z-axis
            try:
                base_orientation = quaternions.qmult(
                    normal_to_forward_quat(normal),
                    euler.euler2quat(-np.pi, 0, np.pi),
                )
            except np.linalg.LinAlgError as e:
                logging.warning(e)
                base_orientation = euler.euler2quat(0, 0, 0)
            z_angle = numpy_random.uniform(np.pi, -np.pi)
            z_orn = euler.euler2quat(0, 0, z_angle)
            base_orientation = quaternions.qmult(base_orientation, z_orn)
            pregrasp_distance = numpy_random.uniform(0.1, 0.2)
            if self.pushin_more:
                # prioritize grasps that push in more, if possible, but slowly
                # back off to prevent all grasps colliding.
                pushin_distance = (len(candidate_indices) - i) / len(
                    candidate_indices
                ) * (
                    env.config.max_pushin_dist - env.config.min_pushin_dist
                ) + env.config.min_pushin_dist
            else:
                # prioritize grasps that push in less. Useful for "pushing"
                pushin_distance = (
                    i
                    / len(candidate_indices)
                    * (env.config.max_pushin_dist - env.config.min_pushin_dist)
                    + env.config.min_pushin_dist
                )

            # compute grasp pose relative to object
            link_pose = state.get_pose(key=self.link_path)
            grasp_pose = Pose(
                position=position - normal * pushin_distance,
                orientation=base_orientation,
            ).transform(np.linalg.inv(link_pose.matrix))
            candidate = GraspLinkPoseAction(
                link_path=self.link_path,
                pose=grasp_pose,
                with_backup=self.with_backup,
                backup_distance=pushin_distance + pregrasp_distance,
            )
            try:
                candidate.check_feasibility(env=env)
                logging.info(
                    f"[{i}|{env.episode_id}] "
                    + f"pushin_distance: {pushin_distance} ({self.pushin_more})"
                )
                break
            except Action.InfeasibleAction as e:
                errors.append(e)
                candidate = None
                continue
        del normals, link_point_cloud, grasp_to_ee, candidate_indices, p

        # see `tests/test_memory_leak.py``, this needs to be here,
        # otherwise, a few MBs will leak each time
        if candidate is not None:
            return ActionListPolicy(actions=[candidate])
        raise Action.InfeasibleAction(
            action_class_name=type(self).__name__,
            message=f"all candidates failed {self}:"
            + ",".join(list({str(e) for e in errors})[:3]),
        )


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class PlaceOnLinkPoseAction(LinkPoseAction):
    preplace_dist: float

    @property
    def str_dict(self) -> Dict[str, Any]:
        return {**super().str_dict, "dist": f"{self.preplace_dist:.2f}"}

    def to_end_effector_actions(
        self,
        obs: Observation,
        numpy_random: np.random.RandomState,
        env: Env,
    ) -> List[EndEffectorAction]:
        link_pose = obs.state.get_pose(key=self.link_path)
        place_pose = self.pose.transform(link_pose.matrix)
        rotmat = quaternions.quat2mat(place_pose.orientation)
        place_direction = rotmat @ np.array([0, 0, 1])
        preplace_pos = place_pose.position + place_direction * self.preplace_dist
        orn = get_best_orn_for_gripper(
            reference_orn=euler.euler2quat(0, 0, 0),
            query_orn=place_pose.orientation,
        )
        return [
            EndEffectorAction(
                gripper_command=True,
                end_effector_position=preplace_pos,
                end_effector_orientation=orn,
                allow_contact=False,
            ),
            EndEffectorAction(
                gripper_command=True,
                end_effector_position=place_pose.position,
                end_effector_orientation=orn,
                allow_contact=True,
            ),
            EndEffectorAction(
                gripper_command=False,
                end_effector_position=place_pose.position,
                end_effector_orientation=orn,
                allow_contact=True,
            ),
            EndEffectorAction(
                gripper_command=False,
                end_effector_position=preplace_pos,
                end_effector_orientation=orn,
                allow_contact=True,
            ),
        ]


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class PlaceOnLinkAction(HighLevelLinkAction):
    link_path: str
    action_primitive_mode: bool = False

    def handle_action_primitive_mode(
        self,
        link_point_cloud: PointCloud,
        state: EnvState,
    ):
        xyz_pts = link_point_cloud.xyz_pts
        # get all points within a planar radius of the center of the obj
        max_pt = xyz_pts.max(axis=0)
        min_pt = xyz_pts.min(axis=0)
        center = (max_pt - min_pt) / 2 + min_pt
        assert center.shape == (3,)
        dists = np.linalg.norm(xyz_pts[:, :2] - center[:2], axis=1)
        radius = np.percentile(dists, 1)
        region_pts = xyz_pts[dists < radius]
        # get the highest point in that region
        highest_pt_idx = np.argmax(region_pts[:, 2])
        position = region_pts[highest_pt_idx]
        link_pose = state.get_pose(key=self.link_path)
        position[2] += 0.05  # place slightly above surface
        place_pose = Pose(
            position=position,
            orientation=euler.euler2quat(0, 0, 0),
        ).transform(np.linalg.inv(link_pose.matrix))
        return ActionListPolicy(
            actions=[
                PlaceOnLinkPoseAction(
                    link_path=self.link_path,
                    pose=place_pose,
                    preplace_dist=0.1,
                )
            ]
        )

    def get_feasible_candidate(self, link_point_cloud: PointCloud, env: Env) -> Policy:
        obs = env.obs
        state = obs.state
        numpy_random = env.policy_numpy_random
        if self.action_primitive_mode:
            return self.handle_action_primitive_mode(
                state=state, link_point_cloud=link_point_cloud
            )

        grasp_to_ee = state.end_effector_pose.position - link_point_cloud.xyz_pts
        grasp_to_ee /= np.linalg.norm(grasp_to_ee, axis=1)[:, None]
        normals = link_point_cloud.normals
        filter_out_mask = normals[:, 2] < env.config.pointing_up_normal_threshold

        xyz_pts = link_point_cloud.xyz_pts[~filter_out_mask]
        normals = normals[~filter_out_mask]
        candidate: Action = DoNothingAction()
        link_pose = obs.state.get_pose(key=self.link_path)
        # TODO sampling can be biased towards points that belong to larger surface areas
        num_candidates = (~filter_out_mask).sum()
        if num_candidates == 0:
            raise Action.InfeasibleAction(
                action_class_name=type(self).__name__,
                message=f"no candidates for {self.link_path}",
            )
        logging.debug(f"PlaceOnLinkAction has {num_candidates} candidates")
        for attempt, idx in enumerate(
            numpy_random.choice(
                num_candidates, size=env.config.num_action_candidates, replace=True
            )
        ):
            position = xyz_pts[idx].copy()
            # compute base orientation, randomize along Z-axis
            normal = normals[idx].copy()
            normal /= np.linalg.norm(normal)
            try:
                base_orientation = quaternions.qmult(
                    normal_to_forward_quat(normal),
                    euler.euler2quat(-np.pi, 0, np.pi),
                )
            except np.linalg.LinAlgError as e:
                logging.warning(e)
                base_orientation = euler.euler2quat(0, 0, 0)
            z_angle = numpy_random.uniform(np.pi, -np.pi)
            z_orn = euler.euler2quat(0, 0, z_angle)
            base_orientation = quaternions.qmult(base_orientation, z_orn)

            # compute grasp pose relative to object
            place_pose = Pose(
                position=position
                + normal
                * numpy_random.uniform(
                    env.config.place_height_min, env.config.place_height_max
                ),
                orientation=base_orientation,
            ).transform(np.linalg.inv(link_pose.matrix))
            # NOTE: if too low, then gripper + object can also bump and move the
            # object it's placing on top of, like the previously stacked blocks or
            # drawer
            preplace_dist = numpy_random.uniform(
                env.config.preplace_dist_min, env.config.preplace_dist_max
            )
            candidate = PlaceOnLinkPoseAction(
                link_path=self.link_path,
                pose=place_pose,
                preplace_dist=preplace_dist,
            )
            try:
                candidate.check_feasibility(env=env)
                logging.info(f"[{attempt}] preplace_dist: {preplace_dist}")
                return ActionListPolicy(actions=[candidate])
            except Action.InfeasibleAction as e:
                logging.debug(e)
                continue
        raise Action.InfeasibleAction(
            action_class_name=type(self).__name__,
            message=f"all candidates failed: {self}",
        )


class Robot(ABC):
    def __init__(self, init_joints: np.ndarray):
        self.last_joints = init_joints

    @abstractproperty
    def gripper_close_ctrl_val(self) -> float:
        pass

    @abstractproperty
    def gripper_open_ctrl_val(self) -> float:
        pass

    @abstractproperty
    def curr_gripper_ctrl_val(self) -> float:
        pass

    def move_to_joints(self, target_joints: np.ndarray):
        self.last_joints = target_joints

    @abstractproperty
    def joint_config(self) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_kinematics(self, pose: Pose) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def forward_kinematics(
        self, joints: np.ndarray, return_ee_pose: bool = False
    ) -> Optional[Pose]:
        pass

    @abstractmethod
    def get_joint_velocities(self) -> np.ndarray:
        pass

    def gripper_ctrl_to_binary_gripper_command(
        self, gripper_ctrl: float, threshold: float = 0.1
    ) -> bool:
        """
        if its close enough to the close value then we consider it closed
        """
        gripper_range = self.gripper_open_ctrl_val - self.gripper_close_ctrl_val
        dist_from_close = gripper_ctrl - self.gripper_close_ctrl_val
        normalized_dist = np.abs(dist_from_close / gripper_range)
        return normalized_dist < threshold

    @property
    def binary_gripper_command(self) -> bool:
        return self.gripper_ctrl_to_binary_gripper_command(
            gripper_ctrl=self.curr_gripper_ctrl_val
        )

    @abstractproperty
    def end_effector_pose(self) -> Pose:
        pass

    @abstractmethod
    def get_joint_target_control(self) -> np.ndarray:
        pass

    @abstractmethod
    def check_collision(self, joints: np.ndarray) -> bool:
        pass

    @abstractproperty
    def end_effector_rest_orientation(self) -> np.ndarray:
        pass


@dataclasses.dataclass(config=AllowArbitraryTypes)
class EnvSample:
    trajectory: Optional[Trajectory] = None
    trajectory_df: pd.DataFrame = pd.DataFrame()
    trajectory_step_df: pd.DataFrame = pd.DataFrame()


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class EnvSamplerState:
    task: Task
    time: float

    def step(self, time: float) -> EnvSamplerState:
        return EnvSamplerState(
            task=self.task,
            time=self.time + time,
        )


@dataclasses.dataclass(frozen=True)
class EnvSamplerConfig:
    allow_nondeterminism: bool
    done_on_success: bool
    obs_horizon: int
    max_time: float
    use_all_subtrajectories: bool
    visualization_fps: int
    visualization_cam: Optional[str]
    visualization_resolution: Tuple[int, int]
    dump_failed_trajectories: bool


class TaskSampler:
    def __init__(self, tasks: typing.Sequence[Task]):
        self.tasks: typing.List[Task] = list(tasks)

    def sample(
        self, obs: Observation, seed: int, task: typing.Optional[Task] = None
    ) -> Task:
        np_random = np.random.RandomState(seed)
        return self.tasks[np_random.choice(len(self.tasks))]


class ResetTask(Task):
    def __init__(self, allow_gripper_self_collision: bool = False):
        self.allow_gripper_self_collision = allow_gripper_self_collision
        super().__init__(desc="reset after fail")

    def check_success(self, traj: Trajectory) -> bool:
        return self.is_robot_collision_free(state=traj.final_state)

    def get_robot_contacts(
        self, state: EnvState, allow_gripper_self_collision: bool
    ) -> Set[Contact]:
        robot_contacts = state.object_states[state.robot_name].contacts
        if allow_gripper_self_collision:
            robot_contacts = set(
                filter(
                    lambda contact: not (
                        contact.other_link in state.robot_gripper_links
                        and contact.self_link in state.robot_gripper_links
                    ),
                    robot_contacts,
                )
            )
        return robot_contacts

    def is_robot_collision_free(
        self, state: EnvState, allow_gripper_self_collision: Optional[bool] = None
    ):
        if allow_gripper_self_collision is None:
            allow_gripper_self_collision = self.allow_gripper_self_collision
        robot_contacts = self.get_robot_contacts(
            state=state, allow_gripper_self_collision=allow_gripper_self_collision
        )
        return len(robot_contacts) == 0


class EnvSampler:
    """
    Manages env objects, and helps generates trajectories
    """

    def __init__(
        self,
        env: Env,
        task_sampler: TaskSampler,
        do_reset: bool = True,
    ):
        self.env = env
        # TODO migrate task sampler to Evaluation?
        self.task_sampler = task_sampler
        self.do_reset = do_reset

    def loop(
        self,
        episode_id: int,
        obs: Observation,
        episode: List[TrajectoryStep],
        control_states: List[Observation],
        renders: List[np.ndarray],
        policy: Policy,
        trajectory: Trajectory,
        task: Task,
        config: EnvSamplerConfig,
        visualization_callback: Callable[[Env], None],
        must_finish_action_list: bool = True,
        is_root_trajectory: bool = False,
    ) -> Tuple[Observation, bool, Trajectory]:
        is_action_list = issubclass(type(policy), ActionListPolicy)
        policy_seed = self.env.policy_numpy_random.randint(int(2**20))
        assert obs.time == self.env.time, (
            f"obs (obs.time: {obs.time})" + f" is outdated (env.time: {self.env.time})"
        )
        if is_action_list:
            action = typing.cast(ActionListPolicy, policy).actions[len(episode)]
        else:
            trajectory_control_states = trajectory.control_states
            if len(trajectory_control_states) > 0:
                assert (
                    trajectory_control_states[-1].time
                    == self.env.time - 1 / self.env.config.ctrl.frequency
                ), "control states should lag behind env.time by 1/config.ctrl.frequency"
            # policy observation should get the history
            # along with the latest obs
            policy_obs = ObservationWithHistory.from_sequence(
                (
                    trajectory_control_states[-(config.obs_horizon - 1) :]
                    if len(trajectory.control_states) > 0
                    else []
                )  # history
                + [obs]  # latest obs
            )
            assert policy_obs.time == self.env.time, "policy receiving outdated obs"
            assert (
                np.diff([o.time for o in policy_obs.sequence])
                == 1 / self.env.config.ctrl.frequency
            ).all(), "observation frequency doesn't match control frequency"
            action = policy(
                obs=policy_obs,
                task=task,
                seed=policy_seed,
            )

        self.env.step_fn_callbacks["visualization_callback"] = (
            config.visualization_fps,
            visualization_callback,
        )
        self.env.step_fn_callbacks["control_states_callback"] = (
            self.env.config.ctrl.frequency,
            lambda env: control_states.append(env.get_obs()),
        )
        next_obs, done, info = self.env.step(
            action=action,
            active_task=task,
            sampler_config=config,
        )
        if "visualization_callback" in self.env.step_fn_callbacks:
            del self.env.step_fn_callbacks["visualization_callback"]
        if "control_states_callback" in self.env.step_fn_callbacks:
            del self.env.step_fn_callbacks["control_states_callback"]
        episode.append(
            TrajectoryStep(
                obs=obs,
                action=action,
                next_obs=next_obs,
                done=done,
                info=info,
                compressed_renders=copy(renders),  # type: ignore
                visualization_dim=(
                    config.visualization_resolution[0],
                    config.visualization_resolution[1],
                ),
                control_states=(control_states),
            )
        )
        trajectory = Trajectory(
            episode=tuple(episode),
            episode_id=episode_id,
            task=task,
            policy_id=repr(policy),
        )
        success = trajectory.is_successful
        if (
            config.done_on_success
            and success
            and not (is_action_list and must_finish_action_list)
        ):
            info["log"] += "done (success)"
            if is_root_trajectory:
                self.env.done = True
            done = True
        if next_obs.time >= config.max_time:
            info["log"] += "done (out of time)"
            done = True

        if is_action_list and len(episode) >= len(
            typing.cast(ActionListPolicy, policy).actions
        ):
            info["log"] += "done (out of actions)"
            done = True
        from scalingup.policy.scalingup import ScalingUpDataGen

        if issubclass(type(policy), ScalingUpDataGen):
            scalingup_explorer = typing.cast(ScalingUpDataGen, policy)
            if not scalingup_explorer.task_tree_inference.retry_until_success:  # type: ignore
                info["log"] += "done (exploration done)"
                done = True
        return next_obs, done, trajectory

    def sample(
        self,
        episode_id: int,
        policy: Policy,
        config: EnvSamplerConfig,
        task: Optional[Task] = None,
        return_trajectory: bool = True,
        root_path: Optional[str] = None,
        must_finish_action_list: bool = True,
        is_root_trajectory: bool = False,
    ) -> EnvSample:
        assert self.do_reset or self.env.episode_id == episode_id
        if self.do_reset:
            obs = self.env.reset(episode_id=episode_id)
        else:
            obs = self.env.get_obs()
        if task is None:
            task = self.task_sampler.sample(
                obs=obs, seed=self.env.task_numpy_random.randint(int(2**20))
            )
        assert task is not None
        assert config.max_time > 0
        episode: List[TrajectoryStep] = []

        renders: List[np.ndarray] = []
        control_states: List[Observation] = []

        def visualization_callback(env: Env):
            if config.visualization_cam is None:  # type: ignore
                return
            renders.append(
                env.get_rgb(
                    camera_name=config.visualization_cam,
                    obs_dim=config.visualization_resolution,
                )
            )

        env_sample_hash = hex(hash((policy, episode_id, task, config)))[-6:]
        done = False
        # if do nothing, does the current task succeed?
        trajectory = Trajectory(
            episode=(
                TrajectoryStep(
                    obs=obs,
                    action=DoNothingAction(),
                    next_obs=obs,
                    done=done,
                    info={},
                    compressed_renders=[],
                    visualization_dim=(0, 0),
                    control_states=[],
                ),
            ),
            episode_id=episode_id,
            task=task,
            policy_id=repr(policy),
        )
        logging.debug(
            f"[{env_sample_hash}] EnvSampler enter {task.desc} with obs.time {obs.time}"
        )
        if config.done_on_success and trajectory.is_successful:
            logging.info(
                f"[{env_sample_hash}] EnvSampler done (success) "
                + f"{task.desc!r} with obs.time {obs.time}"
            )
            # only end the entire episode if the root task is done
            # otherwise only the current subtask is done so only end
            # current subtrajectory of the episode
            policy = ActionListPolicy(
                actions=[
                    DoNothingAction(
                        end_episode=is_root_trajectory,
                        end_subtrajectory=not is_root_trajectory,
                    )
                ]
            )
        while not done:
            obs, done, trajectory = self.loop(
                episode_id=episode_id,
                obs=obs,
                episode=episode,
                control_states=control_states,
                renders=renders,
                policy=policy,
                trajectory=trajectory,
                task=task,
                config=config,
                visualization_callback=visualization_callback,
                must_finish_action_list=must_finish_action_list,
                is_root_trajectory=is_root_trajectory,
            )
            renders.clear()
            control_states.clear()
        logging.debug(
            f"[{env_sample_hash}] EnvSampler done "
            + f"{task.desc!r} at time {self.env.time}"
        )
        trajectory = Trajectory(
            episode=tuple(episode),
            episode_id=episode_id,
            task=task,
            policy_id=repr(policy),
        )

        # NOTE: this is a point of non-determinism.
        # if we really needed determinism, we would
        # proceed in cycles of data collection then
        # updating the policy instead of the current
        # asynchronous but more efficient behaviour
        if config.allow_nondeterminism:
            policy.update(traj=trajectory)
            if config.use_all_subtrajectories:
                for subtrajectory in trajectory.subtrajectories:
                    policy.update(traj=subtrajectory)
        return self.parse_stats(
            trajectory=trajectory,
            config=config,
            return_trajectory=return_trajectory,
            root_path=root_path,
        )

    def parse_stats(
        self,
        trajectory: Trajectory,
        config: EnvSamplerConfig,
        return_trajectory: bool = True,
        root_path: Optional[str] = None,
        strip_renders: bool = True,
        dump_pickle: bool = False,
    ) -> EnvSample:
        trajectory_data: Dict[str, List[Union[str, int, float, Optional[Video]]]] = {
            "episode_id": [],
            "task": [],
            "task_id": [],
            "return": [],
            "success": [],
            "dense_success": [],
            "episode_len": [],
            "duration": [],
            "video": [],
            "policy_id": [],
            "log": [],
        }

        trajectory_step_data: List[Dict[str, Any]] = []
        per_step_columns = [
            "episode_id",
            "episode_step",
            "log",
            "success",
            "dense_success",
            "action_type",
            "task",
            "task_id",
        ]
        all_subtrajectories = (
            [trajectory, *trajectory.subtrajectories]
            if config.use_all_subtrajectories
            else [trajectory]
        )
        unique_subtrajectories: Dict[int, Trajectory] = {}
        # allow skipping of functionally equivalent trajectories

        for traj in all_subtrajectories:
            if traj.duration == 0.0:
                # e.g. bad llm planning, where action has already
                # been done
                continue
            try:
                flat_traj = traj.flatten()
            except ValidationError:
                # trajectory is empty
                continue
            hash_val = hash(
                (
                    flat_traj.episode,
                    traj.is_successful,
                    traj.task.desc,
                    traj.task.__class__.__name__,
                )
            )
            if hash_val not in unique_subtrajectories:
                unique_subtrajectories[hash_val] = traj

        for traj_idx, traj in enumerate(unique_subtrajectories.values()):
            if len(traj) == 0:
                continue
            task_desc = split_state_phrase(traj.task.desc)[1]
            success = traj.is_successful
            dense_success = traj.compute_subtrajectory_success_rate()
            returns = traj.task.get_returns(traj)
            trajectory_data["episode_id"].append(traj.episode_id)
            trajectory_data["task"].append(task_desc)
            trajectory_data["task_id"].append(hex(hash(traj.task)))
            trajectory_data["return"].append(returns)
            trajectory_data["success"].append(float(success))
            trajectory_data["dense_success"].append(dense_success)
            trajectory_data["episode_len"].append(len(traj))
            trajectory_data["duration"].append(traj.duration)
            trajectory_data["policy_id"].append(traj.policy_id)
            trajectory_data["video"].append(None)
            trajectory_data["log"].append(
                "\n".join(
                    traj_step.info["log"]
                    for traj_step in traj.episode
                    if len(traj_step.info["log"]) > 0
                )
            )
            if root_path is not None:
                traj_path = f"{root_path}/traj-" + f"{traj.episode_id:05d}-{traj_idx:05d}"
                video_path = f"{traj_path}.mp4"
                traj.dump_video(output_path=video_path)
                if success or config.dump_failed_trajectories:
                    try:
                        control_trajectory = ControlTrajectory.from_trajectory(
                            trajectory=traj
                        )
                        control_trajectory.dump(path=f"{traj_path}.mdb")
                    except ValueError as e:
                        # trajectory is probably too short
                        logging.error(e)
                if os.path.exists(video_path):
                    trajectory_data["video"][-1] = Video(video_path)
                if strip_renders:
                    traj.strip_renders()
                if dump_pickle:
                    traj.dump(path=f"{traj_path}.pkl")

            for traj_step in traj.episode:
                action_info = traj_step.action.get_info()
                per_step_columns += list(
                    set(action_info.keys()).difference(per_step_columns)
                )
                trajectory_step_data.append(
                    {
                        **action_info,
                        "episode_id": traj.episode_id,
                        "time": traj_step.obs.time,
                        "log": traj_step.info["log"],
                        "success": float(success),
                        "dense_success": dense_success,
                        "task": task_desc,
                        "task_id": hex(hash(traj.task)),
                        "action_type": type(traj_step.action).__name__,
                    }
                )
        trajectory_df = pd.DataFrame.from_dict(trajectory_data)
        trajectory_step_df = pd.DataFrame.from_dict(
            {
                col: [step_data.get(col, None) for step_data in trajectory_step_data]
                for col in per_step_columns
            }
        )
        return EnvSample(
            trajectory_df=trajectory_df,
            trajectory_step_df=trajectory_step_df,
            trajectory=trajectory if return_trajectory else None,
        )


@dataclasses.dataclass
class EnvConfig:
    obs_cameras: List[str]
    obs_dim: Tuple[int, int]
    ctrl: ControlConfig
    settle_time_after_dropped_obj: float
    ee_action_num_grip_steps: int
    num_action_candidates: int
    # grasp action
    max_pushin_dist: float = 0.05
    min_pushin_dist: float = -0.01
    # revolution action
    num_steps_multiplier: float = 75.0
    min_steps: int = 8
    rotate_gripper_threshold: float = 0.2
    # ee control
    solve_ee_inplace: bool = False
    # place action
    pointing_up_normal_threshold: float = 0.95
    place_height_min: float = 0.02
    place_height_max: float = 0.25
    preplace_dist_min: float = 0.05
    preplace_dist_max: float = 0.25
    # move end effector action options
    fallback_on_rrt_fail: bool = False
    # move end effector action options
    end_on_failed_execution: bool = True
    # action primitives
    grasp_primitive_z_pushin: float = 0.02
    grasp_primitive_z_backup: float = 0.2


class Env(ABC):
    def __init__(
        self,
        robot: Robot,
        step_fn: Callable,
        config: EnvConfig,
        motion_plan_subsampler: MotionPlanSubsampler,
        step_fn_callbacks: Optional[Dict[str, Tuple[int, Callable[[Env], None]]]] = None,
        discretizer: Optional[Discretizer] = None,
        num_setup_variations: Optional[int] = None,
        num_pose_variations: Optional[int] = None,
        num_visual_variations: Optional[int] = None,
    ):
        logging.info(config)
        self.num_setup_variations = num_setup_variations
        self.num_pose_variations = num_pose_variations
        self.num_visual_variations = num_visual_variations
        self.done: bool = False
        self.obs: Observation
        self.motion_plan_subsampler = motion_plan_subsampler
        self.discretizer = discretizer
        self.step_fn = step_fn
        self.config = config
        self.__step_counter = 0
        self.prestep_fn_callbacks: Dict[str, Tuple[int, Callable[[Env], None]]] = (
            step_fn_callbacks if step_fn_callbacks is not None else {}
        )
        self.step_fn_callbacks: Dict[str, Tuple[int, Callable[[Env], None]]] = (
            step_fn_callbacks if step_fn_callbacks is not None else {}
        )
        self.robot = robot
        # control handler
        self.config.ctrl = config.ctrl
        self.control_buffer = ControlAction(
            value=np.zeros((1, self.config.ctrl.dof)),
            config=self.config.ctrl,
            timestamps=np.zeros((1, 1)),
        )
        self.episode_id = 0

    @property
    def episode_id(self):
        return self._episode_id

    @episode_id.setter
    def episode_id(self, episode_id: int):
        self._episode_id = episode_id
        np.random.seed(self._episode_id)
        random.seed(self._episode_id)
        torch.manual_seed(self._episode_id)
        self.pose_numpy_random = np.random.RandomState(
            seed=self._episode_id
            if self.num_pose_variations is None
            else self._episode_id % self.num_pose_variations
        )
        self.setup_numpy_random = np.random.RandomState(
            seed=self._episode_id
            if self.num_setup_variations is None
            else self._episode_id % self.num_setup_variations
        )
        self.visual_numpy_random = np.random.RandomState(
            seed=self._episode_id
            if self.num_visual_variations is None
            else self._episode_id % self.num_visual_variations
        )
        self.policy_numpy_random = np.random.RandomState(seed=self._episode_id)
        self.rrt_numpy_random = np.random.RandomState(seed=self._episode_id)
        self.task_numpy_random = np.random.RandomState(seed=self._episode_id)

    def step(
        self,
        action: Action,
        active_task: Task,
        sampler_config: EnvSamplerConfig,
    ) -> Tuple[Observation, bool, Dict[str, Any]]:
        start_time = self.time
        logging.info(f"{start_time:.03f}: " + str(action))
        assert not self.done
        assert start_time < sampler_config.max_time, (
            f"out of time at {sampler_config.max_time} but "
            + f"current time is {self.time}"
        )
        info: Dict[str, Any] = {"log": "", "subtrajectories": []}
        should_end_episode = False
        try:
            should_end_episode = action(
                env=self,
                active_task=active_task,
                sampler_config=sampler_config,
                info=info,
            )
        except Action.InfeasibleAction as e:
            logging.error(e)
            info["log"] += str(e) + "|"
            self.done = e.stop_episode
        except Action.FailedExecution as e:
            if self.config.end_on_failed_execution:
                should_end_episode = True
            logging.error(e)
            open_gripper_ctrl = self.control_buffer.get_target_ctrl(t=self.time)[1].copy()
            open_gripper_ctrl[-1] = self.robot.gripper_open_ctrl_val

            executed_ctrl_mask = (
                self.control_buffer.timestamps
                <= self.time + 1 / self.config.ctrl.frequency
            )
            # Cancel all pending controls
            self.control_buffer = ControlAction(
                value=self.control_buffer.value[executed_ctrl_mask],
                timestamps=self.control_buffer.timestamps[executed_ctrl_mask],
                config=self.config.ctrl,
                target_ee_actions=self.control_buffer.target_ee_actions[
                    : sum(executed_ctrl_mask)
                ]
                if self.control_buffer.target_ee_actions is not None
                else None,
            )
            # NOTE assumes last dim of ctrl vector controls gripper
            ctrl_cycles = int(
                np.ceil(
                    self.config.settle_time_after_dropped_obj * self.config.ctrl.frequency
                )
            )
            timestamps = (
                np.linspace(
                    self.time,
                    self.time + self.config.settle_time_after_dropped_obj,
                    ctrl_cycles,
                    endpoint=False,
                )
                + 1 / self.config.ctrl.frequency
            )
            ctrl = ControlAction(
                value=np.stack([open_gripper_ctrl] * ctrl_cycles),
                timestamps=timestamps,
                config=self.config.ctrl,
                target_ee_actions=[self.control_buffer.get_target_ctrl(t=self.time)[2]]
                * ctrl_cycles,
            )
            self.execute_ctrl(ctrl=ctrl)

        if len(info["log"]) > 0:
            logging.warning(info["log"])
        assert self.time >= start_time, (
            f" start_time: {start_time}," + f", env.time: {self.time}, action: {action}"
        )
        next_obs = self.get_obs()
        self.obs = next_obs
        return (
            self.obs,
            self.done or should_end_episode,
            info,
        )

    @abstractmethod
    def get_state(self) -> EnvState:
        pass

    def is_reset_state_valid(self, obs: Observation) -> bool:
        return True

    def reset(self, episode_id: int) -> Observation:
        gc.collect()
        self.episode_id = episode_id
        self.__step_counter = 0
        self.done = False
        self.obs = self.get_obs()
        Env.is_reset_state_valid(self, obs=self.obs)
        return self.obs

    @property
    def time(self) -> float:
        return self.__step_counter * self.dt

    @abstractmethod
    def render(
        self,
        obs_cameras: Optional[List[str]] = None,
        obs_dim: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, VisionSensorOutput]:
        pass

    @abstractmethod
    def get_rgb(
        self,
        camera_name: str,
        obs_dim: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def servol(self, command: np.ndarray):
        """
        responsible for handling the low-level control
        joint v.s. ee ctrl, then sending the commands over to the robot
        """
        pass

    @abstractmethod
    def get_control(self) -> np.ndarray:
        pass

    @abstractmethod
    def motion_plan_rrt(
        self,
        goal_conf: np.ndarray,
        current_gripper_command: bool,
        goal_gripper_command: bool,
    ) -> Optional[List[np.ndarray]]:
        pass

    @abstractmethod
    def get_exec_vars(
        self, active_task: Task, sampler_config: EnvSamplerConfig
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass

    def discretize_action(self, pos: np.ndarray, orn: np.ndarray):
        assert self.discretizer is not None
        pos = (
            self.discretizer.undiscretize_pos(
                self.discretizer.discretize_pos(torch.from_numpy(pos)[None, ...])
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        orn_degrees = np.array(euler.quat2euler(orn)) * 180.0 / np.pi
        reconstructed_ee_euler_degrees = (
            self.discretizer.undiscretize_rot(
                self.discretizer.discretize_rot(torch.from_numpy(orn_degrees)[None, ...])
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        orn_degrees = reconstructed_ee_euler_degrees
        orn = euler.euler2quat(*(orn_degrees * np.pi / 180))
        return pos, orn

    def get_obs(self, state: Optional[EnvState] = None) -> Observation:
        if state is None:
            state = self.get_state()
        _, target_ctrl_val, target_ee_action = self.control_buffer.get_target_ctrl(
            t=self.time
        )
        annotated_ctrl = self.config.ctrl.annotate_ctrl(
            target_ctrl_val,
        )
        if target_ee_action is not None:
            annotated_ctrl[
                "target_ee_action/pos"
            ] = target_ee_action.end_effector_position
            annotated_ctrl[
                "target_ee_action/quat"
            ] = target_ee_action.end_effector_orientation
            annotated_ctrl[
                "target_ee_action/allow_contact"
            ] = target_ee_action.allow_contact
            annotated_ctrl[
                "target_ee_action/gripper_command"
            ] = target_ee_action.gripper_command
        return Observation(
            state=state,
            images=self.render(obs_cameras=self.config.obs_cameras),
            episode_id=self.episode_id,
            time=self.time,
            control=annotated_ctrl,
        )

    @abstractmethod
    def joint_pos_control_to_ee_pose_control(
        self, target_ctrl_val: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        pass

    @property
    def dt(self) -> float:
        return 0.01

    def __low_level_step(self):
        for fps, callback in self.prestep_fn_callbacks.values():
            if self.__step_counter % int((1 / self.dt) / fps) == 0:
                callback(self)
        self.servol(
            self.control_buffer.get_interpolated_joint_ctrl(
                t=self.time, robot=self.robot, env_config=self.config
            )
        )
        try:
            self.step_fn()
        except dm_control.rl.control.PhysicsError as e:
            logging.error(e)
            self.done = True
        for fps, callback in self.step_fn_callbacks.values():
            if self.__step_counter % int((1 / self.dt) / fps) == 0:
                callback(self)
        self.__step_counter += 1

    def execute_ctrl(
        self,
        ctrl: ControlAction,
        early_stop_checker: Optional[Callable[[], bool]] = None,
    ):
        self.control_buffer = self.control_buffer.combine(ctrl)
        if self.time > ctrl.end_time:
            return

        # early stop checker can check for dropped grasps.
        # it should be checked here instead of in for loop below
        # because this checker generates data for training the policy,
        # which gets trained on and operates at the ctrl frequency.
        def early_stop_checker_handler(env: Env):
            if early_stop_checker is not None and early_stop_checker():
                del self.prestep_fn_callbacks["early_stopper_callback"]
                raise Action.DroppedGraspedObj(
                    drop_time=self.time,
                    num_ctrl_cycles_left=int(
                        (ctrl.timestamps[-1] - self.time) * self.config.ctrl.frequency
                    ),
                )

        self.prestep_fn_callbacks["early_stopper_callback"] = (
            self.config.ctrl.frequency,
            early_stop_checker_handler,
        )
        while self.time < self.control_buffer.end_time:
            self.__low_level_step()

    def get_dropped_grasp_early_stop_checker(
        self, goal_gripper_command: bool
    ) -> Optional[Callable[[], bool]]:
        return None

    def move_end_effector(
        self,
        target_pose: Pose,
        gripper_command: bool,
        allow_contact: bool,
        info: Dict[str, Any],
        use_early_stop_checker: bool,
        wait_time: Optional[float] = None,
    ):
        if wait_time is None:
            # need to wait for over one control frequency cycle
            # so that actions get register, so use two for safety
            wait_time = 0
        early_stop_checker = None
        if use_early_stop_checker:
            early_stop_checker = self.get_dropped_grasp_early_stop_checker(
                goal_gripper_command=gripper_command
            )

        current_ee_command = EndEffectorAction(
            end_effector_position=target_pose.position,
            end_effector_orientation=target_pose.orientation,
            gripper_command=gripper_command,
            allow_contact=allow_contact,
        )

        """
        How gripper command (open/close) is handled: Only move open/close at the
        last control cycle of the motion plan.
        """
        start_gripper_comm = self.control_buffer.ee_gripper_comm[-1]
        target_gripper_comm = (
            self.robot.gripper_close_ctrl_val
            if gripper_command
            else self.robot.gripper_open_ctrl_val
        )

        # always stop of the currently grasped object is dropped
        target_joint = self.robot.inverse_kinematics(pose=target_pose)
        if target_joint is None:
            if info is not None:
                info["log"] += "IK fail"
            self.done = True
            return

        # Step 1, plan in joint space
        simplified_joint_plan: Optional[np.ndarray] = None

        if not allow_contact:
            # TODO switch to error based time budget estimation
            # set this in advance to hint to rrt planner
            joint_plan = None
            try:
                joint_plan = self.motion_plan_rrt(
                    goal_conf=target_joint,
                    current_gripper_command=self.robot.binary_gripper_command,
                    goal_gripper_command=gripper_command,
                )
            except Action.FailedExecution as e:
                logging.error(e)

            if joint_plan is not None:
                simplified_joint_plan = self.motion_plan_subsampler(
                    joint_plan=np.stack(joint_plan), robot=self.robot
                )
            elif not self.config.fallback_on_rrt_fail:
                # after falling, just stop trying
                return

        if simplified_joint_plan is None:
            # either rrt failed, or no motion planning is needed
            simplified_joint_plan = self.motion_plan_subsampler(
                joint_plan=np.array([self.robot.joint_config, target_joint]),
                robot=self.robot,
            )
        assert simplified_joint_plan is not None
        # Step 2, plan for gripper
        gripper_plan = np.array([start_gripper_comm] * len(simplified_joint_plan))
        gripper_plan[-1] = target_gripper_comm

        # Step 3, convert plan in the control representation
        if self.config.ctrl.control_type == ControlType.JOINT:
            ctrl_value = np.concatenate(
                [
                    simplified_joint_plan,
                    gripper_plan[:, None],
                ],
                axis=1,
            )
        else:
            ee_ctrl_plan = []
            for joint_ctrl, gripper_ctrl in zip(
                simplified_joint_plan,
                gripper_plan,
            ):
                ee_pose = self.robot.forward_kinematics(joint_ctrl, return_ee_pose=True)
                assert ee_pose is not None
                ee_ctrl_plan.append(
                    np.concatenate(
                        [
                            ee_pose.position,
                            self.config.ctrl.quat2ctrl_orn(
                                quat=ee_pose.orientation,
                            ),
                            [gripper_ctrl],
                        ],
                        axis=0,
                    )
                )
            ctrl_value = np.stack(ee_ctrl_plan)

        if wait_time > 0:
            wait_cycles = int(np.round(wait_time * self.config.ctrl.frequency))
            ctrl_value = np.concatenate(
                [ctrl_value, *(wait_cycles * [ctrl_value[-1][None, :]])], axis=0
            )

        timestamps = np.linspace(
            self.time,
            self.time + len(ctrl_value) / self.config.ctrl.frequency,
            len(ctrl_value),
            endpoint=False,
        )

        # timestamps are time to reach the control value in position control.
        # since we want each new control value to be reached within the next
        # control cycle, we offset by 1/control frequency
        timestamps += 1 / self.config.ctrl.frequency

        # Step 4, execute
        self.execute_ctrl(
            ctrl=ControlAction(
                value=ctrl_value,
                config=self.config.ctrl,
                timestamps=timestamps,
                target_ee_actions=len(ctrl_value) * [current_ee_command],  # type: ignore
            ),
            early_stop_checker=early_stop_checker,
        )


class TaskGenerator(ABC):
    @abstractmethod
    def infer_from_desc(self, task_desc: str, state: EnvState) -> Task:
        pass


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class ObservationWithHistory(Observation):
    history: List[Observation]

    @property
    def sequence(self) -> List[Observation]:
        return [
            *self.history,
            Observation(
                state=self.state,
                images=self.images,
                episode_id=self.episode_id,
                time=self.time,
                control=self.control,
            ),
        ]

    @classmethod
    def from_sequence(cls, sequence: List[Observation]) -> ObservationWithHistory:
        curr_obs = sequence[-1]
        return ObservationWithHistory(
            state=curr_obs.state,
            images=curr_obs.images,
            episode_id=curr_obs.episode_id,
            time=curr_obs.time,
            control=curr_obs.control,
            history=sequence[:-1],
        )


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class PartialObservation(Observation):
    @validator("images")
    @classmethod
    def no_ground_truth_segmentation(cls, v: Dict[str, VisionSensorOutput]):
        for sensor_output in v.values():
            if len(sensor_output.compressed_segmentation) > 0:
                raise ValueError("Ground truth segmentation is not allowed")
        return v

    @validator("state")
    @classmethod
    def no_object_states(cls, v: EnvState):
        if len(v.object_states) > 0:
            raise ValueError(
                f"Object states are not allowed, but has {v.object_states.keys()}"
            )
        return v

    @classmethod
    def from_obs(cls, obs: Observation):
        return PartialObservation(
            time=obs.time,
            episode_id=obs.episode_id,
            control=obs.control,
            images={
                k: VisionSensorOutput(
                    compressed_rgb=sensor_output.compressed_rgb,
                    compressed_depth=sensor_output.compressed_depth,
                    compressed_segmentation={},  # no ground truth segmentation
                    pos=sensor_output.pos,
                    obs_dim=sensor_output.obs_dim,
                    rot_mat=sensor_output.rot_mat,
                    fovy=sensor_output.fovy,
                )
                for k, sensor_output in obs.images.items()
            },
            state=EnvState(
                object_states={},  # no privileged information about other objects
                robot_name="",  # discard
                robot_gripper_links=[],  # discard
                gripper_command=obs.state.gripper_command,
                end_effector_pose=obs.state.end_effector_pose,
                grasped_objects=obs.state.grasped_objects,
                robot_joint_velocities=obs.state.robot_joint_velocities,
            ),
        )

    def to_dict(
        self,
        include_rgb: bool = True,
        include_depth: bool = True,
        include_matrices: bool = True,
    ) -> Dict[str, Union[np.ndarray, float, bool, int]]:
        # dump all data into numpy dict
        retval = {
            "time": self.time,
            "episode_id": self.episode_id,
            "gripper_command": self.state.gripper_command,
            "end_effector_position": self.state.end_effector_pose.position,
            "end_effector_orientation": self.state.end_effector_pose.orientation,
            **{f"control/{k}": v for k, v in self.control.items()},
            "robot_joint_velocities": np.array(self.state.robot_joint_velocities),
            "robot_grasped_objects": ",".join(sorted(self.state.grasped_objects)),
        }
        for view_name, sensor_output in self.images.items():
            if include_rgb:
                retval[f"images/{view_name}/rgb"] = sensor_output.rgb
            if include_depth:
                retval[f"images/{view_name}/depth"] = sensor_output.depth
            if include_matrices:
                retval[f"images/{view_name}/pos"] = sensor_output.pos
                retval[f"images/{view_name}/rot_mat"] = sensor_output.rot_mat
                retval[f"images/{view_name}/fovy"] = sensor_output.fovy
        return retval


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True)
class ControlTrajectoryStep(TrajectoryStep):
    @validator("obs")
    @classmethod
    def observations_are_partial(cls, v: Observation):
        if type(v) is not PartialObservation:
            raise ValueError(f"Observation {v} is not PartialObservation")
        return v

    @validator("compressed_renders")
    @classmethod
    def renders_is_empty(cls, v: List[np.ndarray]):
        if len(v) > 0:
            raise ValueError("ControlTrajectoryStep should not have render frames")
        return v

    @validator("action")
    @classmethod
    def action_is_empty(cls, v: Action):
        # already stored in observation
        if type(v) is not DoNothingAction:
            raise ValueError(f"Action {v} is not Action")
        return v

    @validator("control_states")
    @classmethod
    def no_control_states(cls, v: List[Observation]):
        if len(v) > 0:
            raise ValueError(
                "ControlTrajectoryStep should not have intermediate observations"
            )
        return v

    @property
    def partial_obs(self) -> PartialObservation:
        return typing.cast(PartialObservation, self.obs)


@dataclasses.dataclass(config=AllowArbitraryTypes, frozen=True, eq=True)
class ControlTrajectory(Trajectory):
    success: bool
    perfect: bool
    task_desc: str
    is_inferred_task: bool
    subtrajectory_success_rate: float
    control_frequency: int = -1  # estimate automatically

    @classmethod
    def estimate_control_frequency(cls, control_states: List[Observation]) -> int:
        timestamps = np.array([control_state.time for control_state in control_states])
        diff = np.diff(np.stack([timestamps[:-1], timestamps[1:]]), axis=0)
        control_frequency = np.mean((1 / diff).reshape(-1))
        if not np.allclose(np.rint(control_frequency), control_frequency):
            raise ValueError(f"control frequency {control_frequency} is not an integer!")
        return int(np.rint(control_frequency))

    @validator("episode", each_item=True)
    @classmethod
    def trajectory_contain_control_steps(cls, v: TrajectoryStep):
        if type(v) is not ControlTrajectoryStep:
            raise ValueError(f"Trajectory step {v} is not ControlTrajectoryStep")
        return v

    # don't dump next obs to save space
    @classmethod
    def from_trajectory(cls, trajectory: Trajectory):
        subtrajectory_success_rate = trajectory.compute_subtrajectory_success_rate()
        perfect = trajectory.is_perfect
        trajectory = trajectory.flatten()
        episode = []
        if len(trajectory.control_states) < 2:
            raise ValueError("Control trajectory must have at least two control states")
        control_frequency = ControlTrajectory.estimate_control_frequency(
            trajectory.control_states
        )
        for traj_step in trajectory.episode:
            step_control_states = traj_step.control_states
            for idx, control_state in enumerate(step_control_states):
                next_obs = (
                    step_control_states[idx + 1]
                    if idx < len(step_control_states) - 1
                    else traj_step.next_obs
                )
                episode.append(
                    ControlTrajectoryStep(
                        info=traj_step.info,
                        obs=PartialObservation.from_obs(obs=control_state),
                        action=DoNothingAction(),
                        done=traj_step.done,
                        next_obs=PartialObservation.from_obs(obs=next_obs),
                        compressed_renders=[],
                        visualization_dim=(0, 0),
                        control_states=[],
                    )
                )
        return ControlTrajectory(
            episode_id=trajectory.episode_id,
            policy_id=trajectory.policy_id,
            task=Task(desc=""),
            episode=tuple(episode),
            control_frequency=control_frequency,
            # success checking depends on state, so need to evaluate this now
            success=trajectory.is_successful,
            perfect=perfect,
            task_desc=trajectory.task.desc,
            subtrajectory_success_rate=subtrajectory_success_rate,
            is_inferred_task="inferred_success_fn_code_str" in trajectory.task.info,
        )

    @property
    def control_sequence(self) -> Tuple[ControlTrajectoryStep, ...]:
        return typing.cast(Tuple[ControlTrajectoryStep, ...], self.episode)

    def parse_ctrl_sequence(
        self,
    ) -> Tuple[
        Dict[str, List[Union[np.ndarray, float, bool, int]]],
        Dict[str, List[Union[np.ndarray, float, bool, int]]],
    ]:
        obs_datasets: Dict[str, List[Union[np.ndarray, float, bool, int]]] = {}
        state_tensor_datasets: Dict[str, List[Union[np.ndarray, float, bool, int]]] = {}
        random_state = np.random.RandomState(self.episode_id)
        for control_step in self.control_sequence:
            obs_dict = control_step.partial_obs.to_dict(
                include_rgb=False,
                include_depth=False,
                include_matrices=False,
            )
            for k, v in obs_dict.items():
                if k not in obs_datasets:
                    obs_datasets[k] = []
                obs_datasets[k].append(v)
            from scalingup.data.dataset import StateTensor

            state_tensor = StateTensor.from_obs(
                partial_obs=control_step.partial_obs,
                # no subsampling should happen so dont need
                # controlled pseudorandom number generator
                numpy_random=random_state,  # type: ignore
                obs_cameras=list(control_step.partial_obs.images.keys()),
                num_obs_pts=PCD_CHUNK_SIZE * 235,
                pos_bounds=((0.0, -0.8, -0.1), (0.8, 0.8, 0.7)),
                occupancy_dim=(75, 150, 75),
            )
            attrs = list(StateTensor.__annotations__.keys())
            for attr in attrs:
                if type(getattr(state_tensor, attr)) == dict:
                    assert attr == "views"
                    for k, v in getattr(state_tensor, attr).items():
                        if f"{attr}/{k}" not in state_tensor_datasets:
                            state_tensor_datasets[f"{attr}/{k}"] = []
                        state_tensor_datasets[f"{attr}/{k}"].append(
                            v.cpu().numpy().astype(RGB_DTYPE)
                        )
                else:
                    if attr not in state_tensor_datasets:
                        state_tensor_datasets[attr] = []
                    numpy_arr = getattr(state_tensor, attr).cpu().numpy()

                    if attr == "input_rgb_pts":
                        numpy_arr = (numpy_arr * 255).astype(RGB_DTYPE)
                    elif attr == "occupancy":
                        assert numpy_arr.dtype == bool
                    state_tensor_datasets[attr].append(numpy_arr)
        return obs_datasets, state_tensor_datasets

    def dump(
        self,
        path: str,
        protocol: Any = open,
        pickled_data_compressor: Any = None,
        protocol_kwargs: Optional[Dict[str, Any]] = None,
        compressor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # NOTE: these params were tuned in scripts/eval_compressors.py
        if pickled_data_compressor is None:
            # NOTE: this compressor uses the least CPU
            pickled_data_compressor = Blosc(
                cname="blosclz", clevel=9, shuffle=Blosc.NOSHUFFLE, blocksize=0
            )
            """
            NOTE: this compressor uses more CPU but saves more disk space
            pickled_data_compressor = Blosc(
                cname="zstd", clevel=7, shuffle=Blosc.SHUFFLE, blocksize=0
            )
            """

        obs_datasets, state_tensor_datasets = self.parse_ctrl_sequence()
        with zarr.LMDBStore(path=path, subdir=False) as store:
            root = zarr.group(store=store)
            root.attrs["task_desc"] = self.task_desc
            root.attrs["length"] = len(self.control_sequence)
            root.attrs["success"] = self.success
            root.attrs["perfect"] = self.perfect
            root.attrs["is_inferred_task"] = self.is_inferred_task
            root.attrs["subtrajectory_success_rate"] = self.subtrajectory_success_rate
            root.attrs["control_frequency"] = self.control_frequency
            root.attrs["policy_id"] = self.policy_id
            for k, v in obs_datasets.items():
                if type(v[0]) is np.ndarray:
                    data = np.stack(v, axis=0)
                    root.create_dataset(
                        k,
                        data=data,
                        chunks=(1, *data.shape[1:]),
                        dtype=data.dtype,
                        compressor=pickled_data_compressor,
                    )
                elif type(v[0]) is str:
                    root.create_dataset(
                        k, data=np.array(v), dtype=object, object_codec=VLenUTF8()
                    )
                else:
                    root.create_dataset(k, data=np.array(v), chunks=False)

            state_tensor_group = root.create_group("state_tensor")
            # shuffle up points for chunking
            num_pts = state_tensor_datasets["input_xyz_pts"][0].shape[0]  # type: ignore
            shuffled_indices = np.arange(num_pts)
            np.random.RandomState(0).shuffle(shuffled_indices)
            for k, v in state_tensor_datasets.items():
                if type(v[0]) is np.ndarray:
                    compressor = pickled_data_compressor
                    data = np.stack(v, axis=0)
                    if k == "input_xyz_pts" or k == "input_rgb_pts":
                        chunks = (1, PCD_CHUNK_SIZE, 3)
                        data = data[:, shuffled_indices, :]
                    elif k.startswith("views"):
                        # c h w ==> h w c
                        data = data.transpose(0, 2, 3, 1)
                        chunks = (1, *data.shape[1:])  # type: ignore
                        compressor = Jpeg2k(level=50, numthreads=1)
                    else:
                        chunks = v[0].shape
                    state_tensor_group.create_dataset(
                        k,
                        data=data,
                        chunks=chunks,
                        dtype=v[0].dtype,
                        compressor=compressor,
                    )

    @classmethod
    def load(
        cls,
        path: str,
        protocol: Any = open,
        pickled_data_decompressor: Any = brotli.decompress,
        protocol_kwargs: Optional[Dict[str, Any]] = None,
        decompressor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        raise NotImplementedError()


@dataclasses.dataclass
class MotionPlanSubsamplerConfig:
    err_threshold: float
    max_speed: float


class MotionPlanSubsampler:
    """
    Subsamples a motion plan to reduce the number of waypoints,
    based on error from the high resolution waypoints and maximum
    allowable speed.
    """

    def __init__(
        self,
        ctrl_config: ControlConfig,
        configs: Dict[str, MotionPlanSubsamplerConfig],
        min_ctrl_cycles: int,
        min_speed_ratio: float,
    ):
        self.ctrl_config = ctrl_config
        self.configs = configs
        self.min_ctrl_cycles = min_ctrl_cycles
        self.min_speed_ratio = min_speed_ratio

    def compute_joint_speed(self, prev_joint: np.ndarray, next_joint) -> float:
        return np.abs(next_joint - prev_joint).max()

    def __call__(self, joint_plan: np.ndarray, robot: Robot) -> np.ndarray:
        assert len(joint_plan) > 1
        simplified_joint_plan = [joint_plan[0]]
        ee_pose = robot.forward_kinematics(joint_plan[0], return_ee_pose=True)
        assert ee_pose is not None
        simplified_joint_plan_ee_pose = [ee_pose]
        if self.ctrl_config.control_type == ControlType.JOINT:
            raise NotImplementedError()
        elif self.ctrl_config.control_type == ControlType.END_EFFECTOR:
            for i in range(1, len(joint_plan)):
                next_joint = joint_plan[i]
                ee_pose = robot.forward_kinematics(next_joint, return_ee_pose=True)
                assert ee_pose is not None
                pos_dist = float(
                    np.linalg.norm(
                        ee_pose.position - simplified_joint_plan_ee_pose[-1].position
                    )
                )
                pos_speed = pos_dist * self.ctrl_config.frequency
                pos_speed_limit_ratio = pos_speed / self.configs["ee_pos"].max_speed

                orn_dist = Pose.orientation_distance(
                    q1=simplified_joint_plan_ee_pose[-1].orientation,
                    q2=ee_pose.orientation,
                )
                orn_speed = orn_dist * self.ctrl_config.frequency
                orn_speed_limit_ratio = orn_speed / self.configs["ee_orn"].max_speed
                speed_limit_ratio = max(pos_speed_limit_ratio, orn_speed_limit_ratio)

                if speed_limit_ratio < 1.0 and speed_limit_ratio > self.min_speed_ratio:
                    # enough to add a point
                    simplified_joint_plan.append(next_joint)
                    simplified_joint_plan_ee_pose.append(ee_pose)
                    continue

                num_chunks = int(np.ceil(speed_limit_ratio))
                if num_chunks > 1:
                    # moving too fast
                    # break up into smaller chunks
                    for j in range(1, num_chunks):
                        alpha = j / num_chunks
                        simplified_joint_plan.append(
                            simplified_joint_plan[-1] * (1 - alpha) + next_joint * alpha
                        )
                        ee_pose = robot.forward_kinematics(
                            simplified_joint_plan[-1], return_ee_pose=True
                        )
                        assert ee_pose is not None
                        simplified_joint_plan_ee_pose.append(ee_pose)

        else:
            raise NotImplementedError(
                "Unknown control type: %s" % self.ctrl_config.control_type
            )
        simplified_joint_plan.append(joint_plan[-1])
        if len(simplified_joint_plan) < self.min_ctrl_cycles + 1:
            simplified_joint_plan.extend(
                [simplified_joint_plan[-1]]
                * (self.min_ctrl_cycles - len(simplified_joint_plan) + 1)
            )
        logging.info(
            f"Subsampled from {len(joint_plan)} to {len(simplified_joint_plan)} waypoints"
        )
        return np.stack(simplified_joint_plan)
