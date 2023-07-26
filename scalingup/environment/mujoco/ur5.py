from abc import abstractproperty
import os
from typing import Any, List, Optional
from dm_control.mujoco.engine import Physics

import numpy as np
from dm_control import mujoco
from transforms3d import euler
from scalingup.environment.mujoco.mujocoRobot import MujocoRobot
from scalingup.environment.mujoco.utils import get_part_path
from scalingup.utils.core import EndEffectorAction, ControlConfig


class UR5(MujocoRobot):
    def __init__(
        self,
        mj_physics: mujoco.Physics,
        bodyid: int,
        gripper_link_paths: List[str],
        gripper_joint_name: str,
        gripper_actuator_name: str,
        init_joints: Optional[np.ndarray] = None,
        prefix: str = "",
    ):
        super().__init__(
            mj_physics=mj_physics,
            init_joints=init_joints
            if init_joints is not None
            else self.home_joint_ctrl_val,
            prefix=prefix,
            bodyid=bodyid,
            gripper_link_paths=gripper_link_paths,
            gripper_joint_name=gripper_joint_name,
            gripper_actuator_name=gripper_actuator_name,
        )

    @property
    def link_names(self) -> List[str]:
        return [
            "base",
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
        ]

    @abstractproperty
    def ee_link_names(self) -> List[str]:
        pass

    @property
    def home_ctrl_qpos(self) -> np.ndarray:
        return self.home_joint_ctrl_val

    @property
    def curr_gripper_ctrl_val(self, physics: Optional[Physics] = None) -> float:
        if physics is None:
            physics = self.mj_physics
        assert physics is not None
        actuator = physics.model.actuator(
            os.path.join(self.prefix, self.gripper_actuator_name)
        )
        return self.mj_physics.control()[actuator.id]

    @property
    def curr_gripper_qpos(self, physics: Optional[Physics] = None) -> float:
        if physics is None:
            physics = self.mj_physics
        assert physics is not None
        joint = physics.model.joint(os.path.join(self.prefix, self.gripper_joint_name))
        assert len(joint.qposadr) == 1
        qpos = physics.data.qpos[joint.qposadr[0]]
        return qpos

    @property
    def home_joint_ctrl_val(self) -> np.ndarray:
        return np.array([-1.7, -2.45, 2.48, -1.57, -1.51, 0])

    def get_joint_target_control(self) -> np.ndarray:
        return np.append(
            self.last_joints,
            self.curr_gripper_ctrl_val,
        )

    @property
    def end_effector_links(self) -> List[Any]:
        return [
            self.mj_physics.model.body(os.path.join(self.prefix, link_name))
            for link_name in self.ee_link_names
        ]

    @property
    def end_effector_link_ids(self) -> List[int]:
        return [link.id for link in self.end_effector_links]

    @property
    def joint_names(self) -> List[str]:
        return [
            os.path.join(self.prefix, joint_name)
            for joint_name in [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]
        ]

    @property
    def link_paths(self) -> List[str]:
        return [os.path.join(self.prefix, link_name) for link_name in self.link_names]

    @property
    def ee_link_paths(self) -> List[str]:
        return [os.path.join(self.prefix, link_name) for link_name in self.ee_link_names]

    @property
    def robot_collision_link_paths(self) -> List[str]:
        return self.link_paths + self.ee_link_paths

    @property
    def end_effector_rest_orientation(self) -> np.ndarray:
        return euler.euler2quat(np.pi, 0, 0)


class UR5Robotiq(UR5):
    def __init__(
        self,
        mj_physics: mujoco.Physics,
        bodyid: int,
        prefix: str = "",
    ):
        end_effector_links = filter(
            lambda body: "robotiq_right_finger" in body.name
            or "robotiq_left_finger" in body.name,
            map(mj_physics.model.body, range(mj_physics.model.nbody)),
        )
        gripper_link_paths = [
            get_part_path(model=mj_physics.model, body=body)
            for body in end_effector_links
        ]
        super().__init__(
            mj_physics=mj_physics,
            prefix=prefix,
            bodyid=bodyid,
            gripper_link_paths=gripper_link_paths,
            gripper_joint_name="right_driver_joint",
            gripper_actuator_name="fingers_actuator",
        )

    @property
    def home_joint_ctrl_val(self) -> np.ndarray:
        return np.concatenate(
            (super().home_joint_ctrl_val, np.array([self.gripper_open_ctrl_val])), axis=0
        )

    @property
    def gripper_close_ctrl_val(self) -> float:
        return 255.0

    @property
    def gripper_open_ctrl_val(self) -> float:
        return 0.0

    @property
    def ee_link_names(self) -> List[str]:
        return [
            "Robotiq 2f85",
            "robotiq_base",
            "right_driver",
            "right_coupler",
            "right_spring_link",
            "right_follower",
            "robotiq_right_finger",
            "right_silicone_pad",
            "left_driver",
            "left_coupler",
            "left_spring_link",
            "left_follower",
            "robotiq_left_finger",
            "left_silicone_pad",
        ]


class UR5WSG50Finray(UR5):
    def __init__(
        self,
        mj_physics: mujoco.Physics,
        bodyid: int,
        prefix: str = "",
    ):
        end_effector_links = filter(
            lambda body: "right_finger" in body.name or "left_finger" in body.name,
            map(mj_physics.model.body, range(mj_physics.model.nbody)),
        )
        gripper_link_paths = [
            get_part_path(model=mj_physics.model, body=body)
            for body in end_effector_links
        ]
        super().__init__(
            mj_physics=mj_physics,
            prefix=prefix,
            bodyid=bodyid,
            init_joints=super().home_joint_ctrl_val,
            gripper_link_paths=gripper_link_paths,
            gripper_joint_name="wsg50/right_driver_joint",
            gripper_actuator_name="wsg50/gripper",
        )

    @property
    def home_joint_ctrl_val(self) -> np.ndarray:
        return np.concatenate(
            (super().home_joint_ctrl_val, np.array([self.gripper_open_ctrl_val])), axis=0
        )

    @property
    def home_ctrl_qpos(self) -> np.ndarray:
        return np.concatenate(
            (
                super().home_joint_ctrl_val,
                np.array([self.gripper_open_ctrl_val, self.gripper_open_ctrl_val]),
            ),
            axis=0,
        )

    @property
    def gripper_close_ctrl_val(self) -> float:
        return 0.0

    @property
    def gripper_open_ctrl_val(self) -> float:
        return 0.055

    @property
    def ee_link_names(self) -> List[str]:
        return [
            "base",
            "right_finger",
            "left_finger",
        ]

    @property
    def end_effector_site_name(self) -> str:
        # TODO make sure self.prefix is frozen
        return os.path.join(self.prefix, "wsg50", "end_effector")

    @property
    def end_effector_rest_orientation(self) -> np.ndarray:
        return euler.euler2quat(np.pi, 0, -np.pi / 2)

    @property
    def end_effector_links(self) -> List[Any]:
        return [
            self.mj_physics.model.body(os.path.join(self.prefix, "wsg50", link_name))
            for link_name in self.ee_link_names
        ]

    @property
    def ee_link_paths(self) -> List[str]:
        return [
            os.path.join(self.prefix, "wsg50", link_name)
            for link_name in self.ee_link_names
        ]
