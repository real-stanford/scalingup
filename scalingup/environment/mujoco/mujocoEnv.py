from __future__ import annotations

import gc
import logging
import os
import time
import typing
from abc import abstractmethod
from copy import copy, deepcopy
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Type
import pytorch3d as pt3d
import cv2
import mujoco
import numpy as np
from dm_control import mjcf
from dm_control.mjcf import RootElement
from dm_control.mujoco.engine import Physics
from dm_control.rl.control import PhysicsError
from mujoco import FatalError as mujocoFatalError  # type: ignore
from transforms3d import affines, euler, quaternions
import torch
from scalingup.algo.end_effector_policy_utils import Discretizer
from scalingup.algo.virtual_grid import Point3D
from scalingup.environment.mujoco.mujocoRobot import MujocoRobot
from scalingup.environment.mujoco.rrt import MujocoRRT
from scalingup.environment.mujoco.ur5 import UR5, UR5Robotiq, UR5WSG50Finray
from scalingup.environment.mujoco.utils import (
    MujocoObjectInstanceConfig,
    get_body_aabbs,
    get_fixed_children,
    get_part_path,
    parse_contact_data,
)
from scalingup.utils.constants import LINK_SEPARATOR_TOKEN, MJCF_NEST_TOKEN
from scalingup.utils.core import (
    Action,
    ControlAction,
    ControlConfig,
    ControlType,
    DegreeOfFreedomRange,
    EndEffectorAction,
    Env,
    EnvSamplerConfig,
    EnvState,
    JointState,
    JointType,
    LinkState,
    ObjectState,
    Observation,
    Pose,
    QPosRange,
    RotationType,
    Task,
    TrajectoryStep,
    Velocity,
    VisionSensorOutput,
)
from scalingup.environment.mujoco.domain_randomizer import (
    DomainRandomizationConfig,
    domain_randomize,
)


class MujocoEnv(Env):
    def __init__(
        self,
        robot_cls: Type[MujocoRobot],
        visibility_checker_cam_name: str,
        assert_visible_objs_at_reset: Optional[Set[str]] = None,
        dynamic_mujoco_model: bool = False,
        domain_rand_config: Optional[DomainRandomizationConfig] = None,
        override_pos_ctrl_kp: Optional[float] = None,
        override_pos_ctrl_force: Optional[float] = None,
        num_setup_variations: Optional[int] = None,
        num_pose_variations: Optional[int] = None,
        num_visual_variations: Optional[int] = None,
        **kwargs,
    ):
        self.num_setup_variations = num_setup_variations
        self.num_pose_variations = num_pose_variations
        self.num_visual_variations = num_visual_variations
        self.episode_id = 0
        self.override_pos_ctrl_kp = override_pos_ctrl_kp
        self.override_pos_ctrl_force = override_pos_ctrl_force
        self.robot_model_name: str
        self.domain_rand_config = domain_rand_config
        self.dynamic_mujoco_model = dynamic_mujoco_model
        if self.domain_rand_config is not None:
            assert (
                self.dynamic_mujoco_model
            ), "Domain randomization requires dynamic model"

        self.mj_physics, robot_bodyid = self.setup()
        robot = robot_cls(
            mj_physics=self.mj_physics, prefix=self.robot_model_name, bodyid=robot_bodyid
        )  # type: ignore
        self.assert_visible_objs_at_reset = (
            assert_visible_objs_at_reset
            if assert_visible_objs_at_reset is not None
            else set()
        )
        self.visibility_checker_cam_name = visibility_checker_cam_name
        super().__init__(
            step_fn=self.mj_physics.step,
            robot=robot,
            num_setup_variations=num_setup_variations,
            num_pose_variations=num_pose_variations,
            num_visual_variations=num_visual_variations,
            **kwargs,
        )
        self.rrt = MujocoRRT(
            physics=self.mj_physics.copy(share_model=True),
            robot=copy(self.mujoco_robot),
            env_config=self.config,
        )

    @abstractmethod
    def setup_model(self) -> Tuple[RootElement, str]:
        raise NotImplementedError()

    def setup_objs(self, world_model: RootElement) -> QPosRange:
        return []

    def setup(self) -> Tuple[Physics, int]:
        world_model, robot_root_link_name = self.setup_model()
        self.obj_qpos_ranges: QPosRange = self.setup_objs(world_model=world_model)
        if self.override_pos_ctrl_kp is not None:
            logging.info(f"Overriding pos ctrl kp to {self.override_pos_ctrl_kp}")
            for actuator in world_model.actuator.all_children():
                actuator.kp = str(self.override_pos_ctrl_kp)
        if self.override_pos_ctrl_force is not None:
            logging.info(f"Overriding pos ctrl force to {self.override_pos_ctrl_force}")
            for actuator in world_model.actuator.all_children():
                actuator.forcerange = (
                    f"-{self.override_pos_ctrl_force} {self.override_pos_ctrl_force}"
                )

        if self.domain_rand_config is not None:
            world_model = domain_randomize(
                world_model,
                config=self.domain_rand_config,
                np_random=self.visual_numpy_random,
            )
        mj_physics = mjcf.Physics.from_mjcf_model(world_model)
        robot_bodyid = mj_physics.model.body(
            mj_physics.model.body(robot_root_link_name).rootid
        ).id
        assert mj_physics is not None
        return mj_physics, robot_bodyid

    def update_mj_physics(self, mj_physics: Physics):
        self.mujoco_robot.mj_physics = mj_physics
        self.step_fn = mj_physics.step
        self.mj_physics = mj_physics
        self.rrt.physics = mj_physics.copy(share_model=True)
        self.rrt.robot = copy(self.mujoco_robot)
        self.rrt.robot.mj_physics = self.rrt.physics

    def get_rgb(
        self,
        camera_name: str,
        obs_dim: Optional[Tuple[int, int]] = None,
        physics: Optional[Physics] = None,
    ) -> np.ndarray:
        if obs_dim is None:
            obs_dim = self.config.obs_dim
        if physics is None:
            physics = self.mj_physics
        cam = physics.model.camera(camera_name)
        return physics.render(
            height=obs_dim[0],
            width=obs_dim[1],
            depth=False,
            camera_id=cam.id,
        )

    def render(
        self,
        obs_cameras: Optional[List[str]] = None,
        obs_dim: Optional[Tuple[int, int]] = None,
        max_retries: int = 100,
        physics: Optional[Physics] = None,
    ) -> Dict[str, VisionSensorOutput]:
        outputs = {}
        if obs_dim is None:
            obs_dim = self.config.obs_dim
        if physics is None:
            physics = self.mj_physics
        assert physics is not None
        if obs_cameras is None:
            obs_cameras = [
                physics.model.camera(cam_id).name for cam_id in range(physics.model.ncam)
            ]
        assert physics is not None
        assert obs_cameras is not None
        for cam_name in sorted(obs_cameras):
            cam = physics.model.camera(cam_name)
            cam_data = physics.data.camera(cam_name)
            cam_pos = cam_data.xpos.reshape(3)
            cam_rotmat = cam_data.xmat.reshape(3, 3)
            for i in range(max_retries):
                try:
                    # NOTE: rgb render much more expensive than others
                    # If optimizing, look into disable rgb rendering for
                    # passes which are not needed
                    rgb = physics.render(
                        height=obs_dim[0],
                        width=obs_dim[1],
                        depth=False,
                        camera_id=cam.id,
                    )
                    depth = physics.render(
                        height=obs_dim[0],
                        width=obs_dim[1],
                        depth=True,
                        camera_id=cam.id,
                    )
                    segmentation_map = physics.render(
                        height=obs_dim[0],
                        width=obs_dim[1],
                        segmentation=True,
                        camera_id=cam.id,
                    )
                    outputs[cam_name] = self.process_renders(
                        rgb=rgb,
                        depth=depth,
                        segmentation_map=segmentation_map,
                        physics=physics,
                        cam_pos=(cam_pos[0], cam_pos[1], cam_pos[2]),
                        cam_rotmat=cam_rotmat,
                        fovy=float(cam.fovy[0]),
                    )
                    break
                except mujocoFatalError as e:
                    if i == max_retries - 1:
                        raise e
                    logging.error(e)
                    time.sleep(5)
        return outputs

    def process_renders(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        segmentation_map: np.ndarray,
        physics: Physics,
        cam_pos: Point3D,
        cam_rotmat: np.ndarray,
        fovy: float,
        erode_seg_mask: bool = False,
    ) -> VisionSensorOutput:
        obs_dim = depth.shape
        segmentation = {
            get_part_path(physics.model, physics.model.body(bodyid)): np.zeros(
                (obs_dim[0], obs_dim[1]), dtype=bool
            )
            for bodyid in range(physics.model.nbody)
        }
        for geomid in np.unique(segmentation_map[..., 0]):
            if geomid == -1:
                continue
            geom_mask = segmentation_map[..., 0] == geomid
            if erode_seg_mask:
                # dm_control has a bug where masks returned are slightly
                # incorrect along the edges, so just erode those away
                erode_size = max(2, int(min(obs_dim) / 512))
                geom_mask = cv2.erode(
                    src=geom_mask.astype(np.uint8),
                    kernel=cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (erode_size, erode_size)
                    ),
                ).astype(bool)
            body_name = get_part_path(
                physics.model,
                physics.model.body(physics.model.geom(geomid).bodyid),
            )

            segmentation[body_name] = np.logical_or(geom_mask, segmentation[body_name])
        return VisionSensorOutput(
            compressed_rgb=rgb,  # type: ignore
            compressed_depth=depth,  # type: ignore
            compressed_segmentation=segmentation,  # type: ignore
            pos=cam_pos,
            rot_mat=cam_rotmat,
            fovy=fovy,
            obs_dim=(obs_dim[0], obs_dim[1]),
        )

    def get_state(self) -> EnvState:
        obj_link_states: Dict[str, Dict[str, LinkState]] = {}
        obj_joint_states: Dict[str, Dict[str, JointState]] = {}
        model = self.mj_physics.model
        data = self.mj_physics.data

        obj_link_contacts = parse_contact_data(physics=self.mj_physics)

        for bodyid in range(model.nbody):
            body_model = model.body(bodyid)
            body_data = data.body(bodyid)  # type: ignore
            pose = Pose(
                position=body_data.xpos.copy(),
                orientation=body_data.xquat.copy(),
            )
            root_name = model.body(body_model.rootid).name
            if root_name not in obj_link_states:
                obj_link_states[root_name] = {}
            if root_name not in obj_joint_states:
                obj_joint_states[root_name] = {}
            part_path = get_part_path(self.mj_physics.model, body_model)
            obj_link_states[root_name][part_path] = LinkState(
                link_path=part_path,
                obj_name=root_name,
                pose=pose,
                velocity=Velocity(
                    linear_velocity=body_data.cvel[3:].copy(),
                    angular_velocity=body_data.cvel[:3].copy(),
                ),
                contacts=obj_link_contacts[root_name][part_path]
                if (
                    root_name in obj_link_contacts
                    and part_path in obj_link_contacts[root_name]
                )
                else set(),
                aabbs=get_body_aabbs(
                    model=model,
                    data=data,
                    bodyid=bodyid,
                ),
            )
        for jointid in range(model.njnt):
            joint_model = model.joint(jointid)
            joint_type = JointType.REVOLUTE
            if joint_model.type[0] == 2:
                joint_type = JointType.PRISMATIC
            elif joint_model.type[0] == 3:
                joint_type = JointType.REVOLUTE
            elif joint_model.type[0] == 0:
                continue
            else:
                raise NotImplementedError(str(joint_model))
            joint_axis = joint_model.axis
            min_value, max_value = joint_model.range
            current_value = self.mj_physics.data.qpos[joint_model.qposadr[0]]  # type: ignore
            child_model = model.body(joint_model.bodyid)
            parent_model = model.body(child_model.parentid)
            root_name = model.body(child_model.rootid).name
            parent_data = data.body(parent_model.id)  # type: ignore
            joint2world = (
                affines.compose(
                    T=parent_data.xpos,
                    R=quaternions.quat2mat(parent_data.xquat),
                    Z=np.ones(3),
                )
                @ affines.compose(
                    T=child_model.pos,
                    R=quaternions.quat2mat(child_model.quat),
                    Z=np.ones(3),
                )
                @ affines.compose(T=joint_model.pos, R=np.identity(3), Z=np.ones(3))
            )
            joint_pos = affines.decompose(joint2world)[0]
            obj_joint_states[root_name][joint_model.name] = JointState(
                name=joint_model.name,
                joint_type=joint_type,
                min_value=min_value,
                max_value=max_value,
                current_value=current_value,
                axis=(joint_axis[0], joint_axis[1], joint_axis[2]),
                orientation=quaternions.qmult(parent_data.xquat, child_model.quat),
                position=(joint_pos[0], joint_pos[1], joint_pos[2]),
                parent_link=get_part_path(self.mj_physics.model, parent_model),
                child_link=get_part_path(self.mj_physics.model, child_model),
            )
            for grandchild_model in get_fixed_children(
                self.mj_physics.model, child_model
            ):
                obj_joint_states[root_name][joint_model.name] = JointState(
                    name=joint_model.name,
                    joint_type=joint_type,
                    min_value=min_value,
                    max_value=max_value,
                    current_value=current_value,
                    axis=(joint_axis[0], joint_axis[1], joint_axis[2]),
                    orientation=quaternions.qmult(
                        parent_data.xquat, grandchild_model.quat
                    ),
                    position=(joint_pos[0], joint_pos[1], joint_pos[2]),
                    parent_link=get_part_path(self.mj_physics.model, parent_model),
                    child_link=get_part_path(self.mj_physics.model, grandchild_model),
                )
        object_states = {
            obj_name: ObjectState(
                link_states=link_states, joint_states=obj_joint_states[obj_name]
            )
            for obj_name, link_states in obj_link_states.items()
        }
        grasped_obj_id = self.mujoco_robot.get_grasped_obj_id(
            physics=self.mj_physics, return_root_id=False
        )
        return deepcopy(
            EnvState(
                object_states=object_states,
                end_effector_pose=self.mujoco_robot.end_effector_pose,
                gripper_command=self.mujoco_robot.binary_gripper_command,
                robot_name=self.mujoco_robot.body_model.name,
                robot_gripper_links=self.mujoco_robot.gripper_link_paths,
                grasped_objects=frozenset(
                    {
                        get_part_path(
                            model=model,
                            body=model.body(grasped_obj_id),
                        )
                    }
                    if grasped_obj_id != -1
                    else {}
                ),
                robot_joint_velocities=self.mujoco_robot.get_joint_velocities().tolist(),
            )
        )

    def get_dropped_grasp_early_stop_checker(
        self, goal_gripper_command: bool, grasp_pose_error_threshold: float = 0.1
    ) -> Optional[Callable[[], bool]]:
        if (
            goal_gripper_command
            and self.mujoco_robot.get_grasped_obj_id(physics=self.mj_physics) != -1
        ):
            original_grasp_pose: Pose = deepcopy(
                self.mujoco_robot.get_grasp_pose(physics=self.mj_physics)  # type: ignore
            )
            assert original_grasp_pose is not None

            def early_stop_checker():
                grasp_pose = self.mujoco_robot.get_grasp_pose(physics=self.mj_physics)
                if grasp_pose is None:
                    return True
                grasp_shifted = (
                    grasp_pose.distance(original_grasp_pose) > grasp_pose_error_threshold
                )
                return grasp_shifted

            return early_stop_checker
        return None

    @property
    def mujoco_robot(self) -> MujocoRobot:
        return typing.cast(MujocoRobot, self.robot)

    def randomize(self):
        pass

    def randomize_ctrl(self):
        pass

    def step_until_stable(
        self,
        max_iterations: int = 10000,
        min_iters: int = 100,
        max_velocity: float = 1e-5,
    ):
        for i in range(max_iterations):
            self.mj_physics.step()
            if self.mj_physics.data.qvel.max() < max_velocity and i > min_iters:
                break

    @property
    def max_reset_attempts(self) -> int:
        return 50

    def reset(self, episode_id: int) -> Observation:
        # NOTE crucial that episode_id is set at the beginning of
        # reset because it seeds the random states controlling
        # environment randomization
        self.episode_id = episode_id
        # TODO find better way to reset control buffer
        if self.dynamic_mujoco_model:
            del self.mj_physics
            gc.collect()
            # create new physics to allow randomly loading new objects
            mj_physics, robot_bodyid = self.setup()
            self.update_mj_physics(mj_physics)
            self.mujoco_robot.bodyid = robot_bodyid
            gc.collect()

        # Get all robots to have a init ctrl value
        with self.mj_physics.reset_context():
            self.mj_physics.reset()
            self.randomize()
            self.randomize_ctrl()
            self.mj_physics.forward()
            for _ in range(int(0.5 / self.dt)):
                self.servol(
                    self.control_buffer.get_interpolated_joint_ctrl(
                        t=self.time, robot=self.robot, env_config=self.config
                    )
                )
        for i in range(self.max_reset_attempts):
            self.done = False
            try:
                self.step_until_stable()
            except PhysicsError as e:
                logging.warning(e)
                self.mj_physics.reset()
                self.randomize_ctrl()
                self.mj_physics.forward()
                for _ in range(int(0.5 / self.dt)):
                    self.servol(
                        self.control_buffer.get_interpolated_joint_ctrl(
                            t=self.time, robot=self.robot, env_config=self.config
                        )
                    )
            self.obs = super().reset(episode_id=episode_id)
            if self.is_reset_state_valid(obs=self.obs):
                break
            elif i == self.max_reset_attempts - 1:
                message = (
                    f"[{episode_id:03d}] Failed to place all objects on "
                    + f"the table after {self.max_reset_attempts} attempts."
                )
                logging.error(message)
                raise RuntimeError(message)
            self.randomize()
            self.mj_physics.forward()
        self.done = False
        return self.obs

    def is_reset_state_valid(self, obs: Observation) -> bool:
        if len(self.assert_visible_objs_at_reset) == 0:
            all_main_objs_visible = True
        else:
            checker_view = self.render(obs_cameras=[self.visibility_checker_cam_name])[
                self.visibility_checker_cam_name
            ]
            all_main_objs_visible = self.assert_visible_objs_at_reset is None or (
                all(
                    checker_view.get_segmentation(
                        obj_name
                    ).any()  # if any pixel is visible, that's enough
                    for obj_name in self.assert_visible_objs_at_reset
                )
            )
        return super().is_reset_state_valid(obs=obs) and all_main_objs_visible

    def get_exec_vars(
        self, active_task: Task, sampler_config: EnvSamplerConfig
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        end_effector_pose = self.robot.end_effector_pose
        global_vars = {
            "np": np,
            "ee_gripper_command": False,
            "ee_position": end_effector_pose.position,
            "ee_orientation": end_effector_pose.orientation,
            "Pose": Pose,
            "gripper_rest_orientation": self.robot.end_effector_rest_orientation,
            "subtrajectory_steps": [],
            "max_time": sampler_config.max_time,
        }

        def sample_one_step():
            if self.done or self.time >= global_vars["max_time"]:
                return
            # this shouldn't matter for CodeAction
            obs = self.get_obs()
            action = EndEffectorAction(
                end_effector_position=global_vars["ee_position"],
                end_effector_orientation=global_vars["ee_orientation"],
                gripper_command=global_vars["ee_gripper_command"],
                allow_contact=True,
            )

            next_obs, self.done, info = self.step(
                action=action,
                active_task=active_task,
                sampler_config=sampler_config,
            )
            global_vars["subtrajectory_steps"].append(
                TrajectoryStep(
                    obs=obs,
                    action=action,
                    next_obs=next_obs,
                    done=self.done,
                    info=info,
                    compressed_renders=[],
                    visualization_dim=(0, 0),
                    control_states=[],
                )
            )

        def get_object_pose(obj_name: str, state: Optional[EnvState] = None) -> Pose:
            if state is None:
                state = self.get_state()
            return state.get_pose(key=obj_name)

        def activate_gripper():
            global_vars["ee_gripper_command"] = True
            sample_one_step()

        def release_gripper():
            global_vars["ee_gripper_command"] = False
            sample_one_step()

        def move_gripper_to_pose(pose: Pose):
            global_vars["ee_position"] = deepcopy(pose.position)
            global_vars["ee_orientation"] = deepcopy(pose.orientation)
            sample_one_step()

        def compose_orientations(o1: np.ndarray, o2: np.ndarray):
            if o1.shape == (3,):
                # most likely in degrees
                o1 = o1.astype(float) * np.pi / 180.0
                o1 = (o1 + np.pi) % (2 * np.pi) - np.pi
                o1 = euler.euler2quat(*o1)
            if o2.shape == (3,):
                # most likely in degrees
                o2 = o2.astype(float) * np.pi / 180.0
                o2 = (o2 + np.pi) % (2 * np.pi) - np.pi
                o2 = euler.euler2quat(*o2)
            return quaternions.qmult(
                o2,
                o1,
            )

        return global_vars, {
            "move_gripper_to_pose": move_gripper_to_pose,
            "get_object_pose": get_object_pose,
            "activate_gripper": activate_gripper,
            "release_gripper": release_gripper,
            "compose_orientations": compose_orientations,
        }

    @property
    def dt(self) -> float:
        return self.mj_physics.timestep()

    def motion_plan_rrt(
        self,
        goal_conf: np.ndarray,
        current_gripper_command: bool,
        goal_gripper_command: bool,
    ) -> Optional[List[np.ndarray]]:
        self.rrt.np_random = np.random.RandomState(
            self.rrt_numpy_random.randint(0, 2**10)
        )
        self.rrt.robot.last_joints = self.mujoco_robot.last_joints
        start_conf = self.robot.joint_config
        plan, rrt_log = self.rrt.plan(
            start_conf=start_conf,
            goal_conf=goal_conf,
            qpos=self.mj_physics.data.qpos,
            qvel=self.mj_physics.data.qvel,
            ctrl=self.mj_physics.data.ctrl,
            current_gripper_command=current_gripper_command,
            goal_gripper_command=goal_gripper_command,
        )
        if plan is None:
            raise Action.FailedExecution(message=rrt_log)
        else:
            logging.info(f"RRT succeeded with {len(plan)} steps")
        return plan

    @property
    def joints(self):
        return list(map(self.mj_physics.model.joint, range(self.mj_physics.model.njnt)))

    def servol(self, command: np.ndarray):
        self.mj_physics.set_control(command)

    def get_control(self) -> np.ndarray:
        return self.mj_physics.control()

    @staticmethod
    def add_obj_from_model(
        obj_model: RootElement,
        world_model: RootElement,
        position: Optional[Point3D] = None,
        euler: Optional[Point3D] = None,
        add_free_joint: Optional[bool] = None,
    ) -> RootElement:
        all_children = obj_model.worldbody.all_children()  # type: ignore
        assert len(all_children) == 1
        # remove existing free joint
        world_model.attach(obj_model)
        obj_body = world_model.worldbody.all_children()[-1]  # type: ignore
        if add_free_joint is None:
            add_free_joint = obj_body.all_children()[0].freejoint is not None
        # clear transforms TODO: loop over all relevant attrs
        # https://mujoco.readthedocs.io/en/stable/XMLreference.html#world-body-r
        # try:
        if add_free_joint:
            try:
                del obj_body.all_children()[0].freejoint
            except AttributeError:
                pass
            try:
                del obj_body.all_children()[0].pos
                del obj_body.all_children()[0].euler
                del obj_body.all_children()[0].quat
                del obj_body.all_children()[0].axisangle
                del obj_body.all_children()[0].xyaxes
                del obj_body.all_children()[0].zaxis
            except AttributeError:
                pass
            obj_body.add("freejoint")

        if position is not None:
            obj_body.all_children()[-1].pos = " ".join(map(str, position))
        if euler is not None:
            obj_body.all_children()[-1].euler = " ".join(map(str, euler))
        return obj_body

    @staticmethod
    def rename_model(model: RootElement, name: str):
        model = deepcopy(model)
        model.model = name
        model.worldbody.all_children()[-1].name = name
        return model


class MujocoUR5Env(MujocoEnv):
    def __init__(
        self,
        robot_cls: Type[UR5],
        num_setup_variations: Optional[int] = None,
        num_pose_variations: Optional[int] = None,
        ground_xml_path: str = "scalingup/environment/mujoco/assets/ground.xml",
        init_ee_pos: Tuple[float, float, float] = (0.2, 0.0, 0.3),
        init_ee_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        **kwargs,
    ):
        self.ground_xml_path = ground_xml_path
        super().__init__(robot_cls=robot_cls, **kwargs)
        assert len(self.obj_qpos_indices) == len(self.obj_qpos_ranges)
        joint_values = self.robot.inverse_kinematics(
            pose=Pose(
                position=np.array(init_ee_pos),
                orientation=euler.euler2quat(*init_ee_euler),
            )
        )
        assert joint_values is not None
        self.robot_init_joint_pos: np.ndarray = joint_values
        self.robot_init_ctrl: ControlAction
        if self.config.ctrl.control_type == ControlType.JOINT:
            # also add gripper control
            self.robot_init_ctrl = ControlAction(
                value=np.stack(
                    [
                        np.concatenate(
                            [joint_values, [self.ur5.gripper_open_ctrl_val]], axis=0
                        )
                    ]
                ),
                timestamps=np.array([0.0]),
                config=self.config.ctrl,
            )
        else:
            assert self.config.ctrl.control_type == ControlType.END_EFFECTOR
            if self.config.ctrl.rotation_type == RotationType.QUATERNION:
                init_ee_orn = euler.euler2quat(*init_ee_euler)
            elif self.config.ctrl.rotation_type == RotationType.ROT_MAT:
                init_ee_orn = euler.euler2mat(*init_ee_euler).reshape(-1)
            elif self.config.ctrl.rotation_type == RotationType.UPPER_ROT_MAT:
                init_ee_orn = euler.euler2mat(*init_ee_euler)[:2].reshape(-1)
            else:
                raise NotImplementedError
            ctrl_val = np.concatenate(
                [init_ee_pos, init_ee_orn, [self.ur5.gripper_open_ctrl_val]],
                axis=0,
            )
            self.robot_init_ctrl = ControlAction(
                value=np.stack([ctrl_val]),
                timestamps=np.array([0.0]),
                config=self.config.ctrl,
                target_ee_actions=[
                    EndEffectorAction(
                        end_effector_position=np.array(init_ee_pos),
                        end_effector_orientation=euler.euler2quat(*init_ee_euler),
                        gripper_command=False,  # open
                        allow_contact=True,
                    )
                ],
            )

    @property
    def ur5(self) -> UR5:
        assert issubclass(type(self.robot), UR5)
        return typing.cast(UR5, self.robot)

    def joint_pos_control_to_ee_pose_control(
        self, target_ctrl_val: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        non_gripper_joint_pos = target_ctrl_val[:6]
        gripper_joint_pos = target_ctrl_val[6]
        ee_pose = self.mujoco_robot.forward_kinematics(
            joints=non_gripper_joint_pos, return_ee_pose=True
        )
        assert ee_pose is not None
        return ee_pose.position, ee_pose.orientation, gripper_joint_pos

    def randomize_ctrl(self):
        self.control_buffer = deepcopy(self.robot_init_ctrl)

    @property
    def robot_qpos_indices(self) -> List[int]:
        return sorted(
            sum(
                (
                    list(
                        np.arange(
                            joint.qposadr[0],
                            joint.qposadr[0] + (6 if joint.type[0] == 0 else 1),
                        )
                    )
                    for joint in filter(
                        lambda joint: self.mj_physics.model.body(
                            joint.bodyid[0]
                        ).name.startswith(self.robot_model_name),
                        self.joints,
                    )
                ),
                [],
            )
        )

    @property
    def obj_qpos_indices(self) -> List[int]:
        return sorted(
            sum(
                (
                    list(
                        np.arange(
                            joint.qposadr[0],
                            joint.qposadr[0] + (6 if joint.type[0] == 0 else 1),
                        )
                    )
                    # all the joints whose name don't start with robot name
                    # belong to the object
                    for joint in filter(
                        lambda joint: not self.mj_physics.model.body(
                            joint.bodyid[0]
                        ).name.startswith(self.robot_model_name),
                        self.joints,
                    )
                ),
                [],
            )
        )

    def randomize(self):
        # NOTE assume that the first `len(self.robot_init_joint_pos)`
        # joints are the robot joints
        self.mj_physics.data.qpos[
            self.robot_qpos_indices[: len(self.robot_init_joint_pos)]
        ] = self.robot_init_joint_pos
        obj_qpos_indices = self.obj_qpos_indices
        obj_qpos = [
            self.pose_numpy_random.uniform(qpos_range.lower, qpos_range.upper)
            for qpos_range in self.obj_qpos_ranges
        ]
        self.mj_physics.data.qpos[obj_qpos_indices] = obj_qpos


class MujocoUR5WSG50FinrayEnv(MujocoUR5Env):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(robot_cls=UR5WSG50Finray, **kwargs)
        assert len(self.obj_qpos_indices) == len(self.obj_qpos_ranges)

    def setup_model(self) -> Tuple[RootElement, str]:
        # load world scene
        world_model = mjcf.from_path(self.ground_xml_path)

        # load robot scene
        robot_model = mjcf.from_path(
            "scalingup/environment/mujoco/assets/menagerie/universal_robots_ur5e/ur5e.xml",
        )
        del robot_model.keyframe
        robot_model.worldbody.light.clear()
        attachment_site = robot_model.find("site", "attachment_site")
        assert attachment_site is not None
        gripper = mjcf.from_path(
            "scalingup/environment/mujoco/assets/wsg50/wsg50_finray_milibar.xml",
        )

        cam_mount_site = gripper.find("site", "cam_mount")
        cam = mjcf.from_path(
            "scalingup/environment/mujoco/assets/menagerie/realsense_d435i/d435i_with_cam.xml"
        )
        cam_mount_site.attach(cam)

        attachment_site.attach(gripper)

        self.robot_model_name = robot_model.model
        world_model.attach(robot_model)

        robot_root_link_name = os.path.join(
            self.robot_model_name,
            robot_model.worldbody.all_children()[0].name,  # type: ignore
        )

        return world_model, robot_root_link_name


class MujocoUR5Robotiq85fEnv(MujocoUR5Env):
    def __init__(
        self,
        num_setup_variations: Optional[int] = None,
        num_pose_variations: Optional[int] = None,
        **kwargs,
    ):
        self.num_setup_variations = num_setup_variations
        self.num_pose_variations = num_pose_variations
        self.episode_id = 0
        super().__init__(
            robot_cls=UR5Robotiq,
            num_setup_variations=num_setup_variations,
            num_pose_variations=num_pose_variations,
            **kwargs,
        )
        assert len(self.obj_qpos_indices) == len(self.obj_qpos_ranges)

    def setup_model(self) -> Tuple[RootElement, str]:
        # load world scene
        world_model = mjcf.from_path(self.ground_xml_path)

        # load robot scene
        robot_scene_model = mjcf.from_path(
            "scalingup/environment/mujoco/assets/ur5-robotiq.xml",
        )

        del robot_scene_model.keyframe
        self.robot_model_name = robot_scene_model.model
        world_model.attach(robot_scene_model)

        robot_root_link_name = os.path.join(
            self.robot_model_name,
            robot_scene_model.worldbody.all_children()[0].name,  # type: ignore
        )
        return world_model, robot_root_link_name


class MujocoUR5EnvFromObjConfigList(MujocoUR5WSG50FinrayEnv):
    def __init__(self, obj_instance_configs: List[MujocoObjectInstanceConfig], **kwargs):
        self.obj_instance_configs = obj_instance_configs
        super().__init__(**kwargs)

    def setup_objs(self, world_model: RootElement) -> QPosRange:
        # load objects
        self.assert_visible_objs_at_reset = set()
        obj_qpos_ranges = super().setup_objs(world_model=world_model)
        for obj_instance_config in self.obj_instance_configs:
            obj_model = mjcf.from_path(obj_instance_config.asset_path)
            if obj_instance_config.name is not None:
                obj_model = self.rename_model(
                    model=obj_model, name=obj_instance_config.name
                )

            if obj_instance_config.color_config is not None:
                obj_class = obj_instance_config.obj_class
                color_name = obj_instance_config.color_config.name
                obj_model.model = f"{color_name}_{obj_class}"
                assert len(obj_model.find_all("body")) == 1
                obj_model.find_all("body")[0].name = f"{color_name}_{obj_class}"
                # add material node to change color
                material_name = f"{color_name}_{obj_class}"
                obj_model.asset.add(
                    "material",
                    name=material_name,
                    rgba=(*obj_instance_config.color_config.rgb, 1),
                )
                for geom in obj_model.find_all("geom"):
                    geom.material = material_name

            self.add_obj_from_model(
                obj_model=obj_model,
                world_model=world_model,
                position=obj_instance_config.position,
                euler=obj_instance_config.euler,
                add_free_joint=obj_instance_config.add_free_joint,
            )
            obj_qpos_ranges.extend(
                obj_instance_config.qpos_range
                if obj_instance_config.qpos_range is not None
                else (
                    DegreeOfFreedomRange(lower=lower, upper=upper)
                    for lower, upper in [
                        # 3D position
                        (0.4, 0.5),
                        (-0.05, 0.05),
                        (0.1, 0.2),
                        # euler rotation
                        (-np.pi, np.pi),
                        (-np.pi, np.pi),
                        (-np.pi, np.pi),
                    ]
                )
            )
            part_path = "".join(
                [
                    obj_model.model,
                    MJCF_NEST_TOKEN,
                    LINK_SEPARATOR_TOKEN,
                    obj_model.model,
                    MJCF_NEST_TOKEN,
                    obj_model.model,
                ],
            )
            self.assert_visible_objs_at_reset.add(part_path)
        return obj_qpos_ranges
