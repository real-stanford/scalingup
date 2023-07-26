import logging
import os
from abc import abstractproperty
from typing import Any, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from dm_control.mujoco.engine import Physics
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from transforms3d import quaternions
from scalingup.utils.core import Pose, Robot


class MujocoRobot(Robot):
    def __init__(
        self,
        mj_physics,
        init_joints: np.ndarray,
        bodyid: int,
        gripper_link_paths: List[str],
        gripper_joint_name: str,
        gripper_actuator_name: str,
        prefix: str = "",
    ):
        self.mj_physics = mj_physics
        self.ik_mj_physics = mj_physics.copy(share_model=True)
        self.prefix = prefix
        self.bodyid = bodyid
        self.gripper_link_paths = gripper_link_paths
        self.gripper_joint_name = gripper_joint_name
        self.gripper_actuator_name = gripper_actuator_name
        super().__init__(init_joints=init_joints)

    @property
    def joint_config(self) -> np.ndarray:
        return self.mj_physics.data.qpos[self.joint_qpos_indices]

    def get_joint_velocities(self) -> np.ndarray:
        return self.mj_physics.data.qvel[[joint.dofadr[0] for joint in self.joints]]

    @property
    def body_model(self):
        # TODO make sure self.bodyid is frozen
        return self.mj_physics.model.body(self.bodyid)

    @property
    def end_effector_site_name(self) -> str:
        # TODO make sure self.prefix is frozen
        return os.path.join(self.prefix, "end_effector")

    @abstractproperty
    def end_effector_links(self) -> List[Any]:
        raise NotImplementedError()

    @abstractproperty
    def joint_names(self) -> List[str]:
        raise NotImplementedError()

    @abstractproperty
    def robot_collision_link_paths(self) -> List[str]:
        raise NotImplementedError()

    @property
    def end_effector_link_ids(self) -> List[int]:
        return [link.id for link in self.end_effector_links]

    def get_end_effector_pose(self, physics: Optional[Physics] = None) -> Pose:
        if physics is None:
            physics = self.mj_physics
        assert physics is not None
        # account for rest pose
        return Pose(
            position=physics.data.site(self.end_effector_site_name).xpos.copy(),
            orientation=quaternions.qmult(
                quaternions.mat2quat(
                    physics.named.data.site_xmat[self.end_effector_site_name].copy()
                ),
                quaternions.qinverse(self.end_effector_rest_orientation),
            ),
        )

    @property
    def end_effector_pose(self) -> Pose:
        return self.get_end_effector_pose(physics=self.mj_physics)

    @property
    def joints(self) -> List[Any]:
        return [
            self.mj_physics.model.joint(joint_name) for joint_name in self.joint_names
        ]

    def get_grasped_obj_child_links(
        self,
        physics: Optional[Physics] = None,
        grasp_obj_id: Optional[int] = None,
        detect_grasp: bool = True,
    ) -> List[Any]:
        if physics is None:
            physics = self.mj_physics
        assert physics is not None

        if detect_grasp or grasp_obj_id is None:
            grasp_obj_id = self.get_grasped_obj_id(physics=physics)
        if grasp_obj_id == -1:
            return []
        return [
            body
            for body in map(
                self.mj_physics.model.body, range(self.mj_physics.model.nbody)
            )
            if body.rootid[0] == grasp_obj_id
        ]

    def get_collision_link_ids(
        self,
        physics: Optional[Physics] = None,
        grasp_obj_id: Optional[int] = None,
        detect_grasp: bool = True,
    ) -> List[int]:
        if physics is None:
            physics = self.mj_physics
        assert physics is not None
        collision_link_ids = [
            self.mj_physics.model.body(link_name).id
            for link_name in self.robot_collision_link_paths
        ]
        for link in self.get_grasped_obj_child_links(
            physics=physics, grasp_obj_id=grasp_obj_id, detect_grasp=detect_grasp
        ):
            collision_link_ids.append(link.id)
        return collision_link_ids

    def get_disabled_collision_pairs(
        self,
        physics: Optional[Physics] = None,
        grasp_obj_id: Optional[int] = None,
        detect_grasp: bool = True,
    ) -> Set[FrozenSet[int]]:
        if physics is None:
            physics = self.mj_physics
        assert physics is not None
        disabled_collision_pairs = set()
        for ee_link in self.end_effector_links:
            for other_ee_link in self.end_effector_links:
                if other_ee_link.id == ee_link.id:
                    continue
                disabled_collision_pairs.add(frozenset([other_ee_link.id, ee_link.id]))
        for link in self.get_grasped_obj_child_links(
            physics=physics, grasp_obj_id=grasp_obj_id, detect_grasp=detect_grasp
        ):
            for ee_link in self.end_effector_links:
                disabled_collision_pairs.add(frozenset([ee_link.id, link.id]))
        return disabled_collision_pairs

    @property
    def joint_qpos_indices(self) -> List[int]:
        return [joint.qposadr[0] for joint in self.joints]

    def inverse_kinematics(self, pose: Pose, inplace=False) -> Optional[np.ndarray]:
        pose = Pose(
            position=pose.position,
            orientation=quaternions.qmult(
                pose.orientation, self.end_effector_rest_orientation
            ),
        )
        if inplace:
            self.ik_mj_physics.data.qpos[:] = self.mj_physics.data.qpos[:].copy()
            # even if doing in place, should use different physics to ensure
            # main simulation's qpos follows physics
            result = qpos_from_site_pose(
                physics=self.ik_mj_physics,
                site_name=self.end_effector_site_name,
                target_pos=pose.position,
                target_quat=pose.orientation,
                joint_names=self.joint_names,
                tol=1e-7,
                max_steps=300,
                inplace=inplace,
            )
        else:
            result = qpos_from_site_pose(
                physics=self.mj_physics,
                site_name=self.end_effector_site_name,
                target_pos=pose.position,
                target_quat=pose.orientation,
                joint_names=self.joint_names,
                tol=1e-7,
                max_steps=300,
                inplace=inplace,
            )
        if not result.success:
            return None
        return result.qpos[self.joint_qpos_indices].copy()

    def get_grasped_obj_id(
        self, physics: Optional[Physics] = None, return_root_id: bool = True
    ) -> int:
        if physics is None:
            physics = self.mj_physics
        if not self.binary_gripper_command:
            # gripper is not closed
            return -1

        # NOTE current heuristic only considers contacts
        # A more complete solution analyzes contact normals,
        # forces, an
        # d specific pairs of contacts corresponding
        # to specific grasps
        collided_bodies1 = self.mj_physics.model.geom_bodyid[
            self.mj_physics.data.contact.geom1
        ]
        collided_bodies2 = self.mj_physics.model.geom_bodyid[
            self.mj_physics.data.contact.geom2
        ]
        ungraspable_ids = [0]  # world
        ungraspable_ids += [
            self.mj_physics.model.body(link_name).id
            for link_name in self.robot_collision_link_paths
        ]
        ungraspable_ids += [
            body.id
            for body in map(
                self.mj_physics.model.body, range(self.mj_physics.model.nbody)
            )
            if body.rootid[0] in ungraspable_ids
        ]
        end_effector_link_ids = self.end_effector_link_ids
        for body1, body2 in zip(collided_bodies1, collided_bodies2):
            if body1 in end_effector_link_ids and body2 not in ungraspable_ids:
                if not return_root_id:
                    return body2
                root = self.mj_physics.model.body(
                    self.mj_physics.model.body(body2).rootid[0]
                )
                if root.jntnum[0] == 0:  # NOTE don't grasp fixed bodies
                    continue
                return root.id
            elif body2 in end_effector_link_ids and body1 not in ungraspable_ids:
                if not return_root_id:
                    return body1
                root = self.mj_physics.model.body(
                    self.mj_physics.model.body(body1).rootid[0]
                )
                if root.jntnum[0] == 0:  # NOTE don't grasp fixed bodies
                    continue
                return root.id
        return -1

    def get_grasp_pose(self, physics: Optional[Physics] = None) -> Optional[Pose]:
        if physics is None:
            physics = self.mj_physics
        assert physics is not None
        grasped_obj_id = self.get_grasped_obj_id(physics=physics)
        if grasped_obj_id == -1:
            return None
        grasped_obj_model = physics.model.body(grasped_obj_id)
        grasped_obj_data = physics.data.body(grasped_obj_id)

        self.grasped_obj_qpos_adr = physics.model.joint(
            grasped_obj_model.jntadr[0]
        ).qposadr[0]
        obj_pose = Pose(
            position=grasped_obj_data.xpos.copy(),
            orientation=quaternions.mat2quat(grasped_obj_data.xmat.copy().reshape(3, 3)),
        )
        return Pose.from_matrix(
            np.linalg.inv(self.get_end_effector_pose(physics=physics).matrix)
            @ obj_pose.matrix
        )

    def check_grasp(self, physics: Optional[Physics] = None) -> bool:
        if physics is None:
            physics = self.mj_physics
        assert physics is not None
        # TODO refactor this to ensure grasp information is always
        # up to date, while avoid calling every time step. However,
        # ideally not entangle between environment and robot classes
        # in any hard-coded way.
        grasped_obj_id = self.get_grasped_obj_id(physics=physics)
        if grasped_obj_id == -1:
            return False
        return True

    def set_joint_config(
        self,
        joints: np.ndarray,
        return_ee_pose: bool = False,
        physics: Optional[Physics] = None,
        grasp_pose: Optional[Pose] = None,
    ) -> Optional[Pose]:
        if physics is None:
            physics = self.mj_physics
        assert physics is not None
        physics.data.qpos[self.joint_qpos_indices] = joints
        physics.forward()
        pose: Optional[Pose] = None
        if grasp_pose is not None:
            pose = self.get_end_effector_pose(physics=physics)
            new_pose = Pose.from_matrix(pose.matrix @ grasp_pose.matrix)
            physics.data.qpos[
                self.grasped_obj_qpos_adr : self.grasped_obj_qpos_adr + 3
            ] = new_pose.position
            physics.data.qpos[
                self.grasped_obj_qpos_adr + 3 : self.grasped_obj_qpos_adr + 7
            ] = new_pose.orientation
            physics.forward()
        if return_ee_pose and pose is None:
            pose = self.get_end_effector_pose(physics=physics)
        return pose

    def get_unfiltered_collided_pairs(
        self,
        joints: Optional[np.ndarray] = None,
        physics: Optional[Physics] = None,
        grasp_obj_id: Optional[int] = None,
        grasp_pose: Optional[Pose] = None,
        detect_grasp: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: support contact distance margin. Read more about margin/gap at
        # https://mujoco.readthedocs.io/en/stable/XMLreference.html?highlight=margin#body-geom
        if physics is None:
            if joints is None:
                # just checking current state
                physics = self.mj_physics
            else:
                # checking another config, which probably means current physics
                # shouldn't be affected
                physics = self.mj_physics.copy(share_model=True)
        assert physics is not None
        if joints is None:
            joints = physics.data.qpos[self.joint_qpos_indices]
        assert joints is not None
        if detect_grasp:
            grasp_obj_id = self.get_grasped_obj_id(physics=physics)
            grasp_pose = self.get_grasp_pose(physics=physics)
        disabled_collision_pairs = self.get_disabled_collision_pairs(
            physics=physics, grasp_obj_id=grasp_obj_id, detect_grasp=False
        )

        self.set_joint_config(
            joints,
            physics=physics,
            grasp_pose=grasp_pose,
        )

        collided_body1 = physics.model.geom_bodyid[physics.data.contact.geom1].copy()
        collided_body2 = physics.model.geom_bodyid[physics.data.contact.geom2].copy()
        if len(disabled_collision_pairs) > 0:
            enabled_collision_mask = np.ones_like(collided_body1).astype(bool)
            for idx in range(len(collided_body1)):
                body1 = collided_body1[idx]
                body2 = collided_body2[idx]
                if frozenset([body1, body2]) in disabled_collision_pairs:
                    enabled_collision_mask[idx] = False
            collided_body1 = collided_body1[enabled_collision_mask]
            collided_body2 = collided_body2[enabled_collision_mask]
        return collided_body1, collided_body2

    def get_unfiltered_collided_pairs_names(
        self,
        joints: Optional[np.ndarray] = None,
        physics: Optional[Physics] = None,
        grasp_obj_id: Optional[int] = None,
        grasp_pose: Optional[Pose] = None,
        detect_grasp: bool = True,
    ) -> List[Tuple[str, str]]:
        collided_body1, collided_body2 = self.get_unfiltered_collided_pairs(
            joints=joints,
            physics=physics,
            grasp_obj_id=grasp_obj_id,
            grasp_pose=grasp_pose,
            detect_grasp=detect_grasp,
        )
        return [
            (
                self.mj_physics.model.body(link1).name + f" ({link1})",
                self.mj_physics.model.body(link2).name + f" ({link2})",
            )
            for link1, link2 in zip(collided_body1, collided_body2)
        ]

    def get_collided_link_ids(
        self,
        joints: Optional[np.ndarray] = None,
        physics: Optional[Physics] = None,
        grasp_obj_id: Optional[int] = None,
        grasp_pose: Optional[Pose] = None,
        detect_grasp: bool = True,
    ) -> Set[int]:
        collided_body1, collided_body2 = self.get_unfiltered_collided_pairs(
            joints=joints,
            physics=physics,
            grasp_obj_id=grasp_obj_id,
            grasp_pose=grasp_pose,
            detect_grasp=detect_grasp,
        )
        collided_link_ids = set(collided_body1).union(collided_body2)
        filtered_collision_link_ids = collided_link_ids.intersection(
            self.get_collision_link_ids(
                physics=physics, grasp_obj_id=grasp_obj_id, detect_grasp=False
            )
        )
        if len(filtered_collision_link_ids) > 0:
            logging.debug(
                "MujocoRobot collision:"
                + "\n".join(
                    self.mj_physics.model.body(link1).name
                    + " \t | \t "
                    + self.mj_physics.model.body(link2).name
                    for link1, link2 in zip(collided_body1, collided_body2)
                )
            )
        return filtered_collision_link_ids

    def check_collision(
        self,
        joints: Optional[np.ndarray] = None,
        physics: Optional[Physics] = None,
        grasp_obj_id: Optional[int] = None,
        grasp_pose: Optional[Pose] = None,
        detect_grasp: bool = True,
    ) -> bool:
        collided_link_ids = self.get_collided_link_ids(
            joints=joints,
            physics=physics,
            grasp_obj_id=grasp_obj_id,
            grasp_pose=grasp_pose,
            detect_grasp=detect_grasp,
        )
        if len(collided_link_ids) > 0:
            return True
        return False

    def forward_kinematics(
        self, joints: np.ndarray, return_ee_pose: bool = False
    ) -> Optional[Pose]:
        physics = self.mj_physics.copy(share_model=True)
        return self.set_joint_config(
            joints, return_ee_pose=return_ee_pose, physics=physics
        )
