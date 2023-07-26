import logging
import typing
import numpy as np
from scalingup.environment.mujoco.mujocoEnv import (
    MujocoEnv,
    MujocoUR5EnvFromObjConfigList,
)
from scalingup.environment.mujoco.table_top import TableTopMujocoEnv
from scalingup.environment.mujoco.utils import (
    MujocoObjectColorConfig,
    MujocoObjectInstanceConfig,
)
from scalingup.utils.core import (
    ControlAction,
    DegreeOfFreedomRange,
    Env,
    JointState,
    JointType,
)
from scalingup.utils.state_api import check_joint_activated


class CatapultMujocoEnv(TableTopMujocoEnv):
    def __init__(
        self, end_after_activated_time: float, pose_randomization: bool = False, **kwargs
    ):
        self.end_after_activated_time = end_after_activated_time
        catapult_config = MujocoObjectInstanceConfig(
            obj_class="catapult",
            asset_path="scalingup/environment/mujoco/assets/custom/catapult/catapult.xml",
            qpos_range=[
                DegreeOfFreedomRange(upper=0.0, lower=0.0),
                DegreeOfFreedomRange(upper=0.0, lower=0.0),
            ],
            position=(0.5, 0.2, 0.075),
        )
        block_config = MujocoObjectInstanceConfig(
            obj_class="block",
            asset_path="scalingup/environment/mujoco/assets/custom/block.xml",
            qpos_range=[
                DegreeOfFreedomRange(upper=0.2, lower=0.4),
                DegreeOfFreedomRange(upper=0.3, lower=0.4),
                DegreeOfFreedomRange(upper=0.1, lower=0.1),
                DegreeOfFreedomRange(upper=np.pi, lower=-np.pi),
                DegreeOfFreedomRange(upper=np.pi, lower=-np.pi),
                DegreeOfFreedomRange(upper=np.pi, lower=-np.pi),
            ]
            if pose_randomization
            else [
                DegreeOfFreedomRange(upper=0.3, lower=0.3),
                DegreeOfFreedomRange(upper=0.35, lower=0.35),
                DegreeOfFreedomRange(upper=0.01, lower=0.01),
                # testing y value
                # DegreeOfFreedomRange(upper=0.5, lower=0.5),
                # testing x value
                # DegreeOfFreedomRange(upper=0.48, lower=0.48),
                # DegreeOfFreedomRange(upper=0.47, lower=0.47),
                # DegreeOfFreedomRange(upper=0.45, lower=0.45),
                # DegreeOfFreedomRange(upper=0.44, lower=0.44),
                # DegreeOfFreedomRange(upper=0.43, lower=0.43),
                # DegreeOfFreedomRange(upper=0.35, lower=0.35),
                # DegreeOfFreedomRange(upper=0.40, lower=0.40),
                # testing z value
                # DegreeOfFreedomRange(upper=0.08, lower=0.08),
                DegreeOfFreedomRange(upper=0, lower=0),
                DegreeOfFreedomRange(upper=0, lower=0),
                DegreeOfFreedomRange(upper=0, lower=0),
            ],
            color_config=MujocoObjectColorConfig(
                name="yellow",
                rgb=(1.0, 1.0, 0.0),
            ),
        )
        closest_box = MujocoObjectInstanceConfig(
            obj_class="bowl",
            asset_path="scalingup/environment/mujoco/assets/custom/target_box.xml",
            qpos_range=[],
            position=(0.5, -0.1, 0.07),
            name="closest_box",
        )
        middle_box = MujocoObjectInstanceConfig(
            obj_class="bowl",
            asset_path="scalingup/environment/mujoco/assets/custom/target_box.xml",
            qpos_range=[],
            position=(0.5, -0.3, 0.07),
            name="middle_box",
        )
        furthest_box = MujocoObjectInstanceConfig(
            obj_class="bowl",
            asset_path="scalingup/environment/mujoco/assets/custom/target_box.xml",
            qpos_range=[],
            position=(0.5, -0.5, 0.07),
            name="furthest_box",
        )
        super().__init__(
            obj_instance_configs=[
                catapult_config,
                block_config,
                closest_box,
                furthest_box,
                middle_box,
            ],
            **kwargs,
        )

        def catapult_mechanism(env: Env):
            """
            on top of checking if the catapult is activated, this function also
            makes sure the block is not picked and placed into the box
            """
            assert issubclass(type(env), CatapultMujocoEnv)
            catapult_env = typing.cast(CatapultMujocoEnv, env)
            physics = catapult_env.mj_physics
            # check pick and place first
            grasp_id = catapult_env.mujoco_robot.get_grasped_obj_id(physics)
            ee_pose = catapult_env.mujoco_robot.end_effector_pose
            if grasp_id != -1 and ee_pose.position[1] < 0:
                logging.info("policy attempting to place block into box")
                env.done = True
                return

            catapult_joint = physics.model.joint("catapult/button_slider")
            min_value, max_value = catapult_joint.range
            activated = check_joint_activated(
                joint_state=JointState(
                    current_value=physics.data.qpos[catapult_joint.qposadr[0]],
                    min_value=min_value,
                    max_value=max_value,
                    # VALUES BELOW NOT USED
                    name="",
                    joint_type=JointType.PRISMATIC,
                    axis=(0, 0, 0),
                    position=(0, 0, 0),
                    orientation=np.array([0, 0, 0, 0]),
                    parent_link="",
                    child_link="",
                )
            )
            if (
                activated
                and self.mj_physics.model.ptr.eq("catapult/catapult_trigger").active
            ):
                logging.info("catapult activated")
                self.mj_physics.model.ptr.eq("catapult/catapult_trigger").active = 0
                ctrl_cycles = int(
                    np.ceil(self.end_after_activated_time * env.config.ctrl.frequency)
                )
                ctrl_val = self.control_buffer.get_target_ctrl(t=self.time)[1].copy()
                executed_ctrl_mask = (
                    self.control_buffer.timestamps
                    <= self.time + 1 / env.config.ctrl.frequency
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

                start_time = self.control_buffer.timestamps[-1]
                ctrl = ControlAction(
                    value=np.stack([ctrl_val] * ctrl_cycles),
                    timestamps=np.linspace(
                        start_time,
                        start_time + ctrl_cycles / env.config.ctrl.frequency,
                        ctrl_cycles,
                        endpoint=False,
                    )
                    + 1 / env.config.ctrl.frequency,
                    config=env.config.ctrl,
                    target_ee_actions=[
                        self.control_buffer.get_target_ctrl(t=self.time)[2]
                    ]
                    * ctrl_cycles,
                )
                env.control_buffer = env.control_buffer.combine(ctrl)
                env.done = True

        self.step_fn_callbacks["catapult_mechanism"] = (
            int(1 / self.dt),
            catapult_mechanism,
        )

    def randomize(self):
        self.mj_physics.model.ptr.eq("catapult/catapult_trigger").active = 1
        return super().randomize()

    def reset(self, episode_id: int = 0):
        # get spring to stabilize
        self.step_until_stable(min_iters=10000, max_velocity=1e-10)
        return super().reset(episode_id=episode_id)
