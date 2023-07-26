from typing import List
import numpy as np
from dm_control import mjcf
from dm_control.mjcf import RootElement
from scalingup.environment.mujoco.table_top import TableTopMujocoEnv
from scalingup.utils.constants import LINK_SEPARATOR_TOKEN, MJCF_NEST_TOKEN
from scalingup.utils.core import (
    DegreeOfFreedomRange,
    EnvState,
    Observation,
    QPosRange,
    Task,
    Trajectory,
)
from scalingup.utils.state_api import check_inside, check_on_top_of


class TableTopBinGraspingMujocoEnv(TableTopMujocoEnv):
    def __init__(
        self,
        asset_paths: List[str],
        bin_x_offset: float,
        min_bin_y_dist: float,
        max_bin_y_dist: float,
        bin_z_rot_margin: float,
        obj_x_margin: float,
        obj_y_margin: float,
        tote_path: str = "scalingup/environment/mujoco/assets/custom/tote/real-tote.xml",
        **kwargs,
    ):
        self.asset_paths = asset_paths
        self.bin_x_offset = bin_x_offset
        self.min_bin_y_dist = min_bin_y_dist
        self.max_bin_y_dist = max_bin_y_dist
        self.obj_x_margin = obj_x_margin
        self.obj_y_margin = obj_y_margin
        self.bin_z_rot_margin = bin_z_rot_margin
        self.tote_path = tote_path
        super().__init__(
            obj_instance_configs=[],
            dynamic_mujoco_model=True,
            **kwargs,
        )

    def setup_objs(self, world_model: RootElement) -> QPosRange:
        obj_qpos_ranges = super().setup_objs(world_model=world_model)
        tote_asset = mjcf.from_path(self.tote_path)
        center_x = self.table_x_pos - self.bin_x_offset
        y_dist = self.setup_numpy_random.uniform(
            low=self.min_bin_y_dist,
            high=self.max_bin_y_dist,
        )
        z_rot = self.setup_numpy_random.uniform(
            low=-self.bin_z_rot_margin,
            high=self.bin_z_rot_margin,
        )
        self.add_obj_from_model(
            obj_model=self.rename_model(
                tote_asset,
                name="left_bin",
            ),
            world_model=world_model,
            position=(
                center_x,
                y_dist,
                0.05,
            ),
            euler=(0, 0, z_rot),
        )
        y_dist = self.setup_numpy_random.uniform(
            low=self.min_bin_y_dist,
            high=self.max_bin_y_dist,
        )
        z_rot = self.setup_numpy_random.uniform(
            low=-self.bin_z_rot_margin,
            high=self.bin_z_rot_margin,
        )
        self.add_obj_from_model(
            obj_model=self.rename_model(
                tote_asset,
                name="right_bin",
            ),
            world_model=world_model,
            position=(
                center_x,
                -y_dist,
                0.05,
            ),
            euler=(0, 0, z_rot),
        )

        # load objects
        asset_paths = self.sample_objects()
        assert len(asset_paths) > 0
        self.assert_visible_objs_at_reset = set()
        for asset_path in asset_paths:
            obj_model = mjcf.from_path(asset_path)
            obj_model = self.rename_model(model=obj_model, name="toy")
            obj_body = self.add_obj_from_model(
                obj_model=obj_model,
                world_model=world_model,
                add_free_joint=True,
            )
            for geom_element in obj_body.find_all("geom"):
                geom_element.condim = "4"
                geom_element.friction = "0.9 0.8"
            obj_qpos_ranges.extend(
                DegreeOfFreedomRange(lower=lower, upper=upper)
                for lower, upper in [
                    # 3D position
                    (
                        center_x - self.obj_x_margin,
                        center_x + self.obj_x_margin,
                    ),
                    (-y_dist - self.obj_y_margin, -y_dist + self.obj_y_margin),
                    (0.2, 0.3),
                    # euler rotation
                    (-np.pi, np.pi),
                    (-np.pi, np.pi),
                    (-np.pi, np.pi),
                ]
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

    def sample_objects(self):
        return self.setup_numpy_random.choice(
            self.asset_paths,
            size=1,
            replace=False,
        )

    def is_reset_state_valid(self, obs: Observation) -> bool:
        right_bin_path = "right_bin/|right_bin/right_bin"
        return super().is_reset_state_valid(obs=obs) and check_on_top_of(
            state=obs.state,
            on_top_obj_name="toy/|toy/toy",
            on_bottom_obj_name=right_bin_path,
            context_name_to_link_path={
                "toy/|toy/toy": "toy/|toy/toy",
                right_bin_path: right_bin_path,
            },
            threshold=0.5,
        )
