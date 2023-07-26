import numpy as np

from scalingup.environment.mujoco.mujocoEnv import MujocoUR5EnvFromObjConfigList
from scalingup.environment.mujoco.table_top import TableTopMujocoEnv
from scalingup.environment.mujoco.utils import MujocoObjectInstanceConfig
from scalingup.utils.core import DegreeOfFreedomRange


class DrawerMujocoEnv(TableTopMujocoEnv):
    def __init__(
        self,
        pose_randomization: bool = False,
        position_randomization: bool = False,
        table_length: float = 1.165,
        table_width: float = 0.58,
        table_x_pos: float = 0.4,
        drawer_xml_path: str = "scalingup/environment/mujoco/assets/custom/small_drawer.xml",
        **kwargs,
    ):
        configs = [
            MujocoObjectInstanceConfig(
                obj_class="drawer",
                asset_path=drawer_xml_path,
                qpos_range=[
                    DegreeOfFreedomRange(upper=0.0, lower=0.0),
                    DegreeOfFreedomRange(upper=0.0, lower=0.0),
                    DegreeOfFreedomRange(upper=0.0, lower=0.0),
                ],
                position=(0.0, 0.7, 0.05),
                euler=(0, 0, -np.pi / 5),
            ),
        ]
        assets = [
            (
                "scalingup/environment/mujoco/assets/google_scanned_objects/household_items/Womens_Multi_13/model.xml",
                "vitamin bottle",
            ),
            (
                "scalingup/environment/mujoco/assets/google_scanned_objects/household_items/Wishbone_Pencil_Case/model.xml",
                "pencil case",
            ),
            (
                "scalingup/environment/mujoco/assets/google_scanned_objects/household_items/Crayola_Crayons_24_count/model.xml",
                "crayon box",
            ),
            (
                "scalingup/environment/mujoco/assets/google_scanned_objects/toys/Breyer_Horse_Of_The_Year_2015/model.xml",
                "horse toy",
            ),
        ]
        assert len(assets) > 1
        start_y_pos = table_width * 0.2  # away from the drawer
        end_y_pos = -table_width * 0.8
        for i, (asset_path, asset_class) in enumerate(assets):
            normalized_idx = i / (len(assets) - 1)
            if pose_randomization:
                configs.append(
                    MujocoObjectInstanceConfig(
                        obj_class=asset_class,
                        asset_path=asset_path,
                        qpos_range=[
                            DegreeOfFreedomRange(
                                upper=table_x_pos + table_width / 3,
                                lower=table_x_pos
                                - table_width / 5,  # not too close to robot base
                            ),
                            DegreeOfFreedomRange(
                                upper=start_y_pos,
                                lower=end_y_pos,
                            ),
                            DegreeOfFreedomRange(
                                upper=(i + 1) * 0.05 + 0.2, lower=i * 0.05 + 0.2
                            ),
                            DegreeOfFreedomRange(upper=np.pi, lower=-np.pi),
                            DegreeOfFreedomRange(upper=np.pi, lower=-np.pi),
                            DegreeOfFreedomRange(upper=np.pi, lower=-np.pi),
                        ],
                        add_free_joint=True,
                    ),
                )
            elif position_randomization:
                configs.append(
                    MujocoObjectInstanceConfig(
                        obj_class=asset_class,
                        asset_path=asset_path,
                        qpos_range=[
                            DegreeOfFreedomRange(
                                upper=table_x_pos + table_width / 5 + 0.05,
                                lower=table_x_pos + table_width / 5 - 0.05,
                            ),
                            DegreeOfFreedomRange(
                                upper=end_y_pos
                                + (start_y_pos - end_y_pos) * normalized_idx
                                + 0.05,
                                lower=end_y_pos
                                + (start_y_pos - end_y_pos) * normalized_idx
                                - 0.05,
                            ),
                            DegreeOfFreedomRange(upper=0.05, lower=0.05),
                            DegreeOfFreedomRange(upper=0, lower=0),
                            DegreeOfFreedomRange(upper=0, lower=0),
                            DegreeOfFreedomRange(upper=0, lower=-0),
                        ],
                        add_free_joint=True,
                    ),
                )
            else:
                configs.append(
                    MujocoObjectInstanceConfig(
                        obj_class=asset_class,
                        asset_path=asset_path,
                        qpos_range=[
                            DegreeOfFreedomRange(
                                upper=table_x_pos + table_width / 5,
                                lower=table_x_pos + table_width / 5,
                            ),
                            DegreeOfFreedomRange(
                                upper=end_y_pos
                                + (start_y_pos - end_y_pos) * normalized_idx,
                                lower=end_y_pos
                                + (start_y_pos - end_y_pos) * normalized_idx,
                            ),
                            DegreeOfFreedomRange(upper=0.05, lower=0.05),
                            DegreeOfFreedomRange(upper=0, lower=0),
                            DegreeOfFreedomRange(upper=0, lower=0),
                            DegreeOfFreedomRange(upper=0, lower=-0),
                        ],
                        add_free_joint=True,
                    ),
                )
        super().__init__(
            obj_instance_configs=configs,
            **kwargs,
        )
