import logging
from typing import Dict, FrozenSet, Callable, Optional
from dm_control import mjcf
from dm_control.mjcf import RootElement
from scalingup.environment.mujoco.mujocoEnv import (
    MujocoUR5EnvFromObjConfigList,
    MujocoUR5WSG50FinrayEnv,
)
from scalingup.utils.constants import MJCF_NEST_TOKEN
from scalingup.utils.core import (
    DegreeOfFreedomRange,
    EnvState,
    Observation,
    QPosRange,
)
from scalingup.utils.state_api import check_on_top_of


class TableTopMujocoEnv(MujocoUR5EnvFromObjConfigList):
    def __init__(
        self,
        table_asset_path: str = "scalingup/environment/mujoco/assets/custom/table.xml",
        table_length: float = 1.165,
        table_width: float = 0.58,
        table_x_pos: float = 0.4,
        end_episode_on_drop: bool = True,
        early_stop_condition: Optional[Callable[[EnvState], bool]] = None,
        **kwargs,
    ):
        self.table_asset_path = table_asset_path
        self.table_length = table_length
        self.table_width = table_width
        self.table_x_pos = table_x_pos
        self.end_episode_on_drop = end_episode_on_drop
        self.early_stop_condition = early_stop_condition
        super().__init__(**kwargs)

    def setup_objs(self, world_model: RootElement) -> QPosRange:
        obj_qpos_ranges = super().setup_objs(world_model=world_model)
        table_asset = mjcf.from_path(self.table_asset_path)
        self.add_obj_from_model(
            obj_model=table_asset,
            world_model=world_model,
            position=(
                self.table_x_pos,
                0.0,
                0.0,
            ),
        )
        return obj_qpos_ranges

    @staticmethod
    def get_are_objs_on_ground(
        state: EnvState,
        non_obj_keywords: FrozenSet[str] = frozenset({"_area", "table", "world", "ur5e"}),
        verbose: bool = False,
    ) -> Dict[str, bool]:
        objects = {
            obj_name: obj_state
            for obj_name, obj_state in state.object_states.items()
            if all(kw not in obj_name for kw in non_obj_keywords)
        }
        is_obj_on_ground = {}
        for obj_name in objects.keys():
            name = obj_name.split(MJCF_NEST_TOKEN)[0]
            is_obj_on_ground[name] = check_on_top_of(
                state=state,
                on_top_obj_name=obj_name,
                on_bottom_obj_name="ground",
                context_name_to_link_path={
                    "ground": "world",
                    obj_name: f"{name}/|{name}/{name}",
                },
            )
            if verbose and is_obj_on_ground[name]:
                logging.info(f"{obj_name} is on the ground")
        return is_obj_on_ground

    def get_state(self) -> EnvState:
        state = super().get_state()
        if self.end_episode_on_drop and any(
            TableTopMujocoEnv.get_are_objs_on_ground(state=state).values()
        ):
            self.done = True
        if self.early_stop_condition is not None and self.early_stop_condition(state):
            self.done = True
        return state

    def is_reset_state_valid(self, obs: Observation) -> bool:
        return super().is_reset_state_valid(obs=obs) and not any(
            TableTopMujocoEnv.get_are_objs_on_ground(state=obs.state).values()
        )
