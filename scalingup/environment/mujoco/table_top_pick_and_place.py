from typing import List
import numpy as np
from dm_control import mjcf
from dm_control.mjcf import RootElement
from scalingup.environment.mujoco.mujocoEnv import (
    DynamicMujocoUR5Env,
    MujocoUR5EnvFromObjConfigList,
)
from scalingup.environment.mujoco.table_top import TableTopMujocoEnv
from scalingup.environment.mujoco.utils import MujocoObjectInstanceConfig
from scalingup.utils.core import (
    DegreeOfFreedomRange,
    EnvState,
    Observation,
    QPosRange,
    Task,
    Trajectory,
)
from scalingup.utils.state_api import check_on_top_of


class TableTopPickAndPlace(TableTopMujocoEnv, MujocoUR5EnvFromObjConfigList):
    pass
