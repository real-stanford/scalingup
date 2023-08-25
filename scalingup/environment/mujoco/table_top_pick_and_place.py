from typing import List
import numpy as np
from dm_control import mjcf
from dm_control.mjcf import RootElement
from scalingup.environment.mujoco.mujocoEnv import (
    # DynamicMujocoUR5Env,
    MujocoUR5EnvFromObjConfigList,
    MujocoFR5EnvFromObjConfigList,
    MujocoUR5Robotiq85fEnvFromObjConfigList,
    MujocoFR5Robotiq85fEnvFromObjConfigList
)
from scalingup.environment.mujoco.table_top import (
    TableTopMujocoEnv,
    TableTopFR5MujocoEnv,
    TableTopRobotiq85MujocoEnv, 
    TableTopFR5Robotiq85MujocoEnv
)
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

class TableTopPickAndPlaceFR5(TableTopFR5MujocoEnv, MujocoFR5EnvFromObjConfigList):
    pass

class TableTopPickAndPlaceRobotiq85(TableTopRobotiq85MujocoEnv, MujocoUR5Robotiq85fEnvFromObjConfigList):
    pass

class TableTopPickAndPlaceFR5Robotiq85(TableTopFR5Robotiq85MujocoEnv, MujocoFR5Robotiq85fEnvFromObjConfigList):
    pass