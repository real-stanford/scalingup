from typing import Dict, Optional, cast
import torch
from scalingup.algo.end_effector_policy_utils import Discretizer
from scalingup.utils.core import Action, EndEffectorAction, Observation, Task, Trajectory
from scalingup.utils.core import Policy
from pathlib import Path
import pickle
from rich.progress import track
import numpy as np
from transforms3d import euler


class BufferReplayPolicy(Policy):
    def __init__(self, rootdir: str, discretizer: Optional[Discretizer]):
        self.rootdir = rootdir
        # from episode id to trajectory path
        self.index: Dict[int, str] = {}
        for path in track(
            list(map(str, sorted(Path(rootdir).rglob("0*.pkl")))),
            description=f"Scanning {self.rootdir}",
        ):
            traj = Trajectory.load(path)
            assert traj.episode_id not in self.index
            self.index[traj.episode_id] = path
        self.discretizer = discretizer

    def _get_action(
        self,
        obs: Observation,
        task: Task,
    ) -> Action:
        raise NotImplementedError(
            "BufferReplayPolicy has not been updated to "
            + "timestep API from episode step API"
        )
        traj = Trajectory.load(self.index[obs.episode_id])
        if len(traj.flatten()) == 0:
            return traj.episode[0].action
        traj = traj.flatten()
        action = traj.episode[min(obs.episode_step, len(traj) - 1)].action
        assert type(action) == EndEffectorAction
        ee_action = cast(EndEffectorAction, action)
        if self.discretizer is not None:
            reconstructed_pos = (
                self.discretizer.undiscretize_pos(
                    pos_onehot=self.discretizer.discretize_pos(
                        pos=torch.tensor(ee_action.end_effector_position)[None,]
                    )
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            q = ee_action.end_effector_orientation
            euler_degrees = np.array(euler.quat2euler(q)) * 180.0 / np.pi
            reconstructed_rot_degrees = (
                self.discretizer.undiscretize_rot(
                    rot_scores=self.discretizer.discretize_rot(
                        rot=torch.tensor(euler_degrees)[None,]
                    )
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            reconstructed_rot_radians = reconstructed_rot_degrees * np.pi / 180
            ee_action = EndEffectorAction(
                end_effector_position=reconstructed_pos,
                end_effector_orientation=euler.euler2quat(*reconstructed_rot_radians),
                allow_contact=ee_action.allow_contact,
                gripper_command=ee_action.gripper_command,
            )
        return ee_action
