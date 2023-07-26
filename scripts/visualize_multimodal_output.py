from __future__ import annotations

import logging
import typing
from typing import Optional
import open3d as o3d
import hydra
import numpy as np
import torch
from scalingup.algo.diffusion import DiffusionPolicy
from scalingup.data.window_dataset import StateSequenceTensor
from scalingup.environment.mujoco.mujocoEnv import MujocoEnv
from scalingup.utils.generic import setup_logger
from scalingup.utils.core import (
    ObservationWithHistory,
    PartialObservation,
    Task,
)
import matplotlib as mpl


def run_policy(
    obs: ObservationWithHistory,
    policy: DiffusionPolicy,
    task: Task,
    output_path: str,
    divisions: int = 20,
    seeds: int = 64,
):
    actions = []
    assert type(obs) == ObservationWithHistory
    obs_with_history = typing.cast(ObservationWithHistory, obs)
    state_sequence: StateSequenceTensor = StateSequenceTensor.from_obs_sequence(
        control_sequence=[
            PartialObservation.from_obs(obs) for obs in obs_with_history.sequence
        ],
        pad_left_count=max(
            policy.rollout_config.proprio_obs_horizon - len(obs_with_history.sequence),
            0,
        ),
        pad_right_count=0,
        pos_bounds=policy.scene_bounds,
        num_obs_pts=policy.num_obs_pts,
        numpy_random=policy.numpy_random,
        obs_cameras=policy.obs_cameras,
    )
    # make sure it is batched
    if not state_sequence.is_batched:
        state_sequence = StateSequenceTensor.collate([state_sequence])

    for seed in range(seeds):
        policy.set_seed(seed)
        action_pred = (
            policy.diffuse(
                state_sequence=state_sequence.to(
                    device=policy.device, dtype=policy.dtype
                ),
                task_names=[task.desc],
                task_metric=torch.tensor(
                    [[policy.towards_token]], device=policy.device, dtype=policy.dtype
                ),
            )
            .detach()
            .squeeze()
            .cpu()
            .numpy()
        )
        actions.append(action_pred)
    xyz_pts = []
    rgb_pts = []
    for action in actions:
        action_xyz_pts = []
        # densely interpolate
        prev_pt = action[0, :3]
        for pt in action[1:, :3]:
            for i in np.arange(0, 1, 1 / divisions):
                action_xyz_pts.append(prev_pt + i * (pt - prev_pt))
            prev_pt = pt
        xyz_pts.extend(action_xyz_pts)
        rgb_pts.extend(mpl.colormaps["jet"](np.linspace(0, 1, len(action_xyz_pts))))
    # write ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(xyz_pts).astype(float))
    pcd.colors = o3d.utility.Vector3dVector(np.array(rgb_pts)[:, :3])
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
    logging.info(f"Saved to {output_path}")


def do_task(
    evaluation: str,
    checkpoint_path: str,
    init_qpos: Optional[np.ndarray] = None,
    init_ctrl: Optional[np.ndarray] = None,
    task_idx: int = 0,
):
    # setup env
    # setup policy
    with hydra.initialize(
        config_path="../scalingup/config",
        version_base="1.2",
    ):
        conf = hydra.compose(
            config_name="inference",
            overrides=[
                f"evaluation={evaluation}",
                "policy=diffusion",
                f"policy.path={checkpoint_path}",
            ],
        )
        setup_logger()
        env: MujocoEnv = hydra.utils.instantiate(conf.evaluation.env)
        obs = env.reset(episode_id=0)
        if init_qpos is not None:
            env.mj_physics.data.qpos[:] = init_qpos
        if init_ctrl is not None:
            env.mj_physics.data.ctrl[:] = init_ctrl
        env.mj_physics.forward()
        obs = ObservationWithHistory.from_sequence([env.get_obs()])
        task = hydra.utils.instantiate(conf.evaluation.task_sampler.tasks)[task_idx]
        policy = hydra.utils.instantiate(conf.policy, remote=False)
    output_path = f"{evaluation}_{task.desc}_actions.ply"
    run_policy(
        obs=obs,
        output_path=output_path,
        policy=policy,
        task=task,
    )


def generate_mailbox_vis(ckpt_path: str):
    init_qpos = np.array(
        [
            -2.82735,
            -1.94587,
            2.23741,
            -1.75831,
            -0.94307,
            -1.13094,
            0.0268723,
            0.0273854,
            2.00001,
            -0.000850527,
            0.417921,
            -0.170588,
            0.153297,
            -0.0403438,
            0.165815,
            -0.97075,
            -0.168887,
        ]
    )
    do_task(
        evaluation="mailbox",
        checkpoint_path=ckpt_path,
        init_qpos=init_qpos,
    )


def generate_transport(ckpt_path: str):
    do_task(evaluation="bin_transport_test", checkpoint_path=ckpt_path)


def generate_drawer(ckpt_path: str):
    approaching_drawer_qpos = np.array(
        [
            -1.75924,
            -1.50084,
            1.45149,
            -0.502712,
            -0.754404,
            -0.942423,
            0.0542528,
            0.0541143,
            0,
            0,
            0,
            0.520489,
            -0.471634,
            0.0497909,
            0.999999,
            6.37268e-05,
            -0.00170076,
            -6.92322e-06,
            0.504329,
            -0.241494,
            0.0485881,
            0.999966,
            -0.000182925,
            0.00821377,
            -3.90489e-06,
            0.474715,
            -0.125312,
            0.049633,
            0.999992,
            -0.000282486,
            -0.00409593,
            -1.79568e-06,
            0.545909,
            0.112142,
            0.0491327,
            0.999987,
            0.00113812,
            0.00504238,
            -0.000107965,
        ]
    )
    # top middle bottom
    for i in range(3):
        do_task(
            evaluation="drawer",
            checkpoint_path=ckpt_path,
            task_idx=i,
            init_qpos=approaching_drawer_qpos,
        )
    # different objects, start off with opened middle drawer
    opened_middle_drawer_qpos = np.array(
        [
            -2.01056,
            -2.00896,
            1.7981,
            -1.50668,
            -1.63354,
            -0.734214,
            0.0542112,
            0.054181,
            0,
            0.239732,
            0,
            0.520489,
            -0.471634,
            0.0497909,
            0.999999,
            6.37268e-05,
            -0.00170076,
            -6.92406e-06,
            0.504329,
            -0.241494,
            0.0485881,
            0.999966,
            -0.000187539,
            0.00821341,
            -3.94692e-06,
            0.474715,
            -0.125312,
            0.049633,
            0.999992,
            -0.000282503,
            -0.00409636,
            -1.61836e-06,
            0.545909,
            0.112142,
            0.0491327,
            0.999987,
            0.00113812,
            0.00504238,
            -0.000107969,
        ]
    )
    for i in [1, 4, 7, 10]:
        do_task(
            evaluation="drawer",
            checkpoint_path=ckpt_path,
            task_idx=i,
            init_qpos=opened_middle_drawer_qpos,
        )


def generate_catapult(ckpt_path: str):
    with hydra.initialize(
        config_path="../scalingup/config",
        version_base="1.2",
    ):
        conf = hydra.compose(
            config_name="inference",
            overrides=[
                f"evaluation=catapult",
                "policy=diffusion",
                f"policy.path={ckpt_path}",
            ],
        )
        setup_logger()
        env: MujocoEnv = hydra.utils.instantiate(conf.evaluation.env)
        obs = env.reset(episode_id=0)
        tasks = hydra.utils.instantiate(conf.evaluation.task_sampler.tasks)
        policy: DiffusionPolicy = hydra.utils.instantiate(conf.policy, remote=False)
        sampler_config = hydra.utils.instantiate(conf.evaluation.sampler_config)
    done = False
    prev_obs = [obs]
    while not done:
        seed = 0
        if obs.time < 11.25:
            action = policy(
                obs=ObservationWithHistory.from_sequence(prev_obs),
                task=tasks[0],
                seed=seed,
            )
            obs, done, _ = env.step(
                action=action, active_task=tasks[0], sampler_config=sampler_config
            )
            prev_obs.append(obs)
            prev_obs = prev_obs[-2:]
            continue
        for task in tasks:
            run_policy(
                obs=ObservationWithHistory.from_sequence(prev_obs),
                output_path=f"catapult_{task.desc}_actions.ply",
                policy=policy,
                task=task,
            )
        exit()


if __name__ == "__main__":
    generate_catapult(ckpt_path="/path/to/your/checkpoint.ckpt")
