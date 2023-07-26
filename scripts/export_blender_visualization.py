from __future__ import annotations
import logging
import os
import pickle
import typing
from typing import Any, Dict, List
import hydra
import numpy as np
import torch
from scalingup.algo.diffusion import DiffusionPolicy
from scalingup.data.window_dataset import (
    StateSequenceTensor,
)
from scalingup.environment.mujoco.mujocoEnv import MujocoEnv
from scalingup.utils.generic import setup_logger
from scalingup.utils.core import (
    Env,
    EnvSampler,
    EnvSamplerConfig,
    ObservationWithHistory,
    PartialObservation,
)
import hashlib
from transforms3d import euler
from rich.progress import track


def do_task(
    evaluation: str,
    checkpoint_path: str,
    output_path: str,
    episode_id: int = 0,
    num_seeds: int = 32,
    fps: int = 6,
):
    """
    evaluation: which domain to evaluate on (drawer, mailbox, table_top_bus_balance, etc.)
    checkpoint_path: path to checkpoint
    output_path: path to save output
    episode_id: episode to evaluate
    num_seeds: number of action sequences to predict per observation
    fps: frames per second to export the simulation and policy predictions
    6 is a good number for all domains except for catapult, which has the dynamic
    catapult. For that, use 12 or 24 fps.
    """
    setup_logger()
    # sha256 hash of input_params
    input_params = f"{evaluation}_{checkpoint_path}_{episode_id}"
    input_hash = hashlib.sha256(input_params.encode()).hexdigest()

    eval_id = f"{evaluation}_{input_hash[:6]}"
    output_path = os.path.join(output_path, eval_id)
    # disable logger
    logging.info(f"Saving to {output_path}, {episode_id}")
    # disable logger
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    dense_obs_path = os.path.join(output_path, "dense_obs.pkl")
    anim_data_path = os.path.join(output_path, "anim_data.pkl")
    trajectory_path = os.path.join(output_path, "trajectory.pkl")
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
                "policy.action_horizon=8",
            ],
        )
        env: MujocoEnv = hydra.utils.instantiate(conf.evaluation.env)
        obs = env.reset(episode_id=episode_id)
        body_to_name = {
            bodyid: env.mj_physics.model.body(bodyid).name
            for bodyid in range(env.mj_physics.model.nbody)
        }
        policy: DiffusionPolicy = hydra.utils.instantiate(conf.policy, remote=False)
        if not (
            os.path.exists(dense_obs_path)
            and os.path.exists(anim_data_path)
            and os.path.exists(trajectory_path)
        ):
            # setup policy
            task_sampler = hydra.utils.instantiate(conf.evaluation.task_sampler)
            sampler_config: EnvSamplerConfig = hydra.utils.instantiate(
                conf.evaluation.sampler_config
            )
            sampler = EnvSampler(env=env, task_sampler=task_sampler)
            # register callbacks
            dense_obs = []

            def dense_obs_callbacks(env: Env):
                dense_obs.append((env.time, env.get_obs()))

            env.step_fn_callbacks["dense_obs_callback"] = (fps, dense_obs_callbacks)
            anim_data: Dict[str, List[Any]] = {
                body_to_name[bodyid]: [] for bodyid in range(env.mj_physics.model.nbody)
            }

            logging.info(body_to_name)

            def anim_pose_callback(env: Env):
                mj_env = typing.cast(MujocoEnv, env)
                for bodyid in range(mj_env.mj_physics.model.nbody):
                    pos = mj_env.mj_physics.data.xpos[bodyid].copy()
                    quat = mj_env.mj_physics.data.xquat[bodyid].copy()

                    anim_data[body_to_name[bodyid]].append(
                        (mj_env.time, pos, euler.quat2euler(quat))
                    )

            env.step_fn_callbacks["anim_pose_callback"] = (fps, anim_pose_callback)
            env_sample = sampler.sample(
                policy=policy,
                episode_id=episode_id,
                return_trajectory=True,
                config=sampler_config,
            )
            assert env_sample.trajectory is not None
            # save
            pickle.dump(
                dense_obs,
                open(os.path.join(output_path, "dense_obs.pkl"), "wb"),
            )
            pickle.dump(
                anim_data,
                open(os.path.join(output_path, "anim_data.pkl"), "wb"),
            )
            trajectory = env_sample.trajectory

            pickle.dump(
                trajectory,
                open(os.path.join(output_path, "trajectory.pkl"), "wb"),
            )
        else:
            dense_obs = pickle.load(open(dense_obs_path, "rb"))
            anim_data = pickle.load(open(anim_data_path, "rb"))
            trajectory = pickle.load(open(trajectory_path, "rb"))

    trajectory.dump_video(
        os.path.join(output_path, "video.mp4"),
    )
    if not trajectory.is_successful:
        logging.warning("Trajectory is not successful")
        # dump video
        return
    action_path = os.path.join(output_path, "timestamped_actions.pkl")
    if not os.path.exists(action_path):
        # rerun all actions with traces
        timestamps = np.array([timestamp for timestamp, _ in dense_obs])
        timestamps = np.round(timestamps * fps) / fps
        timestamped_obs = {
            timestamp: obs for timestamp, (_, obs) in zip(timestamps, dense_obs)
        }
        timestamped_actions: Dict[float, List[np.ndarray]] = {
            timestamp: [] for timestamp in timestamps
        }
        rollout_config = policy.rollout_config
        assert rollout_config.proprio_obs_horizon == 2
        assert rollout_config.vision_obs_horizon == 1
        ctrl_config = policy.ctrl_config

        for timestamp, (_, obs) in track(zip(timestamps, dense_obs)):
            # find prev obs
            obs_sequence = [obs, obs]
            prev_obs_timestamp = timestamp - 1 / ctrl_config.frequency
            prev_obs_timestamp = np.round(prev_obs_timestamp * fps) / fps
            if prev_obs_timestamp in timestamped_obs:
                obs_sequence[0] = timestamped_obs[prev_obs_timestamp]
            obs_with_history = ObservationWithHistory.from_sequence(obs_sequence)
            state_sequence: StateSequenceTensor = StateSequenceTensor.from_obs_sequence(
                control_sequence=[
                    PartialObservation.from_obs(obs) for obs in obs_with_history.sequence
                ],
                pad_left_count=max(
                    policy.rollout_config.proprio_obs_horizon
                    - len(obs_with_history.sequence),
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

            for seed in range(num_seeds):
                policy.set_seed(seed)
                action_pred = (
                    policy.diffuse(
                        state_sequence=state_sequence.to(
                            device=policy.device, dtype=policy.dtype
                        ),
                        task_names=[trajectory.task.desc],
                        task_metric=torch.tensor(
                            [[policy.towards_token]],
                            device=policy.device,
                            dtype=policy.dtype,
                        ),
                    )
                    .detach()
                    .squeeze()
                    .cpu()
                    .numpy()
                )
                timestamped_actions[timestamp].append(action_pred)

            pickle.dump(
                timestamped_actions,
                open(action_path, "wb"),
            )
        #  save action
        pickle.dump(
            timestamped_actions,
            open(action_path, "wb"),
        )


if __name__ == "__main__":
    for i in range(10):
        do_task(
            evaluation="catapult",
            checkpoint_path="path/to/your/checkpoint",
            output_path="catapult_traces/",
            episode_id=i,
            fps=24,
        )
