from __future__ import annotations

import logging
import os
import pickle
import typing
from copy import deepcopy
from time import time
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Union
import dataclasses

import hydra
import numpy as np
import ray
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler, DDIMSchedulerOutput
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput
from diffusers.training_utils import EMAModel

from scalingup.algo.algo import TrainableScalingUpAlgo
from scalingup.algo.diffusion_utils import ConditionalUnet1D
from scalingup.algo.state_encoder.state_sequence_encoder import StateSequenceEncoder
from scalingup.algo.state_encoder.vision_encoder import (
    PointCloudEncoder,
    VolumeEncoder,
)
from scalingup.algo.utils import replace_bn_with_gn
from scalingup.data.dataset import TensorDataClass
from scalingup.data.window_dataset import (
    NormalizationConfig,
    PolicyWindowRolloutConfig,
    StateSequenceTensor,
    TrajectoryWindowDataset,
    TrajectoryWindowTensor,
)
from scalingup.policy.scalingup import ScalingUpDataGen
from scalingup.utils.core import (
    ControlAction,
    ControlConfig,
    Observation,
    ObservationWithHistory,
    PartialObservation,
    Policy,
    RayPolicy,
    Task,
    split_state_phrase,
)
from scalingup.utils.text_encoders import ClipLanguageConditioned

NoiseScheduler = Union[DDPMScheduler, DDIMScheduler]
NoiseSchedulerOutput = Union[DDPMSchedulerOutput, DDIMSchedulerOutput]
CORRUPT_SUCCESS_TOKEN = -1.0


class DiffusionPolicy(Policy):
    def __init__(
        self,
        state_sequence_encoder: StateSequenceEncoder,
        noise_pred_net: ConditionalUnet1D,
        device: torch.device,
        dtype: torch.dtype,
        action_dim: int,
        should_condition_on_task_metric: bool,
        rollout_config: PolicyWindowRolloutConfig,
        noise_scheduler: NoiseScheduler,
        action_norm_config: NormalizationConfig,
        ctrl_config: ControlConfig,
        supported_tasks: FrozenSet[str],
        obs_cameras: List[str],
        # TODO: multiple arguments related to the same feature, should aggregate
        # into data class or a separate model
        task_metric_guidance: float,
        suppress_token: float,  # 0 for failure, -1 for unconditional
        towards_token: float,  # you want successful behavior (likely 1 or close to 1)
        task_metric_dim: int,
        **kwargs,
    ):
        Policy.__init__(self, supported_tasks=supported_tasks)
        self.state_sequence_encoder = state_sequence_encoder.eval()
        self.noise_pred_net = noise_pred_net.eval()
        self.device = device
        self.dtype = dtype
        self.rollout_config = rollout_config
        self.noise_scheduler = noise_scheduler
        self.action_dim = action_dim
        self.should_condition_on_task_metric = should_condition_on_task_metric
        self.task_metric_dim = task_metric_dim
        self.num_obs_pts = (
            state_sequence_encoder.vision_encoder.num_obs_pts
            if state_sequence_encoder.vision_encoder is not None
            and issubclass(type(state_sequence_encoder.vision_encoder), PointCloudEncoder)
            else None
        )
        self.task_metric_guidance = task_metric_guidance
        self.suppress_token = suppress_token
        self.towards_token = towards_token
        self.action_norm_config = action_norm_config
        self.ctrl_config = ctrl_config
        self.scene_bounds = (
            self.state_sequence_encoder.vision_encoder.discretizer.scene_bounds
            if type(self.state_sequence_encoder.vision_encoder) == VolumeEncoder
            else None
        )
        self.obs_cameras = obs_cameras

    def is_task_supported(self, task_desc: str) -> bool:
        task_desc = (
            task_desc
            if not self.state_sequence_encoder.remove_with_statement
            else split_state_phrase(task_desc)[1]
        )
        return super().is_task_supported(task_desc)

    def diffuse(
        self,
        state_sequence: StateSequenceTensor,
        task_names: List[str],
        task_metric: torch.Tensor,  # success, reward, etc.
    ) -> torch.Tensor:
        """_summary_

        Args:
            state_sequence (StateSequenceTensor): _description_
            task_names (List[str]): _description_

        Returns:
            torch.Tensor: **unnormalized** action
        """
        # task_metric: (B, 1)
        # repeat task_metric
        task_metric = task_metric.repeat(1, self.task_metric_dim)
        # task_metric: (B, task_metric_dim)
        with torch.no_grad():
            state_feature = self.state_sequence_encoder(
                state_sequence=state_sequence, task_names=task_names
            )
            batch_size = state_sequence.batch_size
            # (B, obs_horizon * obs_dim)

            generator = self.get_torch_random_generator(device=self.device)
            # initialize action from Gaussian noise
            normalized_action: torch.Tensor = torch.randn(
                (
                    batch_size,  # batch size
                    self.rollout_config.prediction_horizon,
                    self.action_dim,
                ),  # type: ignore
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )

            suppress_state_feature = None
            if self.task_metric_guidance > 0.0:
                suppress_state_feature = torch.cat(
                    (
                        state_feature,
                        torch.full_like(
                            input=task_metric, fill_value=self.suppress_token
                        ),
                    ),
                    dim=-1,
                )
            if self.should_condition_on_task_metric:
                state_feature = torch.cat((state_feature, task_metric), dim=-1)

            # init scheduler
            self.noise_scheduler.set_timesteps(
                len(self.noise_scheduler.timesteps), self.device
            )

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=normalized_action,
                    timestep=k,
                    global_cond=state_feature,
                )
                if suppress_state_feature is not None:
                    suppress_noise_pred = self.noise_pred_net(
                        sample=normalized_action,
                        timestep=k,
                        global_cond=suppress_state_feature,
                    )
                    noise_pred = (
                        1.0 + self.task_metric_guidance
                    ) * noise_pred - self.task_metric_guidance * suppress_noise_pred

                # inverse diffusion step (remove noise)
                normalized_action = typing.cast(
                    NoiseSchedulerOutput,
                    self.noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=int(k),
                        sample=normalized_action,
                        generator=generator,
                    ),
                ).prev_sample
        return self.action_norm_config.unnormalize(normalized_action)

    def _get_action(self, obs: Observation, task: Task) -> ControlAction:
        assert type(obs) == ObservationWithHistory
        obs_with_history = typing.cast(ObservationWithHistory, obs)
        state_sequence: StateSequenceTensor = StateSequenceTensor.from_obs_sequence(
            control_sequence=[
                PartialObservation.from_obs(obs) for obs in obs_with_history.sequence
            ],
            pad_left_count=max(
                self.rollout_config.proprio_obs_horizon - len(obs_with_history.sequence),
                0,
            ),
            pad_right_count=0,
            pos_bounds=self.scene_bounds,
            num_obs_pts=self.num_obs_pts,
            numpy_random=self.numpy_random,
            obs_cameras=self.obs_cameras,
        )
        # make sure it is batched
        if not state_sequence.is_batched:
            state_sequence = StateSequenceTensor.collate([state_sequence])

        action_pred = self.diffuse(
            state_sequence=state_sequence.to(device=self.device, dtype=self.dtype),
            task_names=[task.desc],
            task_metric=torch.tensor(
                [[self.towards_token]], device=self.device, dtype=self.dtype
            ),
        )
        action_pred = (
            action_pred.detach().to("cpu").numpy()[0]
        )  # (pred_horizon, action_dim)
        # only take action_horizon number of actions
        start = self.rollout_config.proprio_obs_horizon - 1
        end = start + self.rollout_config.action_horizon
        action = action_pred[start:end, :]
        return ControlAction(
            value=action,
            config=self.ctrl_config,
            timestamps=np.linspace(
                obs.time, obs.time + len(action) / self.ctrl_config.frequency, len(action)
            )
            + 1 / self.ctrl_config.frequency,
            use_early_stop_checker=False,
        )

    @classmethod
    def from_path(
        cls,
        path: str,
        inference_device: Union[str, torch.device],
        **kwargs,
    ) -> Policy:
        conf = pickle.load(
            open(
                os.path.dirname(os.path.dirname(path)) + "/conf.pkl",
                "rb",
            )
        )
        conf.algo.replay_buffer.should_reindex = False
        algo_kwargs = {
            k: (hydra.utils.instantiate(v) if "Config" in type(v).__name__ else v)
            for k, v in conf.algo.items()
            if not (k.endswith("_") and k.startswith("_"))
        }
        algo_kwargs["device"] = "cpu"
        algo = DiffusionScalingUpAlgo.load_from_checkpoint(
            **algo_kwargs, strict=False, checkpoint_path=path, map_location="cpu"
        )
        policy = deepcopy(
            algo.get_policy(
                inference_device=inference_device,
                **kwargs,
            )
        )
        logging.info(f"loaded from checkpoint: {path!r}")
        return policy


class DiffusionScalingUpAlgo(TrainableScalingUpAlgo):
    def __init__(
        self,
        state_sequence_encoder: StateSequenceEncoder,
        noise_pred_net: ConditionalUnet1D,
        action_dim: int,
        ctrl_config: ControlConfig,
        replay_buffer: TrajectoryWindowDataset,
        optimizer: Any,
        rollout_config: PolicyWindowRolloutConfig,
        noise_scheduler: NoiseScheduler,
        num_inference_steps: int,
        scalingup_explorer: ScalingUpDataGen,
        should_condition_on_task_metric: bool,
        task_metric_dim: int,
        policy_task_metric_guidance: float,
        policy_suppress_token: float,
        policy_towards_token: float,
        task_metric_corrupt_prob: float,
        supported_policy_tasks: Optional[List[str]],
        lr_scheduler: Optional[Any] = None,
        ema_power: float = 0.75,
        ema_update_after_step: int = 0,
        num_validation_batches: int = 16,
        num_validation_seeds: int = 3,
        num_timesteps_per_batch: int = 1,
        validation_accuracy_thresholds: Optional[List[float]] = None,
        action_groups: Optional[Dict[str, slice]] = None,
        float32_matmul_precision: str = "high",
        use_remote_policy: bool = True,
    ):
        super().__init__(
            replay_buffer=replay_buffer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            float32_matmul_precision=float32_matmul_precision,
        )
        self.action_dim = action_dim
        # TODO add note here about replacement
        state_sequence_encoder = replace_bn_with_gn(state_sequence_encoder)
        noise_pred_net = replace_bn_with_gn(noise_pred_net)
        if os.environ.get("TORCH_COMPILE", "0") == "0":
            logging.info("Not compiling model. To enable, set `TORCH_COMPILE=1`")
            self.ema_nets = nn.ModuleDict(
                {
                    "state_sequence_encoder": state_sequence_encoder,
                    "noise_pred_net": noise_pred_net,
                }
            )
        else:
            logging.info("`torch.compile`-ing the model")
            # cache text features before compiling
            task_names = set(replay_buffer.task_names)
            for task_desc in task_names:
                state_sequence_encoder.get_text_feature(task_desc)
            self.ema_nets = nn.ModuleDict(
                {
                    "state_sequence_encoder": state_sequence_encoder,
                    # NOTE: state sequence encoder will handle its own compile
                    "noise_pred_net": torch.compile(noise_pred_net, mode="max-autotune"),
                }  # type: ignore
            )
        self.ema = EMAModel(
            model=self.ema_nets,
            power=ema_power,
            update_after_step=ema_update_after_step,
        )
        self.rollout_config = rollout_config
        logging.info(str(self.rollout_config))
        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = num_inference_steps
        self.num_timesteps_per_batch = num_timesteps_per_batch
        self.scalingup_explorer = scalingup_explorer
        self.should_condition_on_task_metric = should_condition_on_task_metric
        self.task_metric_dim = task_metric_dim
        self.task_metric_corrupt_prob = task_metric_corrupt_prob
        self.policy_task_metric_guidance = policy_task_metric_guidance
        self.policy_suppress_token = policy_suppress_token
        self.policy_towards_token = policy_towards_token
        self.ctrl_config = ctrl_config
        self.action_groups = action_groups if action_groups is not None else {}
        if len(self.window_replay_buffer) > 0:
            assert (
                self.ctrl_config.frequency == self.window_replay_buffer.control_frequency
            )
        self.num_validation_batches = num_validation_batches
        self.num_validation_seeds = num_validation_seeds
        self.validation_accuracy_thresholds = (
            validation_accuracy_thresholds
            if validation_accuracy_thresholds is not None
            else np.logspace(-3, -5, num=3).tolist()
        )
        self.validation_batches: Dict[str, List[TrajectoryWindowTensor]] = {}
        self.validation_batches["sim"] = self.get_validation_batches(
            self.num_validation_batches
        )
        self.supported_policy_tasks = supported_policy_tasks
        self.use_remote_policy = use_remote_policy

    @property
    def action_norm_config(self) -> NormalizationConfig:
        return self.window_replay_buffer.action_norm_config

    @property
    def window_replay_buffer(self) -> TrajectoryWindowDataset:
        assert type(self.replay_buffer) is TrajectoryWindowDataset
        return typing.cast(TrajectoryWindowDataset, self.replay_buffer)

    @property
    def state_sequence_encoder(self) -> StateSequenceEncoder:
        return typing.cast(
            StateSequenceEncoder, self.ema_nets.get_submodule("state_sequence_encoder")
        )

    @property
    def noise_pred_net(self) -> ConditionalUnet1D:
        return typing.cast(
            ConditionalUnet1D, self.ema_nets.get_submodule("noise_pred_net")
        )

    def get_stats(self, tensor_data: TensorDataClass) -> Dict[str, Any]:
        traj_window = typing.cast(TrajectoryWindowTensor, tensor_data).to(
            dtype=self.dtype
        )
        batch_size = traj_window.batch_size
        # encoder state sequence
        state_feature = self.state_sequence_encoder(
            state_sequence=traj_window.state_sequence, task_names=traj_window.task_names
        )

        # prepare target action sequence
        normalized_action = self.action_norm_config.normalize(
            traj_window.action_sequence.tensor
        ).to(device=self.device)
        assert normalized_action.shape == (
            batch_size,
            self.rollout_config.prediction_horizon,
            self.action_dim,
        ), (
            f"expected shape ({batch_size}, {self.rollout_config.prediction_horizon}, "
            + f"{self.action_dim}), got {normalized_action.shape}"
        )

        normalized_action = normalized_action.repeat(self.num_timesteps_per_batch, 1, 1)
        # sample noise to add to actions
        noise = torch.randn(
            (
                batch_size * self.num_timesteps_per_batch,
                self.rollout_config.prediction_horizon,
                self.action_dim,
            ),
            device=self.device,
        )

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,  # type: ignore
            (batch_size * self.num_timesteps_per_batch,),
            device=self.device,
        ).long()

        # add noise to the clean images according to the noise magnitude at each
        # diffusion iteration (this is the forward diffusion process)
        noisy_actions = self.noise_scheduler.add_noise(
            original_samples=typing.cast(torch.FloatTensor, normalized_action),
            noise=typing.cast(torch.FloatTensor, noise),
            timesteps=typing.cast(torch.IntTensor, timesteps),
        )
        if self.should_condition_on_task_metric:
            task_metric = traj_window.task_metrics.clone().to(dtype=self.dtype)
            assert task_metric.shape == (batch_size, 1)
            if self.task_metric_corrupt_prob > 0.0:
                corrupt_mask = (
                    torch.rand_like(task_metric) < self.task_metric_corrupt_prob
                )
                task_metric[corrupt_mask] = CORRUPT_SUCCESS_TOKEN
            # task_metric: (B, 1)
            # repeat task_metric
            task_metric = task_metric.repeat(1, self.task_metric_dim)
            # task_metric: (B, task_metric_dim)
            state_feature = torch.cat((state_feature, task_metric), dim=-1)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_actions,
            timesteps,
            global_cond=state_feature.repeat(self.num_timesteps_per_batch, 1),
        )

        # L2 loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        return {"train/loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        try:
            self.ema.step(self.ema_nets)
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                self.ema.averaged_model = self.ema.averaged_model.to(self.device)
            else:
                raise e
        self.log(
            "train/ema_decay",
            self.ema.decay,
        )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if os.environ.get("TORCH_COMPILE", "0") == "0":
            # need to handle torch compile, for instance:
            # noise_pred_net._orig_mod.final_conv.1.bias
            # noise_pred_net.final_conv.1.bias
            checkpoint["state_dict"] = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["state_dict"].items()
            }
            checkpoint["state_dict"]["ema_model"] = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["state_dict"]["ema_model"].items()
            }
        action_norm_config: NormalizationConfig = checkpoint["action_norm_config"]
        self.window_replay_buffer.min_action_values = action_norm_config.min_value
        self.window_replay_buffer.max_action_values = action_norm_config.max_value

        keys = [k for k in checkpoint["state_dict"].keys() if "lang_embs_cache" in k]
        texts = [k.split("lang_embs_cache.")[-1] for k in keys]
        logging.info(
            f"Loading {len(texts)} cached text features: "
            + ", ".join(f"{t!r}" for t in texts)
        )
        self.state_sequence_encoder.lang_embs_cache = torch.nn.ParameterDict(
            {
                text: torch.nn.Parameter(checkpoint["state_dict"][k], requires_grad=False)
                for text, k in zip(texts, keys)
            }
        )
        return super().on_load_checkpoint(checkpoint)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        retval = super().load_state_dict(state_dict, strict=False)
        self.ema.averaged_model.load_state_dict(state_dict["ema_model"], strict=False)
        return retval

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["action_norm_config"] = self.action_norm_config
        checkpoint["state_dict"]["ema_model"] = self.ema.averaged_model.state_dict()

    def get_diffusion_policy(
        self,
        inference_device: Union[torch.device, str] = "cuda",
        remote: Optional[bool] = None,
        action_norm_config: Optional[NormalizationConfig] = None,
        task_metric_guidance: Optional[float] = None,
        suppress_token: Optional[float] = None,
        towards_token: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        action_horizon: Optional[int] = None,
    ) -> DiffusionPolicy:
        if remote is None:
            remote = self.use_remote_policy
        averaged_models = self.ema.averaged_model
        inference_noise_scheduler = deepcopy(self.noise_scheduler)
        inference_noise_scheduler.set_timesteps(
            num_inference_steps=self.num_inference_steps
            if num_inference_steps is None
            else num_inference_steps,
            device=inference_device,
        )
        state_sequence_encoder: StateSequenceEncoder = (
            averaged_models["state_sequence_encoder"].to(inference_device).eval()
        )
        state_sequence_encoder.lang_embs_cache = (
            self.state_sequence_encoder.lang_embs_cache.to(inference_device)
        )
        kwargs = {
            "state_sequence_encoder": state_sequence_encoder,
            "noise_pred_net": averaged_models["noise_pred_net"]
            .to(inference_device)
            .eval(),
            "device": inference_device,
            "dtype": self.dtype,
            "action_dim": self.action_dim,
            "noise_scheduler": inference_noise_scheduler,
            "rollout_config": self.rollout_config
            if action_horizon is None
            else PolicyWindowRolloutConfig(
                **{
                    **dataclasses.asdict(self.rollout_config),
                    "action_horizon": action_horizon,
                }
            ),
            "ctrl_config": self.ctrl_config,
            "should_condition_on_task_metric": self.should_condition_on_task_metric,
            "task_metric_dim": self.task_metric_dim,
            "suppress_token": self.policy_suppress_token
            if suppress_token is None
            else suppress_token,
            "towards_token": self.policy_towards_token
            if towards_token is None
            else towards_token,
            "task_metric_guidance": self.policy_task_metric_guidance
            if task_metric_guidance is None
            else task_metric_guidance,
            "action_norm_config": self.action_norm_config.to(
                device=torch.device(inference_device)
            )
            if action_norm_config is None
            else action_norm_config.to(device=inference_device),
            "supported_tasks": frozenset(
                list(self.state_sequence_encoder.lang_embs_cache.keys())
            )
            if self.supported_policy_tasks is None
            else frozenset(self.supported_policy_tasks),
            "obs_cameras": self.replay_buffer.obs_cameras,
        }

        if remote:
            # automatically set "TORCH_COMPILE=0" in remote actors
            # https://github.com/ray-project/ray/issues/418
            # runtime_env = {
            #     'env_vars': {'ENV_VAR_1': 'env_var_value'}
            # }
            # # Actors
            # @ray.remote(runtime_env=runtime_env)
            policy_class = ray.remote(DiffusionPolicy).options(  # type: ignore
                num_cpus=1, num_gpus=float(inference_device == "cuda")
            )
            diffusion_policy = typing.cast(
                DiffusionPolicy, RayPolicy(policy=policy_class.remote(**kwargs))
            )  # type: ignore
        else:
            diffusion_policy = DiffusionPolicy(**kwargs)
        return diffusion_policy

    def get_policy(self, use_language_wrapper: bool = False, **kwargs):
        diffusion_policy = self.get_diffusion_policy(**kwargs)
        if not use_language_wrapper:
            return diffusion_policy
        self.scalingup_explorer.task_tree_inference.task_policy = diffusion_policy
        return self.scalingup_explorer

    def on_train_epoch_end(self) -> None:
        if self.num_validation_batches > 0:
            policy = self.get_diffusion_policy(
                inference_device=self.device, remote=False, task_metric_guidance=0.0
            )
            validation_stats: Dict[str, List[float]] = {}
            for validation_key, batches in self.validation_batches.items():
                for i in range(len(batches)):
                    batches[i] = batches[i].to(self.device, self.dtype, non_blocking=True)
                for batch in batches:
                    for stat_key, stat_value in self.get_validation_stats(
                        policy=policy,
                        traj_window=batch,
                        num_seeds=self.num_validation_seeds,
                        accuracy_thresholds=self.validation_accuracy_thresholds,
                    ).items():
                        full_key = os.path.join(validation_key, stat_key)
                        if full_key not in validation_stats:
                            validation_stats[full_key] = []
                        validation_stats[full_key].append(stat_value)
            self.log_dict(
                {
                    f"validation/{k}": float(np.mean(v))
                    for k, v in validation_stats.items()
                },
                sync_dist=True,
                on_epoch=True,
                batch_size=self.replay_buffer.batch_size,
            )
        super().on_train_epoch_end()

    def get_validation_batches(
        self, num_batches: int, replay_buffer: Optional[TrajectoryWindowDataset] = None
    ):
        if replay_buffer is None:
            replay_buffer = self.replay_buffer
        logging.info("pre-loading validation batches")
        batches = []
        for batch in replay_buffer.get_loader(
            num_steps_per_update=num_batches,
            batch_size=1,
            persistent_workers=False,
            pin_memory=False,
            repeat=False,
            shuffle=True,
        ):
            batches.append(batch)
            if len(batches) >= self.num_validation_batches:
                break
        logging.info(f"pre-loaded {len(batches)} validation batches")
        return batches

    def get_validation_stats(
        self,
        policy: DiffusionPolicy,
        traj_window: TrajectoryWindowTensor,
        accuracy_thresholds: List[float],
        num_seeds: int = 3,
    ) -> Dict[str, float]:
        stats: Dict[str, List[float]] = {}
        if not traj_window.is_batched:
            traj_window = TrajectoryWindowTensor.collate([traj_window])
        target_action = self.action_norm_config.normalize(
            traj_window.action_sequence.tensor
        )
        for seed in range(num_seeds):
            start = time()
            policy.set_seed(seed)
            predicted_action = self.action_norm_config.normalize(
                policy.diffuse(
                    state_sequence=traj_window.state_sequence,
                    task_names=traj_window.task_names,
                    task_metric=traj_window.task_metrics,
                )
            )
            if "diffusion_time" not in stats:
                stats["diffusion_time"] = []
            stats["diffusion_time"].append(time() - start)
            assert predicted_action.shape == (
                traj_window.batch_size,
                self.rollout_config.prediction_horizon,
                self.action_dim,
            )
            for group_name, group_slice in self.action_groups.items():
                mse = torch.nn.functional.mse_loss(
                    target_action[..., group_slice],
                    predicted_action[..., group_slice],
                ).item()
                if f"{group_name}/mse_loss" not in stats:
                    stats[f"{group_name}/mse_loss"] = []
                stats[f"{group_name}/mse_loss"].append(mse)
                for accuracy_threshold in accuracy_thresholds:
                    key = f"{group_name}/accuracy@{accuracy_threshold}"
                    if key not in stats:
                        stats[key] = []
                    stats[key].append(float(mse < accuracy_threshold))
        # reduce stats
        reduced_stats = {}
        for k, v in stats.items():
            reduced_stats[f"{k}/mean"] = float(np.mean(v))
            if "mse_loss" in k:
                reduced_stats[f"{k}/best"] = min(v)
            elif "accuracy" in k:
                reduced_stats[f"{k}/best"] = max(v)
        return reduced_stats
