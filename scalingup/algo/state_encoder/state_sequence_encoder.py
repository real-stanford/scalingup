import logging
from typing import List, Optional, Set, Union
import typing
import torch
from scalingup.data.window_dataset import PolicyWindowRolloutConfig, StateSequenceTensor
from scalingup.utils.core import split_state_phrase
from scalingup.utils.text_encoders import ClipLanguageConditioned
from scalingup.algo.state_encoder.vision_encoder import (
    ImageEncoder,
    PointCloudEncoder,
    VisionEncoder,
)
import os


class StateSequenceEncoder(ClipLanguageConditioned):
    def __init__(
        self,
        vision_encoder: VisionEncoder,
        task_desc_proj: torch.nn.Module,
        proprioception_keys: Set[str],
        rollout_config: PolicyWindowRolloutConfig,
        proprio_dim: int,
        obs_cameras: List[str],
        remove_with_statement: bool = True,
        should_condition_on_text: bool = True,
        should_condition_on_vision: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_encoder: Optional[VisionEncoder] = None
        self.vision_encoder_type = type(vision_encoder)
        if should_condition_on_vision:
            self.vision_encoder = vision_encoder
            if os.environ.get("TORCH_COMPILE", "0") == "1":
                self.vision_encoder = torch.compile(self.vision_encoder)
        self.task_desc_proj = task_desc_proj if should_condition_on_text else None
        if not should_condition_on_text:
            logging.warning("StateSequenceEncoder: not conditioning on text")
        self.proprioception_keys = proprioception_keys
        self.rollout_config = rollout_config
        self.proprio_dim = proprio_dim
        self.obs_cameras = obs_cameras
        self.remove_with_statement = remove_with_statement

    def forward(
        self, state_sequence: StateSequenceTensor, task_names: List[str]
    ) -> torch.Tensor:
        batch_size = state_sequence.batch_size
        vision_obs_horizon = self.rollout_config.vision_obs_horizon
        proprio_obs_horizon = self.rollout_config.proprio_obs_horizon
        obs_feature_list = []

        # prepare vision data
        if self.vision_encoder is not None:
            assert self.vision_encoder is not None
            if issubclass(self.vision_encoder_type, PointCloudEncoder):
                point_cloud_encoder = typing.cast(PointCloudEncoder, self.vision_encoder)
                input_xyz_pts = state_sequence.input_xyz_pts.view(
                    batch_size * vision_obs_horizon, point_cloud_encoder.num_obs_pts, 3
                )
                input_rgb_pts = state_sequence.input_rgb_pts.view(
                    batch_size * vision_obs_horizon, point_cloud_encoder.num_obs_pts, -1
                )
                vision_features = self.vision_encoder(
                    input_xyz_pts=input_xyz_pts,
                    input_rgb_pts=input_rgb_pts,
                )
                vision_features = vision_features.view(batch_size, vision_obs_horizon, -1)
                assert vision_features.shape[:2] == (
                    batch_size,
                    vision_obs_horizon,
                ), (
                    f"expected shape ({batch_size}, {vision_obs_horizon}), "
                    + f"got {vision_features.shape}"
                )
                obs_feature_list.append(vision_features)
            elif issubclass(self.vision_encoder_type, ImageEncoder):
                image_encoder = typing.cast(ImageEncoder, self.vision_encoder)
                # get the last `vision_obs_horizon` frames
                # of the vision data
                image_feature = image_encoder(
                    views={
                        k: v[:, -self.rollout_config.vision_obs_horizon :, ...]
                        for k, v in state_sequence.views.items()
                    }
                )
                obs_feature_list.append(image_feature.flatten(start_dim=1))
            else:
                raise NotImplementedError(self.vision_encoder_type)

        # prepare proprioception data
        if len(self.proprioception_keys) > 0:
            proprio = state_sequence.get_proprioception_data(
                proprioception_keys=self.proprioception_keys
            )
            assert proprio.shape == (
                batch_size,
                proprio_obs_horizon,
                self.proprio_dim,
            ), (
                f"expected {batch_size}x{proprio_obs_horizon}x{self.proprio_dim},"
                + f" got {proprio.shape}"
            )
            obs_feature_list.append(proprio.flatten(start_dim=1))

        # concatenate vision feature and low-dim obs
        obs_features = torch.cat(obs_feature_list, dim=-1)
        if self.task_desc_proj is None:
            return obs_features
        # prepare task description data
        text_features = []
        for text in task_names:
            text = text if not self.remove_with_statement else split_state_phrase(text)[1]
            text_feature = self.get_text_feature(text=text).to(
                dtype=state_sequence.dtype, device=state_sequence.device
            )
            text_features.append(text_feature)
        assert (
            len(text_features) == batch_size
        ), f"expected {batch_size} text features, got {len(text_features)}"
        task_desc_emb = torch.stack(text_features, dim=0)
        task_desc_emb = task_desc_emb.view(batch_size, -1)
        task_desc_emb = self.task_desc_proj(task_desc_emb)
        assert (
            task_desc_emb.shape[0] == batch_size
        ), f"expected shape ({batch_size}), got {task_desc_emb.shape}"
        return torch.cat([obs_features, task_desc_emb], dim=-1)
