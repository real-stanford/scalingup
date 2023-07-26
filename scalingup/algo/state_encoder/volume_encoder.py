from typing import Tuple
from abc import ABC, abstractmethod
import torch


class PolicyVolumeEncoder(torch.nn.Module, ABC):
    def __init__(
        self,
        proprioception_dim: int,
        grid_shape: Tuple[int, int, int],
        lang_emb_dim: int = 512,
        lang_max_seq_len: int = 77,
        num_rotation_classes: int = 72,
        num_grip_classes: int = 2,  # open or not open
        num_contact_classes: int = 2,  # collisions allowed or not allowed
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lang_max_seq_len = lang_max_seq_len
        self.lang_emb_dim = lang_emb_dim
        self.proprioception_dim = proprioception_dim
        self.num_rotation_classes = num_rotation_classes
        self.num_grip_classes = num_grip_classes
        self.num_contact_classes = num_contact_classes
        self.grid_shape = grid_shape

    @abstractmethod
    def forward(
        self,
        voxelized_obs,
        proprio,
        lang_goal_embs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass
