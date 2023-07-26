import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
import typing
import torch
import logging
import wandb
from pytorch_lightning import LightningModule

from scalingup.data.dataset import ReplayBuffer, TensorDataClass, TrajectoryStepTensor
from scalingup.utils.core import Policy


class ScalingUpAlgo(LightningModule, ABC):
    def __init__(self, replay_buffer: ReplayBuffer):
        super().__init__()
        self.replay_buffer = replay_buffer

    @property
    def logdir(self) -> str:
        return f"{wandb.run.dir}"  # type: ignore

    @property
    def current_epoch_logdir(self) -> str:
        retval = f"{self.logdir}/{self.trainer.current_epoch:03d}/"
        os.makedirs(exist_ok=True, name=retval)
        return retval

    def on_train_epoch_end(self):
        self.log_dict(
            dictionary={
                f"replay_buffer/{k}": v for k, v in self.replay_buffer.summary.items()
            },
            on_epoch=True,
            sync_dist=True,
        )

    @abstractmethod
    def get_policy(self) -> Policy:
        pass

    def train_dataloader(self):
        return self.replay_buffer.get_loader()

    def configure_optimizers(self):
        return []

    @property
    def dtype(self) -> torch.dtype:
        return typing.cast(torch.dtype, super().dtype)


class TrainableScalingUpAlgo(ScalingUpAlgo, ABC):
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        optimizer: Any,
        lr_scheduler: Optional[Any] = None,
        float32_matmul_precision: str = "high",
    ):
        super().__init__(replay_buffer=replay_buffer)
        self.lr_scheduler_partial = lr_scheduler
        self.optimizer_partial = optimizer
        torch.set_float32_matmul_precision(float32_matmul_precision)
        logging.info(f"Using {float32_matmul_precision} precision training")
        if float32_matmul_precision == "medium":
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            logging.info("Use `float32_matmul_precision=medium` for faster training")

    def training_step(self, tensor_data: TensorDataClass, batch_idx: int):
        stats = self.get_stats(tensor_data.to(device=self.device, non_blocking=True))
        total_loss = sum(v for k, v in stats.items() if "loss" in k)
        self.log_dict(
            {f"train/{k}": v for k, v in stats.items()},
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.replay_buffer.batch_size,
        )
        return total_loss

    def configure_optimizers(self):
        optimizer = self.optimizer_partial(params=self.parameters())
        if self.lr_scheduler_partial is None:
            return optimizer
        return [optimizer], [
            {
                "scheduler": self.lr_scheduler_partial(optimizer=optimizer),
                "interval": "step",
            }
        ]

    @abstractmethod
    def get_stats(self, tensor_data: TensorDataClass) -> Dict[str, Any]:
        pass
