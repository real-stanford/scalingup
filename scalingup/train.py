# flake8: noqa
from copy import deepcopy
import logging
from typing import List, Optional, Tuple
import hydra
from omegaconf import OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning import Trainer as LightningTrainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers.wandb import WandbLogger

import wandb
from scalingup.algo.algo import ScalingUpAlgo
from scalingup.inference import setup_inference
from scalingup.utils.generic import setup
import torch
import os
from scalingup.utils.core import EnvSamplerConfig
import dataclasses

rank_idx = os.environ.get("NODE_RANK", 0)

torch.multiprocessing.set_sharing_strategy("file_system")


def setup_trainer(
    conf: OmegaConf,
    callbacks: Optional[List[Callback]] = None,
    wandb_logger: Optional[WandbLogger] = None,
) -> Tuple[LightningTrainer, ScalingUpAlgo]:
    if wandb_logger is None and rank_idx == 0:
        wandb_logger = setup(
            logdir=conf.logdir,
            seed=conf.seed,
            num_processes=conf.num_processes,
            tags=conf.tags,
            notes=conf.notes,
            conf=conf,
        )
    if conf.load_from_path is not None:
        algo = ScalingUpAlgo.load_from_checkpoint(
            checkpoint_path=conf.load_from_path,
            **{
                k: (hydra.utils.instantiate(v) if "Config" in type(v).__name__ else v)
                for k, v in conf.algo.items()
                if not (k.endswith("_") and k.startswith("_"))
            },
            strict=False,
        )
        # TODO also load in ema model.
        # ideally always have it in checkpoint, without manual hacking
    else:
        algo = hydra.utils.instantiate(conf.algo)
    if callbacks is None:
        callbacks = []
    if rank_idx == 0:
        callbacks.extend(
            (
                ModelCheckpoint(
                    dirpath=f"{wandb.run.dir}/checkpoints/",  # type: ignore
                    filename="{epoch:04d}",
                    every_n_epochs=1,
                    save_last=True,
                    save_top_k=10,
                    monitor="epoch",
                    mode="max",
                    save_weights_only=False,
                ),
                RichProgressBar(leave=True),
                LearningRateMonitor(logging_interval="step"),
            )
        )
    trainer: LightningTrainer = hydra.utils.instantiate(
        conf.trainer,
        callbacks=callbacks,
        default_root_dir=wandb.run.dir if wandb.run is not None else None,  # type: ignore
        logger=wandb_logger,
    )
    return (
        trainer,
        algo,
    )


@hydra.main(config_path="config", config_name="train_offline", version_base="1.2")
def train(conf: OmegaConf):
    # only setup if rank is 0
    callbacks = []
    wandb_logger = None
    evaluation = None
    if conf.evaluation is not None and rank_idx == 0:
        _, evaluation, wandb_logger = setup_inference(conf=conf)
        callbacks.append(evaluation)
    trainer, algo = setup_trainer(
        conf=conf, callbacks=callbacks, wandb_logger=wandb_logger
    )
    logging.info("Testing data loader")
    for _ in zip(range(5), algo.replay_buffer.get_loader(num_workers=1, batch_size=1)):
        pass
    if evaluation is not None:
        logging.info("Running debug evaluation")
        algo.trainer = trainer

        eval_criteria = deepcopy(evaluation.eval_criteria)
        evaluation.eval_criteria = None
        num_episodes = evaluation.num_episodes
        evaluation.num_episodes = 1
        sampler_config = deepcopy(evaluation.sampler_config)
        evaluation.sampler_config = EnvSamplerConfig(
            **{  # type: ignore
                **dataclasses.asdict(evaluation.sampler_config),
                "max_time": 1,
            }
        )

        # do a test run
        evaluation.on_train_epoch_end(
            trainer=trainer, algo=algo, update_policy=True, debug_run=True
        )
        evaluation.sampler_config = sampler_config
        evaluation.eval_criteria = eval_criteria
        evaluation.num_episodes = num_episodes
    trainer.fit(algo)
    wandb.finish()  # type: ignore


if __name__ == "__main__":
    train()
