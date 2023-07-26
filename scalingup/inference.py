from __future__ import annotations

import logging
from typing import Optional, Tuple

import hydra
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger

from scalingup.evaluation import SimEvaluation
from scalingup.utils.generic import setup
from scalingup.utils.core import Policy


def setup_inference(
    conf: OmegaConf,
) -> Tuple[Optional[Policy], SimEvaluation, WandbLogger]:
    wandb_logger = setup(
        logdir=conf.logdir,
        seed=conf.seed,
        num_processes=conf.num_processes,
        tags=conf.tags,
        notes=conf.notes,
        conf=conf,
    )
    evaluation: SimEvaluation = hydra.utils.instantiate(conf.evaluation)
    tasks = evaluation.get_tasks()
    for task_idx, task in enumerate(tasks):
        logging.info(f"[{task_idx}] {str(task)}")
    policy: Optional[Policy] = None
    if "policy" in conf:
        policy = hydra.utils.instantiate(conf.policy)
    return policy, evaluation, wandb_logger


@hydra.main(config_path="config", config_name="inference", version_base="1.2")
def main(conf: OmegaConf):
    policy, evaluation, _ = setup_inference(conf=conf)
    logging.info(f"Dumping experience to {wandb.run.dir}")  # type: ignore
    assert policy is not None
    stats = evaluation.run(
        policy=policy, pbar=True, root_path=wandb.run.dir  # type: ignore
    )
    wandb.log(stats)  # type: ignore


if __name__ == "__main__":
    main()
