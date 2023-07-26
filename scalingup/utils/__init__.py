import logging
from omegaconf import OmegaConf
import wandb


def custom_eval(x):
    logging.debug(x)
    return eval(x)


OmegaConf.register_new_resolver("eval", custom_eval)
