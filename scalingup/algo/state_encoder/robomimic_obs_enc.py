import gc
from typing import Dict, List, Optional, Tuple
from robomimic.algo import algo_factory
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
from omegaconf import OmegaConf
from robomimic.config import config_factory
import robomimic.scripts.generate_paper_configs as gpc
from robomimic.scripts.generate_paper_configs import (
    modify_config_for_default_image_exp,
    modify_config_for_default_low_dim_exp,
    modify_config_for_dataset,
)
from robomimic.config.config import Config
import torch

from scalingup.algo.state_encoder.vision_encoder import ImageEncoder, VisionEncoder

"""
From with minimal modifications
https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/common/robomimic_config_util.py
"""


def get_robomimic_config(
    algo_name="bc_rnn", hdf5_type="low_dim", task_name="square", dataset_type="ph"
):
    base_dataset_dir = "/tmp/null"
    filter_key = None

    # decide whether to use low-dim or image training defaults
    modifier_for_obs = modify_config_for_default_image_exp
    if hdf5_type in ["low_dim", "low_dim_sparse", "low_dim_dense"]:
        modifier_for_obs = modify_config_for_default_low_dim_exp

    algo_config_name = "bc" if algo_name == "bc_rnn" else algo_name
    config = config_factory(algo_name=algo_config_name)
    # turn into default config for observation modalities (e.g.: low-dim or rgb)
    config = modifier_for_obs(config)
    # add in config based on the dataset
    config = modify_config_for_dataset(
        config=config,
        task_name=task_name,
        dataset_type=dataset_type,
        hdf5_type=hdf5_type,
        base_dataset_dir=base_dataset_dir,
        filter_key=filter_key,
    )
    # add in algo hypers based on dataset
    algo_config_modifier = getattr(gpc, f"modify_{algo_name}_config_for_dataset")
    config = algo_config_modifier(
        config=config,
        task_name=task_name,
        dataset_type=dataset_type,
        hdf5_type=hdf5_type,
    )
    return config


class RobomimicImageEncoder(torch.nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        crop_shape: Optional[Tuple[int, int]] = None,
        num_kp: int = 32,
    ):
        super().__init__()
        shape_meta = {
            "obs": {
                "rgb": {"shape": [3, *input_shape], "type": "rgb"},
            },
        }
        obs_shape_meta = shape_meta["obs"]
        obs_config: Dict[str, List[str]] = {
            "low_dim": [],
            "rgb": [],
            "depth": [],
            "scan": [],
        }
        obs_key_shapes = {}
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)

            type = attr.get("type", "low_dim")
            if type == "rgb":
                obs_config["rgb"].append(key)
            elif type == "low_dim":
                obs_config["low_dim"].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name="bc_rnn", hdf5_type="image", task_name="square", dataset_type="ph"
        )

        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for modality in config.observation.encoder.values():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality["obs_randomizer_class"] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for modality in config.observation.encoder.values():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw
            config.observation.encoder["rgb"]["core_kwargs"]["pool_kwargs"][
                "num_kp"
            ] = num_kp

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=10,  # not used
            device="cpu",
        )

        self.obs_encoder = policy.nets["policy"].nets["encoder"].nets["obs"]

        # may store references to robomimic.Config, which can't be loaded from pickles
        # and therefore incompatible with ray
        del self.obs_encoder.obs_nets_kwargs

    def __repr__(self):
        return "RobomimicImageEncoder()"

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.obs_encoder({"rgb": image})
