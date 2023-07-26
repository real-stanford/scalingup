from __future__ import annotations

import logging
import os
import pickle
from collections.abc import MutableMapping
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import ray
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger
from rich.logging import RichHandler
from zarr import blosc


def flatten_dict(
    d: MutableMapping, parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, List):
            if len(v) == 0:
                items.append((new_key, []))
            elif isinstance(v[0], MutableMapping):
                for idx in range(len(v)):
                    items.extend(
                        flatten_dict(v[idx], f"{new_key}/{idx}", sep=sep).items()
                    )
            else:
                for idx in range(len(v)):
                    items.append((f"{new_key}/{idx}", v[idx]))
        else:
            items.append((new_key, v))
    return dict(items)


def setup_logger(logging_level=logging.INFO):
    logger = logging.getLogger()
    logger.handlers = []
    handler = RichHandler(markup=True)
    logger.setLevel(logging_level)
    handler.setLevel(logging_level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def resize_and_add_data(dataset, data):
    data_shape = np.array(data.shape)
    dataset_shape = np.array(dataset.shape)
    assert (dataset_shape[1:] == data_shape[1:]).all()
    dataset.resize(dataset_shape[0] + data_shape[0], axis=0)
    dataset[-data_shape[0] :, ...] = data
    return [
        dataset.regionref[dataset_shape[0] + i, ...] for i in np.arange(0, data_shape[0])
    ]


def write_to_hdf5(group, key, value, dtype=None, replace=False):
    if value is None:
        return
    if key in group and replace:
        del group[key]
    if type(value) == str or np.isscalar(value):
        group.attrs[key] = value
    elif type(value) == dict:
        if key in group:
            subgroup = group[key]
        else:
            subgroup = group.create_group(key)
        for subgroup_key, subgroup_value in value.items():
            write_to_hdf5(subgroup, subgroup_key, subgroup_value)
    elif type(value) == list or type(value) == tuple:
        if key in group:
            subgroup = group[key]
        else:
            subgroup = group.create_group(key)
        for i, item in enumerate(value):
            write_to_hdf5(group=subgroup, key=f"{key}_{i}", value=item)
    else:
        group.create_dataset(
            name=key,
            data=value,
            dtype=dtype,
            compression="gzip",
            compression_opts=9,
        )


def add_text_to_image(
    image: np.ndarray,
    texts: Sequence[str],
    positions: Sequence[Tuple[int, int]],
    color="rgb(0, 0, 0)",
    fontsize=18,
):
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", fontsize)
    except OSError:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/lato/Lato-Light.ttf", fontsize
        )
    for text, pos in zip(texts, positions):
        draw.text(
            pos,
            text,
            fill=color,
            font=font,
        )
    return np.array(pil_image)


def exec_safe(code_str, global_vars=None, local_vars=None):
    banned_phrases = ["import"]
    for phrase in banned_phrases:
        assert phrase not in code_str

    if global_vars is None:
        global_vars = {}
    if local_vars is None:
        local_vars = {}
    custom_global_vars = merge_dicts([global_vars, {"exec": empty_fn, "eval": empty_fn}])
    exec(code_str, custom_global_vars, local_vars)


def var_exists(name, all_vars):
    try:
        eval(name, all_vars)
    except NameError:
        exists = False
    else:
        exists = True
    return exists


def merge_dicts(dicts):
    return {k: v for d in dicts for k, v in d.items()}


def empty_fn(*args, **kwargs):
    return None


def setup(
    logdir: str,
    seed: int,
    num_processes: int,
    tags: List[str],
    notes: Optional[str] = None,
    conf: Optional[Union[OmegaConf, DictConfig]] = None,
):
    os.makedirs(logdir, exist_ok=True)
    seed_everything(seed, workers=True)
    wandb_logger = WandbLogger(
        project="scalingup",
        save_dir=logdir,
        config=flatten_dict(OmegaConf.to_container(conf, resolve=True), sep="/")  # type: ignore
        if conf is not None
        else None,
        tags=tags,
        notes=notes,
    )
    assert wandb.run is not None  # type: ignore
    if conf is not None:
        logging.info(f"Dumping conf to `{wandb.run.dir}/conf.pkl`")  # type: ignore
        pickle.dump(conf, open(f"{wandb.run.dir}/conf.pkl", "wb"))  # type: ignore
    return wandb_logger


class AllowArbitraryTypes:
    # TODO look into numpy.typing.NDArray
    # https://numpy.org/devdocs/reference/typing.html#numpy.typing.NDArray
    arbitrary_types_allowed = True


def limit_threads(n: int = 1):
    blosc.set_nthreads(n)
    if n == 1:
        blosc.use_threads = False
    torch.set_num_threads(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
