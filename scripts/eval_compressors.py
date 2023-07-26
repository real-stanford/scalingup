import gc
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import typing
from typing import Any, Dict, List, Optional, Union

import numcodecs
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
import torch
import zarr
from matplotlib import pyplot as plt
from numcodecs import Blosc
from utils import actualsize, get_sample_trajectory, memory_usage, set_up_env

from scalingup.data.window_dataset import copy_from_store
from scalingup.utils.core import ControlTrajectory


"""
RAM usage is astronomically high because trajectories
are long and many high dimensional observations and
visualization frames are rendered and stored uncompressed
in memory.

This hinders large scale data generation.
"""


@pytest.mark.parametrize("control_frequency", range(5, 10))
def test_conversion(control_frequency: int):
    traj = get_sample_trajectory(control_frequency=control_frequency)
    start = time.time()
    control_traj = ControlTrajectory.from_trajectory(traj)
    assert float(time.time() - start) < 0.05
    assert control_traj.control_frequency == control_frequency


def test_compression_algos():
    traj = get_sample_trajectory()
    control_traj = ControlTrajectory.from_trajectory(traj)
    path = "test_control_traj.zarr"
    stats: Dict[str, List[Any]] = {
        "algo": [],
        "config": [],
        "dump_time": [],
        "disk_space": [],
        "rgb_loading_time": [],
        "xyz_loading_time": [],
    }
    compressors = []
    for shuffle in [
        Blosc.SHUFFLE
        # , Blosc.NOSHUFFLE, Blosc.BITSHUFFLE
    ]:
        for algo in [
            "zstd",
            #   "lz4", "blosclz", "lz4hc", "zlib"
        ]:
            for clevel in [
                9,
                # 7,
                # 5,
                # 3,
                # 1,
            ]:
                compressor = Blosc(cname=algo, clevel=clevel, shuffle=shuffle)
                compressors.append(
                    (
                        compressor,
                        "Blosc",
                        f"shuffle={shuffle},clevel={clevel},algo={algo}",
                    )
                )
    for compressor, name, config in compressors:
        print(name, config)
        start = time.time()
        # check size
        control_traj.dump(path, pickled_data_compressor=compressor)
        dump_time = float(time.time() - start)
        root = zarr.open(path)
        start = time.time()
        root["state_tensor/input_rgb_pts"][:]
        rgb_loading_time = float(time.time() - start)
        start = time.time()
        root["state_tensor/input_xyz_pts"][:]
        xyz_loading_time = float(time.time() - start)
        # run subprocess then get output
        result = subprocess.run(["du", "-sh", path], capture_output=True, text=True)
        print(name, config, float(result.stdout.split("M")[0]))
        stats["algo"].append(name)
        stats["config"].append(config)
        stats["dump_time"].append(dump_time)
        stats["disk_space"].append(float(result.stdout.split("M")[0]))
        stats["rgb_loading_time"].append(rgb_loading_time)
        stats["xyz_loading_time"].append(xyz_loading_time)
        df = pd.DataFrame.from_dict(stats)
        df.to_pickle("stats.pkl")
    df = pd.read_pickle("stats.pkl")
    fig, axes = plt.subplots(1, 3)
    df = df[df.config.str.startswith("shuffle=1")]
    df = df[df.config.str.endswith("algo=zstd")]
    sns.scatterplot(
        data=df, x="disk_space", y="xyz_loading_time", hue="config", ax=axes[0]
    )
    sns.scatterplot(
        data=df, x="dump_time", y="xyz_loading_time", hue="config", ax=axes[1]
    )
    sns.scatterplot(data=df, x="disk_space", y="dump_time", hue="config", ax=axes[2])
    plt.tight_layout(pad=0)
    plt.show()
    exit()


def get_loading_stats(root):
    start = time.time()
    root["state_tensor/input_rgb_pts"][:]
    rgb_loading_time = float(time.time() - start)
    start = time.time()
    root["state_tensor/input_xyz_pts"][:]
    xyz_loading_time = float(time.time() - start)
    size = actualsize(root)
    return rgb_loading_time, xyz_loading_time, size / 2**20


def test_io_vs_compression_speed():
    stats: Dict[str, List[Any]] = {
        "rgb_loading_time": [],
        "xyz_loading_time": [],
        "store": [],
        "size": [],
    }
    path = "temp_dataset/test_control_traj.zarr/"
    root = zarr.open(path, "r")
    rgb_loading_time, xyz_loading_time, size = get_loading_stats(root)
    stats["rgb_loading_time"].append(rgb_loading_time)
    stats["xyz_loading_time"].append(xyz_loading_time)
    stats["size"].append(size)
    stats["store"].append("disk")
    compressors = [None]

    for shuffle in [Blosc.SHUFFLE, Blosc.NOSHUFFLE, Blosc.BITSHUFFLE]:
        for algo in ["zstd", "lz4", "blosclz", "lz4hc", "zlib"]:
            for clevel in [
                9,
                7,
                5,
                3,
                1,
            ]:
                compressor = Blosc(cname=algo, clevel=clevel, shuffle=shuffle)
                compressors.append(
                    compressor,
                )
    df = pd.DataFrame()
    for compressor in compressors:
        start = time.time()
        in_memory_root = copy_from_store(
            src=root, dest=zarr.group(store=zarr.MemoryStore()), compressor=compressor
        )
        assert str(in_memory_root.tree()) == str(root.tree())
        print(f"copy time ({compressor}): {float(time.time() - start):.03}s")
        rgb_loading_time, xyz_loading_time, size = get_loading_stats_from_store_cls(
            zarr.ZipStore, in_memory_root
        )
        stats["rgb_loading_time"].append(rgb_loading_time)
        stats["xyz_loading_time"].append(xyz_loading_time)
        stats["size"].append(size)
        stats["store"].append("in_memory (compressor: " + str(compressor) + ")")
        df = pd.DataFrame.from_dict(stats)
        df.to_pickle("io_compression_stats.pkl")
    df = pd.read_pickle("io_compression_stats.pkl")
    df.sort_values(by="rgb_loading_time", inplace=True)
    df = df[1:]  # ignore no compression, which is too large
    df = df[:10]
    print(df)
    fig, axes = plt.subplots(1, 3)
    sns.scatterplot(
        data=df, x="rgb_loading_time", y="xyz_loading_time", hue="store", ax=axes[0]
    )
    sns.scatterplot(data=df, x="rgb_loading_time", y="size", hue="store", ax=axes[1])
    sns.scatterplot(data=df, x="xyz_loading_time", y="size", hue="store", ax=axes[2])
    plt.tight_layout(pad=0)
    plt.show()


def get_loading_stats_from_store_cls(store_cls, path):
    # read once first to activate cache
    with store_cls(path) as store:
        root = zarr.open(store, "r")
        root["state_tensor/input_rgb_pts"][:]
        root["state_tensor/input_xyz_pts"][:]
    start = time.time()
    with store_cls(path) as store:
        root = zarr.open(store, "r")
        root["state_tensor/input_rgb_pts"][:]
        rgb_loading_time = float(time.time() - start)
    start = time.time()
    with store_cls(path) as store:
        root = zarr.open(store, "r")
        root["state_tensor/input_xyz_pts"][:]
        xyz_loading_time = float(time.time() - start)
        size = actualsize(root)
    return rgb_loading_time, xyz_loading_time, size / 2**20


def test_store_cls(store_cls, temp_dataset_path: str):
    stats: Dict[str, List[Any]] = {
        "rgb_loading_time": [],
        "xyz_loading_time": [],
        "store": [],
        "size": [],
    }
    path = "temp_dataset/test_control_traj.zarr/"
    raw_root = zarr.open(path, "r")
    rgb_loading_time, xyz_loading_time, size = get_loading_stats(raw_root)
    stats["rgb_loading_time"].append(rgb_loading_time)
    stats["xyz_loading_time"].append(xyz_loading_time)
    stats["size"].append(size)
    stats["store"].append("disk")
    compressors = [None]

    for shuffle in [
        # Blosc.SHUFFLE,
        Blosc.NOSHUFFLE,
        #   Blosc.BITSHUFFLE
    ]:
        for algo in [
            "zstd",
            #  "lz4",
            "blosclz",
            #  "lz4hc", "zlib"
        ]:
            for clevel in [
                9,
                7,
                5,
            ]:
                compressor = Blosc(cname=algo, clevel=clevel, shuffle=shuffle)
                compressors.append(
                    compressor,
                )
    df = pd.DataFrame()
    for compressor in compressors:
        start = time.time()
        if os.path.exists(temp_dataset_path):
            shutil.rmtree(temp_dataset_path)
        with store_cls(path=temp_dataset_path) as store:
            copy_from_store(
                src=raw_root,
                dest=zarr.group(store=store),
                compressor=compressor,
            )
            print(f"copy time ({compressor}): {float(time.time() - start):.03}s")
        rgb_loading_time, xyz_loading_time, size = get_loading_stats_from_store_cls(
            store_cls, temp_dataset_path
        )
        stats["rgb_loading_time"].append(rgb_loading_time)
        stats["xyz_loading_time"].append(xyz_loading_time)
        stats["size"].append(size)
        stats["store"].append(
            f"{store_cls.__name__} (compressor: " + str(compressor) + ")"
        )
        df = pd.DataFrame.from_dict(stats)
        df.to_pickle("store_stats.pkl")
    df = pd.read_pickle("store_stats.pkl")
    df.sort_values(by="rgb_loading_time", inplace=True)
    # df = df[1:]  # ignore no compression, which is too large
    # df = df[:10]
    print(df)
    fig, axes = plt.subplots(1, 3)
    sns.scatterplot(
        data=df, x="rgb_loading_time", y="xyz_loading_time", hue="store", ax=axes[0]
    )
    sns.scatterplot(data=df, x="rgb_loading_time", y="size", hue="store", ax=axes[1])
    sns.scatterplot(data=df, x="xyz_loading_time", y="size", hue="store", ax=axes[2])
    plt.tight_layout(pad=0)
    plt.show()


# test_store_cls(zarr.ZipStore, "temp_dataset.zip")
test_store_cls(zarr.LMDBStore, "temp_dataset.mdb")
# test_io_vs_compression_speed()
# test_compression_algos()
# test_conversion(5)


def test_in_memory_compression():
    env, _ = set_up_env("drawer")
    env.reset(0)
    rgb = env.get_rgb("front_right")
    raw_megabytes = actualsize(np.ascontiguousarray(rgb)) / 2**20
    compressor = Blosc(
        cname="zstd",
        clevel=9,
        shuffle=Blosc.NOSHUFFLE,
    )
    shape = rgb.shape
    dtype = rgb.dtype
    compressed_rgb = compressor.encode(np.ascontiguousarray(rgb))
    compressed_megabytes = actualsize(compressed_rgb) / 2**20
    decompressed = np.frombuffer(compressor.decode(compressed_rgb), dtype=dtype).reshape(
        shape
    )
    assert np.allclose(rgb, decompressed)
    assert raw_megabytes / compressed_megabytes > 20  # 20x compression ratio


def test_different_compressors():
    env, _ = set_up_env("drawer")
    env.reset(0)
    rgb = np.ascontiguousarray(env.get_rgb("front_right", image_dim=(512, 512)))
    raw_megabytes = actualsize(rgb) / 2**20
    compressors = []
    for shuffle in [Blosc.SHUFFLE, Blosc.NOSHUFFLE, Blosc.BITSHUFFLE]:
        for algo in ["zstd", "lz4", "blosclz", "lz4hc", "zlib"]:
            for clevel in [
                9,
                7,
                5,
                3,
                1,
            ]:
                compressor = Blosc(cname=algo, clevel=clevel, shuffle=shuffle)
                compressors.append(
                    (
                        compressor,
                        "Blosc",
                        f"shuffle={shuffle},clevel={clevel},algo={algo}",
                    )
                )

    shape = rgb.shape
    dtype = rgb.dtype
    stats: Dict[str, List[Any]] = {
        "algo": [],
        "config": [],
        "ratio": [],
        "compression_time": [],
        "decompression_time": [],
    }
    df = pd.DataFrame.from_dict(stats)
    for compressor, name, config in compressors:
        print(name, config)
        start = time.time()
        # check size
        compressed_rgb = compressor.encode(np.ascontiguousarray(rgb))
        compressed_megabytes = actualsize(compressed_rgb) / 2**20

        compression_time = float(time.time() - start)
        start = time.time()
        decompressed = np.frombuffer(
            compressor.decode(compressed_rgb), dtype=dtype
        ).reshape(shape)
        assert np.allclose(rgb, decompressed)
        decompression_time = float(time.time() - start)
        # run subprocess then get output
        stats["algo"].append(name)
        stats["config"].append(config)
        stats["ratio"].append(raw_megabytes / compressed_megabytes)
        stats["compression_time"].append(compression_time)
        stats["decompression_time"].append(decompression_time)
        df = pd.DataFrame.from_dict(stats)
        df.to_pickle("stats.pkl")
        print(df)
    df.sort_values(by="ratio", ascending=False, inplace=True)
    print(df[:10])


if __name__ == "__main__":
    test_different_compressors()
