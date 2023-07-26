from utils import set_up_env
from scalingup.utils.core import (
    GraspLinkAction,
    EnvSampler,
    ActionListPolicy,
    GraspObj,
    PolicyTaskAction,
)
from scalingup.utils.generic import setup_logger
import matplotlib.pyplot as plt

if __name__ == "__main__":
    setup_logger()
    env = set_up_env("bin_transport_train")[0]
    env.num_pose_variations = 2
    env.num_setup_variations = 1
    images = [env.reset(episode_id=i).images["top_down"].rgb for i in range(9)]
    fig, axes = plt.subplots(3, 3)
    for img, ax in zip(images, axes.flatten()):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    env = set_up_env(
        "bin_transport_train",
        extra_overrides=[
            "env/domain_rand_config=all",
            "env.domain_rand_config.dtd_root=/home/huy/scalingup-release-private/dtd/",
        ],
    )[0]
    env.num_pose_variations = 2
    env.num_setup_variations = 1
    images = [env.reset(episode_id=i).images["front"].rgb for i in range(9)]
    fig, axes = plt.subplots(3, 3)
    for img, ax in zip(images, axes.flatten()):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
