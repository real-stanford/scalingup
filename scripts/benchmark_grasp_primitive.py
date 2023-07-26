import numpy as np
from omegaconf import OmegaConf
from scalingup.environment.mujoco.drawer import DrawerMujocoEnv
from scalingup.environment.mujoco.mujocoEnv import MujocoUR5EnvFromObjConfigList
from scalingup.environment.mujoco.utils import MujocoObjectInstanceConfig
from scalingup.evaluation.base import SingleEnvSimEvaluation
from scalingup.utils.generic import setup, setup_logger
from scalingup.utils.core import (
    ActionListPolicy,
    ControlConfig,
    DegreeOfFreedomRange,
    GraspLinkAction,
    GraspObj,
    TaskSampler,
    EnvSamplerConfig,
    ControlConfig,
)
import wandb


def benchmark_grasp(
    asset_class: str,
    asset_path: str,
    link_path: str,
    max_time: float = 20.0,
    num_episodes: int = 100,
    num_processes: int = 20,
):
    env = MujocoUR5EnvFromObjConfigList(
        obs_cameras=["front", "front_right", "top_down"],
        obs_dim=(256, 256),
        ctrl_config=ControlConfig(frequency=2, dof=13),
        ground_xml_path="scalingup/environment/mujoco/assets/zero_ground.xml",
        obj_instance_configs=[
            MujocoObjectInstanceConfig(
                obj_class=asset_class,
                asset_path=asset_path,
                qpos_range=[
                    DegreeOfFreedomRange(upper=0.3, lower=0.6),
                    DegreeOfFreedomRange(upper=0.2, lower=0.2),
                    DegreeOfFreedomRange(upper=0.3, lower=0.2),
                    DegreeOfFreedomRange(upper=np.pi, lower=-np.pi),
                    DegreeOfFreedomRange(upper=np.pi, lower=-np.pi),
                    DegreeOfFreedomRange(upper=np.pi, lower=-np.pi),
                ],
                add_free_joint=True,
            ),
        ],
    )
    setup(
        logdir="scalingup/",
        seed=0,
        num_processes=num_processes,
        tags=["benchmark", "grasp_primitive"],
        conf=OmegaConf.create(
            {
                "link_path": link_path,
                "asset_path": asset_path,
                "asset_class": asset_class,
            }
        ),  # type: ignore
    )
    policy = ActionListPolicy(
        actions=[
            GraspLinkAction(
                link_path=link_path,
            )
        ]
    )
    task = GraspObj(link_path=link_path)
    evaluation = SingleEnvSimEvaluation(
        env=env,
        task_sampler=TaskSampler(tasks=[task]),
        start_episode=0,
        num_episodes=num_episodes,
        sampler_config=EnvSamplerConfig(
            allow_nondeterminism=True,
            done_on_success=False,
            obs_horizon=0,
            max_time=max_time,
            use_all_subtrajectories=False,
            visualization_fps=12,
            visualization_cam="front",
            # visualization_cam="ur5e/wsg50/wrist_mount",
            visualization_resolution=(400, 400),
            dump_failed_trajectories=True,
        ),
        ctrl_config=ControlConfig(frequency=2, dof=7),
        num_processes=num_processes,
        auto_incr_episode_ids=True,
    )
    stats = evaluation.run(policy=policy, pbar=True, root_path=wandb.run.dir)
    wandb.log(stats)  # type: ignore


if __name__ == "__main__":
    num_processes: int = 6
    benchmark_grasp(
        asset_path="scalingup/environment/mujoco/assets/google_scanned_objects/toys/Android_Lego/model.xml",
        asset_class="lego android toy",
        link_path="lego android toy/|lego android toy/lego android toy",
        num_processes=num_processes,
        num_episodes=100,
    )
    exit()
    benchmark_grasp(
        asset_path="scalingup/environment/mujoco/assets/google_scanned_objects/toys/Schleich_African_Black_Rhino/model.xml",
        asset_class="rhino toy",
        link_path="rhino toy/|rhino toy/rhino toy",
        num_processes=num_processes,
        num_episodes=100,
    )
    benchmark_grasp(
        asset_path="scalingup/environment/mujoco/assets/custom/stick.xml",
        asset_class="stick",
        link_path="stick/|stick/stick",
        num_processes=num_processes,
        num_episodes=100,
    )
    benchmark_grasp(
        asset_path="scalingup/environment/mujoco/assets/google_scanned_objects/toys/Schleich_Hereford_Bull/model.xml",
        asset_class="bull toy",
        link_path="bull toy/|bull toy/bull toy",
        num_processes=num_processes,
        num_episodes=100,
    )
    benchmark_grasp(
        asset_path="scalingup/environment/mujoco/assets/google_scanned_objects/household_items/Wishbone_Pencil_Case/model.xml",
        asset_class="purple pencilcase",
        link_path="purple pencilcase/|purple pencilcase/purple pencilcase",
        num_processes=num_processes,
        num_episodes=100,
    )
    benchmark_grasp(
        asset_path="scalingup/environment/mujoco/assets/google_scanned_objects/toys/Sperry_TopSider_pSUFPWQXPp3/model.xml",
        asset_class="brown women's boot",
        link_path="brown women's boot/|brown women's boot/brown women's boot",
        num_processes=num_processes,
        num_episodes=100,
    )
    benchmark_grasp(
        asset_path="scalingup/environment/mujoco/assets/google_scanned_objects/toys/Thomas_Friends_Wooden_Railway_Porter_5JzRhMm3a9o/model.xml",
        asset_class="porter the tank engine toy",
        link_path="porter the tank engine toy/|porter the tank engine toy/porter the tank engine toy",
        num_processes=num_processes,
        num_episodes=10,
    )
    benchmark_grasp(
        asset_path="scalingup/environment/mujoco/assets/google_scanned_objects/shoes/Womens_Bluefish_2Eye_Boat_Shoe_in_Linen_Natural_Sparkle_Suede_kqi81aojcOR/model.xml",
        asset_class="tan boat shoe",
        link_path="tan boat shoe/|tan boat shoe/tan boat shoe",
        num_processes=num_processes,
        num_episodes=100,
    )
    benchmark_grasp(
        asset_path="scalingup/environment/mujoco/assets/google_scanned_objects/household_items/Womens_Multi_13/model.xml",
        asset_class="vitamin bottle",
        link_path="vitamin bottle/|vitamin bottle/vitamin bottle",
        num_processes=num_processes,
        num_episodes=10,
    )
