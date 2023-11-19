from utils import set_up_env
from scalingup.utils.core import (
    GraspLinkAction,
    EnvSampler,
    ActionListPolicy,
    GraspObj,
    PolicyTaskAction,
)
from scalingup.utils.generic import setup_logger
from mujoco import viewer

if __name__ == "__main__":
    setup_logger()
    """
    task sampler picks a random task in the domain, which is `catapult` in this case.
    env sampler runs an environment with a policy and task to generate a trajectory.

    this script also demonstrates how to add a real time GUI viewer.
    """
    env, task_sampler, env_sampler_config = set_up_env("catapult")
    # dm_control's data and model is a wrapper around MuJoCo's data and model
    # so you have to call `.ptr` to get the underlying MuJoCo's data and model
    viewer_handle = viewer.launch_passive(
        env.mj_physics.model.ptr, env.mj_physics.data.ptr
    )

    # register the viewer sync callback to the environment
    def sync_viewer(env):
        viewer_handle.sync()

    env.step_fn_callbacks["viewer_sync_callback"] = (
        24,  # frames per second
        sync_viewer,
    )``
    env_sampler = EnvSampler(env=env, task_sampler=task_sampler)
    # `policy` could be a diffusion policy, a language model, or simply a list of actions
    # to be executed in an open loop manner
    open_loop_grasp_action = GraspLinkAction(
        link_path="yellow_block/|yellow_block/yellow_block"
    )
    open_loop_grasp_policy = ActionListPolicy(actions=[open_loop_grasp_action])
    env_sample = env_sampler.sample(
        episode_id=1,  # everything is fully-deterministic, using numpy random seeds
        policy=open_loop_grasp_policy,
        config=env_sampler_config,
    )
    trajectory = env_sample.trajectory
    trajectory.dump_video(
        output_path="open_loop_grasp_block.mp4",
    )
    closed_loop_grasp_action = PolicyTaskAction(
        policy=open_loop_grasp_policy,
        retry_until_success=True,
        task=GraspObj(link_path="yellow_block/|yellow_block/yellow_block"),
    )
    closed_loop_grasp_policy = ActionListPolicy(actions=[closed_loop_grasp_action])
    env_sample = env_sampler.sample(
        episode_id=0,  # everything is fully-deterministic, using numpy random seeds
        policy=closed_loop_grasp_policy,
        config=env_sampler_config,
    )
    trajectory = env_sample.trajectory
    trajectory.dump_video(
        output_path="closed_loop_grasp_block.mp4",
    )
