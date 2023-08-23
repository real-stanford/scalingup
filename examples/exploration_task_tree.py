from utils import set_up_env, set_up_policy
from scalingup.utils.core import (
    GraspLinkAction,
    EnvSampler,
    ActionListPolicy,
    GraspObj,
    PolicyTaskAction,
)
from scalingup.utils.generic import setup_logger
import rich

if __name__ == "__main__":
    setup_logger()
    # this will create the data generation policy from our paper
    # complete with verify & retry and the 6 DoF exploration primitives
    datagen_policy = set_up_policy(policy_config_name="scalingup")
    # task_name = "drawer"
    task_name = "bin_transport_train"
    env, task_sampler, env_sampler_config = set_up_env(task_name)
    obs = env.reset(episode_id=0)
    task = task_sampler.sample(obs=obs, seed=0)
    # state encoder simply parses through the env state to output a
    # bullet list of objects and the articulation structure so the
    # llm knows which objects (parts) are in the scene and how they
    # are connected
    context, context_name_to_link_path = datagen_policy.state_encoder(obs.state)
    # this is the part where the LLM is used to recursively infer the
    # exploration task tree
    task_tree = datagen_policy.task_tree_inference(
        query=task.desc,
        context=context,
        state=obs.state,
        context_name_to_link_path=context_name_to_link_path,
    )
    rich.print(task_tree)

    env_sampler = EnvSampler(env=env, task_sampler=task_sampler)
    env_sample = env_sampler.sample(
        episode_id=0,
        policy=datagen_policy,
        config=env_sampler_config,
    )
    env_sample.trajectory.dump_video(
        output_path="exploration_task_tree.mp4",
    )
