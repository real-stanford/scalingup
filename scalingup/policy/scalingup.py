from __future__ import annotations
from scalingup.policy.llm_policy_utils import HierachicalStep, LanguageStateEncoder
from scalingup.utils.core import (
    Action,
    Observation,
    Policy,
    PolicyTaskAction,
    RayPolicy,
    Task,
)


class ScalingUpDataGen(Policy):
    def __init__(
        self,
        task_tree_inference: HierachicalStep,
        state_encoder: LanguageStateEncoder,
        **kwargs,
    ):
        self.task_tree_inference = task_tree_inference
        self.state_encoder = state_encoder

    def __str__(self):
        if self.task_tree_inference.task_policy is None:
            return "ScalingUpDataGen()"
        task_policy = self.task_tree_inference.task_policy
        policy_str = str(task_policy)
        supported_tasks = task_policy.get_supported_tasks()
        if len(supported_tasks) == 0:
            task_support_str = "no tasks"
        else:
            task_support_str = ", ".join(
                f"{task_desc!r}" for task_desc in sorted(supported_tasks)
            )
        return f"ScalingUpDataGen({policy_str} for {task_support_str})"

    def _get_action(
        self,
        obs: Observation,
        task: Task,
    ) -> Action:
        context, context_name_to_link_path = self.state_encoder(obs.state)

        task_tree = self.task_tree_inference(
            query=task.desc,
            context=context,
            state=obs.state,
            context_name_to_link_path=context_name_to_link_path,
        )
        return PolicyTaskAction(
            policy=task_tree.get_policy(),
            task=task_tree.task,
            retry_until_success=self.task_tree_inference.retry_until_success,
        )
