from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import typing

import numpy as np
from rich.tree import Tree as RichTree
from scalingup.utils.constants import LINK_SEPARATOR_TOKEN, MJCF_NEST_TOKEN
from scalingup.utils.generic import AllowArbitraryTypes
from scalingup.utils.openai_api import GPT3APIConfig, GPT3Wrapper, Prompt, PromptConfig
from scalingup.utils.core import (
    Action,
    ActionListPolicy,
    EnvState,
    GraspLinkAction,
    GraspObj,
    JointType,
    PlaceOnLinkAction,
    Policy,
    PolicyTaskAction,
    PrismaticJointAction,
    RevoluteJointAction,
    Task,
    TaskGenerator,
    split_state_phrase,
)
from pydantic import dataclasses
from abc import ABC, abstractmethod


@dataclasses.dataclass
class LanguageStateEncoder:
    include_robot: bool = False
    include_only_robot_ee_links: bool = True
    line_prefix: str = ""
    indent_str: str = "   "
    obj_bullet: str = " -"
    link_bullet: str = " +"

    def __call__(
        self,
        state: EnvState,
    ) -> Tuple[str, Dict[str, str]]:
        output: str = "scene:"
        context_name_to_link_path: Dict[str, str] = {}
        for obj_name, obj_state in state.object_states.items():
            if obj_name == "world":
                continue
            if "ur5" in obj_name.lower() and not self.include_robot:
                continue
            output += "\n"
            llm_obj_name = " ".join(obj_name.split("_")).split(MJCF_NEST_TOKEN)[0]

            if "ur5" in obj_name.lower() and self.include_only_robot_ee_links:
                # special case for handling robot
                obj_name = "UR5"
                link_paths = [
                    f"UR5{LINK_SEPARATOR_TOKEN}robotiq_left_finger",
                    f"UR5{LINK_SEPARATOR_TOKEN}robotiq_right_finger",
                ]
                context_name_to_link_path["robotiq left finger"] = link_paths[0]
                context_name_to_link_path["robotiq right finger"] = link_paths[1]
            else:
                link_paths = list(obj_state.link_states.keys())
                if obj_name[-1] != MJCF_NEST_TOKEN:
                    raise NotImplementedError("Only support nested objects for now")
                # nested as a result of using mjcf
                link_paths = [
                    "".join(link_name.split(obj_name)[2:])
                    for link_name in link_paths
                    if len("".join(link_name.split(obj_name)[2:])) > 0
                ]
                for link_path in link_paths:
                    link_name = link_path.split(LINK_SEPARATOR_TOKEN)[-1]
                    link_name = " ".join(link_name.split("_"))
                    link_path = (
                        obj_name
                        + LINK_SEPARATOR_TOKEN
                        + LINK_SEPARATOR_TOKEN.join(
                            obj_name + sublink_path
                            for sublink_path in link_path.split(LINK_SEPARATOR_TOKEN)
                        )
                    )
                    assert link_name not in context_name_to_link_path
                    context_name_to_link_path[link_name] = link_path
                    if not link_name.startswith(obj_name):
                        context_name_to_link_path[
                            llm_obj_name + " " + link_name
                        ] = link_path

            output += f"{self.obj_bullet} {llm_obj_name}"
            for link_path in link_paths:
                link_path = " ".join(link_path.split("_"))
                if link_path == llm_obj_name:
                    #  root link
                    continue
                level = len(link_path.split(LINK_SEPARATOR_TOKEN)) - 1
                link_name = link_path.split(LINK_SEPARATOR_TOKEN)[-1]
                output += f"\n{self.indent_str*level}{self.link_bullet} {link_name}"
        return (
            self.line_prefix + ("\n" + self.line_prefix).join(output.split("\n")),
            context_name_to_link_path,
        )


class LLMCompletionStep:
    def __init__(
        self,
        prompt_config: PromptConfig,
        api_config: GPT3APIConfig,
        temperature: float = 0.0,
    ):
        self.api_config = api_config
        self.prompt_config = prompt_config
        self.temperature = temperature

    def process_output(
        self,
        query: str,
        context: str,
        completion: str,
        state: EnvState,
        context_name_to_link_path: Dict[str, str],
    ) -> Any:
        return completion

    def __call__(
        self,
        query: str,
        context: str,
        state: EnvState,
        context_name_to_link_path: Dict[str, str],
    ) -> Any:
        prompt_str, usage = Prompt(config=self.prompt_config).build_prompt(
            query=query, context=context
        )
        completion = GPT3Wrapper.complete(
            prompt=prompt_str,
            api_config=self.api_config,
            temperature=self.temperature,
            numpy_random=np.random.RandomState(),
        )
        return self.process_output(
            query, context, completion.completion, state, context_name_to_link_path
        )


class LLMMultipleChoiceCompletionStep(LLMCompletionStep):
    def __init__(self, api_config: GPT3APIConfig, **kwargs):
        assert api_config.max_tokens == 0, "MCQ LLM should not be doing completion"
        super().__init__(api_config=api_config, **kwargs)

    @abstractmethod
    def get_choices(
        self,
        query: str,
        context: str,
        state: EnvState,
        context_name_to_link_path: Dict[str, str],
    ) -> Dict[Any, str]:
        """
        Mapping from each option's return value to its prompt
        """
        raise NotImplementedError()

    def process_multiple_choice_output(
        self,
        query: str,
        context: str,
        state: EnvState,
        context_name_to_link_path: Dict[str, str],
        multiple_choice_output: Dict[Any, float],
    ):
        return max(multiple_choice_output.items(), key=lambda item: item[1])[0]

    def __call__(
        self,
        query: str,
        context: str,
        state: EnvState,
        context_name_to_link_path: Dict[str, str],
    ) -> Any:
        choices = self.get_choices(
            query=query,
            context=context,
            state=state,
            context_name_to_link_path=context_name_to_link_path,
        )
        multiple_choice_output = {
            choice_return_value: GPT3Wrapper.complete(
                prompt=choice_prompt_str,
                api_config=self.api_config,
                temperature=self.temperature,
                numpy_random=np.random.RandomState(),
            ).logprob
            for choice_return_value, choice_prompt_str in choices.items()
        }
        return self.process_multiple_choice_output(
            query=query,
            context=context,
            state=state,
            context_name_to_link_path=context_name_to_link_path,
            multiple_choice_output=multiple_choice_output,
        )


class PlannerStep(LLMCompletionStep):
    def process_output(
        self,
        query: str,
        context: str,
        completion: str,
        state: EnvState,
        context_name_to_link_path: Dict[str, str],
    ) -> List[str]:
        steps = completion.split("\nanswer:")[-1].split("\n")
        logging.info(f"planning query: {query!r}")
        for step in steps:
            logging.info(step)
        return ["".join(step.split(". ")[1:]) for step in steps if ". " in step]


class ObjectPartIdentifier(LLMCompletionStep):
    def process_output(
        self,
        query: str,
        context: str,
        completion: str,
        state: EnvState,
        context_name_to_link_path: Dict[str, str],
    ) -> str:
        obj_part = (
            completion.split(self.prompt_config.context_suffix)[-1]
            .split(".")[0]
            .rstrip()
            .lstrip()
        )
        return obj_part


class PickAndPlaceParser(LLMMultipleChoiceCompletionStep):
    def __init__(
        self,
        prompt_config: PromptConfig,
        **kwargs,
    ):
        super().__init__(prompt_config=prompt_config, **kwargs)

    def process_output(
        self,
        query: str,
        context: str,
        completion: str,
        state: EnvState,
        context_name_to_link_path: Dict[str, str],
    ) -> Tuple[str, str]:
        pick = completion.split("pick:")[-1].split(".")[0]
        place = completion.split("place:")[-1].split(".")[0]
        return pick.rstrip().lstrip(), place.rstrip().lstrip()

    def get_choices(
        self,
        query: str,
        context: str,
        state: EnvState,
        context_name_to_link_path: Dict[str, str],
    ) -> Dict[Any, str]:
        raise NotImplementedError()

    def __call__(
        self,
        query: str,
        context: str,
        state: EnvState,
        context_name_to_link_path: Dict[str, str],
        pick_obj: Optional[str] = None,
    ) -> Tuple[str, str]:
        # determine pick location
        prompt_str, usage = Prompt(config=self.prompt_config).build_prompt(
            query=query, context=context
        )
        if pick_obj is None:
            options = {}
            for pick_obj in context_name_to_link_path.keys():
                options[pick_obj] = prompt_str + f" {pick_obj}."
            multiple_choice_output = {
                choice_return_value: GPT3Wrapper.complete(
                    prompt=choice_prompt_str,
                    api_config=self.api_config,
                    temperature=self.temperature,
                    numpy_random=np.random.RandomState(),
                ).logprob
                for choice_return_value, choice_prompt_str in options.items()
            }
            pick_obj = self.process_multiple_choice_output(
                query=query,
                context=context,
                state=state,
                context_name_to_link_path=context_name_to_link_path,
                multiple_choice_output=multiple_choice_output,
            )
        assert pick_obj is not None
        # determine pick location
        options = {}
        for place_obj in context_name_to_link_path.keys():
            if pick_obj == place_obj:
                continue
            options[place_obj] = prompt_str + f" {pick_obj}.\nplace: {place_obj}."
        multiple_choice_output = {
            choice_return_value: GPT3Wrapper.complete(
                prompt=choice_prompt_str,
                api_config=self.api_config,
                temperature=self.temperature,
                numpy_random=np.random.RandomState(),
            ).logprob
            for choice_return_value, choice_prompt_str in options.items()
        }
        place_obj = self.process_multiple_choice_output(
            query=query,
            context=context,
            state=state,
            context_name_to_link_path=context_name_to_link_path,
            multiple_choice_output=multiple_choice_output,
        )
        return pick_obj, place_obj


class AmbiguousTaskHandler(LLMCompletionStep):
    def __init__(
        self,
        remove_with_statement: bool = True,
        **kwargs,
    ):
        self.remove_with_statement = remove_with_statement
        super().__init__(**kwargs)

    def __call__(
        self,
        query: str,
        context: str,
        state: EnvState,
        context_name_to_link_path: Dict[str, str],
    ) -> Any:
        with_statement = ""
        if self.remove_with_statement:
            with_statement, query = split_state_phrase(query)
        completion = super().__call__(
            query=query,
            context=context,
            state=state,
            context_name_to_link_path=context_name_to_link_path,
        )
        return with_statement + completion

    def process_output(
        self,
        query: str,
        context: str,
        completion: str,
        state: EnvState,
        context_name_to_link_path: Dict[str, str],
    ) -> str:
        specific_task_desc = completion.split("answer:")[-1].split(".")[0].strip()
        logging.info(f"potentially ambiguous query: {query!r}")
        logging.info(f"specific query: {specific_task_desc!r}")
        return specific_task_desc


class OneOrMultipleObj(LLMCompletionStep):
    def process_output(
        self,
        query: str,
        context: str,
        completion: str,
        state: EnvState,
        context_name_to_link_path: Dict[str, str],
    ) -> bool:
        logging.info(f"one or multiple ({query}): {completion}")
        return "multiple" in completion.split("answer:")[-1]


class HierachicalStep:
    def __init__(
        self,
        task_generator: TaskGenerator,
        planner: PlannerStep,
        ambiguous_task_handler: AmbiguousTaskHandler,
        one_or_multiple: OneOrMultipleObj,
        plan_grounder: PlanGrounder,
        retry_until_success: bool,
        parent_node: Optional[TaskTreeNode] = None,
        task_policy: Optional[Policy] = None,
    ):
        self.parent_node = parent_node
        self.task_generator = task_generator
        self.planner = planner
        self.ambiguous_task_handler = ambiguous_task_handler
        self.one_or_multiple = one_or_multiple
        self.task_policy = task_policy
        self.plan_grounder = plan_grounder
        assert self.plan_grounder.retry_until_success == retry_until_success
        self.retry_until_success = retry_until_success

    def __call__(
        self,
        query: str,
        context: str,
        state: EnvState,
        context_name_to_link_path: Dict[str, str],
    ):
        query = self.ambiguous_task_handler(
            query=query,
            context=context,
            state=state,
            context_name_to_link_path=context_name_to_link_path,
        )
        task = self.task_generator.infer_from_desc(task_desc=query, state=state)
        if self.task_policy is not None and self.task_policy.is_task_supported(
            task_desc=task.desc
        ):
            return ActionTreeNode(
                task=task,
                children=[],
                policy=self.task_policy,
                retry_until_success=self.retry_until_success,
            )
        is_multiobj = self.one_or_multiple(
            query=query,
            context=context,
            state=state,
            context_name_to_link_path=context_name_to_link_path,
        )

        current_task_node = TaskTreeNode(
            task=task, children=[], retry_until_success=self.retry_until_success
        )
        logging.debug(query)
        if is_multiobj:
            subtasks: List[str] = self.planner(
                query=query,
                context=context,
                state=state,
                context_name_to_link_path=context_name_to_link_path,
            )
            if query in subtasks and len(subtasks) == 1:
                logging.error(f"{query!r} is recursive, manually stopping it")
                is_multiobj = False
            else:
                assert query not in subtasks, f"{query} is in {subtasks}"
                for subtask_description in subtasks:
                    subtask_inference = HierachicalStep(
                        parent_node=current_task_node,
                        planner=self.planner,
                        task_generator=self.task_generator,
                        ambiguous_task_handler=self.ambiguous_task_handler,
                        one_or_multiple=self.one_or_multiple,
                        plan_grounder=self.plan_grounder,
                        task_policy=self.task_policy,
                        retry_until_success=self.retry_until_success,
                    )
                    subtask_inference(
                        query=subtask_description,
                        context=context,
                        state=state,
                        context_name_to_link_path=context_name_to_link_path,
                    )
        if not is_multiobj:
            current_task_node = self.plan_grounder(
                query=query,
                context=context,
                state=state,
                context_name_to_link_path=context_name_to_link_path,
            )

        if self.parent_node is not None:
            self.parent_node.children.append(current_task_node)
        return current_task_node


@dataclasses.dataclass(config=AllowArbitraryTypes)
class TaskTreeNode:
    task: Task
    children: List[TaskTreeNode]
    retry_until_success: bool

    def __hash__(self) -> int:
        return hash(tuple(self.children))

    def __eq__(self, __value: object) -> bool:
        if not issubclass(type(__value), TaskTreeNode):
            return False
        other_task_tree = typing.cast(TaskTreeNode, __value)
        return all(c1 == c2 for c1, c2 in zip(self.children, other_task_tree.children))

    def get_policy(self) -> Policy:
        return ActionListPolicy(
            actions=[
                PolicyTaskAction(
                    policy=child.get_policy(),
                    task=child.task,
                    retry_until_success=self.retry_until_success,
                )
                for child in self.children
            ]
        )

    def print_tree(self, node: Optional[RichTree] = None) -> RichTree:
        if node is None:
            node = RichTree(f"[bold green]{self.task.desc}[/bold green]")
        else:
            node = node.add(f"[bold blue]{self.task.desc}[/bold blue]")
        for child in self.children:
            child.print_tree(node=node)
        return node

    def get_child_tasks(self, parent_tasks: Optional[List[Task]] = None) -> List[Task]:
        if parent_tasks is None:
            # current task is root task
            parent_tasks = [self.task]
        for child in self.children:
            parent_tasks.append(child.task)
            child.get_child_tasks(parent_tasks=parent_tasks)
        return parent_tasks

    def find_node(self, criterion: Callable[[TaskTreeNode], bool]) -> TaskTreeNode:
        # a post-order tree traversal
        # used by task sampler
        for child in self.children:
            if criterion(child):
                return child.find_node(criterion=criterion)
        return self


@dataclasses.dataclass(config=AllowArbitraryTypes)
class ActionTreeNode(TaskTreeNode):
    policy: Policy

    def __hash__(self) -> int:
        return hash(self.policy)

    def __eq__(self, __value: object) -> bool:
        if not issubclass(type(__value), ActionTreeNode):
            return False
        other_action_tree_node = typing.cast(ActionTreeNode, __value)
        return self.policy == other_action_tree_node.policy

    def print_tree(self, node: Optional[RichTree] = None) -> RichTree:
        if node is None:
            node = RichTree(f"[bold green]{self.task.desc}[/bold green]")
        else:
            node = node.add(f"[bold blue]{self.task.desc}[/bold blue]")

        if type(self.policy) == ActionListPolicy:
            for a in typing.cast(ActionListPolicy, self.policy).actions:
                node.add(f"[bold]{str(a)}[/bold]")
        else:
            node.add(f"[bold]{str(self.policy)}[/bold]")
        return node

    def get_policy(self) -> Policy:
        return self.policy


class PlanGrounder(ABC):
    def __init__(
        self,
        obj_part_identifier: ObjectPartIdentifier,
        task_generator: TaskGenerator,
        retry_until_success: bool,
    ):
        self.obj_part_identifier = obj_part_identifier
        self.task_generator = task_generator
        self.retry_until_success = retry_until_success

    @abstractmethod
    def link_path_to_action(
        self,
        link_path: str,
        query: str,
        context: str,
        task: Task,
        state: EnvState,
        context_name_to_link_path: Dict[str, str],
    ) -> ActionTreeNode:
        pass

    def __call__(
        self,
        query: str,
        context: str,
        state: EnvState,
        context_name_to_link_path: Dict[str, str],
    ) -> ActionTreeNode:
        logging.info(f"grounding {query!r}")
        object_part: str = self.obj_part_identifier(
            query=query,
            context=context,
            state=state,
            context_name_to_link_path=context_name_to_link_path,
        )
        link_path = context_name_to_link_path[object_part]
        task = self.task_generator.infer_from_desc(task_desc=query, state=state)
        return self.link_path_to_action(
            link_path=link_path,
            task=task,
            query=query,
            context=context,
            state=state,
            context_name_to_link_path=context_name_to_link_path,
        )


class ExplorationPrimitivePlanGrounder(PlanGrounder):
    def __init__(
        self,
        pick_and_place_parser: PickAndPlaceParser,
        action_primitive_mode: bool,
        planar_primitive_only: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pick_and_place_parser = pick_and_place_parser
        self.action_primitive_mode = action_primitive_mode
        self.planar_primitive_only = planar_primitive_only
        if self.planar_primitive_only:
            assert (
                self.action_primitive_mode
            ), "Planar exploration primitives not supported"
        else:
            assert not self.action_primitive_mode, "6 DOF action primitives not supported"

    def handle_pick_and_place(
        self,
        query: str,
        context: str,
        task: Task,
        state: EnvState,
        context_name_to_link_path: Dict[str, str],
        pick_obj: Optional[str] = None,
    ) -> ActionTreeNode:
        pick_obj, place_location = self.pick_and_place_parser(
            query=query,
            context=context,
            state=state,
            context_name_to_link_path=context_name_to_link_path,
            pick_obj=pick_obj,
        )
        return ActionTreeNode(
            task=task,
            children=[],
            policy=ActionListPolicy(
                actions=[
                    PolicyTaskAction(
                        policy=ActionListPolicy(
                            actions=[
                                GraspLinkAction(
                                    link_path=context_name_to_link_path[pick_obj],
                                    pushin_more=True,
                                    action_primitive_mode=self.action_primitive_mode,
                                ),
                            ]
                        ),
                        task=GraspObj(
                            link_path=context_name_to_link_path[pick_obj],
                            desc_template=f"grasp the {pick_obj}",
                        ),
                        retry_until_success=self.retry_until_success,
                    ),
                    PlaceOnLinkAction(
                        link_path=context_name_to_link_path[place_location],
                        action_primitive_mode=self.action_primitive_mode,
                    ),
                ]
            ),
            retry_until_success=self.retry_until_success,
        )

    def handle_revolute_prismatic_joints(
        self,
        link_path: str,
        joint_type: JointType,
        query: str,
        task: Task,
    ) -> ActionTreeNode:
        towards_max = any(
            keyword in split_state_phrase(task.desc)[1]
            for keyword in {
                "open the",
                "turn on the",
                "press the",
                "raise the",
                "lift the",
            }
        )
        object_part = link_path.split(LINK_SEPARATOR_TOKEN)[-1].split(MJCF_NEST_TOKEN)[-1]
        grasp_action = PolicyTaskAction(
            policy=ActionListPolicy(
                actions=[
                    GraspLinkAction(
                        link_path=link_path,
                        with_backup=False,
                        # a surface level grasp as a proxy for a pushing action
                        pushin_more=not any(
                            kw in query for kw in ["push the", "press the"]
                        ),
                        action_primitive_mode=self.action_primitive_mode,
                    )
                ]
            ),
            task=GraspObj(
                link_path=link_path,
                desc_template=f"grasp the {object_part}",
            ),
            retry_until_success=self.retry_until_success,
        )
        joint_action = (
            RevoluteJointAction(
                link_path=link_path,
                towards_max=towards_max,
                rotate_gripper=False if self.planar_primitive_only else None,
            )
            if joint_type == JointType.REVOLUTE
            else PrismaticJointAction(link_path=link_path, towards_max=towards_max)
        )
        actions = [grasp_action, joint_action]
        return ActionTreeNode(
            task=task,
            children=[],
            policy=ActionListPolicy(actions=actions),
            retry_until_success=self.retry_until_success,
        )

    def link_path_to_action(
        self,
        link_path: str,
        query: str,
        context: str,
        task: Task,
        state: EnvState,
        context_name_to_link_path: Dict[str, str],
    ) -> ActionTreeNode:
        joint_type = state.get_link_joint_type(link_path=link_path)
        # from here, just do simple text processing
        if joint_type == JointType.FREE:
            return self.handle_pick_and_place(
                query=query,
                context=context,
                task=task,
                state=state,
                context_name_to_link_path=context_name_to_link_path,
            )
        else:
            return self.handle_revolute_prismatic_joints(
                joint_type=joint_type,
                link_path=link_path,
                query=query,
                task=task,
            )
