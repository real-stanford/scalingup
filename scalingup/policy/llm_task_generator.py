import dataclasses
from functools import partial
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
from rich.console import Console
from rich.syntax import Syntax
from transforms3d import euler
from scalingup.policy.llm_policy_utils import LanguageStateEncoder, split_state_phrase
from scalingup.utils.constants import LINK_SEPARATOR_TOKEN

from scalingup.utils.generic import exec_safe
from scalingup.utils.openai_api import GPT3APIConfig, GPT3Wrapper, Prompt, PromptConfig
from scalingup.utils.core import (
    Action,
    Contact,
    EnvState,
    JointState,
    Observation,
    Pose,
    Task,
    TaskGenerator,
    Trajectory,
)
from scalingup.utils.state_api import (
    check_on_top_of,
    get_pose,
    check_contact,
    check_closed,
    check_opened,
    check_inside,
    check_activated,
    check_deactivated,
)


global_vars = {
    "np": np,
    "euler": euler,
    "dataclasses": dataclasses,
    "Dict": Dict,
    "List": List,
    "Set": Set,
    "Union": Union,
    "Task": Task,
    "Trajectory": Trajectory,
    "Observation": Observation,
    "Action": Action,
    "Pose": Pose,
    "Contact": Contact,
}
RETRIES: int = 10


class MonolithicLLMTaskGenerator(TaskGenerator):
    def __init__(
        self,
        prompt_config: PromptConfig,
        api_config: GPT3APIConfig,
        temperature: float = 0.0,
        seed: int = 0,
    ):
        self.prompt = Prompt(config=prompt_config)
        self.api_config = api_config
        self.temperature = temperature
        self.numpy_random = np.random.RandomState(seed=seed)

    def infer_from_desc(self, task_desc: str, state: EnvState) -> Task:
        usage_prompt = """
# task description: "{task_desc}"
# scene:
#  - UR5: [robotiq_left_finger, robotiq_right_finger]
#  - banana: [banana]
#  - trash_can: [trash_can, trash_can_opening]
#  - trash_can_lid: [trash_can_lid]
#  - fridge: [fridge, top_fridge_shelf, fridge_door, fridge_door_handle]
""".format(
            task_desc=task_desc
        )
        # TODO make scene objects dynamically part of prompt
        prompt, use_query = self.prompt.build_prompt(query=usage_prompt)
        console = Console(record=True)
        for attempt in range(RETRIES):
            try:
                completion = GPT3Wrapper.complete(
                    prompt=prompt,
                    api_config=self.api_config,
                    temperature=self.temperature,
                    numpy_random=self.numpy_random,
                )
                local_vars: Dict[str, Any] = {}
                exec_safe(
                    completion.completion,
                    local_vars=local_vars,
                    global_vars=global_vars,
                )
                with console.capture():
                    console.print(
                        Syntax(
                            code=completion.completion,
                            lexer="python",
                            theme="monokai",
                            line_numbers=True,
                            indent_guides=True,
                        )
                    )
                logging.info(console.export_text(clear=True))
                return local_vars["task"]
            except SyntaxError as e:
                logging.error(e)
                if attempt == RETRIES - 1:
                    raise e
                logging.info("Retrying in 10 seconds...")
                time.sleep(10)
        raise SyntaxError("this code should not be reachable")


class FactorizedLLMTaskGenerator(TaskGenerator):
    """
    Breaks each component of inferring a task out into
    sub tasks, then performing LLM completion on each
    subtask
    """

    template_code_str = """
class {task_class_name}(Task):
    def __init__(self, **kwargs):
        super().__init__(
            desc="{desc}",
            **kwargs
        )

    def check_success(self, traj: Trajectory) -> bool:
        init_state = traj.init_state
        final_state = traj.final_state
{success_fn_code_str}
"""
    RETRIES: int = 10

    def __init__(
        self,
        prompt: Prompt,
        api_config: GPT3APIConfig,
        state_encoder: LanguageStateEncoder,
        remove_with_statement: bool,
        temperature: float,
        seed: int,
    ):
        self.prompt = prompt
        self.state_encoder = state_encoder
        self.api_config = api_config

        self.temperature = temperature
        self.numpy_random = np.random.RandomState(seed=seed)
        self.remove_with_statement = remove_with_statement

    def infer_from_desc(self, task_desc: str, state: EnvState) -> Task:
        if self.remove_with_statement:
            task_desc = split_state_phrase(sentence=task_desc)[1]
        context, context_name_to_link_path = self.state_encoder(state)
        prompt, usage_prompt = self.prompt.build_prompt(
            query="""
# robot task: {task_desc}
{context}
def""".format(
                task_desc=task_desc, context=context
            )
        )

        extra_global_vars = {
            "get_pose": partial(
                get_pose,
                context_name_to_link_path=context_name_to_link_path,
            ),
            "check_contact": partial(
                check_contact,
                context_name_to_link_path=context_name_to_link_path,
            ),
            "check_closed": partial(
                check_closed,
                context_name_to_link_path=context_name_to_link_path,
            ),
            "check_opened": partial(
                check_opened,
                context_name_to_link_path=context_name_to_link_path,
            ),
            "check_inside": partial(
                check_inside,
                context_name_to_link_path=context_name_to_link_path,
            ),
            "check_on_top_of": partial(
                check_on_top_of,
                context_name_to_link_path=context_name_to_link_path,
            ),
            "check_activated": partial(
                check_activated,
                context_name_to_link_path=context_name_to_link_path,
            ),
            "check_deactivated": partial(
                check_deactivated,
                context_name_to_link_path=context_name_to_link_path,
            ),
        }
        for attempt in range(RETRIES):
            task_cls_name = re.sub("[^0-9a-zA-Z]+", "", task_desc.title())
            try:
                success_fn_code_str = GPT3Wrapper.complete(
                    prompt=prompt,
                    api_config=self.api_config,
                    temperature=self.temperature,
                    numpy_random=self.numpy_random,
                ).completion
                full_success_fn_code_str = usage_prompt + success_fn_code_str

                console = Console(record=True)
                with console.capture():
                    console.print(
                        Syntax(
                            code=full_success_fn_code_str,
                            lexer="python",
                            theme="monokai",
                            line_numbers=True,
                            indent_guides=True,
                        )
                    )
                logging.info(console.export_text(clear=True))

                success_fn_code_str = success_fn_code_str.split("):")[-1]
                success_fn_code_str = re.sub("\n+", "\n", success_fn_code_str)
                success_fn_code_str = success_fn_code_str.replace("\n", "\n    ")
                code_str = FactorizedLLMTaskGenerator.template_code_str.format(
                    task_class_name=task_cls_name,
                    desc=task_desc,
                    success_fn_code_str=success_fn_code_str,
                )
                local_vars: Dict[str, Any] = {}
                exec_safe(
                    code_str=code_str,
                    global_vars={**global_vars, **extra_global_vars},
                    local_vars=local_vars,
                )
                task_cls = local_vars[task_cls_name]
                return task_cls(
                    info={
                        "inferred_success_fn_code_str": success_fn_code_str.replace(
                            "\n      ", "\n"
                        )
                    }
                )
            except SyntaxError as e:
                logging.error(e)
                if attempt == RETRIES - 1:
                    raise e
                logging.info("Retrying in 10 seconds...")
                time.sleep(10)
        raise SyntaxError("this code should not be reachable")
