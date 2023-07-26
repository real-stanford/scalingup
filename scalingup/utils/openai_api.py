from __future__ import annotations

import dataclasses
import difflib
import logging
import os
import pickle
import re
import time
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Set, Tuple
from datetime import datetime
import hashlib
import numpy as np
import openai
import rich
from filelock import FileLock
from openai.error import APIConnectionError, RateLimitError, APIError
from openai.openai_object import OpenAIObject
from pydantic.dataclasses import dataclass
from pydantic import validator
from rich.columns import Columns
from rich.panel import Panel
from scipy.special import softmax

from scalingup import root_dir


def str_diff(a: str, b: str):
    annotate_text_a = ""
    annotate_text_b = ""
    summary = []
    for i, s in enumerate(difflib.ndiff(a, b)):
        if s[0] == " ":
            annotate_text_a += a[i]
            annotate_text_b += b[i]
            continue
        elif s[0] == "-":
            if i < len(a):
                annotate_text_a += f"[red]{a[i]}[/red]"
            if i < len(b):
                annotate_text_b += b[i]
            summary.append('Delete "{}" from position {}'.format(s[-1], i))
        elif s[0] == "+":
            if i < len(a):
                annotate_text_a += a[i]
            if i < len(b):
                annotate_text_b += f"[red]{b[i]}[/red]"
            summary.append('Add "{}" to position {}'.format(s[-1], i))
    rich.print(Columns([Panel(annotate_text_a), Panel(annotate_text_b)]))
    return "\n".join(summary)


class GPT3Completion:
    def __init__(self, path: Sequence[GPT3CompletionNode]):
        self.__path = path

    @property
    def logprob(self) -> float:
        return sum(node.logprob for node in self.__path)

    @property
    def completion(self) -> str:
        return "".join(node.text for node in self.__path)

    def __str__(self) -> str:
        output = "GPT3Completion(\n"
        output += f"\tlen(path): {len(self.__path)}\n"
        output += f"\tlen(completion): {len(self.completion)}\n"
        output += f"\tlogprob: {self.logprob}\n"
        output += f"\tcompletion: `{r'{}'.format(self.completion)}`)"
        return output


@dataclass(frozen=True)
class Token:
    val: str
    logprob: float

    @staticmethod
    def to_string(sequence: Sequence[Token]):
        return "".join(token.val for token in sequence)


@dataclass(frozen=True)
class TokenOptions:
    options: Set[Token]

    def get_branches(self, cutoff: float) -> Set[Token]:
        options = list(self.options)
        token_logprobs = [option.logprob for option in options]
        token_probs = softmax(token_logprobs)
        return {
            token
            for token, token_prob in zip(options, token_probs)
            if token_prob >= cutoff
        }


class GPT3CompletionNode:
    cutoff_prob: float = 1.0
    # track issue with codex at https://github.com/openai/openai-python/issues/133

    def __init__(
        self,
        tokens: Sequence[Token],
        children: Sequence[GPT3CompletionNode],
    ):
        assert len(tokens) > 0
        self.__children = children
        self.__tokens = tokens

    @property
    def logprob(self) -> float:
        return sum(token.logprob for token in self.__tokens)

    @property
    def text(self) -> str:
        return "".join(token.val for token in self.__tokens)

    def get_completions(
        self, init_path: Optional[Sequence[GPT3CompletionNode]] = None
    ) -> List[GPT3Completion]:
        if len(self.__children) > 0:
            # middle level node
            results: List[GPT3Completion] = []
            for child in self.__children:
                results.extend(
                    child.get_completions(
                        init_path=list(init_path) + [self]
                        if init_path is not None
                        else [self]
                    )
                )
            return results
        else:
            # leaf node
            return [
                GPT3Completion(
                    path=list(init_path) + [self] if init_path is not None else [self]
                )
            ]

    def sample(
        self, numpy_random: np.random.RandomState, temperature: float = 0.0
    ) -> GPT3Completion:
        completions = self.get_completions()
        logprobs = np.array([completion.logprob for completion in completions]).astype(
            np.float64
        )
        if temperature == 0.0:
            return completions[np.argmax(logprobs)]
        choice = numpy_random.choice(
            len(completions), p=softmax(logprobs / temperature)  # type: ignore
        )
        return completions[choice]

    @classmethod
    def build_tree(
        cls,
        prompt: str,
        api_config: GPT3APIConfig,
        init_node_tokens: Optional[Sequence[Token]] = None,
    ) -> GPT3CompletionNode:
        """
        Recursively build a tree of GPT3CompletionNodes, branching at every
        token with probability cutoff higher than `GPT3CompletionNode.cutoff_prob`
        """
        prompt += Token.to_string(
            init_node_tokens if init_node_tokens is not None else []
        )
        response = GPT3Wrapper.request(
            prompt=prompt,
            api_config=api_config,
        )
        res_logprobs = response["choices"][0]["logprobs"]
        n_prompt_tokens: int = response["usage"]["prompt_tokens"]
        n_completion_tokens: int = response["usage"]["completion_tokens"]
        prompt_tokens: Sequence[Token] = [
            Token(
                val=res_logprobs["tokens"][i],
                logprob=res_logprobs["token_logprobs"][i]
                if i != 0
                else 0.0,  # first token has logprob None
            )
            for i in range(n_prompt_tokens)
        ]
        assert prompt == Token.to_string(prompt_tokens), str_diff(
            a=prompt, b=Token.to_string(prompt_tokens)
        )
        completion_tokens: list[Token] = [
            Token(
                val=res_logprobs["tokens"][i], logprob=res_logprobs["token_logprobs"][i]
            )
            for i in range(
                n_prompt_tokens,
                n_prompt_tokens + n_completion_tokens,
            )
        ]
        assert (
            Token.to_string(prompt_tokens) + Token.to_string(completion_tokens)
            == response["choices"][0]["text"]
        ), str_diff(
            a=Token.to_string(prompt_tokens) + Token.to_string(completion_tokens),
            b=response["choices"][0]["text"],
        )

        completion_options: list[TokenOptions] = [
            TokenOptions(
                options={
                    Token(val=val, logprob=logprob)
                    for val, logprob in res_logprobs["top_logprobs"][i].items()
                }
            )
            for i in range(n_prompt_tokens, n_prompt_tokens + n_completion_tokens)
        ]
        completion_branches = [
            completion_option.get_branches(cutoff=cls.cutoff_prob)
            for completion_option in completion_options
        ]
        assert len(completion_tokens) == len(completion_options)

        node_tokens: List[Token] = []
        children: List[GPT3CompletionNode] = []

        for completion_token, branches in zip(completion_tokens, completion_branches):
            if len(branches) > 1:
                for branch_token in branches:
                    try:
                        children.append(
                            cls.build_tree(
                                prompt=deepcopy(prompt + Token.to_string(node_tokens)),
                                init_node_tokens=[branch_token],
                                api_config=api_config,
                            )
                        )
                    except AssertionError as e:
                        logging.warning(e)
                        continue
                break
            else:
                node_tokens.append(completion_token)
        if init_node_tokens is not None:
            node_tokens = list(init_node_tokens) + node_tokens
        return GPT3CompletionNode(
            tokens=node_tokens,
            children=children,
        )

    def __iter__(self):
        self.__completion_iter = iter(self.get_completions())
        return self

    def __next__(self):
        return next(self.__completion_iter)


class GPT3Wrapper:
    """
    Wrapper around OpenAI API with additional functionality
    for caching and branching
    """

    MIN_COMPLETIONS_PER_PROMPT = 1
    USE_CACHE = True
    TIMEOUT = 10
    cache_path: str = f"{root_dir}/responses"
    readtime: datetime = datetime.utcfromtimestamp(0.0)

    @classmethod
    def has_key(cls, key: str) -> bool:
        path = os.path.join(cls.cache_path, key + ".pkl")
        return os.path.exists(path)

    @classmethod
    def add_value(cls, key: str, value: OpenAIObject):
        # if folder doesn't exist yet, then create it
        if not os.path.exists(cls.cache_path):
            os.makedirs(cls.cache_path)
        path = os.path.join(cls.cache_path, key + ".pkl")
        # read it
        cache = []
        if cls.has_key(key):
            with open(path, "rb") as f:
                cache = pickle.load(f)
        cache.append(value)
        with open(path, "wb") as f:
            pickle.dump(cache, f)

    @classmethod
    def get_values(cls, key: str) -> List[OpenAIObject]:
        path = os.path.join(cls.cache_path, key + ".pkl")
        # read it
        cache = []
        if cls.has_key(key):
            try:
                with open(path, "rb") as f:
                    cache = pickle.load(f)
            except pickle.UnpicklingError:
                logging.warning(f"Failed to unpickle {path!r}")
                os.remove(path)
        return cache

    @classmethod
    def get_last_value(cls, key: str) -> OpenAIObject:
        return cls.get_values(key)[-1]

    @classmethod
    def add_request(cls, key: str, prompt: str, api_config: GPT3APIConfig):
        while True:
            try:
                if api_config.echo is None:
                    api_config.echo = api_config.max_tokens == 0
                response: OpenAIObject = openai.Completion.create(
                    prompt=prompt,
                    logprobs=5,
                    **dataclasses.asdict(api_config),
                )  # type: ignore
                if (
                    any(
                        completion["finish_reason"] == "length"
                        for completion in response["choices"]
                    )
                    and api_config.max_tokens > 0
                ):
                    logging.error(
                        "Completion truncated because max tokens not long enough"
                    )
                    exit()
                cls.add_value(key=key, value=response)
                return
            except (RateLimitError, APIConnectionError, APIError) as e:
                logging.warning(f"OpenAI API got err {e}")
                logging.warning(f"Retrying after {GPT3Wrapper.TIMEOUT}s.")
                time.sleep(GPT3Wrapper.TIMEOUT)

    @classmethod
    def request(
        cls,
        prompt: str,
        api_config: GPT3APIConfig,
    ) -> OpenAIObject:
        m = hashlib.sha256()
        m.update(prompt.encode("utf-8"))
        m.update(str(api_config).encode("utf-8"))
        key = m.hexdigest()
        cls.add_request(key=key, prompt=prompt, api_config=api_config)
        return cls.get_last_value(key=key)

    @classmethod
    def complete(
        cls,
        prompt: str,
        api_config: GPT3APIConfig,
        numpy_random: np.random.RandomState,
        temperature: float = 1.0,
    ) -> GPT3Completion:
        if not cls.USE_CACHE:
            while True:
                try:
                    response: OpenAIObject = openai.Completion.create(
                        prompt=prompt,
                        logprobs=5,
                        **{
                            **dataclasses.asdict(api_config),
                            "echo": api_config.max_tokens > 0,
                        },
                    )  # type: ignore
                    return GPT3CompletionNode(
                        tokens=[Token(val="", logprob=0.0)],
                        children=[
                            GPT3CompletionNode(
                                tokens=[
                                    Token(
                                        val=response["choices"][0]["text"],
                                        logprob=sum(
                                            filter(
                                                lambda logprob: logprob is not None,
                                                response["choices"][0]["logprobs"][
                                                    "token_logprobs"
                                                ],
                                            )
                                        ),
                                    )
                                ],
                                children=[],
                            )
                        ],
                    ).sample(numpy_random=numpy_random)
                except (RateLimitError, APIConnectionError) as e:
                    logging.warning(f"OpenAI API got err {e}")
                    logging.warning(f"Retrying after {GPT3Wrapper.TIMEOUT}s.")
                    time.sleep(GPT3Wrapper.TIMEOUT)
        # Until Github issue is fixed, use next best alternative
        api_config.echo = False
        m = hashlib.sha256()
        m.update(prompt.encode("utf-8"))
        m.update(str(api_config).encode("utf-8"))
        key = m.hexdigest()
        while (
            not cls.has_key(key=key)
            or len(cls.get_values(key=key)) < GPT3Wrapper.MIN_COMPLETIONS_PER_PROMPT
        ):
            cls.add_request(key=key, prompt=prompt, api_config=api_config)
        completions: Dict[str, float] = {}
        for openai_obj in cls.get_values(key=key):
            # remove multiple new lines, which are equivalent
            text = re.sub("\n{2,}", "\n\n", openai_obj["choices"][0]["text"])
            logprob = sum(
                filter(
                    lambda logprob: logprob is not None,
                    openai_obj["choices"][0]["logprobs"]["token_logprobs"],
                )
            )
            if text not in completions and not text.isspace():
                completions[text] = logprob
        completion_tree = GPT3CompletionNode(
            tokens=[Token(val="", logprob=0.0)],
            children=[
                GPT3CompletionNode(
                    tokens=[
                        Token(
                            val=text,
                            logprob=logprob,
                        )
                    ],
                    children=[],
                )
                for text, logprob in completions.items()
            ],
        )
        return completion_tree.sample(temperature=temperature, numpy_random=numpy_random)


@dataclass
class PromptConfig:
    base_prompt: str
    query_prefix: str = ""
    query_suffix: str = ""
    context_prefix: str = ""
    context_suffix: str = ""
    maintain_session: bool = False


class Prompt:
    def __init__(self, config: PromptConfig, exec_hist: str = ""):
        self.config = config
        self.exec_hist = exec_hist

    def clear_exec_hist(self) -> None:
        self.exec_hist = ""

    def build_prompt(self, query: str, context: str = "") -> Tuple[str, str]:
        prompt = deepcopy(self.config.base_prompt)

        if self.config.maintain_session:
            prompt += f"\n{self.exec_hist}"

        use_query = self.config.query_prefix
        use_query += query
        use_query += self.config.query_suffix
        prompt += f"\n{use_query}"

        if context != "":
            prompt += (
                f"\n{self.config.context_prefix}{context}{self.config.context_suffix}"
            )

        return prompt, use_query

    def __iadd__(self, other: str) -> Prompt:
        self.exec_hist += other
        return self

    def __add__(self, other: str) -> Prompt:
        return Prompt(config=self.config, exec_hist=self.exec_hist + other)


@dataclass
class GPT3APIConfig:
    stop: List[str] = dataclasses.field(default_factory=lambda: list("#"))
    engine: str = "text-davinci-003"
    temperature: float = 0.0
    max_tokens: int = 512
    echo: Optional[bool] = None
    frequency_penalty: float = 0.0

    @validator("frequency_penalty")
    @classmethod
    def valid_frequency_penalty_range(cls, v: float):
        if v < -2.0 or v > 2.0:
            raise ValueError(
                "`frequency_penalty` must be between -2.0 and 2.0: "
                + "https://beta.openai.com/docs/api-reference/completions/"
                + "create#completions/create-frequency_penalty"
            )
        return v
