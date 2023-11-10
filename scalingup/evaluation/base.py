import logging
from abc import ABC, abstractmethod
from itertools import cycle
from time import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import ray
from pytorch_lightning import Trainer as LightningTrainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from rich.console import Console
from rich.table import Table as RichTable

import wandb
from scalingup.algo.algo import ScalingUpAlgo
from scalingup.utils.ray import wait_with_pbar
from scalingup.utils.core import (
    Action,
    ControlConfig,
    Env,
    EnvSample,
    EnvSampler,
    EnvSamplerConfig,
    Policy,
    Task,
    TaskSampler,
)
from wandb import Histogram  # type: ignore
from wandb import Table as WandbTable  # type: ignore
import torch
import pydantic


@pydantic.dataclasses.dataclass(frozen=True)
class EvalCriteria:
    metric: str  # which metric in `trainer.logged_metrics` to use
    value: float  # value to compare against
    maximize: bool  # whether to maximize or minimize the metric

    def should_evaluate(self, logged_metrics: Dict[str, torch.Tensor]) -> bool:
        if self.metric not in logged_metrics:
            logging.info(f"Metric {self.metric} not found in logged metrics")
            return False
        metric = logged_metrics[self.metric].item()
        if self.maximize:
            should_evaluate = metric > self.value
        else:
            should_evaluate = metric < self.value
        if not should_evaluate:
            logging.info(
                "Skipping evaluation."
                + f" {self.metric} is {metric}, but need {self.value}"
            )
        return should_evaluate


class SimEvaluation(ABC, Callback):
    def __init__(
        self,
        start_episode: int,
        num_episodes: int,
        sampler_config: EnvSamplerConfig,
        ctrl_config: ControlConfig,
        num_processes: int = 8,
        policy: Optional[Policy] = None,
        auto_incr_episode_ids: bool = True,
        eval_criteria: Optional[EvalCriteria] = None,
        remote: bool = True,
        auto_reindex_dataset: bool = False,
    ):
        assert (
            wandb.run is not None  # type: ignore
        ), "No run initialized. Do `wandb.init()` before creating an evaluater."
        self.start_episode = start_episode
        self.num_episodes = num_episodes
        self.sampler_config = sampler_config
        self.auto_incr_episode_ids = auto_incr_episode_ids
        self.auto_reindex_dataset = auto_reindex_dataset
        self.policy = policy
        self.trajectory_stats = pd.DataFrame()
        self.num_processes = num_processes
        self.ctrl_config = ctrl_config
        self.remote = remote
        logging.info(f"{self.__class__.__name__} initialized with {self.ctrl_config}")
        if not self.remote:
            logging.info("Running in single thread mode")
        self.eval_criteria = eval_criteria
        if self.eval_criteria is not None:
            logging.info(f"Using criteria {self.eval_criteria!r}")

    @abstractmethod
    def generate_trajectories(
        self,
        policy: Policy,
        root_path: str,
        desc: Optional[str] = None,
        episode_ids: Optional[Sequence[int]] = None,
        pbar: bool = False,
    ) -> List[EnvSample]:
        pass

    @abstractmethod
    def get_tasks(self) -> List[Task]:
        pass

    def run(
        self,
        policy: Policy,
        root_path: str,
        print_summary: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        start_time = time()
        assert (
            wandb.run is not None  # type: ignore
        ), "No run initialized. Make sure `wandb` is still running."
        trajectory_summary_dataframes: List[EnvSample] = self.generate_trajectories(
            policy=policy, root_path=root_path, **kwargs
        )
        if len(trajectory_summary_dataframes) == 0:
            return {}

        trajectory_df = pd.concat(
            [traj_dfs.trajectory_df for traj_dfs in trajectory_summary_dataframes]
        )
        trajectory_step_df = pd.concat(
            [traj_dfs.trajectory_step_df for traj_dfs in trajectory_summary_dataframes]
        )
        self.trajectory_stats = (
            pd.concat([self.trajectory_stats, trajectory_df])
            if self.trajectory_stats is not None
            else trajectory_df
        )

        # summarize
        to_log: Dict[str, Any] = {}
        self.trajectory_stats.to_pickle(f"{root_path}/trajectory_stats.pkl")
        to_log["trajectory"] = WandbTable(dataframe=trajectory_df)
        to_log["per_step"] = WandbTable(dataframe=trajectory_step_df)
        to_log["stats/evaluation_time"] = float(time() - start_time)
        for task_name in trajectory_df.task.unique():
            task_df = trajectory_df[trajectory_df.task == task_name]
            task_returns = np.array(task_df["return"])
            to_log[f"{task_name}/return_hist"] = Histogram(task_returns)  # type: ignore
            to_log[f"{task_name}/return_avg"] = task_returns.mean()
            to_log[f"{task_name}/success_avg"] = np.array(task_df["success"]).mean()
        to_log["all_tasks/success_avg"] = np.array(trajectory_df["success"]).mean()
        if print_summary:
            table = self.get_summary_table(trajectory_stats=trajectory_df)
            console = Console()
            print("\n")
            console.print(table)
        return to_log

    @property
    def task_stats(
        self, window: Optional[int] = None
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, int]]]:
        if window is None:
            window = self.num_episodes * 2
        tasks: Sequence[str] = sorted(self.trajectory_stats.task.unique())
        policies: Sequence[str] = list(self.trajectory_stats.policy_id.unique())
        means: Dict[str, Dict[str, float]] = {}
        for task in tasks:
            logging.debug(task)
            means[task] = {}
            for policy in policies:
                value = float(
                    np.array(
                        self.trajectory_stats[
                            (self.trajectory_stats.policy_id == policy)
                            & (self.trajectory_stats.task == task)
                        ]["success"]
                    )[-window:].mean()
                )
                means[task][policy] = value
                logging.debug(f"\t{policy}: {value*100:0.2f}")

        counts = {
            task: {
                policy: len(
                    self.trajectory_stats[
                        (self.trajectory_stats.policy_id == policy)
                        & (self.trajectory_stats.task == task)
                    ].iloc[-window:]
                )
                for policy in policies
            }
            for task in tasks
        }

        return means, counts

    def get_summary_table(
        self, trajectory_stats: pd.DataFrame, plot_by_policy: bool = False
    ):
        table = RichTable(title="Inference")
        table.add_column("Task", style="magenta")
        if plot_by_policy:
            table.add_column("Policy", justify="left")
        table.add_column("Success", justify="left")
        table.add_column("Dense Success", justify="left")
        table.add_column("Perfect", justify="left")
        table.add_column("Duration", justify="left")
        table.add_column("Count", justify="right")

        per_task: Dict[str, List[float]] = {
            "success": [],
            "dense_success": [],
            "perfect": [],
            "duration": [],
            "count": [],
        }
        for task in sorted(trajectory_stats.task.unique()):
            task_stats = trajectory_stats[trajectory_stats.task == task]
            if plot_by_policy:
                for policy_id in task_stats.policy_id.unique():
                    policy_task_stats = task_stats[task_stats.policy_id == policy_id]
                    table.add_row(
                        task,
                        policy_id,
                        f"{policy_task_stats['success'].mean():.02f}±{policy_task_stats['success'].std():.02f}",
                        f"{policy_task_stats['dense_success'].mean():.02f}±{policy_task_stats['dense_success'].std():.02f}",
                        f"{(policy_task_stats['dense_success']==1).mean():.02f}±{(policy_task_stats['dense_success']==1).std():.02f}",
                        f"{policy_task_stats['duration'].mean():.02f}±{policy_task_stats['duration'].std():.02f}",
                        f"{len(policy_task_stats['duration'])}",
                    )
            else:
                table.add_row(
                    task,
                    f"{task_stats['success'].mean():.02f}±{task_stats['success'].std():.02f}",
                    f"{task_stats['dense_success'].mean():.02f}±{task_stats['dense_success'].std():.02f}",
                    f"{(task_stats['dense_success']==1).mean():.02f}±{(task_stats['dense_success']==1).std():.02f}",
                    f"{task_stats['duration'].mean():.02f}±{task_stats['duration'].std():.02f}",
                    f"{len(task_stats['return'])}",
                )
            per_task["success"].append(task_stats["success"].mean())
            per_task["dense_success"].append(task_stats["dense_success"].mean())
            per_task["perfect"].append((task_stats["dense_success"] == 1).mean())
            per_task["duration"].append(task_stats["duration"].mean())
            per_task["count"].append(len(task_stats["return"]))
        table.add_row(
            "all_tasks",
            f"{np.mean(per_task['success']):.02f}",
            f"{np.mean(per_task['dense_success']):.02f}",
            f"{np.mean(per_task['perfect']):.02f}",
            f"{np.mean(per_task['duration']):.02f}",
            f"{np.sum(per_task['count'])}",
        )
        return table

    def on_train_epoch_end(
        self,
        trainer: LightningTrainer,
        algo: ScalingUpAlgo,
        update_policy: bool = True,
        debug_run: bool = False,
    ):
        algo.trainer = trainer
        if self.eval_criteria is not None and not self.eval_criteria.should_evaluate(
            trainer.logged_metrics
        ):
            return
        if update_policy:
            algo.eval()
            self.policy = algo.get_policy()
            algo.train()
        assert self.policy is not None
        logging.info(
            f"Inference on episodes in [{self.start_episode},{self.num_episodes}]"
            + f" with {str(self.policy)}"
        )
        stats = self.run(
            pbar=False,
            policy=self.policy,
            root_path=algo.current_epoch_logdir,
            print_summary=True,
        )
        if not debug_run:
            wandb.log({f"test/{k}": v for k, v in stats.items()})  # type: ignore
            if self.auto_reindex_dataset and algo.logdir in algo.replay_buffer.rootdir:
                logging.info(f"reindexing dataset at {algo.replay_buffer.rootdir!r}")
                # added new trajectories to replay buffer root, should reindex
                algo.replay_buffer.reindex()
                # update trainer with the latest data loader
                trainer.fit_loop._combined_loader = CombinedLoader(
                    algo.train_dataloader()
                )

            if self.auto_incr_episode_ids:
                self.start_episode += self.num_episodes

    def get_env_task_sampler_string(self, env: Env, task_sampler: TaskSampler):
        return (
            f"[bold yellow]{env.__class__.__name__}[/]"
            + f" with time budget {self.sampler_config.max_time:.01f} seconds "
            + "for tasks "
            + ", ".join([f"{t.desc!r}" for t in task_sampler.tasks])
        )


class SingleEnvSimEvaluation(SimEvaluation):
    def __init__(self, env: Env, task_sampler: TaskSampler, **kwargs):
        self.env = env
        self.task_sampler = task_sampler
        super().__init__(**kwargs)
        logging.info(
            self.get_env_task_sampler_string(
                env=self.env, task_sampler=self.task_sampler
            ),
            extra={"markup": True},
        )

    def get_tasks(self) -> List[Task]:
        return self.task_sampler.tasks

    def run(
        self,
        policy: Policy,
        root_path: str,
        print_summary: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        to_log = super().run(
            policy=policy,
            root_path=root_path,
            print_summary=print_summary,
            **kwargs,
        )
        return to_log

    def generate_trajectories(
        self,
        policy: Policy,
        root_path: str,
        desc: Optional[str] = None,
        episode_ids: Optional[Sequence[int]] = None,
        pbar: bool = False,
    ) -> List[EnvSample]:
        trajectory_summary_dataframes: List[EnvSample] = []
        if episode_ids is None:
            episode_ids = list(
                range(self.start_episode, self.start_episode + self.num_episodes)
            )
        if self.remote:
            if not ray.is_initialized():
                ray.init(log_to_driver=False, local_mode=False)
            sampler_cls = ray.remote(EnvSampler).options(max_restarts=10)
            env_handle = ray.put(self.env)
            env_samplers = [
                sampler_cls.remote(
                    env=env_handle,
                    task_sampler=self.task_sampler,
                )
                for _ in range(min(self.num_processes, len(episode_ids)))
            ]
            if desc is None:
                desc = f"{str(policy)} for {len(episode_ids)} episodes"
            sample_tasks = [
                env_samplers[episode_id % len(env_samplers)].sample.remote(
                    episode_id=episode_id,
                    policy=policy,
                    return_trajectory=False,
                    root_path=root_path,
                    config=self.sampler_config,
                    is_root_trajectory=True,
                )
                for episode_id in episode_ids
            ]
            if pbar:
                trajectory_summary_dataframes.extend(
                    wait_with_pbar(tasks_dict={"inference": sample_tasks})["inference"]
                )
            else:
                for task in sample_tasks:
                    ray.wait(sample_tasks, timeout=0.00001)
                    try:
                        trajectory_summary_dataframes.append(ray.get(task))
                    except OSError as e:
                        logging.error(e)
                        logging.error("most likely out of disk space")
                        ray.stop()
                        exit()
                    except Action.FailedExecution:
                        pass
                    except Exception as e:  # noqa: B902
                        logging.error(e)
        else:
            sampler = EnvSampler(
                env=self.env,
                task_sampler=self.task_sampler,
            )
            # disable logger for sampler
            logging.getLogger().disabled = True
            for episode_id in episode_ids:
                try:
                    trajectory_summary_dataframes.append(
                        sampler.sample(
                            episode_id=episode_id,
                            policy=policy,
                            return_trajectory=False,
                            root_path=root_path,
                            config=self.sampler_config,
                            is_root_trajectory=True,
                        )
                    )
                except Action.FailedExecution:
                    pass
                except Exception as e:  # noqa: B902
                    logging.getLogger().disabled = False
                    logging.error(e)
                    logging.getLogger().disabled = True
            logging.getLogger().disabled = False
        num_failed_episodes = len(episode_ids) - len(trajectory_summary_dataframes)
        if num_failed_episodes > 0:
            logging.warning(f"{num_failed_episodes} failed eval episodes")
        return trajectory_summary_dataframes


class MultiEnvSimEvaluation(SimEvaluation):
    """
    Extends `SimEvaluation` with functionality to run multiple different
    simulation environments, each of which can support a different set
    of tasks
    """

    def __init__(
        self,
        env_task_sampler_pairs: List[Tuple[Env, TaskSampler]],
        num_episodes: int,
        **kwargs,
    ):
        assert num_episodes > len(
            env_task_sampler_pairs
        ), "not enough episodes, should have at least one per env"
        super().__init__(num_episodes=num_episodes, **kwargs)
        self.env_task_sampler_pairs = env_task_sampler_pairs
        for env, task_sampler in env_task_sampler_pairs:
            logging.info(
                self.get_env_task_sampler_string(env=env, task_sampler=task_sampler),
                extra={"markup": True},
            )
        raise NotImplementedError("Currently deprecated, make sure to update yaml files")

    def get_tasks(self) -> List[Task]:
        return sum(
            (task_sampler.tasks for _, task_sampler in self.env_task_sampler_pairs), []
        )

    def run(
        self,
        policy: Policy,
        root_path: str,
        print_summary: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        to_log = super().run(
            policy=policy,
            root_path=root_path,
            print_summary=print_summary,
            **kwargs,
        )
        task_means, _ = self.task_stats
        return to_log

    def generate_trajectories(
        self,
        policy: Policy,
        root_path: str,
        desc: Optional[str] = None,
        episode_ids: Optional[Sequence[int]] = None,
        pbar: bool = False,
    ) -> List[EnvSample]:
        trajectory_summary_dataframes: List[EnvSample] = []
        if episode_ids is None:
            episode_ids = list(
                range(self.start_episode, self.start_episode + self.num_episodes)
            )
        assert self.remote, "single thread mode not supported"
        # round robin for each env task sampler pair
        sampler_cls = ray.remote(EnvSampler).options(max_restarts=10)
        env_samplers = [
            sampler_cls.remote(
                env=env,
                task_sampler=task_sampler,
            )
            for (env, task_sampler), _ in zip(
                cycle(self.env_task_sampler_pairs),
                range(
                    max(
                        min(self.num_processes, len(episode_ids)),
                        len(self.env_task_sampler_pairs),
                    )
                ),
            )
        ]
        if desc is None:
            desc = f"{str(policy)} for {len(episode_ids)} episodes"
        trajectory_samples = [
            env_samplers[episode_id % len(env_samplers)].sample.remote(
                episode_id=episode_id,
                policy=policy,
                return_trajectory=False,
                root_path=root_path,
                config=self.sampler_config,
                is_root_trajectory=True,
            )
            for episode_id in episode_ids
        ]
        if pbar:
            trajectory_summary_dataframes.extend(
                wait_with_pbar(tasks_dict={"inference": trajectory_samples})["inference"]
            )
        else:
            for trajectory_sample in trajectory_samples:
                ray.wait(trajectory_samples, timeout=10)
                try:
                    trajectory_summary_dataframes.append(ray.get(trajectory_sample))
                except OSError as e:
                    logging.error(e)
                    logging.error("most likely out of disk space")
                    ray.stop()
                    exit()
                except Action.FailedExecution:
                    pass
                except Exception as e:  # noqa: B902
                    logging.error(e)
        return trajectory_summary_dataframes
