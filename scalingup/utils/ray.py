import logging
from rich.progress import (
    BarColumn,
    ProgressColumn,
    Progress,
    TaskID,
    Text,
    TextColumn,
    TimeRemainingColumn,
    Task,
)
import signal
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Dict, List, Any, Sequence
from concurrent.futures import wait as futures_wait
import ray


class RayTaskColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?")
        return Text(f"{speed:.02f} tasks/s")


def wait_with_pbar(tasks_dict: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    progress = Progress(
        TextColumn("[bold blue]{task.fields[task_name]}", justify="right"),
        BarColumn(bar_width=20),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        RayTaskColumn(),
        "•",
        TimeRemainingColumn(),
    )

    def wait_ray_tasks(
        task_id: TaskID,
        tasks: List[Any],
        done_event: Event,
    ):
        retvals: List[Any] = []
        # This will break if the response doesn't contain content length
        progress.update(task_id, total=len(tasks))
        ray.wait(tasks, timeout=0.2)
        progress.start_task(task_id)

        while len(tasks) > 0:
            readies, tasks = ray.wait(tasks, timeout=0.2)
            for ready in readies:
                try:
                    retvals.append(ray.get(ready))
                except Exception as e:  # noqa: B902
                    logging.error(e)
                progress.update(task_id, advance=1)
            if done_event.is_set():
                break

        return retvals

    with progress:
        futures = {}
        done_event = Event()

        def handle_sigint(signum, frame):
            done_event.set()

        original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, handle_sigint)
        with ThreadPoolExecutor(max_workers=len(tasks_dict)) as pool:
            for task_name, tasks in tasks_dict.items():
                task_id = progress.add_task(
                    task_name=task_name, description=task_name, start=False
                )
                futures[task_name] = pool.submit(
                    wait_ray_tasks, task_id=task_id, tasks=tasks, done_event=done_event
                )
    signal.signal(signal.SIGINT, original_sigint_handler)
    futures_wait(futures.values())
    return {k: v.result() for k, v in futures.items()}
