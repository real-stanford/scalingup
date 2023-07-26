import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    paths = {
        "planar_noretry": "scripts/plot_data/llm_as_policy_2d_sandy-hill-639.pkl",
        "6dof_noretry": "scripts/plot_data/llm_as_policy_no_retry_wild-shadow-285.pkl",
        "6dof_retry": "scripts/plot_data/llm_as_policy_ours_stellar-aardvark-286.pkl",
        "distill_6dof_no_retry": "scripts/plot_data/distill_no_retry_stellar-dream-248.pkl",
        "ours": "scripts/plot_data/ours-vibrant-puddle-240.pkl",
    }
    weights = []
    data = pd.DataFrame.from_dict({})
    min_time = 0
    time_limit = 100
    num_episodes = 200
    for approach, path in paths.items():
        approach_data = pd.read_pickle(path)
        print(approach, len(approach_data), approach_data["success"].mean())
        approach_data["approach"] = [approach] * len(approach_data)
        # print(approach_data[~approach_data["success"].astype(bool)])
        approach_data.loc[~approach_data["success"].astype(bool), "duration"] = (
            time_limit + 1
        )
        success_rate = sum(approach_data["success"]) / num_episodes
        weights.extend(np.ones(len(approach_data)) / num_episodes)
        if len(data) == 0:
            data = approach_data
        else:
            data = pd.concat([data, approach_data])
    sns.set_style("darkgrid")
    fig, ax = plt.subplots()
    graph = sns.ecdfplot(
        data=data,
        x="duration",
        stat="count",
        hue="approach",
        weights=weights,
        ax=ax,
    )
    plt.xlim(min_time, time_limit)
    plt.ylim(0, 0.8)
    plt.tight_layout(pad=0.0)
    plt.show()
