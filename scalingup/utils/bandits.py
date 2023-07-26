from typing import Dict, List, Optional, Sequence
import numpy as np


class UCB:
    def __init__(
        self,
        arm_names: List[str],
        arm_sums: Optional[np.ndarray] = None,
        arm_counts: Optional[np.ndarray] = None,
        alpha: float = 2.0,
    ):
        self.arm_names = arm_names
        self.alpha = alpha
        self.arm_sums: np.ndarray = (
            np.zeros(len(self.arm_names), dtype=float) if arm_sums is None else arm_sums
        )
        self.arm_counts: np.ndarray = (
            np.zeros(len(self.arm_names), dtype=int) if arm_counts is None else arm_counts
        )

    @property
    def mean(self):
        return self.arm_sums / self.arm_counts

    @property
    def uncertainty(self):
        return np.sqrt(self.alpha * np.log(np.sum(self.arm_counts)) / self.arm_counts)

    @property
    def upper_confidence_bound(self):
        return self.mean + self.uncertainty

    # choice of bandit
    def choose(self) -> str:
        arm_idx = np.argmax(self.upper_confidence_bound)
        return self.arm_names[arm_idx]

    def add_reward(self, arm: str, reward: float):
        arm_idx = self.arm_names.index(arm)
        self.arm_sums[arm_idx] += reward
        self.arm_counts[arm_idx] += 1

    def sort_arms(self, numpy_random: np.random.RandomState) -> List[str]:
        ucbs = self.upper_confidence_bound
        arm_by_ucb: Dict[float, List[str]] = {ucb: [] for ucb in np.unique(ucbs)}
        for ucb, arm_name in zip(ucbs, self.arm_names):
            arm_by_ucb[ucb].append(arm_name)
        sorted_arms = []
        for ucb in sorted(arm_by_ucb.keys(), reverse=True):
            numpy_random.shuffle(arm_by_ucb[ucb])
            sorted_arms.extend(arm_by_ucb[ucb])
        return sorted_arms
