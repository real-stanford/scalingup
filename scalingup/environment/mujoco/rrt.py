# ------------------------------------------------
# This file is adapted from
# https://github.com/caelan/motion-planners
# ------------------------------------------------

import logging
from copy import copy, deepcopy
from itertools import islice
from time import time
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm
from transforms3d import euler, quaternions

from scalingup.environment.mujoco.mujocoRobot import MujocoRobot
from scalingup.utils.core import Pose, EnvConfig

DistanceFunc = Callable[[np.ndarray, np.ndarray], float]
SampleFunc = Callable[[], np.ndarray]
ExtendFunc = Callable[[np.ndarray, np.ndarray], List[np.ndarray]]
CollisionFunc = Callable[[np.ndarray], bool]


def irange(start, stop=None, step=1):  # np.arange
    if stop is None:
        stop = start
        start = 0
    while start < stop:
        yield start
        start += step


def argmin(function, sequence):
    values = list(sequence)
    scores = [function(x) for x in values]
    return values[scores.index(min(scores))]


def pairs(lst):
    return zip(lst[:-1], lst[1:])


def merge_dicts(*args):
    result = {}
    for d in args:
        result.update(d)
    return result
    # return dict(reduce(operator.add, [d.items() for d in args]))


def flatten(iterable_of_iterables):
    return (item for iterables in iterable_of_iterables for item in iterables)


def randomize(sequence, np_random: np.random.RandomState):
    np_random.shuffle(sequence)
    return sequence


def take(iterable, n: Optional[int] = None):
    if n is None:
        n = 0  # NOTE - for some of the uses
    return islice(iterable, n)


def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    enums["names"] = sorted(enums.keys(), key=lambda k: enums[k])
    return type("Enum", (), enums)


def smooth_path(
    path: List[np.ndarray],
    extend_fn: ExtendFunc,
    collision_fn: CollisionFunc,
    np_random: np.random.RandomState,
    iterations: int = 50,
):
    smoothed_path = path
    for _ in range(iterations):
        if len(smoothed_path) <= 2:
            return smoothed_path
        i = np_random.randint(0, len(smoothed_path) - 1)
        j = np_random.randint(0, len(smoothed_path) - 1)
        if abs(i - j) <= 1:
            continue
        if j < i:
            i, j = j, i
        shortcut = list(extend_fn(smoothed_path[i], smoothed_path[j]))
        if (len(shortcut) < (j - i)) and all(not collision_fn(q) for q in shortcut):
            smoothed_path = smoothed_path[: i + 1] + shortcut + smoothed_path[j + 1 :]
    return smoothed_path


class TreeNode(object):
    def __init__(self, config, parent=None):
        # configuration
        self.config = config
        # parent configuration
        self.parent = parent

    def retrace(self):
        """
        Get a list of nodes from start to itself
        :return: a list of nodes
        """
        sequence = []
        node = self
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]

    def __str__(self):
        return "TreeNode(" + str(self.config) + ")"

    __repr__ = __str__


def configs(nodes) -> Optional[List[np.ndarray]]:
    """
    Get the configurations of nodes
    :param nodes: array type of nodes
    :return: a list of configurations
    """
    return [n.config for n in nodes] if nodes is not None else None


def closest_node_to_goal(distance_fn, target, nodes: List[TreeNode]) -> TreeNode:
    return nodes[np.argmin([distance_fn(node.config, target) for node in nodes])]


def rrt(
    start_conf: np.ndarray,
    goal_conf: np.ndarray,  # TODO extend functionality to multiple goal confs
    distance_fn: DistanceFunc,
    sample_fn: SampleFunc,
    extend_fn: ExtendFunc,
    collision_fn: CollisionFunc,
    np_random: np.random.RandomState,
    iterations: int = 2000,
    goal_probability: float = 0.2,
    greedy: bool = True,
) -> Optional[List[np.ndarray]]:
    """
    RRT algorithm
    :param start_conf: start_conf configuration.
    :param goal_conf: goal configuration.
    :param distance_fn: distance_fn function distance_fn(q1, q2). Takes two
                    confugurations, returns a number of distance_fn.
    :param sample_fn: sampling function sample_fn(). Takes nothing, returns a sample
                of configuration.
    :param extend_fn: extend_fn function extend_fn(q1, q2). Extends the tree from q1
                towards q2.
    :param collision_fn: collision checking function. Check whether
                    collision exists for configuration q.
    :param goal_test: to test if q is the goal configuration.
    :param iterations: number of iterations to extend tree towards an sample
    :param goal_probability: bias probability to set the sample to goal
    :param visualize: whether draw nodes and lines for the tree
    :return: a list of configurations
    """
    if collision_fn(start_conf):
        logging.error("rrt fails, start_conf configuration has collision")
        return None

    nodes = [TreeNode(start_conf)]
    for i in irange(iterations):
        goal = np_random.random() < goal_probability or i == 0
        current_target = goal_conf if goal else sample_fn()
        last = closest_node_to_goal(
            distance_fn=distance_fn, target=current_target, nodes=nodes
        )
        for q in extend_fn(last.config, current_target):
            if collision_fn(q):
                break
            last = TreeNode(q, parent=last)
            nodes.append(last)
            if (last.config == goal_conf).all():
                logging.debug("Success")
                return configs(last.retrace())
            if not greedy:
                break
        else:
            if goal:
                logging.error("Impossible")
                return configs(last.retrace())
    return None


def rrt_connect(
    q1: np.ndarray,
    q2: np.ndarray,
    distance_fn: DistanceFunc,
    sample_fn: SampleFunc,
    extend_fn: ExtendFunc,
    collision_fn: CollisionFunc,
    iterations: int,
    greedy: bool,
    timeout: float,
    debug: bool = False,
):
    start_time = time()
    if collision_fn(q1) or collision_fn(q2):
        return None, "start or goal configuration has collision"
    root1, root2 = TreeNode(q1), TreeNode(q2)
    nodes1, nodes2 = [root1], [root2]
    loop = irange(iterations)
    if debug:
        tqdm(loop, total=iterations)
    message = ""
    for _ in loop:
        if float(time() - start_time) > timeout:
            message += f"timeout ({timeout:.01f}s)"
            break
        if len(nodes1) > len(nodes2):
            nodes1, nodes2 = nodes2, nodes1
        current_target = sample_fn()
        last1 = closest_node_to_goal(
            distance_fn=distance_fn, target=current_target, nodes=nodes1
        )
        for q in extend_fn(last1.config, current_target):
            if collision_fn(q):
                break
            last1 = TreeNode(q, parent=last1)
            nodes1.append(last1)
            if not greedy:
                break

        last2 = closest_node_to_goal(
            distance_fn=distance_fn, target=last1.config, nodes=nodes2
        )

        for q in extend_fn(last2.config, last1.config):
            if collision_fn(q):
                break
            last2 = TreeNode(q, parent=last2)
            nodes2.append(last2)
            if not greedy:
                break
        if (last2.config == last1.config).all():
            path1, path2 = last1.retrace(), last2.retrace()
            if path1[0] != root1:
                path1, path2 = path2, path1
            return (
                configs(path1[:-1] + path2[::-1]),
                f" found in {float(time() - start_time):.01f} seconds",
            )
    return None, f"out of iterations ({iterations})" if message == "" else message


def direct_path(
    q1: np.ndarray, q2: np.ndarray, extend_fn: ExtendFunc, collision_fn: CollisionFunc
):
    if collision_fn(q1) or collision_fn(q2):
        return None
    path = [q1]
    for q in extend_fn(q1, q2):
        if collision_fn(q):
            return None
        path.append(q)
    return path


def birrt(
    start_conf: np.ndarray,
    goal_conf: np.ndarray,
    distance_fn: DistanceFunc,
    sample_fn: SampleFunc,
    extend_fn: ExtendFunc,
    collision_fn: CollisionFunc,
    np_random: np.random.RandomState,
    iterations: int,
    smooth: int,
    greedy: bool,
    timeout: float,
    smooth_extend_fn: Optional[ExtendFunc] = None,
):
    start = time()
    path = direct_path(start_conf, goal_conf, extend_fn, collision_fn)
    if path is not None:
        return path, "[RRT] direct path found in {time:.01f} seconds".format(
            time=float(time() - start)
        )
    path, message = rrt_connect(
        q1=start_conf,
        q2=goal_conf,
        distance_fn=distance_fn,
        sample_fn=sample_fn,
        extend_fn=extend_fn,
        collision_fn=collision_fn,
        iterations=iterations,
        greedy=greedy,
        timeout=timeout,
    )
    if path is not None:
        return (
            smooth_path(
                path=path,
                extend_fn=smooth_extend_fn if smooth_extend_fn is not None else extend_fn,
                collision_fn=collision_fn,
                np_random=np_random,
                iterations=smooth,
            ),
            f"[RRT] plan found: {message}",
        )
    return None, message


class RRTSampler:
    def __init__(
        self,
        start_conf: np.ndarray,
        goal_conf: np.ndarray,
        min_values: np.ndarray,
        max_values: np.ndarray,
        numpy_random: np.random.RandomState,
        init_samples: Optional[List[np.ndarray]] = None,
    ):
        self.start_conf = start_conf
        self.goal_conf = goal_conf
        self.min_values = min_values
        self.max_values = max_values
        self.value_range = self.max_values - self.min_values
        self.numpy_random = numpy_random
        self.init_samples = init_samples
        self.curr_sample_idx = 0

    def __call__(self):
        if self.init_samples is not None and self.curr_sample_idx < len(
            self.init_samples
        ):
            self.curr_sample_idx += 1
            return self.init_samples[self.curr_sample_idx - 1]
        return self.numpy_random.uniform(low=self.min_values, high=self.max_values)


class NearJointsNormalSampler(RRTSampler):
    def __init__(self, bias: float, **kwargs):
        self.bias = bias
        super().__init__(**kwargs)

    def __call__(self):
        if self.numpy_random.random() > 0.5:
            return super().__call__()
        center = self.goal_conf if self.numpy_random.random() > 0.5 else self.start_conf
        sample = (
            center
            + self.numpy_random.randn(len(center))
            * (self.max_values - self.min_values)
            * self.bias
        )
        return np.clip(sample, a_min=self.min_values, a_max=self.max_values)


class NearJointsUniformSampler(RRTSampler):
    def __init__(self, bias: float, **kwargs):
        self.bias = bias
        super().__init__(**kwargs)

    def __call__(self):
        if self.numpy_random.random() > 0.5:
            return super().__call__()
        center = self.goal_conf if self.numpy_random.random() > 0.5 else self.start_conf
        sample = center + self.numpy_random.uniform(
            low=np.clip(
                center - self.bias * self.value_range,
                a_min=self.min_values,
                a_max=self.max_values,
            ),
            high=np.clip(
                center + self.bias * self.value_range,
                a_min=self.min_values,
                a_max=self.max_values,
            ),
        )
        return sample


class MujocoRRT:
    def __init__(
        self,
        physics,
        robot: MujocoRobot,
        env_config: EnvConfig,
        seed: int = 0,
    ):
        self.physics = physics
        self.init_state = deepcopy(self.physics.get_state())
        self.np_random = np.random.RandomState(seed)
        self.robot = robot
        self.env_config = env_config
        self.robot.mj_physics = self.physics

        self.minmax = np.array([joint.range for joint in self.robot.joints])
        self.ranges = self.minmax[:, 1] - self.minmax[:, 0]

    def extend_joint_l2(
        self, q1: np.ndarray, q2: np.ndarray, resolution: float = 0.005
    ) -> List[np.ndarray]:
        if (dist := self.joint_l2(q1, q2)) == 0:
            return []
        step = resolution / dist
        return [(q2 - q1) * np.clip(t, 0, 1) + q1 for t in np.arange(0, 1 + step, step)]

    def extend_end_effector_l2(
        self, q1: np.ndarray, q2: np.ndarray, resolution: float = 0.004
    ) -> List[np.ndarray]:
        if (dist := self.end_effector_l2(q1, q2)) == 0:
            return []
        step = resolution / dist
        return [(q2 - q1) * np.clip(t, 0, 1) + q1 for t in np.arange(0, 1 + step, step)]

    def extend_end_effector(
        self, q1: np.ndarray, q2: np.ndarray, resolution: float = 0.005
    ) -> List[np.ndarray]:
        pose1 = self.robot.set_joint_config(q1, return_ee_pose=True)
        pose2 = self.robot.set_joint_config(q2, return_ee_pose=True)
        assert pose1 is not None and pose2 is not None
        w, x, y, z = pose1.orientation
        start_rot = Rotation.from_quat([x, y, z, w])
        w, x, y, z = pose2.orientation
        end_rot = Rotation.from_quat([x, y, z, w])
        orientation_slerp = Slerp(
            times=[0, 1], rotations=Rotation.concatenate([start_rot, end_rot])
        )
        waypoints: List[Optional[np.ndarray]] = []
        dist = self.end_effector_l2(q1, q2)
        for i in np.arange(0, 1, resolution / dist):
            position_i = (pose2.position - pose1.position) * i + pose1.position
            x, y, z, w = orientation_slerp([i])[0].as_quat()
            result = self.robot.inverse_kinematics(
                pose=Pose(position=position_i, orientation=np.array([w, x, y, z])),
                inplace=True,
            )
            if result is None:
                waypoints.append(None)
                break
            waypoints.append(result)
        return waypoints  # type: ignore

    def joint_l2(self, q1: np.ndarray, q2: np.ndarray) -> float:
        return float(np.linalg.norm(q1 - q2))

    def end_effector_l2(
        self, q1: np.ndarray, q2: np.ndarray, orientation_factor: float = 0.2
    ) -> float:
        pose1 = self.robot.set_joint_config(q1, return_ee_pose=True)
        pose2 = self.robot.set_joint_config(q2, return_ee_pose=True)

        assert pose1 is not None and pose2 is not None
        return pose1.distance(pose2, orientation_factor=orientation_factor)

    def plan(
        self,
        start_conf: np.ndarray,
        goal_conf: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
        current_gripper_command: bool,
        goal_gripper_command: bool,
    ) -> Tuple[Optional[List[np.ndarray]], str]:
        self.physics.data.qpos[:] = qpos.copy()
        self.physics.data.qvel[:] = qvel.copy()
        self.physics.data.ctrl[:] = ctrl.copy()
        self.physics.forward()

        start_grasp_obj_id = self.robot.get_grasped_obj_id(physics=self.physics)
        start_grasp_pose = self.robot.get_grasp_pose(physics=self.physics)

        if goal_gripper_command != current_gripper_command:
            ctrl = self.physics.control()
            ctrl[-1] = (
                self.robot.gripper_close_ctrl_val
                if goal_gripper_command
                else self.robot.gripper_open_ctrl_val
            )
            self.physics.set_control(ctrl)
            for _ in range(self.env_config.ee_action_num_grip_steps):
                self.physics.step()

        grasp_obj_id = -1
        grasp_pose = None
        if current_gripper_command and goal_gripper_command:
            # likely motion planning with an already grasped object
            grasp_obj_id = start_grasp_obj_id
            grasp_pose = start_grasp_pose
        elif current_gripper_command and not goal_gripper_command:
            # likely motion planning to release an object
            pass
        elif not current_gripper_command and goal_gripper_command:
            # likely motion planning to grasp an object
            pass
        else:
            # no grasped object motion planning
            pass

        def collision_fn(q: np.ndarray):
            return self.robot.check_collision(
                joints=q,
                physics=self.physics,
                grasp_obj_id=grasp_obj_id,
                grasp_pose=grasp_pose,
                detect_grasp=False,
            )

        logging.debug(f"RRT qpos: {self.physics.data.qpos!r}")
        logging.debug(f"RRT qvel: {self.physics.data.qvel!r}")
        logging.debug(f"RRT ctrl: {self.physics.data.ctrl!r}")
        logging.debug(f"RRT start_conf: {start_conf!r}")
        logging.debug(f"RRT start_conf: {goal_conf!r}")
        logging.debug(f"RRT start_grasp_obj_id: {start_grasp_obj_id!r}")
        logging.debug(f"RRT start_grasp_pose: {start_grasp_pose!r}")

        if collision_fn(start_conf):
            pair_names = self.robot.get_unfiltered_collided_pairs_names(
                physics=self.physics,
                joints=start_conf,
                grasp_pose=start_grasp_pose,
                grasp_obj_id=start_grasp_obj_id,
                detect_grasp=False,
            )
            collision_summary = ", ".join(
                {" and ".join(sorted(pair)) for pair in pair_names}
            )
            return None, f"RRT failed: start config in collision {collision_summary}."

        elif collision_fn(goal_conf):
            pair_names = self.robot.get_unfiltered_collided_pairs_names(
                physics=self.physics,
                joints=goal_conf,
                grasp_pose=grasp_pose,
                grasp_obj_id=grasp_obj_id,
                detect_grasp=False,
            )
            collision_summary = ", ".join(
                {" and ".join(sorted(pair)) for pair in pair_names}
            )
            return None, f"RRT failed: goal config in collision {collision_summary}"
        init_samples = []
        # try top down grasps first
        start_pose = self.robot.set_joint_config(
            joints=start_conf, grasp_pose=grasp_pose, return_ee_pose=True
        )
        goal_pose = self.robot.set_joint_config(
            joints=goal_conf, grasp_pose=grasp_pose, return_ee_pose=True
        )
        assert start_pose is not None and goal_pose is not None
        for z_height in np.arange(0.1, 0.3 + 1e-4, 0.1):
            for z_angle in np.arange(-np.pi / 2, np.pi / 2 + 1e-4, np.pi / 4):
                z_orn = euler.euler2quat(0, 0, z_angle)
                before_goal_pose = Pose(
                    position=goal_pose.position + np.array([0, 0, z_height]),
                    orientation=quaternions.qmult(z_orn, goal_pose.orientation),
                )
                config = self.robot.inverse_kinematics(
                    pose=before_goal_pose,
                    inplace=True,
                )
                if config is not None:
                    init_samples.append(config)
                    continue
                after_start_pose = Pose(
                    position=start_pose.position + np.array([0, 0, z_height]),
                    orientation=quaternions.qmult(z_orn, start_pose.orientation),
                )
                config = self.robot.inverse_kinematics(
                    pose=after_start_pose,
                    inplace=True,
                )
                if config is not None:
                    init_samples.append(config)
                intermediate_pose = Pose(
                    position=(before_goal_pose.position + after_start_pose.position) / 2,
                    orientation=before_goal_pose.orientation,
                )
                config = self.robot.inverse_kinematics(
                    pose=intermediate_pose,
                    inplace=True,
                )
                if config is not None:
                    init_samples.append(config)
        return birrt(
            start_conf=start_conf,
            goal_conf=goal_conf,
            distance_fn=self.end_effector_l2,
            sample_fn=NearJointsUniformSampler(
                bias=0.2,
                start_conf=start_conf,
                goal_conf=goal_conf,
                numpy_random=self.np_random,
                min_values=self.minmax[:, 0],
                max_values=self.minmax[:, 1],
                init_samples=init_samples,
            ),
            extend_fn=self.extend_end_effector_l2,
            collision_fn=collision_fn,
            iterations=1000,
            smooth=100,
            timeout=60,
            greedy=True,
            np_random=self.np_random,
            smooth_extend_fn=self.extend_end_effector_l2,
        )
