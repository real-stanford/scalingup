from typing import Dict, List, Optional

import numpy as np
from scalingup.algo.virtual_grid import Point3D
from scalingup.utils.constants import LINK_SEPARATOR_TOKEN

from scalingup.utils.core import (
    AABB,
    EnvState,
    JointState,
    ObjectState,
)


def get_pose(
    state: EnvState,
    obj_name: str,
    context_name_to_link_path: Dict[str, str],
):
    obj_name = context_name_to_link_path[obj_name]
    return state.get_pose(key=obj_name)


def get_obj(
    state: EnvState,
    obj_name: str,
) -> ObjectState:
    if obj_name in state.object_states:
        return state.object_states[obj_name]
    raise KeyError(
        f"trying to get {obj_name!r} but only has {state.object_states.keys()!r}"
    )


def check_contact(
    state: EnvState,
    obj1: str,
    obj2: str,
    context_name_to_link_path: Dict[str, str],
):
    link1 = context_name_to_link_path[obj1]
    obj1 = link1.split(LINK_SEPARATOR_TOKEN)[0]
    link2 = context_name_to_link_path[obj2]
    link1_contacts = get_obj(state=state, obj_name=obj1).link_states[link1].contacts
    return any(c.other_link == link2 for c in link1_contacts)


def check_joint_activated(joint_state: JointState) -> bool:
    value_range = joint_state.max_value - joint_state.min_value
    return np.abs(joint_state.max_value - joint_state.current_value) < value_range * 0.20


def check_joint_deactivated(joint_state: JointState) -> bool:
    value_range = joint_state.max_value - joint_state.min_value
    return np.abs(joint_state.current_value - joint_state.min_value) < value_range * 0.20


def get_joint_state(
    state: EnvState, obj_name: str, context_name_to_link_path: Dict[str, str]
):
    link_path = context_name_to_link_path[obj_name]
    obj_name = link_path.split(LINK_SEPARATOR_TOKEN)[0]
    obj_state = state.object_states[obj_name]
    # TODO make this into a high level API
    joint_states: List[JointState] = [
        joint_state
        for joint_state in obj_state.joint_states.values()
        if joint_state.child_link == link_path
    ]
    if len(joint_states) == 0:
        joint_states = [
            joint_state
            for joint_state in obj_state.joint_states.values()
            if link_path in joint_state.child_link
        ]
    return joint_states[0]


def check_activated(
    state: EnvState,
    obj_name: str,
    context_name_to_link_path: Dict[str, str],
):
    joint_state = get_joint_state(
        state=state,
        obj_name=obj_name,
        context_name_to_link_path=context_name_to_link_path,
    )
    return check_joint_activated(joint_state=joint_state)


def check_deactivated(
    state: EnvState,
    obj_name: str,
    context_name_to_link_path: Dict[str, str],
):
    joint_state = get_joint_state(
        state=state,
        obj_name=obj_name,
        context_name_to_link_path=context_name_to_link_path,
    )
    return check_joint_deactivated(joint_state=joint_state)


def check_opened(
    state: EnvState,
    obj_name: str,
    context_name_to_link_path: Dict[str, str],
):
    return check_activated(
        state=state,
        obj_name=obj_name,
        context_name_to_link_path=context_name_to_link_path,
    )


def check_closed(
    state: EnvState,
    obj_name: str,
    context_name_to_link_path: Dict[str, str],
):
    return check_deactivated(
        state=state,
        obj_name=obj_name,
        context_name_to_link_path=context_name_to_link_path,
    )


def check_inside(
    state: EnvState,
    containee_obj_name: str,
    container_obj_name: str,
    context_name_to_link_path: Dict[str, str],
    containment_threshold: float = 0.25,
):
    containee_link_path = context_name_to_link_path[containee_obj_name]
    containee_obj_name = containee_link_path.split(LINK_SEPARATOR_TOKEN)[0]
    containee_link_state = state.object_states[containee_obj_name].link_states[
        containee_link_path
    ]
    containee_aabb = AABB.union(containee_link_state.aabbs)

    container_link_path = context_name_to_link_path[container_obj_name]
    container_obj_name = container_link_path.split(LINK_SEPARATOR_TOKEN)[0]
    container_link_state = state.object_states[container_obj_name].link_states[
        container_link_path
    ]
    container_aabb = AABB.union(container_link_state.aabbs)
    containment_score = (
        containee_aabb.intersection(container_aabb) / containee_aabb.volume
    )
    return containment_score > containment_threshold


def check_on_top_of(
    state: EnvState,
    on_top_obj_name: str,
    on_bottom_obj_name: str,
    context_name_to_link_path: Dict[str, str],
    up: Point3D = (0, 0, 1),
    threshold: float = 0.99,
    allow_any: bool = False,
) -> bool:
    link1 = context_name_to_link_path[on_bottom_obj_name]
    obj1 = link1.split(LINK_SEPARATOR_TOKEN)[0]
    link2 = context_name_to_link_path[on_top_obj_name]
    obj2 = link2.split(LINK_SEPARATOR_TOKEN)[0]
    link1_contacts = [
        c
        for c in state.object_states[obj1].link_states[link1].contacts
        if c.other_link == link2 and c.other_name == obj2
    ]
    if len(link1_contacts) == 0:
        return False
    # average normal of contacts from bottom object to top object
    # should point up
    dotproducts = [
        np.dot(np.array(c.normal) / np.linalg.norm(c.normal), up) for c in link1_contacts
    ]
    if allow_any:
        return any(dp > threshold for dp in dotproducts)
    return bool(np.mean(dotproducts) > threshold)
