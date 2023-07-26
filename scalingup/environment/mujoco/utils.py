from __future__ import annotations

from typing import Dict, Optional, Set

import numpy as np
from pydantic import dataclasses
from transforms3d import euler, quaternions

from scalingup.algo.virtual_grid import Point3D
from scalingup.utils.constants import LINK_SEPARATOR_TOKEN
from scalingup.utils.core import (
    AABB,
    RGB,
    VISUAL_GEOM_GROUP,
    Contact,
    EnvState,
    Pose,
    QPosRange,
)


def get_part_path(model, body) -> str:
    rootid = body.rootid
    path = ""
    while True:
        path = body.name + path
        currid = body.id
        if currid == rootid:
            return path
        body = model.body(body.parentid)
        path = LINK_SEPARATOR_TOKEN + path


def get_fixed_children(model, body):
    children = []
    for bodyid in range(model.nbody):
        other_body = model.body(bodyid)
        is_child = other_body.parentid == body.id
        is_fixed = other_body.dofadr[0] == -1
        if is_child and is_fixed:
            children.append(other_body)
            children.extend(get_fixed_children(model, other_body))
    return children


def parse_contact_data(physics) -> Dict[str, Dict[str, Set[Contact]]]:
    obj_link_contacts: Dict[str, Dict[str, Set[Contact]]] = {}
    data = physics.data
    model = physics.model
    for contact_idx in range(len(data.contact.geom1)):
        geom1 = data.contact.geom1[contact_idx]
        geom2 = data.contact.geom2[contact_idx]
        link1 = model.body(model.geom(geom1).bodyid)
        link1name = get_part_path(model, link1)
        link2 = model.body(model.geom(geom2).bodyid)
        link2name = get_part_path(model, link2)

        contact_pos = data.contact.pos[contact_idx].astype(float)
        position: Point3D = (contact_pos[0], contact_pos[1], contact_pos[2])
        normal = data.contact.frame[contact_idx][:3]

        obj1name = link1name.split(LINK_SEPARATOR_TOKEN)[0]
        obj2name = link2name.split(LINK_SEPARATOR_TOKEN)[0]
        if obj1name not in obj_link_contacts:
            obj_link_contacts[obj1name] = {}
        if link1name not in obj_link_contacts[obj1name]:
            obj_link_contacts[obj1name][link1name] = set()
        obj_link_contacts[obj1name][link1name].add(
            Contact(
                other_link=link2name,
                other_name=obj2name,
                self_link=link1name,
                position=position,
                normal=(normal[0], normal[1], normal[2]),
            )
        )
        if obj2name not in obj_link_contacts:
            obj_link_contacts[obj2name] = {}
        if link2name not in obj_link_contacts[obj2name]:
            obj_link_contacts[obj2name][link2name] = set()
        obj_link_contacts[obj2name][link2name].add(
            Contact(
                other_link=link1name,
                other_name=obj1name,
                self_link=link2name,
                position=position,
                normal=(-normal[0], -normal[1], -normal[2]),
            )
        )
    return obj_link_contacts


def get_body_aabbs(model, data, bodyid: int, geom_group_filter: Optional[int] = None):
    """
    geom_group_filter can be VISUAL_GEOM_GROUP
    """
    aabbs = []
    for geom_id in range(
        model.body_geomadr[bodyid],
        model.body_geomadr[bodyid] + model.body_geomnum[bodyid],
    ):
        geom_group = model.geom_group[geom_id]
        if geom_group_filter is not None and geom_group != geom_group_filter:
            continue
        geom_pos = data.geom_xpos[geom_id]
        geom_rot_mat = data.geom_xmat[geom_id].reshape(3, 3)
        geom_pose = Pose(
            position=geom_pos,
            orientation=quaternions.mat2quat(geom_rot_mat),
        )
        aabb_flattened = model.geom_aabb[geom_id, :]
        aabbs.append(
            AABB(
                center=(aabb_flattened[0], aabb_flattened[1], aabb_flattened[2]),
                size=(aabb_flattened[3], aabb_flattened[4], aabb_flattened[5]),
                pose=geom_pose,
            )
        )
    return aabbs


def get_body_bounding_sphere(model, data, bodyid: int):
    spheres = []
    for geom_id in range(
        model.body_geomadr[bodyid],
        model.body_geomadr[bodyid] + model.body_geomnum[bodyid],
    ):
        spheres.append((data.geom_xpos[geom_id], model.geom_rbound[geom_id]))
    return spheres


@dataclasses.dataclass(frozen=True)
class MujocoObjectColorConfig:
    name: str
    rgb: RGB


@dataclasses.dataclass(frozen=True)
class MujocoObjectInstanceConfig:
    obj_class: str
    asset_path: str
    qpos_range: QPosRange
    color_config: Optional[MujocoObjectColorConfig] = None
    position: Optional[Point3D] = None
    euler: Optional[Point3D] = None
    add_free_joint: Optional[bool] = None
    name: Optional[str] = None


def mailbox_dropped(state: EnvState):
    link_state = state.object_states["amazon package/"].link_states[
        "amazon package/|amazon package/amazon package"
    ]
    non_z_rotation = np.abs(np.array(euler.quat2euler(link_state.pose.orientation)[:2]))
    not_pointing_up = (non_z_rotation > np.pi * 0.3).any()
    table_contacts = {c for c in link_state.contacts if "table/" in c.other_name}
    nontable_contacts = {c for c in link_state.contacts if "table/" not in c.other_name}
    touching_ground = len(table_contacts) > 0
    touching_something_else = len(nontable_contacts) > 0
    # if the package fell over on the ground, without being supported by something else
    # (e.g., the gripper) then the episode should end
    return not_pointing_up and touching_ground and not touching_something_else
