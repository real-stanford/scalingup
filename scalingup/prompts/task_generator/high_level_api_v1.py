import dataclasses
from typing import Dict, List, Set, Union
from transforms3d import euler
import numpy as np
from utils import (
    check_contact,
    get_pose,
    check_opened,
    check_closed,
    check_inside,
    EnvState,
)


# robot task: touch the apple
# scene:
# - UR5
#    + robotiq left finger
#    + robotiq right finger
# - apple
#    + apple body
#    + apple stem
def touching_apple(init_state: EnvState, final_state: EnvState):
    return check_contact(
        final_state, "robotiq left finger", "apple body"
    ) and check_contact(final_state, "robotiq right finger", "apple body")


# robot task: release the currently grasped cup
# scene:
# - UR5
#    + robotiq left finger
#    + robotiq right finger
# - cup
#    + cup body
#    + cup handle
def released_cup(init_state: EnvState, final_state: EnvState):
    finally_touching_cup = check_contact(
        final_state, "robotiq left finger", "cup handle"
    ) and check_contact(final_state, "robotiq right finger", "cup handle")
    finally_released_cup = (not finally_touching_cup) and (
        not final_state.gripper_command
    )
    return finally_released_cup


# robot task: move gripper to 10 cm above the soft drink
# scene:
# - UR5
#    + robotiq left finger
#    + robotiq right finger
# - milk carton
#    + milk carton body
#    + milk carton cap
# - coke can
def gripper_is_10cm_above_soft_drink(init_state: EnvState, final_state: EnvState):
    final_coke_can_pose = get_pose(final_state, "coke can")
    final_coke_can_position = final_coke_can_pose.position
    final_gripper_position = final_state.end_effector_pose.position
    distance = np.linalg.norm(
        (final_coke_can_position + [0, 0, 0.1]) - final_gripper_position
    )
    return distance < 0.01


# robot task: rotate the couch by 90 degrees
# scene:
# - UR5
#    + robotiq left finger
#    + robotiq right finger
# - love seat
#    + pillow left
#    + pillow right
# - carpet
def couch_rotated_along_z_axis(init_state: EnvState, final_state: EnvState):
    init_couch_pose = get_pose(init_state, "love seat")
    final_couch_pose = get_pose(final_state, "love seat")
    _, _, init_couch_rot_z_radian = euler.quat2euler(init_couch_pose.orientation)
    _, _, final_couch_rot_z_radian = euler.quat2euler(final_couch_pose.orientation)
    couch_rotation_radian = final_couch_rot_z_radian - init_couch_rot_z_radian
    couch_rotation_degrees = np.rad2deg(couch_rotation_radian)
    return np.abs(couch_rotation_degrees) > 75


# robot task: open the washing machine
# scene:
# - UR5
#    + robotiq left finger
#    + robotiq right finger
# - washing machine
#    + washing machine door
#       + washing machine door handle
#    + control panel
#       + on off button
def washing_machine_opened(init_state: EnvState, final_state: EnvState):
    return check_opened(final_state, "washing machine door")


# robot task: close the washing machine
# scene:
# - UR5
#    + robotiq left finger
#    + robotiq right finger
# - washing machine
#    + washing machine door
#       + washing machine door handle
#    + control panel
#       + on off button
def washing_machine_closed(init_state: EnvState, final_state: EnvState):
    return check_closed(final_state, "washing machine door")


# robot task: move the sock into the washing machine
# scene:
# - UR5
#    + robotiq left finger
#    + robotiq right finger
# - washing machine
#    + washing machine door
#       + washing machine door handle
#    + control panel
#       + on off button
# - sock
def sock_inside_washing_machine(init_state: EnvState, final_state: EnvState):
    return check_inside(final_state, "sock", "washing machine door")


# robot task: move the sock into the washing machine then close the washing machine
# scene:
# - UR5
#    + robotiq left finger
#    + robotiq right finger
# - washing machine
#    + washing machine door
#       + washing machine door handle
#    + control panel
#       + on off button
# - sock
def sock_inside_washing_machine_and_washing_machine_closed(
    init_state: EnvState, final_state: EnvState
):
    sock_inside_washing_machine = check_inside(
        final_state, "sock", "washing machine door"
    )
    washing_machine_door_closed = check_closed(final_state, "washing machine door")
    return sock_inside_washing_machine and washing_machine_door_closed
