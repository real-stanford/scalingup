from utils import (
    check_contact,
    check_activated,
    check_deactivated,
    check_inside,
    check_on_top_of,
    EnvState,
)


"""
instructions:
given a input task description, the goal is to output the success condition for
that task. unless otherwise specified, all objects start in a de-activated state
(e.g., doors, drawers, cabinets, cupboards, and other containers are closed,
lights are off, etc.) unless specified otherwise (e.g., with the door opened).
after performing the task, objects should be reset to original state if possible.
"""


# robot task: touch the apple
# scene:
# - apple
#    + apple body
#    + apple stem
def touching_apple(init_state: EnvState, final_state: EnvState):
    return check_contact(
        final_state, "robotiq left finger", "apple body"
    ) and check_contact(final_state, "robotiq right finger", "apple body")


# robot task: release the cup
# scene:
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


# robot task: move the milk carton into the shelf
# scene:
# - milk carton
# - coke can
# - shelf
def milk_carton_is_on_shelf(init_state: EnvState, final_state: EnvState):
    return check_on_top_of(final_state, "milk carton", "shelf")


# robot task: move the milk carton from the shelf
# scene:
# - milk carton
# - coke can
# - shelf
def milk_carton_is_not_on_shelf(init_state: EnvState, final_state: EnvState):
    return not check_on_top_of(final_state, "milk carton", "shelf")


# robot task: open the washing machine
# scene:
# - washing machine
#    + washing machine door
#       + washing machine door handle
#    + control panel
#       + on off button
def washing_machine_opened(init_state: EnvState, final_state: EnvState):
    return check_activated(final_state, "washing machine door")


# robot task: move the sock into the washing machine
# scene:
# - washing machine
#    + washing machine door
#       + washing machine door handle
#    + control panel
#       + on off button
# - sock
def sock_inside_washing_machine(init_state: EnvState, final_state: EnvState):
    # the washing machine can be opened (activated state) or closed (de-activated
    # state). since its activation state was not specified, the washing machine starts
    # off closed. therefore, it needs to be closed after the sock is moved inside.
    sock_inside_washing_machine = check_inside(final_state, "sock", "washing machine")
    washing_machine_door_closed = check_deactivated(final_state, "washing machine door")
    return sock_inside_washing_machine and washing_machine_door_closed


# robot task: with the washing machine opened, move the sock into the washing machine
# scene:
# - washing machine
#    + washing machine door
#       + washing machine door handle
#    + control panel
#       + on off button
# - sock
def sock_inside_washing_machine_with_washing_machine_opened(
    init_state: EnvState, final_state: EnvState
):
    # the washing machine can be opened (activated state) or closed (de-activated
    # state). the washing machine starts off opened, so it does not need to be closed
    # after the sock is moved inside.
    sock_inside_washing_machine = check_inside(final_state, "sock", "washing machine")
    return sock_inside_washing_machine
