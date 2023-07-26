from typing import List, Optional
from scalingup.utils.constants import LINK_SEPARATOR_TOKEN
from scalingup.utils.state_api import (
    check_activated,
    check_closed,
    check_deactivated,
    check_inside,
    check_on_top_of,
)
from scalingup.utils.core import (
    ActionListPolicy,
    GraspLinkAction,
    Policy,
    PolicyTaskAction,
    PrismaticJointAction,
    Task,
    Trajectory,
)
from scalingup.utils.core import (
    EndEffectorAction,
    EnvState,
    JointState,
    ObjectState,
    Observation,
)
import numpy as np
from transforms3d import euler


class OpenObj(Task):
    def __init__(
        self,
        link_path: str,
        desc_template: str = "open the {link_path}",
    ):
        self.link_path = link_path
        super().__init__(
            desc=desc_template.format(link_path=self.link_path),
        )

    def check_success(self, traj: Trajectory) -> bool:
        final_state = traj.final_state
        # get joint associated with link
        obj_states = final_state.object_states
        obj_name = self.link_path.split(LINK_SEPARATOR_TOKEN)[0]
        if obj_name in obj_states:
            joint_state: JointState = [
                joint_state
                for joint_state in obj_states[obj_name].joint_states.values()
                if joint_state.child_link == self.link_path
            ][0]
        else:
            raise NotImplementedError()
        value_range = joint_state.max_value - joint_state.min_value
        return (
            np.abs(joint_state.max_value - joint_state.current_value) < value_range * 0.05
        )

    def get_reward(self, state: EnvState) -> float:
        # get joint associated with link
        obj_states = state.object_states
        obj_name = self.link_path.split(LINK_SEPARATOR_TOKEN)[0]
        if obj_name in obj_states:
            joint_state: JointState = [
                joint_state
                for joint_state in obj_states[obj_name].joint_states.values()
                if joint_state.child_link == self.link_path
            ][0]
        else:
            raise NotImplementedError()
        value_range = joint_state.max_value - joint_state.min_value
        return np.exp(
            -np.abs(joint_state.max_value - joint_state.current_value) / value_range
        )

    @property
    def name(self) -> str:
        return f"open the {self.link_path}"


class CloseObj(Task):
    def __init__(
        self,
        link_path: str,
        desc_template: str = "close the {link_path}",
    ):
        self.link_path = link_path
        super().__init__(
            desc=desc_template.format(link_path=self.link_path),
        )

    def check_success(self, traj: Trajectory) -> bool:
        final_state = traj.final_state
        # get joint associated with link
        obj_states = final_state.object_states
        obj_name = self.link_path.split(LINK_SEPARATOR_TOKEN)[0]
        if obj_name in obj_states:
            joint_state: JointState = [
                joint_state
                for joint_state in obj_states[obj_name].joint_states.values()
                if joint_state.child_link == self.link_path
            ][0]
        else:
            raise NotImplementedError()
        value_range = joint_state.max_value - joint_state.min_value
        return (
            np.abs(joint_state.current_value - joint_state.min_value) < value_range * 0.05
        )

    def get_reward(self, state: EnvState) -> float:
        # get joint associated with link
        obj_states = state.object_states
        obj_name = self.link_path.split(LINK_SEPARATOR_TOKEN)[0]
        if obj_name in obj_states:
            joint_state: JointState = [
                joint_state
                for joint_state in obj_states[obj_name].joint_states.values()
                if joint_state.child_link == self.link_path
            ][0]
        else:
            raise NotImplementedError()
        value_range = joint_state.max_value - joint_state.min_value
        return np.exp(
            -np.abs(joint_state.current_value - joint_state.min_value) / value_range
        )

    @property
    def name(self) -> str:
        return f"close the {self.link_path}"


class PutObjInContainer(Task):
    def __init__(
        self,
        obj_link_path: str,
        container_link_path: str,
        require_close: bool,
        desc_template: str = "put the {obj_link_path} inside the "
        + "{container_link_path}, then close the {container_link_path}",
    ):
        self.obj_link_path = obj_link_path
        self.container_link_path = container_link_path
        self.require_close = require_close
        super().__init__(
            desc=desc_template.format(
                obj_link_path=self.obj_link_path,
                container_link_path=self.container_link_path,
            ),
        )

    def check_success(
        self,
        traj: Trajectory,
    ) -> bool:
        return check_inside(
            state=traj.final_state,
            containee_obj_name=self.obj_link_path,
            container_obj_name=self.container_link_path,
            context_name_to_link_path={
                self.obj_link_path: self.obj_link_path,
                self.container_link_path: self.container_link_path,
            },
        ) and (
            not self.require_close
            or check_closed(
                state=traj.final_state,
                obj_name=self.container_link_path,
                context_name_to_link_path={
                    self.container_link_path: self.container_link_path,
                },
            )
        )


class SortBlocksIntoPlates(Task):
    def __init__(
        self,
        matching: bool = True,
    ):
        self.matching = matching
        super().__init__(
            desc="sort the blocks onto plates with matching colors"
            if matching
            else "move the blocks onto plates with non-matching colors"
        )

    def check_success(
        self,
        traj: Trajectory,
    ) -> bool:
        final_state = traj.final_state
        block_names = []
        plate_names = []
        for obj_name, obj_state in final_state.object_states.items():
            if "block" in obj_name or "plate" in obj_name:
                assert (
                    len(obj_state.link_states) == 2
                ), "blocks and plates must have only root link and geom link"
            if "block" in obj_name:
                block_names.append(list(obj_state.link_states.keys())[1])
            elif "plate" in obj_name:
                plate_names.append(list(obj_state.link_states.keys())[1])
        block_plate_colors = []
        for block_name in block_names:
            for plate_name in plate_names:
                if check_on_top_of(
                    state=final_state,
                    on_top_obj_name=block_name,
                    on_bottom_obj_name=plate_name,
                    context_name_to_link_path={
                        block_name: block_name,
                        plate_name: plate_name,
                    },
                ):
                    block_plate_colors.append(
                        [block_name.split("_")[0], plate_name.split("_")[0]]
                    )
                    break
        all_blocks_have_plates = len(block_plate_colors) == len(block_names)
        if not all_blocks_have_plates:
            return False
        if self.matching:
            return all(
                block_color == plate_color
                for block_color, plate_color in block_plate_colors
            )
        return all(
            block_color != plate_color for block_color, plate_color in block_plate_colors
        )


class StackBlocks(Task):
    def __init__(self):
        super().__init__(desc="stack the blocks on top of each other")

    def check_success(
        self,
        traj: Trajectory,
    ) -> bool:
        final_state = traj.final_state
        num_blocks_not_on_another_blocks: int = 0
        # if more than one block is not on another block, then task failed
        all_link_names: List[str] = sum(
            (
                list(obj_state.link_states.keys())
                for obj_state in final_state.object_states.values()
            ),
            [],
        )
        for obj_name, obj_state in final_state.object_states.items():
            if "block" not in obj_name:
                continue
            assert (
                len(obj_state.link_states) == 2
            ), "blocks and plates must have only root link and geom link"
            block_name = list(obj_state.link_states.keys())[1]
            # if block is not on another block then
            is_block_on_another_block = (
                len(
                    list(
                        filter(
                            lambda link_name: check_on_top_of(
                                state=final_state,
                                on_top_obj_name=block_name,
                                on_bottom_obj_name=link_name,
                                context_name_to_link_path={
                                    block_name: block_name,
                                    link_name: link_name,
                                },
                            )
                            and "block" in link_name,
                            all_link_names,
                        )
                    )
                )
                > 0
            )
            if not is_block_on_another_block:
                num_blocks_not_on_another_blocks += 1
            if num_blocks_not_on_another_blocks > 1:
                return False
        return True


def find_obj_underneath_target_obj(
    state: EnvState, target_obj_name: str, query_object_names: Optional[List[str]] = None
) -> Optional[str]:
    if query_object_names is None:
        query_object_names = list(state.object_states.keys())
    for query_obj_name in query_object_names:
        if check_on_top_of(
            state=state,
            on_top_obj_name=target_obj_name,
            on_bottom_obj_name=query_obj_name,
            context_name_to_link_path={
                target_obj_name: target_obj_name,
                query_obj_name: query_obj_name,
            },
        ):
            return query_obj_name
    return None


class SortFruitsAndToolsIntoPlates(Task):
    def __init__(
        self,
        matching: bool = True,
    ):
        self.matching = matching
        super().__init__(desc="sort the fruits and tools into separate plates")

    def check_success(
        self,
        traj: Trajectory,
    ) -> bool:
        final_state = traj.final_state
        fruit_names = []
        tool_names = []
        plate_names = []
        for obj_name, obj_state in final_state.object_states.items():
            if "plate" in obj_name:
                plate_names.append(list(obj_state.link_states.keys())[1])
            elif any(
                keyword in obj_name for keyword in ["apple", "banana", "peach", "orange"]
            ):
                fruit_names.append(list(obj_state.link_states.keys())[1])
            elif any(keyword in obj_name for keyword in ["screwdriver", "hammer"]):
                tool_names.append(list(obj_state.link_states.keys())[1])
        tool_plates = [
            find_obj_underneath_target_obj(
                state=final_state,
                target_obj_name=tool_name,
                query_object_names=plate_names,
            )
            for tool_name in tool_names
        ]
        fruit_plates = [
            find_obj_underneath_target_obj(
                state=final_state,
                target_obj_name=fruit_name,
                query_object_names=plate_names,
            )
            for fruit_name in fruit_names
        ]
        if any(tool_plate is None for tool_plate in tool_plates):
            # some tools are not on plates
            return False
        if len(set(tool_plates)) != 1:
            # tools are on more than one plate
            return False
        if any(fruit_plate is None for fruit_plate in fruit_plates):
            # some fruits are not on plates
            return False
        if len(set(fruit_plates)) != 1:
            # fruits are on more than one plate
            return False
        return True


class PutObjOnAnother(Task):
    def __init__(self, on_top_link_path: str, below_link_path: str, desc: str):
        self.on_top_link_path = on_top_link_path
        self.below_link_path = below_link_path
        super().__init__(desc=desc)

    def check_success(
        self,
        traj: Trajectory,
    ) -> bool:
        return check_on_top_of(
            state=traj.final_state,
            on_top_obj_name=self.on_top_link_path,
            on_bottom_obj_name=self.below_link_path,
            context_name_to_link_path={
                self.on_top_link_path: self.on_top_link_path,
                self.below_link_path: self.below_link_path,
            },
        )


class ActivateObj(Task):
    def __init__(self, link_path: str, desc: str):
        self.link_path = link_path
        super().__init__(desc=desc)

    def check_success(self, traj: Trajectory) -> bool:
        return check_activated(
            state=traj.final_state,
            obj_name=self.link_path,
            context_name_to_link_path={
                self.link_path: self.link_path,
            },
        )


class SendMailPackageBack(Task):
    def __init__(
        self,
        package_link_path: str,
        mailbox_link_path: str,
        desc: str = "send the package for return",
        mailbox_flag_link: Optional[str] = None,
    ):
        super().__init__(desc=desc)
        self.package_link_path = package_link_path
        self.mailbox_link_path = mailbox_link_path
        self.mailbox_flag_link = mailbox_flag_link

    def check_success(self, traj: Trajectory) -> bool:
        mailbox_closed = check_deactivated(
            state=traj.final_state,
            obj_name=self.mailbox_link_path,
            context_name_to_link_path={
                self.mailbox_link_path: self.mailbox_link_path,
            },
        )
        package_inside_mailbox = check_inside(
            state=traj.final_state,
            containee_obj_name=self.package_link_path,
            container_obj_name=self.mailbox_link_path,
            context_name_to_link_path={
                self.package_link_path: self.package_link_path,
                self.mailbox_link_path: self.mailbox_link_path,
            },
        )
        mailbox_flag_raised = (
            check_activated(
                state=traj.final_state,
                obj_name=self.mailbox_flag_link,
                context_name_to_link_path={
                    self.mailbox_flag_link: self.mailbox_flag_link,
                },
            )
            if self.mailbox_flag_link is not None
            else True
        )
        return mailbox_closed and package_inside_mailbox and mailbox_flag_raised
