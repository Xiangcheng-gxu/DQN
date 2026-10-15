"""2F-85夹爪：只通过fingers_actuator控制，不直接写两侧手指关节角。"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class Gripper2F85:
    """本模型指令0是全开，255是全闭，不需要再反向换算。

    只写执行器ctrl，再由mj_step求解连杆、肌腱和约束；发出255不代表已经闭合。
    推动前由控制器调用is_closed并等待指垫平行、位置稳定。
    """

    model: mujoco.MjModel
    actuator_name: str = "fingers_actuator"

    # 完全闭合且指垫平行时，底部5mm范围的碰撞包络，不是整个夹爪外壳尺寸。
    # 沿推向16mm，横向22mm；用于在高度图中建立落点检测框。
    CLOSED_TIP_ALONG_PUSH_M = 0.016
    CLOSED_TIP_ACROSS_PUSH_M = 0.022
    COMMAND_OPEN = 0
    COMMAND_CLOSED = 255
    CLOSED_STABLE_TIME_S = 0.1
    CLOSED_POSITION_TOLERANCE_RAD = 0.001
    PAD_ALIGNMENT_TOLERANCE_RAD = 0.001

    def __post_init__(self) -> None:
        actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.actuator_name)
        if actuator_id < 0:
            raise ValueError(f"Actuator {self.actuator_name!r} was not found")
        object.__setattr__(self, "_actuator_id", actuator_id)
        joints = tuple(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{link}_joint")
                       for side in ("left", "right") for link in ("driver", "spring_link", "follower", "coupler"))
        pads = tuple(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_pad") for side in ("left", "right"))
        base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_base")
        if min((*joints, *pads, base)) < 0:
            raise ValueError("Missing required 2F-85 linkage or pad body")
        object.__setattr__(self, "_linkage_joint_ids", joints)
        object.__setattr__(self, "_driver_joint_ids", (joints[0], joints[4]))
        object.__setattr__(self, "_pad_body_ids", pads)
        object.__setattr__(self, "_base_body_id", base)

    @property
    def actuator_id(self) -> int:
        return self._actuator_id

    def mujoco_control(self, command: int) -> float:
        """把0到255指令映射到XML执行器ctrlrange，不反转开合方向。"""
        command = self._validate(command)
        low, high = self.model.actuator_ctrlrange[self.actuator_id]
        return float(low + (high - low) * command / self.COMMAND_CLOSED)

    def set_opening(self, data: mujoco.MjData, command: int) -> None:
        """只写开合指令；调用方还需要执行物理步，手指才会真的运动。"""
        data.ctrl[self.actuator_id] = self.mujoco_control(command)

    def close(self, data: mujoco.MjData) -> None:
        self.set_opening(data, self.COMMAND_CLOSED)

    def is_closed(self, data: mujoco.MjData, *, settled: bool = True) -> bool:
        """同时检查驱动关节闭合量和指垫平行；可选检查关节速度是否稳定。"""
        for joint in self._driver_joint_ids:
            if data.qpos[self.model.jnt_qposadr[joint]] < .775:
                return False
        if max(self.pad_alignment_errors_rad(data)) > self.PAD_ALIGNMENT_TOLERANCE_RAD:
            return False
        if settled:
            for joint in self._linkage_joint_ids:
                if abs(data.qvel[self.model.jnt_dofadr[joint]]) >= .01:
                    return False
        return True

    def pad_alignment_errors_rad(self, data: mujoco.MjData) -> tuple[float, float, float]:
        """返回左指垫倾角、右指垫倾角、两指垫夹角，单位为弧度。"""
        base_z = data.xmat[self._base_body_id].reshape(3, 3)[:, 2]
        left_z, right_z = (data.xmat[b].reshape(3, 3)[:, 2] for b in self._pad_body_ids)
        return tuple(float(np.arccos(np.clip(a @ b, -1, 1)))
                     for a, b in ((base_z, left_z), (base_z, right_z), (left_z, right_z)))

    @staticmethod
    def _validate(command: int) -> int:
        if isinstance(command, bool) or not isinstance(command, int):
            raise TypeError("Gripper command must be an integer from 0 to 255")
        if not Gripper2F85.COMMAND_OPEN <= command <= Gripper2F85.COMMAND_CLOSED:
            raise ValueError("Gripper command must be in the range 0..255")
        return command
