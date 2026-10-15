"""UR5e 运动学：只保留训练用的关节读写、夹爪底部定位和逆运动学。

阅读顺序：bottom_center_pose -> solve_ik_bottom_position -> solve_ik_position。
本文件只计算机械臂该到哪里；真正逐步执行动作的是 push_controller.py。
对外关节角使用度，MuJoCo 内部使用弧度；位置统一使用米。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np


@dataclass(frozen=True)
class EndEffectorPose:
    """末端位姿：三维位置，以及三个局部坐标轴在世界坐标中的方向。"""

    position_m: np.ndarray
    rotation_matrix: np.ndarray


@dataclass(frozen=True)
class IKResult:
    """逆运动学结果。必须检查 success，失败时不能直接拿关节角去运动。"""

    joint_angles_deg: np.ndarray
    position_m: np.ndarray
    rotation_matrix: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    iterations: int
    success: bool


class UR5eRobot:
    # 标定后的控制点是“完全闭合夹爪的底部中心”，不是 XML 的 pinch 原点。
    # 向 pinch 的局部 +Z 偏移 17.626 mm；夹爪朝下时，世界 Z 相应减小。
    PINCH_TO_BOTTOM_M = 0.017626
    JOINT_NAMES = (
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    )
    ACTUATOR_NAMES = (
        "shoulder_pan", "shoulder_lift", "elbow",
        "wrist_1", "wrist_2", "wrist_3",
    )
    END_EFFECTOR_SITE = "pinch"
    IK_REFERENCE_ANGLES_DEG = (33.3, -56.4, 100.0, 167.5, -60.5, -94.9)

    def __init__(self, model: mujoco.MjModel) -> None:
        """把 XML 中的名字转换成数组地址，后续不再反复查找。"""
        self.model = model
        self._joint_ids = tuple(
            self._require_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.JOINT_NAMES)
        self._actuator_ids = tuple(
            self._require_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.ACTUATOR_NAMES)
        self._qpos_addresses = tuple(model.jnt_qposadr[j] for j in self._joint_ids)
        self._dof_addresses = tuple(model.jnt_dofadr[j] for j in self._joint_ids)
        self._pinch_site_id = self._require_id(mujoco.mjtObj.mjOBJ_SITE, self.END_EFFECTOR_SITE)
        # IK 在独立数据中试算，不修改正在运行的真实场景。
        self._ik_data = mujoco.MjData(model)

    def set_joint_angles_deg(self, data, joint_angles_deg) -> EndEffectorPose:
        """设置六个机械臂关节角，不直接改夹爪关节或桌面物体的位置。"""
        angles_rad = np.deg2rad(self._validate_angles(joint_angles_deg))
        for address, angle in zip(self._qpos_addresses, angles_rad):
            data.qpos[address] = angle
        for actuator, angle in zip(self._actuator_ids, angles_rad):
            data.ctrl[actuator] = angle
        # 更新正运动学，让 site_xpos/site_xmat 与新关节角一致。
        mujoco.mj_forward(self.model, data)
        return self.end_effector_pose(data)

    def joint_angles_deg(self, data) -> np.ndarray:
        """从仿真数组读取六个关节角，并从弧度转换为度。"""
        return np.rad2deg([data.qpos[a] for a in self._qpos_addresses])

    def end_effector_pose(self, data) -> EndEffectorPose:
        """读取 pinch 位姿；复制数组，避免后续仿真步把返回值一起改掉。"""
        return EndEffectorPose(
            data.site_xpos[self._pinch_site_id].copy(),
            data.site_xmat[self._pinch_site_id].reshape(3, 3).copy(),
        )

    def bottom_center_pose(self, data) -> EndEffectorPose:
        """用 pinch 位姿换算夹爪底部中心，这才是推动动作的控制点。"""
        pinch = self.end_effector_pose(data)
        return EndEffectorPose(
            pinch.position_m + pinch.rotation_matrix[:, 2] * self.PINCH_TO_BOTTOM_M,
            pinch.rotation_matrix,
        )

    def solve_ik_bottom_position(self, data, target_position_m, *,
                                 target_rotation_matrix, **kwargs):
        """输入底部中心目标，先换算成 pinch 目标，再求六个关节角。"""
        rotation = self._validate_rotation(target_rotation_matrix)
        pinch_position = self._validate_position(target_position_m) - rotation[:, 2] * self.PINCH_TO_BOTTOM_M
        return self.solve_ik_position(
            data, pinch_position, target_rotation_matrix=rotation, **kwargs)

    def solve_ik_position(
        self, data, target_position_m: Sequence[float], *,
        target_rotation_matrix: Sequence[Sequence[float]],
        initial_angles_deg: Sequence[float] | None = None,
        position_tolerance_m: float = 0.003,
        orientation_tolerance_rad: float = 0.03,
        max_iterations: int = 300, damping: float = 0.03,
        max_step_rad: float = 0.12, orientation_weight: float = 0.30,
    ) -> IKResult:
        """阻尼最小二乘 IK，同时接近目标位置和目标朝向。

        每次用雅可比矩阵估计“各关节转一点，末端会移动多少”，
        再修正一小步。阻尼用于减轻奇异位形附近的数值不稳定。
        训练传入上一时刻的关节角作为初值，减少突然切换到另一组解。
        """
        target_position = self._validate_position(target_position_m)
        target_rotation = self._validate_rotation(target_rotation_matrix)
        if (position_tolerance_m <= 0 or orientation_tolerance_rad <= 0
                or max_iterations < 1 or damping <= 0 or max_step_rad <= 0
                or orientation_weight <= 0):
            raise ValueError("IK solver parameters must be positive")
        seed = self.IK_REFERENCE_ANGLES_DEG if initial_angles_deg is None else initial_angles_deg
        scratch = self._ik_data
        scratch.qpos[:] = data.qpos
        scratch.qvel[:] = 0.0
        scratch.ctrl[:] = data.ctrl
        for address, angle in zip(self._qpos_addresses, np.deg2rad(self._validate_angles(seed))):
            scratch.qpos[address] = angle
        mujoco.mj_forward(self.model, scratch)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            current_rotation = scratch.site_xmat[self._pinch_site_id].reshape(3, 3)
            position_error = target_position - scratch.site_xpos[self._pinch_site_id]
            orientation_error = self._orientation_error(current_rotation, target_rotation)
            if (np.linalg.norm(position_error) <= position_tolerance_m
                    and np.linalg.norm(orientation_error) <= orientation_tolerance_rad):
                break
            mujoco.mj_jacSite(self.model, scratch, jacp, jacr, self._pinch_site_id)
            # 只取机械臂的六个自由度，不让 IK 通过移动物体来“解决”误差。
            jacobian = np.vstack((
                jacp[:, self._dof_addresses],
                orientation_weight * jacr[:, self._dof_addresses],
            ))
            error = np.concatenate((position_error, orientation_weight * orientation_error))
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * damping * np.eye(6), error)
            delta = np.clip(delta, -max_step_rad, max_step_rad)
            for index, (joint, address) in enumerate(zip(self._joint_ids, self._qpos_addresses)):
                value = scratch.qpos[address] + delta[index]
                if self.model.jnt_limited[joint]:
                    value = float(np.clip(value, *self.model.jnt_range[joint]))
                scratch.qpos[address] = value
            mujoco.mj_forward(self.model, scratch)
        position = scratch.site_xpos[self._pinch_site_id].copy()
        rotation = scratch.site_xmat[self._pinch_site_id].reshape(3, 3).copy()
        position_error_m = float(np.linalg.norm(target_position - position))
        orientation_error_rad = float(np.linalg.norm(self._orientation_error(rotation, target_rotation)))
        success = (position_error_m <= position_tolerance_m
                   and orientation_error_rad <= orientation_tolerance_rad)
        return IKResult(
            np.rad2deg([scratch.qpos[a] for a in self._qpos_addresses]),
            position, rotation, position_error_m, orientation_error_rad, iterations, success,
        )

    def _require_id(self, object_type, name) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Required MuJoCo object {name!r} was not found")
        return object_id

    @staticmethod
    def _orientation_error(current, target) -> np.ndarray:
        """把两个朝向的差转换成三维旋转误差，单位为弧度。"""
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, (target @ current.T).ravel())
        error = np.empty(3)
        mujoco.mju_quat2Vel(error, quaternion, 1.0)
        return error

    @staticmethod
    def _validate_angles(values) -> np.ndarray:
        if len(values) != 6:
            raise ValueError("UR5e requires exactly six joint angles in degrees")
        result = np.asarray(values, dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("Joint angles must be finite numbers")
        return result

    @staticmethod
    def _validate_position(values) -> np.ndarray:
        if len(values) != 3:
            raise ValueError("Cartesian targets require exactly three coordinates in metres")
        result = np.asarray(values, dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("Cartesian target coordinates must be finite numbers")
        return result

    @staticmethod
    def _validate_rotation(values) -> np.ndarray:
        result = np.asarray(values, dtype=float)
        if result.shape != (3, 3) or not np.isfinite(result).all():
            raise ValueError("Target orientation must be a finite 3x3 rotation matrix")
        if (not np.allclose(result.T @ result, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(result), 1.0, atol=1e-5)):
            raise ValueError("Target orientation must be a proper rotation matrix")
        return result
