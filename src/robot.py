"""本项目 UR5e 的角度制正运动学与逆运动学。"""
from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Sequence
import mujoco
import numpy as np

@dataclass(frozen=True)
class EndEffectorPose:
    position_m: np.ndarray
    rotation_matrix: np.ndarray

@dataclass(frozen=True)
class IKResult:
    joint_angles_deg: np.ndarray
    position_m: np.ndarray
    rotation_matrix: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    iterations: int
    success: bool

@dataclass(frozen=True)
class _JointMotion:
    start_deg: np.ndarray
    target_deg: np.ndarray
    start_time_s: float
    duration_s: float

@dataclass
class _PoseCycle:
    poses_deg: tuple[np.ndarray, ...]
    hold_s: float
    transition_s: float
    start_time_s: float
    active_index: int = 0
    motion_index: int = -1

@dataclass
class _IKPositionCycle:
    targets_m: tuple[np.ndarray, ...]
    target_rotation_matrix: np.ndarray
    hold_angles_deg: np.ndarray
    hold_s: float
    transition_s: float
    start_time_s: float
    active_index: int = 0
    motion_index: int = -1

class UR5eRobot:
    # Closed-gripper calibration: bottom centre is along pinch local +Z.
    PINCH_TO_BOTTOM_M = 0.017626
    JOINT_NAMES = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    ACTUATOR_NAMES = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
    END_EFFECTOR_SITE = "pinch"
    IK_REFERENCE_ANGLES_DEG = (33.3, -56.4, 100.0, 167.5, -60.5, -94.9)

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self._joint_ids = tuple(self._require_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.JOINT_NAMES)
        self._actuator_ids = tuple(self._require_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.ACTUATOR_NAMES)
        self._qpos_addresses = tuple(model.jnt_qposadr[joint_id] for joint_id in self._joint_ids)
        self._dof_addresses = tuple(model.jnt_dofadr[joint_id] for joint_id in self._joint_ids)
        self._pinch_site_id = self._require_id(mujoco.mjtObj.mjOBJ_SITE, self.END_EFFECTOR_SITE)
        self._ik_data = mujoco.MjData(model)
        self._motion: _JointMotion | None = None
        self._pose_cycle: _PoseCycle | None = None
        self._ik_cycle: _IKPositionCycle | None = None

    def set_joint_angles_deg(self, data: mujoco.MjData, joint_angles_deg: Sequence[float]) -> EndEffectorPose:
        angles_rad = np.deg2rad(self._validate_angles(joint_angles_deg))
        for qpos_adr, angle_rad in zip(self._qpos_addresses, angles_rad): data.qpos[qpos_adr] = angle_rad
        for actuator_id, angle_rad in zip(self._actuator_ids, angles_rad): data.ctrl[actuator_id] = angle_rad
        mujoco.mj_forward(self.model, data)
        return self.end_effector_pose(data)

    def bottom_center_pose(self, data: mujoco.MjData) -> EndEffectorPose:
        pinch = self.end_effector_pose(data)
        return EndEffectorPose(
            pinch.position_m + pinch.rotation_matrix[:, 2] * self.PINCH_TO_BOTTOM_M,
            pinch.rotation_matrix,
        )

    def solve_ik_bottom_position(self, data, target_position_m, *, target_rotation_matrix, **kwargs):
        rotation = self._validate_rotation(target_rotation_matrix)
        pinch_position = self._validate_position(target_position_m) - rotation[:, 2] * self.PINCH_TO_BOTTOM_M
        return self.solve_ik_position(data, pinch_position, target_rotation_matrix=rotation, **kwargs)

    def begin_joint_motion_deg(self, data: mujoco.MjData, target_angles_deg: Sequence[float], duration_s: float = 2.0, *, now_s: float | None = None) -> None:
        if not np.isfinite(duration_s) or duration_s <= 0: raise ValueError("duration_s must be a positive finite number")
        start_deg = self.joint_angles_deg(data)
        target_deg = self._validate_angles(target_angles_deg).copy()
        for index, joint_id in enumerate(self._joint_ids):
            if not self.model.jnt_limited[joint_id]:
                target_deg[index] = start_deg[index] + ((target_deg[index] - start_deg[index] + 180.0) % 360.0 - 180.0)
        self._motion = _JointMotion(start_deg, target_deg, time.perf_counter() if now_s is None else float(now_s), float(duration_s))

    def update_joint_motion(self, data: mujoco.MjData, *, now_s: float | None = None) -> bool:
        if self._motion is None: return True
        now = time.perf_counter() if now_s is None else float(now_s)
        progress = np.clip((now - self._motion.start_time_s) / self._motion.duration_s, 0.0, 1.0)
        alpha = progress * progress * (3.0 - 2.0 * progress)
        self.set_joint_angles_deg(data, (1.0 - alpha) * self._motion.start_deg + alpha * self._motion.target_deg)
        if progress >= 1.0:
            self._motion = None
            return True
        return False

    def task_orientation_mode(self, data: mujoco.MjData) -> str:
        """判断当前 pinch 局部 Z 轴相对桌面是垂直还是水平。"""
        tool_axis = self.end_effector_pose(data).rotation_matrix[:, 2]
        return "vertical" if abs(tool_axis[2]) >= np.linalg.norm(tool_axis[:2]) else "horizontal"

    def task_orientation_rotation(self, data: mujoco.MjData) -> np.ndarray:
        """返回最接近当前姿态的严格水平或垂直末端朝向。"""
        rotation = self.end_effector_pose(data).rotation_matrix
        tool_axis = rotation[:, 2]
        if self.task_orientation_mode(data) == "vertical":
            desired_z = np.array([0.0, 0.0, 1.0 if tool_axis[2] >= 0.0 else -1.0])
            desired_x = rotation[:, 0] - np.dot(rotation[:, 0], desired_z) * desired_z
            if np.linalg.norm(desired_x) < 1e-8:
                desired_x = np.array([1.0, 0.0, 0.0])
            desired_x /= np.linalg.norm(desired_x)
            desired_y = np.cross(desired_z, desired_x)
            desired_x = np.cross(desired_y, desired_z)
        else:
            desired_z = np.array([tool_axis[0], tool_axis[1], 0.0])
            desired_z /= np.linalg.norm(desired_z)
            desired_y = rotation[:, 1] - np.dot(rotation[:, 1], desired_z) * desired_z
            if np.linalg.norm(desired_y) < 1e-8:
                desired_y = np.array([0.0, 0.0, -1.0 if rotation[1, 2] < 0.0 else 1.0])
            desired_y /= np.linalg.norm(desired_y)
            desired_x = np.cross(desired_y, desired_z)
        return np.column_stack((desired_x, desired_y, desired_z))
    def solve_ik_position(self, data: mujoco.MjData, target_position_m: Sequence[float], *, target_rotation_matrix: Sequence[Sequence[float]] | None = None, initial_angles_deg: Sequence[float] | None = None, position_tolerance_m: float = 0.003, orientation_tolerance_rad: float = 0.03, max_iterations: int = 300, damping: float = 0.03, max_step_rad: float = 0.12, orientation_weight: float = 0.30) -> IKResult:
        target_position = self._validate_position(target_position_m)
        target_rotation = self.task_orientation_rotation(data) if target_rotation_matrix is None else self._validate_rotation(target_rotation_matrix)
        if position_tolerance_m <= 0 or orientation_tolerance_rad <= 0 or max_iterations < 1 or damping <= 0 or max_step_rad <= 0 or orientation_weight <= 0: raise ValueError("IK solver parameters must be positive")
        seed = self.IK_REFERENCE_ANGLES_DEG if initial_angles_deg is None else initial_angles_deg
        scratch = self._ik_data
        scratch.qpos[:] = data.qpos
        scratch.qvel[:] = 0.0
        scratch.ctrl[:] = data.ctrl
        for qpos_adr, angle_rad in zip(self._qpos_addresses, np.deg2rad(self._validate_angles(seed))): scratch.qpos[qpos_adr] = angle_rad
        mujoco.mj_forward(self.model, scratch)
        jacp = np.zeros((3, self.model.nv)); jacr = np.zeros((3, self.model.nv))
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            current_rotation = scratch.site_xmat[self._pinch_site_id].reshape(3, 3)
            position_error = target_position - scratch.site_xpos[self._pinch_site_id]
            orientation_error = self._orientation_error(current_rotation, target_rotation)
            if np.linalg.norm(position_error) <= position_tolerance_m and np.linalg.norm(orientation_error) <= orientation_tolerance_rad: break
            mujoco.mj_jacSite(self.model, scratch, jacp, jacr, self._pinch_site_id)
            jacobian = np.vstack((jacp[:, self._dof_addresses], orientation_weight * jacr[:, self._dof_addresses]))
            error = np.concatenate((position_error, orientation_weight * orientation_error))
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * damping * np.eye(6), error)
            delta = np.clip(delta, -max_step_rad, max_step_rad)
            for index, (joint_id, qpos_adr) in enumerate(zip(self._joint_ids, self._qpos_addresses)):
                value = scratch.qpos[qpos_adr] + delta[index]
                if self.model.jnt_limited[joint_id]: value = float(np.clip(value, *self.model.jnt_range[joint_id]))
                scratch.qpos[qpos_adr] = value
            mujoco.mj_forward(self.model, scratch)
        position = scratch.site_xpos[self._pinch_site_id].copy()
        rotation = scratch.site_xmat[self._pinch_site_id].reshape(3, 3).copy()
        position_error_m = float(np.linalg.norm(target_position - position))
        orientation_error_rad = float(np.linalg.norm(self._orientation_error(rotation, target_rotation)))
        return IKResult(np.rad2deg([scratch.qpos[a] for a in self._qpos_addresses]), position, rotation, position_error_m, orientation_error_rad, iterations, position_error_m <= position_tolerance_m and orientation_error_rad <= orientation_tolerance_rad)

    def begin_ik_position_motion(self, data: mujoco.MjData, target_position_m: Sequence[float], duration_s: float = 2.0, *, target_rotation_matrix: Sequence[Sequence[float]] | None = None, now_s: float | None = None) -> IKResult:
        result = self.solve_ik_position(data, target_position_m, target_rotation_matrix=target_rotation_matrix, initial_angles_deg=self.joint_angles_deg(data))
        if not result.success: raise RuntimeError(f"IK did not reach target; position={result.position_error_m:.4f} m, orientation={result.orientation_error_rad:.4f} rad")
        self.begin_joint_motion_deg(data, result.joint_angles_deg, duration_s, now_s=now_s)
        return result

    def start_pose_cycle_deg(self, data: mujoco.MjData, poses_deg: Sequence[Sequence[float]], hold_s: float = 1.0, transition_s: float = 2.0, *, now_s: float | None = None) -> None:
        if len(poses_deg) < 2: raise ValueError("A pose cycle requires at least two poses")
        self._validate_cycle_times(hold_s, transition_s)
        poses = tuple(self._validate_angles(pose) for pose in poses_deg)
        self._pose_cycle = _PoseCycle(poses, float(hold_s), float(transition_s), time.perf_counter() if now_s is None else float(now_s))
        self._motion = None
        self.set_joint_angles_deg(data, poses[0])

    def update_pose_cycle(self, data: mujoco.MjData, *, now_s: float | None = None) -> None:
        if self._pose_cycle is None: raise RuntimeError("No pose cycle has been started")
        cycle = self._pose_cycle; now = time.perf_counter() if now_s is None else float(now_s)
        index, phase = self._cycle_phase(now, cycle.start_time_s, cycle.hold_s, cycle.transition_s, len(cycle.poses_deg))
        if index != cycle.active_index:
            cycle.active_index, cycle.motion_index, self._motion = index, -1, None
            self.set_joint_angles_deg(data, cycle.poses_deg[index])
        if phase >= cycle.hold_s:
            if cycle.motion_index != index:
                self.begin_joint_motion_deg(data, cycle.poses_deg[(index + 1) % len(cycle.poses_deg)], cycle.transition_s, now_s=now)
                cycle.motion_index = index
            self.update_joint_motion(data, now_s=now)

    def start_ik_position_cycle(self, data: mujoco.MjData, target_positions_m: Sequence[Sequence[float]], hold_s: float = 1.0, transition_s: float = 2.0, *, now_s: float | None = None) -> None:
        if len(target_positions_m) < 2: raise ValueError("An IK target cycle requires at least two positions")
        self._validate_cycle_times(hold_s, transition_s)
        targets = tuple(self._validate_position(target) for target in target_positions_m)
        now = time.perf_counter() if now_s is None else float(now_s)
        self._ik_cycle = _IKPositionCycle(
            targets,
            self.task_orientation_rotation(data),
            self.joint_angles_deg(data),
            float(hold_s),
            float(transition_s),
            now,
        )
        self._motion = None

    def update_ik_position_cycle(self, data: mujoco.MjData, *, now_s: float | None = None) -> None:
        if self._ik_cycle is None: raise RuntimeError("No IK target cycle has been started")
        cycle = self._ik_cycle; now = time.perf_counter() if now_s is None else float(now_s)
        index, phase = self._cycle_phase(now, cycle.start_time_s, cycle.hold_s, cycle.transition_s, len(cycle.targets_m))
        if phase < cycle.hold_s:
            self.set_joint_angles_deg(data, cycle.hold_angles_deg)
            return
        if phase >= cycle.hold_s and cycle.motion_index != index:
            result = self.begin_ik_position_motion(data, cycle.targets_m[index], cycle.transition_s, target_rotation_matrix=cycle.target_rotation_matrix, now_s=now)
            cycle.hold_angles_deg = result.joint_angles_deg.copy()
            cycle.motion_index = index
        self.update_joint_motion(data, now_s=now)

    @property
    def is_joint_motion_active(self) -> bool: return self._motion is not None
    def joint_angles_deg(self, data: mujoco.MjData) -> np.ndarray: return np.rad2deg([data.qpos[a] for a in self._qpos_addresses])
    def end_effector_pose(self, data: mujoco.MjData) -> EndEffectorPose: return EndEffectorPose(data.site_xpos[self._pinch_site_id].copy(), data.site_xmat[self._pinch_site_id].reshape(3, 3).copy())
    def _cycle_phase(self, now_s: float, start_time_s: float, hold_s: float, transition_s: float, count: int) -> tuple[int, float]:
        elapsed = max(0.0, now_s - start_time_s); duration = hold_s + transition_s
        return int(elapsed // duration) % count, elapsed % duration
    def _require_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0: raise ValueError(f"Required MuJoCo object {name!r} was not found")
        return object_id
    @staticmethod
    def _orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, (target @ current.T).ravel())
        error = np.empty(3)
        mujoco.mju_quat2Vel(error, quaternion, 1.0)
        return error
    @staticmethod
    def _validate_angles(values: Sequence[float]) -> np.ndarray:
        if len(values) != 6: raise ValueError("UR5e requires exactly six joint angles in degrees")
        result = np.asarray(values, dtype=float)
        if not np.isfinite(result).all(): raise ValueError("Joint angles must be finite numbers")
        return result
    @staticmethod
    def _validate_position(values: Sequence[float]) -> np.ndarray:
        if len(values) != 3: raise ValueError("Cartesian targets require exactly three coordinates in metres")
        result = np.asarray(values, dtype=float)
        if not np.isfinite(result).all(): raise ValueError("Cartesian target coordinates must be finite numbers")
        return result
    @staticmethod
    def _validate_rotation(values: Sequence[Sequence[float]]) -> np.ndarray:
        result = np.asarray(values, dtype=float)
        if result.shape != (3, 3) or not np.isfinite(result).all(): raise ValueError("Target orientation must be a finite 3x3 rotation matrix")
        if not np.allclose(result.T @ result, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(result), 1.0, atol=1e-5): raise ValueError("Target orientation must be a proper rotation matrix")
        return result
    @staticmethod
    def _validate_cycle_times(hold_s: float, transition_s: float) -> None:
        if not np.isfinite(hold_s) or hold_s < 0: raise ValueError("hold_s must be a finite non-negative number")
        if not np.isfinite(transition_s) or transition_s <= 0: raise ValueError("transition_s must be a positive finite number")
