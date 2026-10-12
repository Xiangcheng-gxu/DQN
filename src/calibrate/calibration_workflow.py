"""可复用且精确的标定板运动学演示流程。"""

from __future__ import annotations

from dataclasses import dataclass
import time

import mujoco
import numpy as np

from .calibration_scene import BoardAttachment, CalibrationBoardGraspScene
from gripper import Gripper2F85
from robot import UR5eRobot


@dataclass(frozen=True)
class CalibrationWorkflowConfig:
    initial_joints_deg: tuple[float, ...]
    target_positions_m: tuple[tuple[float, float, float], ...]
    close_duration_s: float = 0.5
    attach_delay_s: float = 0.3
    target_hold_s: float = 1.0
    target_transition_s: float = 2.0
    return_transition_s: float = 2.0
    final_settle_s: float = 1.0


@dataclass(frozen=True)
class CalibrationWorkflowUpdate:
    initial_board_position_m: np.ndarray | None = None
    final_board_position_m: np.ndarray | None = None


class CalibrationBoardWorkflow:
    """附着标定板，依次访问笛卡尔目标点，再通过关节正运动学回归。"""

    def __init__(
        self,
        robot: UR5eRobot,
        gripper: Gripper2F85,
        scene: CalibrationBoardGraspScene,
        config: CalibrationWorkflowConfig,
    ) -> None:
        self.robot = robot
        self.gripper = gripper
        self.scene = scene
        self.config = config
        self._attachment: BoardAttachment | None = None
        self._path_started_at: float | None = None
        self._fk_return_started = False
        self._finished = False
        self._finished_at_data_time: float | None = None
        self._final_position_reported = False

    @property
    def is_finished(self) -> bool:
        return self._finished

    def prepare(
        self,
        data: mujoco.MjData,
        *,
        open_driver_qpos: float,
        grip_force_limit: float,
    ) -> None:
        self.robot.set_joint_angles_deg(data, self.config.initial_joints_deg)
        self.gripper.model.actuator_forcerange[self.gripper.actuator_id] = (
            -grip_force_limit,
            grip_force_limit,
        )
        self.scene.set_gripper_open_pose(data, open_driver_qpos)
        self.scene.disable_internal_pad_collisions()

    def before_step(
        self, data: mujoco.MjData, *, now_s: float | None = None
    ) -> CalibrationWorkflowUpdate:
        now = time.perf_counter() if now_s is None else float(now_s)
        if self._attachment is None:
            self.robot.set_joint_angles_deg(data, self.config.initial_joints_deg)
        else:
            finished_now = self._advance_motion(data, now)
            if finished_now:
                self._finished_at_data_time = data.time

        self._sync_board(data)
        # 标定流程需要连续开合轨迹，故在此局部计算执行器控制值；
        # Gripper2F85 本身保持仅提供 open()/close() 的最小接口。
        command = self._gripper_command(data.time)
        actuator_id = mujoco.mj_name2id(self.gripper.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
        low, high = self.gripper.model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = high - (high - low) * command / 255.0

        if (
            self._attachment is None
            and data.time >= self.config.close_duration_s + self.config.attach_delay_s
        ):
            # 在物理积分前记录夹持关系，此时是精确的初始正运动学姿态。
            self._attachment = self.scene.capture_attachment(data)
            initial_position = self.scene.board_position(data)
            self.robot.start_ik_position_cycle(
                data,
                self.config.target_positions_m,
                hold_s=self.config.target_hold_s,
                transition_s=self.config.target_transition_s,
                now_s=now,
            )
            self._path_started_at = now
            self._sync_board(data)
            return CalibrationWorkflowUpdate(initial_board_position_m=initial_position)
        return CalibrationWorkflowUpdate()

    def after_step(self, data: mujoco.MjData) -> CalibrationWorkflowUpdate:
        if self._finished:
            self.robot.set_joint_angles_deg(data, self.config.initial_joints_deg)
        self._sync_board(data)

        if (
            not self._final_position_reported
            and self._finished_at_data_time is not None
            and data.time >= self._finished_at_data_time + self.config.final_settle_s
        ):
            self._final_position_reported = True
            return CalibrationWorkflowUpdate(
                final_board_position_m=self.scene.board_position(data)
            )
        return CalibrationWorkflowUpdate()

    def _advance_motion(self, data: mujoco.MjData, now_s: float) -> bool:
        if self._finished:
            self.robot.set_joint_angles_deg(data, self.config.initial_joints_deg)
            return False
        if self._path_started_at is None:
            raise RuntimeError("Calibration motion has not been started")

        ik_duration_s = len(self.config.target_positions_m) * (
            self.config.target_hold_s + self.config.target_transition_s
        )
        if now_s < self._path_started_at + ik_duration_s:
            self.robot.update_ik_position_cycle(data, now_s=now_s)
            return False
        if not self._fk_return_started:
            self.robot.begin_joint_motion_deg(
                data,
                self.config.initial_joints_deg,
                self.config.return_transition_s,
                now_s=now_s,
            )
            self._fk_return_started = True
            return False
        if self.robot.update_joint_motion(data, now_s=now_s):
            self._finished = True
            self.robot.set_joint_angles_deg(data, self.config.initial_joints_deg)
            return True
        return False

    def _sync_board(self, data: mujoco.MjData) -> None:
        if self._attachment is None:
            self.scene.hold_board(data)
        else:
            self.scene.apply_attachment(data, self._attachment)

    def _gripper_command(self, time_s: float) -> int:
        progress = min(max(time_s / self.config.close_duration_s, 0.0), 1.0)
        return round(255 * (1.0 - progress))
