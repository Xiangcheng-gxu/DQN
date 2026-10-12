"""可复用的 MuJoCo 标定板夹持场景组装与附着辅助工具。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from robot import UR5eRobot


@dataclass(frozen=True)
class BoardAttachment:
    qpos_address: int
    dof_address: int
    relative_position: np.ndarray
    relative_rotation: np.ndarray


class CalibrationBoardGraspScene:
    """将可复用标定板临时加入已有项目场景。"""

    BOARD_BODY_NAME = "calibration_board"
    BOARD_FREEJOINT_NAME = "calibration_board_freejoint"
    GRIPPER_BODY_NAME = "robotiq_base"
    PAD_NAMES = ("left_pad1", "left_pad2", "right_pad1", "right_pad2")

    def __init__(
        self,
        scene_path: Path,
        board_path: Path,
        initial_joints_deg: tuple[float, ...],
        board_half_length_m: float,
        edge_insertion_m: float,
    ) -> None:
        self.scene_path = scene_path
        self.board_path = board_path
        self.initial_joints_deg = initial_joints_deg
        self.board_half_length_m = board_half_length_m
        self.edge_insertion_m = edge_insertion_m
        self._model: mujoco.MjModel | None = None
        self._board_id: int | None = None
        self._board_qpos_address: int | None = None
        self._board_dof_address: int | None = None
        self._held_board_pose: np.ndarray | None = None

    def create(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        base_model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        base_data = mujoco.MjData(base_model)
        pinch_pose = UR5eRobot(base_model).set_joint_angles_deg(
            base_data, self.initial_joints_deg
        )
        board_center = pinch_pose.position_m + pinch_pose.rotation_matrix[:, 2] * (
            self.board_half_length_m - self.edge_insertion_m
        )
        runtime_board = self.scene_path.parent / ".calibration_runtime_board.xml"
        runtime_scene = self.scene_path.parent / ".calibration_runtime_scene.xml"
        board_xml = re.sub(
            r'<body name="calibration_board" pos="[^"]+">',
            '<body name="calibration_board" pos="{:.6f} {:.6f} {:.6f}">'.format(
                *board_center
            ),
            self.board_path.read_text(encoding="utf-8"),
            count=1,
        )
        runtime_board.write_text(board_xml, encoding="utf-8")
        scene_xml = self.scene_path.read_text(encoding="utf-8").replace(
            "</worldbody>",
            '  <include file=".calibration_runtime_board.xml"/>\n</worldbody>',
            1,
        )
        runtime_scene.write_text(scene_xml, encoding="utf-8")
        try:
            model = mujoco.MjModel.from_xml_path(str(runtime_scene))
        finally:
            runtime_scene.unlink(missing_ok=True)
            runtime_board.unlink(missing_ok=True)

        data = mujoco.MjData(model)
        board_joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, self.BOARD_FREEJOINT_NAME
        )
        self._model = model
        self._board_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.BOARD_BODY_NAME
        )
        self._board_qpos_address = model.jnt_qposadr[board_joint_id]
        self._board_dof_address = model.jnt_dofadr[board_joint_id]
        self._held_board_pose = data.qpos[
            self._board_qpos_address : self._board_qpos_address + 7
        ].copy()
        mujoco.mj_forward(model, data)
        return model, data

    def set_gripper_open_pose(self, data: mujoco.MjData, driver_qpos: float) -> None:
        model = self._require_model()
        for joint_name in ("right_driver_joint", "left_driver_joint"):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            data.qpos[model.jnt_qposadr[joint_id]] = driver_qpos
        mujoco.mj_forward(model, data)
        self._held_board_pose = data.qpos[
            self._board_qpos_address : self._board_qpos_address + 7
        ].copy()

    def disable_internal_pad_collisions(self) -> None:
        model = self._require_model()
        for geom_name in self.PAD_NAMES:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            model.geom_conaffinity[geom_id] = 0

    def board_position(self, data: mujoco.MjData) -> np.ndarray:
        return data.xpos[self._board_id].copy()

    def hold_board(self, data: mujoco.MjData) -> None:
        self._apply_board_pose(data, self._held_board_pose)

    def capture_attachment(self, data: mujoco.MjData) -> BoardAttachment:
        model = self._require_model()
        gripper_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.GRIPPER_BODY_NAME
        )
        gripper_rotation = data.xmat[gripper_id].reshape(3, 3)
        return BoardAttachment(
            qpos_address=self._board_qpos_address,
            dof_address=self._board_dof_address,
            relative_position=gripper_rotation.T
            @ (data.xpos[self._board_id] - data.xpos[gripper_id]),
            relative_rotation=gripper_rotation.T
            @ data.xmat[self._board_id].reshape(3, 3),
        )

    def apply_attachment(self, data: mujoco.MjData, attachment: BoardAttachment) -> None:
        model = self._require_model()
        gripper_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.GRIPPER_BODY_NAME
        )
        # 夹爪包含耦合关节。第一次设置位姿后，正运动学可能更新夹爪 body
        # 坐标系，因此再使用稳定后的夹爪坐标系计算一次标定板位姿。
        for _ in range(2):
            gripper_rotation = data.xmat[gripper_id].reshape(3, 3)
            board_position = data.xpos[gripper_id] + gripper_rotation @ attachment.relative_position
            board_rotation = gripper_rotation @ attachment.relative_rotation
            board_quaternion = np.empty(4)
            mujoco.mju_mat2Quat(board_quaternion, board_rotation.ravel())
            self._apply_board_pose(
                data,
                np.concatenate((board_position, board_quaternion)),
                attachment.qpos_address,
                attachment.dof_address,
            )

    def _apply_board_pose(
        self,
        data: mujoco.MjData,
        pose: np.ndarray,
        qpos_address: int | None = None,
        dof_address: int | None = None,
    ) -> None:
        qpos = self._board_qpos_address if qpos_address is None else qpos_address
        dof = self._board_dof_address if dof_address is None else dof_address
        data.qpos[qpos : qpos + 7] = pose
        data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(self._require_model(), data)

    def _require_model(self) -> mujoco.MjModel:
        if self._model is None:
            raise RuntimeError("Call create() before using the calibration scene")
        return self._model
