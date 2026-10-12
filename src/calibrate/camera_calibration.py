"""遵循真实 UR5 流程的 MuJoCo RGB-D 相机标定。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import cv2
import mujoco
import numpy as np

from .calibration_scene import CalibrationBoardGraspScene
from gripper import Gripper2F85
from robot import UR5eRobot


@dataclass(frozen=True)
class RealUR5CalibrationConfig:
    """真实项目中相对桌面的 5 cm 步长、45 点标定网格。"""

    initial_joints_deg: tuple[float, ...] = (
        0.0,
        -135.0,
        135.0,
        0.0,
        90.0,
        -90.0,
    )
    workspace_limits_m: tuple[tuple[float, float], ...] = (
        (-0.55, -0.45),
        (-0.05, 0.15),
        (0.10, 0.20),
    )
    grid_step_m: float = 0.05
    tool_positions_m: tuple[tuple[float, float, float], ...] | None = None
    tabletop_height_m: float = 0.75
    checkerboard_inner_corners: tuple[int, int] = (9, 6)
    square_size_m: float = 0.03
    camera_name: str = "topdown_depth"
    image_width: int = 640
    image_height: int = 480
    open_driver_qpos: float = 0.0026
    grip_force_limit: float = 100.0
    collision_safe_seed_joints_deg: tuple[float, ...] = (
        0.0,
        -60.0,
        120.0,
        -90.0,
        90.0,
        -90.0,
    )


@dataclass(frozen=True)
class CameraCalibrationResult:
    output_dir: Path
    valid_views: int
    total_views: int
    depth_scale: float
    camera_to_world: np.ndarray
    world_to_camera: np.ndarray
    intrinsics: np.ndarray
    rmse_m: float


class MuJoCoCameraCalibrator:
    """在当前 MuJoCo 场景中执行真实 UR5 风格的 RGB-D 标定流程。"""

    BOARD_FEATURE_Z_M = 0.00301

    def __init__(
        self,
        scene_path: Path,
        board_path: Path,
        output_dir: Path,
        config: RealUR5CalibrationConfig = RealUR5CalibrationConfig(),
    ) -> None:
        self.scene_path = Path(scene_path)
        self.board_path = Path(board_path)
        self.output_dir = Path(output_dir)
        self.config = config

    def run(self) -> CameraCalibrationResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        scene = CalibrationBoardGraspScene(
            scene_path=self.scene_path,
            board_path=self.board_path,
            initial_joints_deg=self.config.initial_joints_deg,
            board_half_length_m=0.165,
            edge_insertion_m=0.030,
        )
        model, data = scene.create()
        robot = UR5eRobot(model)
        gripper = Gripper2F85(model)
        robot.set_joint_angles_deg(data, self.config.initial_joints_deg)
        model.actuator_forcerange[gripper.actuator_id] = (
            -self.config.grip_force_limit,
            self.config.grip_force_limit,
        )
        scene.set_gripper_open_pose(data, self.config.open_driver_qpos)
        scene.disable_internal_pad_collisions()
        scene.hold_board(data)
        attachment = scene.capture_attachment(data)
        target_rotation = robot.end_effector_pose(data).rotation_matrix

        intrinsics = self._intrinsics(model)
        renderer = mujoco.Renderer(
            model,
            height=self.config.image_height,
            width=self.config.image_width,
        )
        measured_points: list[np.ndarray] = []
        observed_points: list[np.ndarray] = []
        tool_positions: list[np.ndarray] = []

        try:
            for index, tool_position in enumerate(self._tool_positions(), start=1):
                solution = robot.solve_ik_position(
                    data,
                    tool_position,
                    target_rotation_matrix=target_rotation,
                    initial_angles_deg=self.config.collision_safe_seed_joints_deg,
                )
                if not solution.success:
                    raise RuntimeError(
                        f"IK failed at calibration point {index}: "
                        f"position error={solution.position_error_m:.6f} m"
                    )
                robot.set_joint_angles_deg(data, solution.joint_angles_deg)
                scene.apply_attachment(data, attachment)
                rgb, depth_m = self._render(renderer, data)
                found, corners = self._find_corners(rgb)
                visualization = rgb.copy()
                if found:
                    cv2.drawChessboardCorners(
                        visualization,
                        self.config.checkerboard_inner_corners,
                        corners.reshape(-1, 1, 2),
                        True,
                    )
                    pixel = corners.reshape(-1, 2).mean(axis=0)
                    depth = self._depth_at(depth_m, pixel)
                    if depth > 0.0 and np.isfinite(depth):
                        observed_points.append(
                            self._deproject(pixel, depth, intrinsics)
                        )
                        measured_points.append(
                            self._board_feature_position(scene, data)
                        )
                        tool_positions.append(tool_position.copy())
                cv2.imwrite(
                    str(image_dir / f"{index:06d}.png"),
                    cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR),
                )
        finally:
            renderer.close()

        if len(measured_points) < 3:
            raise RuntimeError(
                f"Only {len(measured_points)} valid checkerboard observations; need at least 3."
            )

        measured = np.asarray(measured_points)
        observed = np.asarray(observed_points)
        depth_scale, world_to_camera, rmse_m = self._optimize_depth_scale(
            measured, observed
        )
        camera_to_world = np.linalg.inv(world_to_camera)
        self._save(
            intrinsics,
            depth_scale,
            world_to_camera,
            camera_to_world,
            rmse_m,
            np.asarray(tool_positions),
            measured,
            observed,
        )
        return CameraCalibrationResult(
            output_dir=self.output_dir,
            valid_views=len(measured),
            total_views=len(self._tool_positions()),
            depth_scale=depth_scale,
            camera_to_world=camera_to_world,
            world_to_camera=world_to_camera,
            intrinsics=intrinsics,
            rmse_m=rmse_m,
        )

    def run_visual_demo(
        self,
        *,
        motion_duration_s: float = 1.5,
        capture_hold_s: float = 1.0,
        grasp_settle_s: float = 0.8,
    ) -> CameraCalibrationResult:
        """以可视化实时运动方式执行同一套 45 点标定。"""
        if motion_duration_s <= 0.0 or capture_hold_s < 0.0 or grasp_settle_s < 0.0:
            raise ValueError("Motion and grasp times must be non-negative; motion must be positive")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        scene = CalibrationBoardGraspScene(
            scene_path=self.scene_path,
            board_path=self.board_path,
            initial_joints_deg=self.config.initial_joints_deg,
            board_half_length_m=0.165,
            edge_insertion_m=0.030,
        )
        model, data = scene.create()
        robot = UR5eRobot(model)
        gripper = Gripper2F85(model)
        robot.set_joint_angles_deg(data, self.config.initial_joints_deg)
        model.actuator_forcerange[gripper.actuator_id] = (
            -self.config.grip_force_limit,
            self.config.grip_force_limit,
        )
        scene.set_gripper_open_pose(data, self.config.open_driver_qpos)
        scene.disable_internal_pad_collisions()
        scene.hold_board(data)
        attachment = None
        target_rotation = robot.end_effector_pose(data).rotation_matrix
        positions = self._tool_positions()
        # 初始姿态直接前往最低网格点会让水平标定板扫过桌面。
        # 因此先抬升到安全高度，再按真实网格轨迹移动。
        motion_waypoints = (
            np.array((positions[0][0], positions[0][1], 1.05)),
            *positions,
        )
        intrinsics = self._intrinsics(model)
        renderer = mujoco.Renderer(
            model,
            height=self.config.image_height,
            width=self.config.image_width,
        )
        measured_points: list[np.ndarray] = []
        observed_points: list[np.ndarray] = []
        tool_positions: list[np.ndarray] = []
        target_index = 0
        moving = False
        hold_started_at: float | None = None
        held_joint_angles: np.ndarray | None = None
        completed = False
        result: CameraCalibrationResult | None = None
        completion_announced_at: float | None = None

        def begin_next_motion(now_s: float) -> np.ndarray:
            solution = robot.solve_ik_position(
                data,
                motion_waypoints[target_index],
                target_rotation_matrix=target_rotation,
                initial_angles_deg=self.config.collision_safe_seed_joints_deg,
            )
            if not solution.success:
                raise RuntimeError(
                    f"IK failed at calibration point {target_index + 1}: "
                    f"position error={solution.position_error_m:.6f} m"
                )
            robot.begin_joint_motion_deg(
                data,
                solution.joint_angles_deg,
                motion_duration_s,
                now_s=now_s,
            )
            return solution.joint_angles_deg

        from mujoco import viewer

        try:
            with viewer.launch_passive(model, data) as simulation_viewer:
                simulation_viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                simulation_viewer.cam.lookat[:] = (-0.50, 0.0, 0.95)
                simulation_viewer.cam.distance = 2.4
                simulation_viewer.cam.azimuth = -90.0
                simulation_viewer.cam.elevation = -35.0
                while simulation_viewer.is_running():
                    started_at = time.perf_counter()
                    gripper.close(data)
                    if attachment is None:
                        robot.set_joint_angles_deg(data, self.config.initial_joints_deg)
                        scene.hold_board(data)
                        if data.time >= grasp_settle_s:
                            # 仅在可见的夹爪手指闭合后记录夹持关系。
                            attachment = scene.capture_attachment(data)
                            scene.apply_attachment(data, attachment)
                    elif not completed:
                        if not moving and hold_started_at is None:
                            held_joint_angles = begin_next_motion(started_at)
                            moving = True
                        elif moving:
                            if robot.update_joint_motion(data, now_s=started_at):
                                moving = False
                                hold_started_at = data.time
                        elif held_joint_angles is not None:
                            robot.set_joint_angles_deg(data, held_joint_angles)
                            if data.time >= hold_started_at + capture_hold_s:
                                if target_index > 0:
                                    self._capture_calibration_view(
                                        renderer,
                                        scene,
                                        data,
                                        target_index,
                                        positions[target_index - 1],
                                        intrinsics,
                                        image_dir,
                                        measured_points,
                                        observed_points,
                                        tool_positions,
                                    )
                                target_index += 1
                                hold_started_at = None
                                if target_index == len(motion_waypoints):
                                    completed = True
                                    result = self._finalize_result(
                                        intrinsics,
                                        measured_points,
                                        observed_points,
                                        tool_positions,
                                        len(positions),
                                    )
                                    completion_announced_at = time.perf_counter()
                                    print("标定完成，窗口将在 5 秒后自动关闭。")

                    if attachment is None:
                        scene.hold_board(data)
                    else:
                        scene.apply_attachment(data, attachment)
                    mujoco.mj_step(model, data)
                    if attachment is None:
                        robot.set_joint_angles_deg(data, self.config.initial_joints_deg)
                        scene.hold_board(data)
                    elif not moving and held_joint_angles is not None:
                        robot.set_joint_angles_deg(data, held_joint_angles)
                    if attachment is not None:
                        scene.apply_attachment(data, attachment)
                    simulation_viewer.sync()
                    if (
                        completion_announced_at is not None
                        and time.perf_counter() - completion_announced_at >= 5.0
                    ):
                        break
                    remaining = model.opt.timestep - (time.perf_counter() - started_at)
                    if remaining > 0.0:
                        time.sleep(remaining)
        finally:
            renderer.close()

        if result is None:
            raise RuntimeError("Calibration demo ended before all 45 views were captured")
        return result

    def _capture_calibration_view(
        self,
        renderer: mujoco.Renderer,
        scene: CalibrationBoardGraspScene,
        data: mujoco.MjData,
        image_index: int,
        tool_position: np.ndarray,
        intrinsics: np.ndarray,
        image_dir: Path,
        measured_points: list[np.ndarray],
        observed_points: list[np.ndarray],
        tool_positions: list[np.ndarray],
    ) -> None:
        rgb, depth_m = self._render(renderer, data)
        found, corners = self._find_corners(rgb)
        visualization = rgb.copy()
        if found:
            cv2.drawChessboardCorners(
                visualization,
                self.config.checkerboard_inner_corners,
                corners.reshape(-1, 1, 2),
                True,
            )
            pixel = corners.reshape(-1, 2).mean(axis=0)
            depth = self._depth_at(depth_m, pixel)
            if depth > 0.0 and np.isfinite(depth):
                observed_points.append(self._deproject(pixel, depth, intrinsics))
                measured_points.append(self._board_feature_position(scene, data))
                tool_positions.append(tool_position.copy())
        cv2.imwrite(
            str(image_dir / f"{image_index:06d}.png"),
            cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR),
        )

    def _finalize_result(
        self,
        intrinsics: np.ndarray,
        measured_points: list[np.ndarray],
        observed_points: list[np.ndarray],
        tool_positions: list[np.ndarray],
        total_views: int,
    ) -> CameraCalibrationResult:
        if len(measured_points) < 3:
            raise RuntimeError(
                f"Only {len(measured_points)} valid checkerboard observations; need at least 3."
            )
        measured = np.asarray(measured_points)
        observed = np.asarray(observed_points)
        depth_scale, world_to_camera, rmse_m = self._optimize_depth_scale(
            measured, observed
        )
        camera_to_world = np.linalg.inv(world_to_camera)
        self._save(
            intrinsics,
            depth_scale,
            world_to_camera,
            camera_to_world,
            rmse_m,
            np.asarray(tool_positions),
            measured,
            observed,
        )
        return CameraCalibrationResult(
            output_dir=self.output_dir,
            valid_views=len(measured),
            total_views=total_views,
            depth_scale=depth_scale,
            camera_to_world=camera_to_world,
            world_to_camera=world_to_camera,
            intrinsics=intrinsics,
            rmse_m=rmse_m,
        )

    def _tool_positions(self) -> tuple[np.ndarray, ...]:
        if self.config.tool_positions_m is not None:
            return tuple(
                np.asarray(position, dtype=float) for position in self.config.tool_positions_m
            )

        axes = [
            np.linspace(lower, upper, round(1 + (upper - lower) / self.config.grid_step_m))
            for lower, upper in self.config.workspace_limits_m
        ]
        x, y, z = np.meshgrid(*axes)
        positions = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        positions[:, 2] += self.config.tabletop_height_m
        return tuple(position for position in positions)

    def _intrinsics(self, model: mujoco.MjModel) -> np.ndarray:
        camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, self.config.camera_name
        )
        if camera_id < 0:
            raise ValueError(f"Camera {self.config.camera_name!r} was not found")
        fy = self.config.image_height / (
            2.0 * np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2.0)
        )
        return np.array(
            [
                [fy, 0.0, (self.config.image_width - 1.0) / 2.0],
                [0.0, fy, (self.config.image_height - 1.0) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def _render(
        self, renderer: mujoco.Renderer, data: mujoco.MjData
    ) -> tuple[np.ndarray, np.ndarray]:
        renderer.update_scene(data, camera=self.config.camera_name)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        depth_m = renderer.render().copy()
        renderer.disable_depth_rendering()
        return rgb, depth_m

    def _find_corners(self, rgb: np.ndarray) -> tuple[bool, np.ndarray]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray,
            self.config.checkerboard_inner_corners,
            cv2.CALIB_CB_ADAPTIVE_THRESH,
        )
        if not found:
            found, corners = cv2.findChessboardCornersSB(
                gray,
                self.config.checkerboard_inner_corners,
                cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
            )
        if not found or corners is None:
            return False, np.empty((0, 2), dtype=np.float32)
        refined = cv2.cornerSubPix(
            gray,
            corners.reshape(-1, 1, 2),
            (5, 5),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        return True, refined.reshape(-1, 2)

    @staticmethod
    def _depth_at(depth_m: np.ndarray, pixel: np.ndarray) -> float:
        x = int(np.clip(round(float(pixel[0])), 0, depth_m.shape[1] - 1))
        y = int(np.clip(round(float(pixel[1])), 0, depth_m.shape[0] - 1))
        return float(depth_m[y, x])

    @staticmethod
    def _deproject(pixel: np.ndarray, depth_m: float, intrinsics: np.ndarray) -> np.ndarray:
        return np.array(
            [
                (pixel[0] - intrinsics[0, 2]) * depth_m / intrinsics[0, 0],
                (pixel[1] - intrinsics[1, 2]) * depth_m / intrinsics[1, 1],
                depth_m,
            ],
            dtype=float,
        )

    def _board_feature_position(
        self, scene: CalibrationBoardGraspScene, data: mujoco.MjData
    ) -> np.ndarray:
        board_id = mujoco.mj_name2id(
            data.model, mujoco.mjtObj.mjOBJ_BODY, scene.BOARD_BODY_NAME
        )
        rotation = data.xmat[board_id].reshape(3, 3)
        return data.xpos[board_id] + rotation @ np.array(
            [0.0, 0.0, self.BOARD_FEATURE_Z_M]
        )

    @staticmethod
    def _rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        source_center = source.mean(axis=0)
        target_center = target.mean(axis=0)
        u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0.0:
            vt[-1, :] *= -1.0
            rotation = vt.T @ u.T
        translation = target_center - rotation @ source_center
        return rotation, translation

    @classmethod
    def _rigid_transform_rmse(
        cls, measured: np.ndarray, observed: np.ndarray, depth_scale: float
    ) -> tuple[float, np.ndarray]:
        scaled_observed = observed * depth_scale
        rotation, translation = cls._rigid_transform(measured, scaled_observed)
        registered = (rotation @ measured.T).T + translation
        rmse = float(np.sqrt(np.mean(np.sum((registered - scaled_observed) ** 2, axis=1))))
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = translation
        return rmse, world_to_camera

    @classmethod
    def _optimize_depth_scale(
        cls, measured: np.ndarray, observed: np.ndarray
    ) -> tuple[float, np.ndarray, float]:
        lower, upper = 0.5, 1.5
        for _ in range(80):
            left = lower + (upper - lower) / 3.0
            right = upper - (upper - lower) / 3.0
            left_error, _ = cls._rigid_transform_rmse(measured, observed, left)
            right_error, _ = cls._rigid_transform_rmse(measured, observed, right)
            if left_error <= right_error:
                upper = right
            else:
                lower = left
        depth_scale = (lower + upper) / 2.0
        rmse_m, world_to_camera = cls._rigid_transform_rmse(
            measured, observed, depth_scale
        )
        return depth_scale, world_to_camera, rmse_m

    def _save(
        self,
        intrinsics: np.ndarray,
        depth_scale: float,
        world_to_camera: np.ndarray,
        camera_to_world: np.ndarray,
        rmse_m: float,
        tool_positions: np.ndarray,
        measured_points: np.ndarray,
        observed_points: np.ndarray,
    ) -> None:
        np.savetxt(self.output_dir / "camera_intrinsics.txt", intrinsics)
        np.savetxt(self.output_dir / "camera_depth_scale.txt", [depth_scale])
        np.savetxt(self.output_dir / "camera_pose.txt", camera_to_world)
        np.savetxt(self.output_dir / "world_to_camera.txt", world_to_camera)
        np.savez(
            self.output_dir / "camera_parameters.npz",
            intrinsics=intrinsics,
            distortion_coefficients=np.zeros(5),
            depth_scale=depth_scale,
            camera_to_world=camera_to_world,
            world_to_camera=world_to_camera,
        )
        np.save(self.output_dir / "tool_positions_m.npy", tool_positions)
        np.save(self.output_dir / "measured_board_points_m.npy", measured_points)
        np.save(self.output_dir / "observed_camera_points_m.npy", observed_points)
        report = {
            "workflow": "real_ur5_calibrate.py RGB-D rigid-transform calibration",
            "config": asdict(self.config),
            "valid_views": int(len(measured_points)),
            "rmse_m": rmse_m,
            "depth_scale": depth_scale,
            "camera_height_above_table_m": float(camera_to_world[2, 3] - self.config.tabletop_height_m),
        }
        (self.output_dir / "calibration_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
