"""强化学习环境：负责创建一局、执行一步、计算奖励和判断结束。

训练主循环只调用 reset/step，不需要知道 MuJoCo 的关节和图像细节。
reset 返回初始状态；step 返回下一状态、奖励、终止、截断和调试信息。
"""
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from action_primitives import TARGET_HEIGHT_OFFSET_M, PushPrimitiveSet
from features import HeightmapCamera
from push_controller import (DEFAULT_DESCENT_SPEED_M_S, DEFAULT_PUSH_SPEED_M_S,
                             MotionError, PushController)
from push_points import PushPointError
from scene import WORKSPACE_LIMITS, WORKSPACE_REFERENCE_Z_M, advance_clutter, create_clutter_environment

ACTION_HEIGHT_MODE = "target_relative_v2"
ACTION_POINT_MODE = "contour_2f85_half_width_plus_5mm_v1"


@dataclass
class EnvironmentConfig:
    """环境参数。新训练入口会覆盖障碍物数；未暴露到命令行的参数在此修改。"""

    object_count: int | tuple[int, int] = 3  # 仅障碍物数量，每局另有1个目标。
    max_steps: int = 8  # 每局最多尝试8次动作，空推也占一次。
    action_height_mode: str = ACTION_HEIGHT_MODE
    action_point_mode: str = ACTION_POINT_MODE
    push_distance_m: float = .08
    push_speed_m_s: float = DEFAULT_PUSH_SPEED_M_S
    descent_speed_m_s: float = DEFAULT_DESCENT_SPEED_M_S
    success_clearance_m: float = .02  # 目标与所有障碍物的底面边缘间距需大于2cm。
    change_height_m: float = .008  # 高度变化至少8mm的像素才计为变化。
    change_pixels: int = 500  # 变化像素不超过500个，视为空推。
    clustering_steps: int = 100  # 每局开始时，水平吸引聚拢的物理步数。

    def __post_init__(self):
        if isinstance(self.object_count, list):
            self.object_count = tuple(self.object_count)
        if isinstance(self.object_count, int) and not isinstance(self.object_count, bool):
            counts = (self.object_count,)
        elif isinstance(self.object_count, tuple) and len(self.object_count) == 2:
            counts = self.object_count
        else:
            raise ValueError("object_count must be an integer or a two-item interval")
        if any(isinstance(n, bool) or not isinstance(n, int) for n in counts) or min(counts) < 1 or counts[0] > counts[-1]:
            raise ValueError("object_count must be >=1, or an ordered interval")
        if self.action_height_mode != ACTION_HEIGHT_MODE:
            raise ValueError("Unsupported action height convention")
        if self.action_point_mode != ACTION_POINT_MODE:
            raise ValueError("Unsupported push-point convention")
        if (not 1 <= self.max_steps <= 8 or not np.isfinite((self.push_distance_m, self.push_speed_m_s, self.descent_speed_m_s)).all()
                or min(self.push_distance_m, self.push_speed_m_s, self.descent_speed_m_s) <= 0):
            raise ValueError("Invalid episode/motion configuration")
        if self.success_clearance_m <= 0 or self.change_pixels < 0 or self.change_height_m <= 0 or self.clustering_steps < 1:
            raise ValueError("Invalid success/change/clutter configuration")

    @classmethod
    def from_checkpoint(cls, config):
        """恢复模型对应的环境，防止旧动作语义与当前动作不一致。"""
        if config.get("action_height_mode") != ACTION_HEIGHT_MODE:
            raise ValueError("Checkpoint uses a different action-height convention. Start a new run with a0-a3=-5mm and a4-a7=+5mm; old replay cannot be resumed.")
        if config.get("action_point_mode") != ACTION_POINT_MODE:
            raise ValueError("Checkpoint uses a different push-point convention. Start a new run with contour points, half 2F-85 narrow width + 5mm, and five-candidate selection; old replay cannot be resumed.")
        config = dict(config)
        config["max_steps"] = min(config.get("max_steps", 8), 8)
        return cls(**config)


class ClutterPushEnv:
    """scene管物理，camera管观察，planner找推点，controller执行，环境给奖励。"""
    def __init__(self, config=None, *, seed=1, viewer=False, ias_enabled=True):
        self.config = config or EnvironmentConfig()
        self.rng = np.random.default_rng(seed)
        self.show_viewer = viewer
        self.ias_enabled = ias_enabled
        self.scene = self.camera = self.viewer = None
        self.steps = 0
        self.finished = True
        self.render_ticks = 0
        self.action_mask = np.ones(8, dtype=bool)

    def _display(self):
        # viewer只负责观察；不开窗口时仍执行全部物理步，不做实时等待。
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise KeyboardInterrupt
            self.render_ticks += 1
            if self.render_ticks % 10 == 0:
                self.viewer.sync()

    def reset(self):
        """重建场景、聚拢、闭合夹爪、拍照；每一局都从新的随机布局开始。"""
        self.close()
        # 初始布局若放不下或本来就已分离，最多重试12次，避免无限循环。
        last_error = None
        for _ in range(12):
            try:
                self.scene = create_clutter_environment(
                    self.config.object_count, rng=self.rng,
                    clustering_steps=self.config.clustering_steps)
                while not advance_clutter(self.scene):
                    pass
                self.controller = PushController(self.scene, push_speed_m_s=self.config.push_speed_m_s,
                                                 descent_speed_m_s=self.config.descent_speed_m_s)
                self.controller.close_and_settle()
                if self._clearance() >= self.config.success_clearance_m:
                    raise RuntimeError("Initial target is already separated")
                break
            except RuntimeError as error:
                last_error = error
        else:
            raise RuntimeError(f"Cannot create valid clutter after 12 attempts: {last_error}")
        if self.show_viewer:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.scene.model, self.scene.data)
            self.viewer.cam.lookat[:] = (-.5, 0, WORKSPACE_REFERENCE_Z_M+.08)
            self.viewer.cam.distance = 1.3
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -35
        self.controller.on_step = self._display
        self.camera = HeightmapCamera(self.scene)
        self.observation = self.camera.capture()
        if self.observation.target_center_xy is None:
            self.close()
            raise RuntimeError("Target missing from initial RGB-D observation")
        self.steps, self.finished = 0, False
        self.action_mask[:] = True
        self.planner = PushPrimitiveSet(WORKSPACE_REFERENCE_Z_M,
                                       push_distance_m=self.config.push_distance_m)
        return self.observation.state.copy(), {"obstacles": len(self.scene.objects)-1, "clearance_m": self._clearance()}

    def _positions(self):
        s = self.scene
        return np.array([s.data.xpos[mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, f"{o.kind}_{i}"), :2]
                         for i, o in enumerate(s.objects)])

    def _clearance(self):
        """用真实物体位置计算奖励：目标到底面最近障碍物的边缘距离。"""
        positions = self._positions()
        sides = np.array([o.side_m for o in self.scene.objects])
        gaps = np.maximum(abs(positions[1:]-positions[0])-(sides[1:, None]+sides[0])/2, 0)
        return float(np.linalg.norm(gaps, axis=1).min())

    def _out_of_bounds(self):
        """只要任意物体的底面超出XY工作区，就算越界，不只检查中心。"""
        positions = self._positions()
        half = np.array([o.side_m/2 for o in self.scene.objects])[:, None]
        return bool(np.any(positions-half < WORKSPACE_LIMITS[:2, 0]) or
                    np.any(positions+half > WORKSPACE_LIMITS[:2, 1]))

    def step(self, action: int):
        """一次动作尝试也算一步；无法生成推点时不下降，按空推返回。"""
        if self.finished:
            raise RuntimeError("Call reset before step")
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= action < 8:
            raise ValueError("Action must be an integer in 0..7")
        # 先保留动作前的观察，动作后才能比较高度图到底改变了多少。
        previous = self.observation
        before = self._positions().copy()
        primitive, failure, failure_reason = None, None, "motion_failed"
        try:
            primitive = self.planner.plan_observation(action, previous)
            self.controller.push(primitive)
        except PushPointError as error:
            failure, failure_reason = str(error), "push_point_failed"
        except MotionError as error:
            failure = str(error)
        self.steps += 1
        self.observation = self.camera.capture()
        clearance = self._clearance()
        changed_pixels = int(np.count_nonzero(abs(self.observation.heightmap-previous.heightmap) >= self.config.change_height_m))
        success = False
        if failure_reason == "push_point_failed":
            # 找不到安全落点不代表机械臂坏了：不运动，给空推奖励，训练可继续。
            reward, terminated, reason = -5., False, "empty_push"
        elif failure is not None:
            reward, terminated, reason = -10., True, failure_reason
        elif self._out_of_bounds():
            reward, terminated, reason = -10., True, "out_of_workspace"
        elif self.observation.target_center_xy is None:
            reward, terminated, reason = -10., True, "target_lost"
        elif clearance > self.config.success_clearance_m:
            reward, terminated, reason, success = 10., True, "separated", True
        elif changed_pixels <= self.config.change_pixels:
            reward, terminated, reason = -5., False, "empty_push"
        else:
            reward, terminated, reason = -1., False, "pushed"
        # IAS只辅助训练：无变化就暂时屏蔽该动作，场景明显变化后重新开放全部动作。
        # 它只影响选动作，不在TD目标计算时屏蔽Q值；评估入口关闭IAS。
        if self.ias_enabled:
            if changed_pixels > self.config.change_pixels:
                self.action_mask[:] = True
            elif failure is None or failure_reason == "push_point_failed":
                self.action_mask[action] = False
            if not terminated and not self.action_mask.any():
                terminated, reason = True, "no_effective_actions"
        # terminated：任务真的结束；truncated：只是耗尽本局步数，两者学习含义不同。
        truncated = self.steps >= self.config.max_steps and not terminated
        self.finished = terminated or truncated
        selection = None if primitive is None else primitive.point_selection
        info = {"success": success, "reason": "step_limit" if truncated else reason,
                "clearance_m": clearance, "changed_pixels": changed_pixels,
                "max_displacement_m": float(np.linalg.norm(self._positions()-before, axis=1).max()),
                "action_mask": self.action_mask.copy(),
                "orientation_error_rad": self.controller.max_orientation_error_rad,
                "position_error_m": self.controller.max_position_error_m, "motion_error": failure,
                "target_height_m": previous.target_height_m,
                "push_height_m": previous.target_height_m+(-TARGET_HEIGHT_OFFSET_M if action < 4 else TARGET_HEIGHT_OFFSET_M),
                "push_start_xy_m": None if primitive is None else primitive.start_position_m[:2].copy(),
                "push_point_index": None if selection is None else selection.selected_index,
                "push_point_max_height_m": None if selection is None else float(selection.max_heights_m[selection.selected_index])}
        return self.observation.state.copy(), reward, terminated, truncated, info

    def state_dict(self):
        """保存环境参数和随机状态；不保存本局中途的机器人运动过程。"""
        return {"config": asdict(self.config), "rng": self.rng.bit_generator.state}

    def close(self):
        """释放窗口和离屏相机资源，下一次reset会创建新的场景。"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.camera is not None:
            self.camera.close()
            self.camera = None
