"""把a0到a7转换成夹爪底部中心的起点、终点和推动方向。

a0-a3：目标边缘外找落点，低于目标顶面5mm，向目标内部推。
a4-a7：不找推点，从目标中心出发，高于目标顶面5mm，向外推。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from features import HEIGHTMAP_RESOLUTION_M, Observation
from gripper import Gripper2F85
from push_points import PushPointError, PushPointSelection, sample_target_edges, select_push_point
from scene import WORKSPACE_LIMITS

ACTION_COUNT = 8
PUSH_DISTANCE_M = 0.08
# 所有动作高度均相对于视觉测得的目标顶面，不相对于相机或pinch原点。
TARGET_HEIGHT_OFFSET_M = 0.005
PUSH_POINT_CLEARANCE_M = 0.005
# 基准落点 = 目标边缘外偏13mm = 夹爪沿推向半宽8mm + 间隙5mm。
TARGET_EDGE_OFFSET_M = Gripper2F85.CLOSED_TIP_ALONG_PUSH_M/2 + PUSH_POINT_CLEARANCE_M
# 检测框总尺寸为17×24mm；是总共加1/2mm，不是每侧加，不修改实体模型。
PUSH_FOOTPRINT_ALONG_M = Gripper2F85.CLOSED_TIP_ALONG_PUSH_M + .001
PUSH_FOOTPRINT_ACROSS_M = Gripper2F85.CLOSED_TIP_ACROSS_PUSH_M + .002


@dataclass(frozen=True)
class PushPrimitive:
    """一个水平推动动作；起终点使用世界坐标米，控制点是夹爪底部中心。"""

    action_id: int
    start_position_m: np.ndarray
    end_position_m: np.ndarray
    push_direction_rad: float
    point_selection: PushPointSelection | None = None


class PushPrimitiveSet:
    """规划器不执行运动；训练和评估都通过plan_observation使用相同动作规则。"""

    def __init__(
        self,
        tabletop_z_m: float,
        *,
        push_distance_m: float = PUSH_DISTANCE_M,
        target_edge_offset_m: float = TARGET_EDGE_OFFSET_M,
    ) -> None:
        if not np.isfinite((tabletop_z_m, push_distance_m, target_edge_offset_m)).all() or tabletop_z_m <= 0 or push_distance_m <= 0 or target_edge_offset_m < 0:
            raise ValueError("tabletop height and push distances must be positive")
        self.tabletop_z_m = float(tabletop_z_m)
        self.push_distance_m = float(push_distance_m)
        self.target_edge_offset_m = float(target_edge_offset_m)

    def plan(
        self,
        action_id: int,
        target_center_xy_m: Sequence[float],
        target_boundary_xy_m: Sequence[Sequence[float]] | None = None,
        *,
        target_height_m: float,
    ) -> PushPrimitive:
        """由目标中心/四边缘建立几何动作，不负责检查落点能否下降。

        四个边缘点的顺序为+X、+Y、-X、-Y；a4-a7不使用这些边缘点。
        target_height_m是目标相对桌面的高度，内部再加桌面世界Z。
        """
        if isinstance(action_id, bool) or not isinstance(action_id, (int, np.integer)) or not 0 <= action_id < ACTION_COUNT:
            raise ValueError("action_id must be an integer in 0..7")
        if not np.isfinite(target_height_m) or target_height_m < TARGET_HEIGHT_OFFSET_M:
            raise ValueError("Target height must be finite and at least 5 mm")
        offset = -TARGET_HEIGHT_OFFSET_M if action_id < 4 else TARGET_HEIGHT_OFFSET_M
        contact_z = self.tabletop_z_m + float(target_height_m) + offset
        center = self._xy(target_center_xy_m, "target_center_xy_m")
        direction_index = action_id % 4
        outward_direction = direction_index * np.pi / 2.0
        outward_unit = np.array((np.cos(outward_direction), np.sin(outward_direction)))
        if action_id < 4:
            boundary = np.asarray(target_boundary_xy_m, dtype=float)
            if boundary.shape != (4, 2) or not np.isfinite(boundary).all():
                raise ValueError("target_boundary_xy_m must have shape (4, 2) in [+X, +Y, -X, -Y] order")
            # 目标边缘加13mm外偏，得到夹爪底部中心的基准落点。
            start_xy = boundary[direction_index] + outward_unit * self.target_edge_offset_m
            push_direction = (outward_direction + np.pi) % (2.0 * np.pi)
        else:
            # a4-a7直接从目标中心向外推，不沿用边缘落点。
            start_xy = center
            push_direction = outward_direction
        end_xy = start_xy + self.push_distance_m * np.array((np.cos(push_direction), np.sin(push_direction)))
        return PushPrimitive(
            action_id=action_id,
            start_position_m=np.array((start_xy[0], start_xy[1], contact_z)),
            end_position_m=np.array((end_xy[0], end_xy[1], contact_z)),
            push_direction_rad=float(push_direction),
        )

    def plan_observation(self, action_id: int, observation: Observation) -> PushPrimitive:
        """从当前视觉观察找推点；仅a0-a3执行五候选落点筛选。

        筛选只改变XY，不改变规定的目标顶面±5mm动作高度。
        找不到推点则抛PushPointError，由环境归类为空推，不强行下降。
        """
        # a4-a7验证参数后直接返回，不搜索边缘、不建立五个检测框。
        if isinstance(action_id, bool) or not isinstance(action_id, (int, np.integer)) or not 0 <= action_id < ACTION_COUNT:
            raise ValueError("action_id must be an integer in 0..7")
        if action_id >= 4:
            return self.plan(action_id, observation.target_center_xy,
                             target_height_m=observation.target_height_m)
        if observation.target_contour_px is None:
            raise PushPointError("Target contour missing for action 0--3")
        pixels = sample_target_edges(observation.target_contour_px)
        boundaries = WORKSPACE_LIMITS[:2, 0]+(pixels+.5)*HEIGHTMAP_RESOLUTION_M
        primitive = self.plan(action_id, observation.target_center_xy, boundaries,
                              target_height_m=observation.target_height_m)
        selection = select_push_point(
            observation.heightmap, primitive.start_position_m[:2], primitive.push_direction_rad,
            origin_xy_m=WORKSPACE_LIMITS[:2, 0], resolution_m=HEIGHTMAP_RESOLUTION_M,
            along_push_m=PUSH_FOOTPRINT_ALONG_M,
            across_push_m=PUSH_FOOTPRINT_ACROSS_M,
            max_allowed_height_m=primitive.start_position_m[2]-self.tabletop_z_m)
        shift = np.r_[selection.selected_xy_m-primitive.start_position_m[:2], 0.]
        return replace(primitive, start_position_m=primitive.start_position_m+shift,
                       end_position_m=primitive.end_position_m+shift, point_selection=selection)

    @staticmethod
    def _xy(values: Sequence[float], name: str) -> np.ndarray:
        result = np.asarray(values, dtype=float)
        if result.shape != (2,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must contain two finite coordinates")
        return result
