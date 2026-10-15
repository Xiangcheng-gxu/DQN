"""推点获取：红色目标轮廓 -> 四边缘 -> 五个候选框 -> 最低可下降位置。"""
from dataclasses import dataclass

import cv2
import numpy as np


CANDIDATE_SHIFT_M = 0.01  # 每次偏移10mm；在2mm/像素的图上相当于5像素。


class PushPointError(RuntimeError):
    """没有可靠边缘或可下降候选点；抛出后不能继续调用推动控制器。"""


@dataclass(frozen=True)
class TargetContour:
    points: np.ndarray
    center_pixel_xy: np.ndarray


@dataclass(frozen=True)
class PushPointSelection:
    candidates_xy_m: np.ndarray
    max_heights_m: np.ndarray
    selected_index: int

    @property
    def selected_xy_m(self) -> np.ndarray:
        return self.candidates_xy_m[self.selected_index].copy()


def extract_target_contour(target_mask: np.ndarray) -> TargetContour | None:
    """对掩码做3×3模糊和阈值化，再取主轮廓与整数质心。

    主轮廓按顶点数最多选择，面积只用于打破并列；过滤零面积噪声。
    """
    mask = np.asarray(target_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("target_mask must be a 2-D array")
    if not mask.any():
        return None
    image = cv2.GaussianBlur(mask.astype(np.uint8)*255, (3, 3), 0)
    threshold = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 0]
    if not contours:
        return None
    contour = max(contours, key=lambda c: (len(c), cv2.contourArea(c)))
    moments = cv2.moments(contour)
    center = np.array((int(moments["m10"]/moments["m00"]),
                       int(moments["m01"]/moments["m00"])), dtype=float)
    return TargetContour(contour.copy(), center)


def _line_pixels(start, end):
    """沿一条线逐像素行走，从目标质心走到边缘外，供轮廓交点搜索。"""
    x, y = (int(v) for v in start)
    ex, ey = (int(v) for v in end)
    dx, dy = abs(ex-x), -abs(ey-y)
    sx, sy = (1 if x < ex else -1), (1 if y < ey else -1)
    error = dx+dy
    while True:
        yield x, y
        if x == ex and y == ey:
            return
        twice = 2*error
        if twice >= dy:
            error += dy
            x += sx
        if twice <= dx:
            error += dx
            y += sy


def sample_target_edges(contour: np.ndarray) -> np.ndarray:
    """从质心向四边搜索轮廓交点，仅供a0-a3使用。

    按2×2邻域找实际命中像素，再向外补半个像素，得到像素格的外缘。
    找不到交点就报告失败，不能用(0,0)假装找到推点。
    """
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        raise PushPointError("Target contour has no area")
    cx, cy = int(moments["m10"]/moments["m00"]), int(moments["m01"]/moments["m00"])
    x, y, w, h = cv2.boundingRect(contour)
    endpoints = ((x+w, int(y+h/2)), (cx, y+h), (x, int(y+h/2)), (cx, y))
    points = []
    outward_directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
    for endpoint, outward in zip(endpoints, outward_directions):
        for px, py in _line_pixels((cx, cy), endpoint):
            neighbours = ((px, py), (px+1, py), (px, py+1), (px+1, py+1))
            hits = [p for p in neighbours if cv2.pointPolygonTest(contour, p, False) == 0]
            if hits:
                points.append(np.asarray(hits[0], dtype=float)+.5*np.asarray(outward))
                break
        else:
            raise PushPointError("Cannot find all four target contour intersections")
    return np.asarray(points, dtype=float)


def gripper_footprint_mask(shape, center_xy_m, push_direction_rad, *,
                           origin_xy_m, resolution_m, along_push_m, across_push_m):
    """在高度图像素中心采样旋转矩形，返回哪些像素落入夹爪检测框。

    短边沿推向、长边垂直推向；实际指端16×22mm，外部传入含余量17×24mm。
    这不是整个夹爪外壳，更不是整条推动轨迹的碰撞包络。
    """
    rows, columns = np.indices(shape)
    dx = origin_xy_m[0]+(columns+.5)*resolution_m-center_xy_m[0]
    dy = origin_xy_m[1]+(rows+.5)*resolution_m-center_xy_m[1]
    c, s = np.cos(push_direction_rad), np.sin(push_direction_rad)
    return ((abs(c*dx+s*dy) <= along_push_m/2+1e-12) &
            (abs(-s*dx+c*dy) <= across_push_m/2+1e-12))


def select_push_point(heightmap, nominal_xy_m, push_direction_rad, *,
                      origin_xy_m, resolution_m, along_push_m, across_push_m,
                      max_allowed_height_m=None) -> PushPointSelection:
    """比较基准、+X、+Y、-X、-Y五个候选框内的最高点。

    框完整落在图内、数值有效、最高点不超过指端下降高度，才可以入选。
    在入选点中取最高点最低的一个，并列取靠前者；低矮障碍物不一定被拒绝。
    无候选时抛出PushPointError；不得用边界外补零冒充空白位置。
    """
    heights = np.asarray(heightmap)
    if max_allowed_height_m is not None and not np.isfinite(max_allowed_height_m):
        raise ValueError("Allowed descent height must be finite")
    origin = np.asarray(origin_xy_m, dtype=float)
    nominal = np.asarray(nominal_xy_m, dtype=float)
    dimensions = (resolution_m, along_push_m, across_push_m)
    if (heights.ndim != 2 or min(heights.shape) == 0 or origin.shape != (2,) or nominal.shape != (2,)
            or not np.isfinite((*origin, *nominal, push_direction_rad, *dimensions)).all()
            or min(dimensions) <= 0):
        raise ValueError("Invalid heightmap, coordinates or gripper dimensions")
    # 偏移沿世界XY，而不是随夹爪转动的局部坐标；检测框本身随推向旋转。
    shifts = CANDIDATE_SHIFT_M*np.array(((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)))
    candidates = nominal+shifts
    scores = np.full(5, np.inf)
    direction = np.array((np.cos(push_direction_rad), np.sin(push_direction_rad)))
    cross = np.array((-direction[1], direction[0]))
    half_extent = (abs(direction)*along_push_m+abs(cross)*across_push_m)/2
    high = origin+np.array(heights.shape[::-1])*resolution_m
    for i, point in enumerate(candidates):
        if np.any(point-half_extent < origin-1e-12) or np.any(point+half_extent > high+1e-12):
            continue
        mask = gripper_footprint_mask(heights.shape, point, push_direction_rad,
                                     origin_xy_m=origin, resolution_m=resolution_m,
                                     along_push_m=along_push_m, across_push_m=across_push_m)
        covered = heights[mask]
        if covered.size and np.isfinite(covered).all():
            maximum = float(covered.max())
            if max_allowed_height_m is None or maximum <= max_allowed_height_m:
                scores[i] = maximum
    if not np.isfinite(scores).any():
        raise PushPointError("No safe descent point: all five footprints are obstructed, outside the heightmap or invalid")
    return PushPointSelection(candidates, scores, int(np.argmin(scores)))
