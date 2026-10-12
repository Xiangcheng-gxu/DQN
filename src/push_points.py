"""Contour push points and reference-style five-candidate height scoring."""
from dataclasses import dataclass

import cv2
import numpy as np


CANDIDATE_SHIFT_M = 0.01  # Reference: five pixels at 2 mm/pixel.


class PushPointError(RuntimeError):
    """No usable target edge or fully observable candidate footprint."""


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
    """Use the reference blur, threshold, main contour and integer centroid.

    The reference selects the contour with the most boundary vertices, not
    the largest area. Area only breaks ties here; zero-area noise is ignored.
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
    """Bresenham traversal from centroid to the reference ray endpoint."""
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
    """Reference contour rays, corrected to the outer edge of the hit cell.

    Called only for actions 0--3. Missing intersections raise explicitly;
    never fabricate a push point at pixel (0, 0). The reference tests a 2x2
    neighbourhood but returns the traversal pixel, potentially one pixel
    inside the contour. Return the actual hit instead, plus half a pixel
    outward: the contour describes cell centres, not their physical edges.
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
    """Closed fingertip rectangle, local Y along push and local X across it.

    Sample at heightmap cell centres; the 13 mm offset is not rounded to an
    integer pixel. Unlike the reference's 16 x 16 mm square, 2F-85 is 16 x 22.
    """
    rows, columns = np.indices(shape)
    dx = origin_xy_m[0]+(columns+.5)*resolution_m-center_xy_m[0]
    dy = origin_xy_m[1]+(rows+.5)*resolution_m-center_xy_m[1]
    c, s = np.cos(push_direction_rad), np.sin(push_direction_rad)
    return ((abs(c*dx+s*dy) <= along_push_m/2+1e-12) &
            (abs(-s*dx+c*dy) <= across_push_m/2+1e-12))


def select_push_point(heightmap, nominal_xy_m, push_direction_rad, *,
                      origin_xy_m, resolution_m, along_push_m, across_push_m) -> PushPointSelection:
    """Score original,+X,+Y,-X,-Y candidates by footprint maximum height.

    Select the smallest maximum, breaking ties in reference order. This is
    local height ranking, NOT a guarantee of a collision-free entire path.
    Fully out-of-view/clipped or non-finite footprints cannot receive a low
    score from implicit zero padding.
    """
    heights = np.asarray(heightmap)
    origin = np.asarray(origin_xy_m, dtype=float)
    nominal = np.asarray(nominal_xy_m, dtype=float)
    dimensions = (resolution_m, along_push_m, across_push_m)
    if (heights.ndim != 2 or min(heights.shape) == 0 or origin.shape != (2,) or nominal.shape != (2,)
            or not np.isfinite((*origin, *nominal, push_direction_rad, *dimensions)).all()
            or min(dimensions) <= 0):
        raise ValueError("Invalid heightmap, coordinates or gripper dimensions")
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
            scores[i] = covered.max()
    if not np.isfinite(scores).any():
        raise PushPointError("All five push-point footprints are outside the heightmap or invalid")
    return PushPointSelection(candidates, scores, int(np.argmin(scores)))
