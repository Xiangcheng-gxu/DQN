"""Discrete push primitives matching the 0--7 action layout of the reference policy."""

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
# Bottom-centre offset relative to the observed target top surface.
TARGET_HEIGHT_OFFSET_M = 0.005
PUSH_POINT_CLEARANCE_M = 0.005
# Bottom-centre to leading narrow edge (8 mm), plus 5 mm clearance.
TARGET_EDGE_OFFSET_M = Gripper2F85.CLOSED_TIP_ALONG_PUSH_M/2 + PUSH_POINT_CLEARANCE_M


@dataclass(frozen=True)
class PushPrimitive:
    """One Cartesian, horizontal 2F-85 push command."""

    action_id: int
    start_position_m: np.ndarray
    end_position_m: np.ndarray
    push_direction_rad: float
    point_selection: PushPointSelection | None = None


class PushPrimitiveSet:
    """Plan the eight discrete push actions used by the reference policy.

    Actions 0--3 push inwards from the target's +X, +Y, -X, -Y boundaries.
    Actions 4--7 start at the target centre and push respectively toward
    +X, +Y, -X, -Y.  Positions refer to the calibrated gripper bottom centre.
    Actions 0--3 operate 5 mm BELOW the target top; 4--7 operate 5 mm ABOVE.
    """

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
        """Return the world-frame push command for action ``0`` through ``7``.

        For actions 0--3, ``target_boundary_xy_m`` contains four edge points in
        [+X, +Y, -X, -Y] order, matching ``utils.push_sample`` in the reference
        implementation. Actions 4--7 do not require or inspect edge points.
        Training and demos use ``plan_observation`` for contour sampling and
        five-candidate selection; this method only builds Cartesian geometry.
        ``target_height_m`` is the target top height relative to the tabletop.
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
            # 2F-85: half the narrow dimension plus the requested 5 mm gap.
            start_xy = boundary[direction_index] + outward_unit * self.target_edge_offset_m
            push_direction = (outward_direction + np.pi) % (2.0 * np.pi)
        else:
            # Reference action 4--7: start at the target centre and push outward.
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
        """The shared RGB-D action planner for training and action demos.

        The reference local height score chooses XY only. It does not replace
        the user's target_top -/+ 5 mm bottom-centre height convention.
        """
        # Validate action and centre even when no contour exists. For 4--7,
        # return immediately: no edge sampling, footprint or candidate search.
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
            along_push_m=Gripper2F85.CLOSED_TIP_ALONG_PUSH_M,
            across_push_m=Gripper2F85.CLOSED_TIP_ACROSS_PUSH_M)
        shift = np.r_[selection.selected_xy_m-primitive.start_position_m[:2], 0.]
        return replace(primitive, start_position_m=primitive.start_position_m+shift,
                       end_position_m=primitive.end_position_m+shift, point_selection=selection)

    @staticmethod
    def _xy(values: Sequence[float], name: str) -> np.ndarray:
        result = np.asarray(values, dtype=float)
        if result.shape != (2,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must contain two finite coordinates")
        return result
