"""View clutter generation or the eight discrete 2F-85 push primitives."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from action_primitives import ACTION_COUNT, PushPrimitiveSet
from features import HeightmapCamera
from push_controller import PushController
from scene import (
    TARGET_COLOR,
    WORKSPACE_REFERENCE_Z_M,
    advance_clutter,
    create_clutter_environment,
    create_fixed_scene,
    create_object,
)

APPROACH_HEIGHT_M = 0.10  # Reference pushing_point_margin.
WAYPOINT_DURATION_S = 0.30
DESCENT_DURATION_S = 0.15
PUSH_DURATION_S = 0.15
WAYPOINT_SETTLE_S = 0.25  # Let the arm reach the pose before the next segment.


@dataclass
class _ActionDemo:
    scene: object
    joint_waypoints_deg: list[np.ndarray]
    cartesian_waypoints: list[tuple[np.ndarray, np.ndarray]]
    waypoint_index: int = 0
    motion_started: bool = False
    waypoint_settled_at: float | None = None
    close_started_sim_time: float = 0.0
    initial_joint_angles_deg: np.ndarray | None = None
    arm_reset_after_close: bool = False
    segment_started_at: float = 0.0
    segment_start_position: np.ndarray | None = None


def _create_action_demo(seed: int | None) -> _ActionDemo:
    """Create one fixed target and all reference-policy 0--7 push waypoints."""
    target = create_object(TARGET_COLOR, rng=np.random.default_rng(seed), dynamic=False)
    scene = create_fixed_scene(objects=(target,))
    scene.gripper.close(scene.data)
    close_started_sim_time = scene.data.time
    initial_joint_angles_deg = scene.robot.joint_angles_deg(scene.data)
    # Match training: acquire RGB-D only after physical parallel closure, so
    # initially open fingers do not alter the candidate footprint heights.
    PushController(scene).close_and_settle()
    # Use exactly the training RGB-D point acquisition, not simulator-known
    # square dimensions. The fixed target allows all eight poses to share it.
    camera = HeightmapCamera(scene)
    try:
        observation = camera.capture()
    finally:
        camera.close()
    if observation.target_center_xy is None:
        raise RuntimeError("Target missing from action-demo RGB-D observation")
    actions = PushPrimitiveSet(WORKSPACE_REFERENCE_Z_M)
    base_rotation = scene.robot.task_orientation_rotation(scene.data)
    # The local Y axis is the 2F-85 side face.  Align that face, rather than
    # the gripper front (local X), with the horizontal push direction.
    base_side_angle = float(np.arctan2(base_rotation[1, 1], base_rotation[0, 1]))
    cartesian_waypoints: list[tuple[np.ndarray, np.ndarray]] = []
    for action_id in range(ACTION_COUNT):
        primitive = actions.plan_observation(action_id, observation)
        if primitive.point_selection is not None:
            selection = primitive.point_selection
            print(f"a{action_id}: point={primitive.start_position_m[:2].round(4)} "
                  f"candidate={selection.selected_index} local_h={selection.max_heights_m[selection.selected_index]*1000:.1f}mm")
        else:
            print(f"a{action_id}: target_center={primitive.start_position_m[:2].round(4)}")
        yaw = primitive.push_direction_rad - base_side_angle
        yaw_rotation = np.array(((np.cos(yaw), -np.sin(yaw), 0.0),
                                 (np.sin(yaw), np.cos(yaw), 0.0),
                                 (0.0, 0.0, 1.0)))
        push_rotation = yaw_rotation @ base_rotation
        cartesian_waypoints.extend((
            (primitive.start_position_m + (0.0, 0.0, APPROACH_HEIGHT_M), push_rotation),
            (primitive.start_position_m, push_rotation),
            (primitive.end_position_m, push_rotation),
            (primitive.end_position_m + (0.0, 0.0, APPROACH_HEIGHT_M), push_rotation),
        ))
    # Solve the complete rotated path before the viewer opens.  At run time we
    # only interpolate the resulting joints, so changing from one push-facing
    # orientation to the next cannot make online IK fail mid-demonstration.
    planning_data = mujoco.MjData(scene.model)
    planning_data.qpos[:] = scene.data.qpos
    planning_data.ctrl[:] = scene.data.ctrl
    mujoco.mj_forward(scene.model, planning_data)
    joint_waypoints_deg: list[np.ndarray] = []

    def solve_waypoint(waypoint, target_rotation):
        return scene.robot.solve_ik_bottom_position(
            planning_data, waypoint, target_rotation_matrix=target_rotation,
            initial_angles_deg=scene.robot.joint_angles_deg(planning_data),
            position_tolerance_m=0.00001, orientation_tolerance_rad=0.00001,
            max_iterations=1000, damping=0.01,
        )

    for waypoint_index, (waypoint, target_rotation) in enumerate(cartesian_waypoints):
        # Continue planning from the preceding lifted waypoint, matching
        # direct action-to-action transit without a physical home detour.
        result = solve_waypoint(waypoint, target_rotation)
        if not result.success and waypoint_index % 4 == 0:
            # Offline fallback for an approach-pose IK branch only. This
            # never inserts a home waypoint or changes the live robot pose.
            scene.robot.set_joint_angles_deg(planning_data, initial_joint_angles_deg)
            result = solve_waypoint(waypoint, target_rotation)
        if not result.success:
            raise RuntimeError(
                f"Unable to plan rotated push waypoint {waypoint_index}; "
                f"position={result.position_error_m:.4f} m, "
                f"orientation={result.orientation_error_rad:.4f} rad"
            )
        joint_waypoints_deg.append(result.joint_angles_deg.copy())
        scene.robot.set_joint_angles_deg(planning_data, result.joint_angles_deg)

    return _ActionDemo(scene, joint_waypoints_deg, cartesian_waypoints, close_started_sim_time=close_started_sim_time,
                       initial_joint_angles_deg=initial_joint_angles_deg)


def _advance_action_demo(demo: _ActionDemo) -> bool:
    """Advance one IK-controlled waypoint; return True after all eight actions."""
    scene = demo.scene
    scene.gripper.close(scene.data)
    closed = scene.gripper.is_closed(scene.data)
    if scene.data.time - demo.close_started_sim_time < scene.gripper.CLOSE_SETTLE_TIME_S or not closed:
        if scene.data.time - demo.close_started_sim_time > 8.0:
            raise RuntimeError("Gripper did not fully close and settle; arm motion blocked")
        for _ in range(10):
            scene.gripper.close(scene.data)
            scene.step()
        return False
    if not demo.arm_reset_after_close:
        scene.robot.set_joint_angles_deg(scene.data, demo.initial_joint_angles_deg)
        demo.arm_reset_after_close = True
    if demo.waypoint_index >= len(demo.joint_waypoints_deg):
        mujoco.mj_forward(scene.model, scene.data)
        return True
    duration_s = (WAYPOINT_DURATION_S, DESCENT_DURATION_S,
                  PUSH_DURATION_S, WAYPOINT_DURATION_S)[demo.waypoint_index % 4]
    if not demo.motion_started:
        demo.segment_started_at = time.perf_counter()
        demo.segment_start_position = scene.robot.bottom_center_pose(scene.data).position_m.copy()
        if demo.waypoint_index % 4 == 0:
            scene.robot.begin_joint_motion_deg(
                scene.data, demo.joint_waypoints_deg[demo.waypoint_index], duration_s,
            )
        demo.motion_started = True
    if demo.waypoint_index % 4 == 0:
        waypoint_done = scene.robot.update_joint_motion(scene.data)
    else:
        progress = float(np.clip((time.perf_counter() - demo.segment_started_at) / duration_s, 0, 1))
        alpha = progress * progress * (3 - 2 * progress)
        target_position, target_rotation = demo.cartesian_waypoints[demo.waypoint_index]
        position = (1 - alpha) * demo.segment_start_position + alpha * target_position
        result = scene.robot.solve_ik_bottom_position(
            scene.data, position, target_rotation_matrix=target_rotation,
            initial_angles_deg=scene.robot.joint_angles_deg(scene.data),
            position_tolerance_m=0.00001, orientation_tolerance_rad=0.00001,
            max_iterations=1000, damping=0.01,
        )
        if not result.success:
            raise RuntimeError(f"Fixed-orientation push IK failed at waypoint {demo.waypoint_index}")
        scene.robot.set_joint_angles_deg(scene.data, result.joint_angles_deg)
        waypoint_done = progress >= 1.0
    scene.gripper.close(scene.data)
    mujoco.mj_forward(scene.model, scene.data)
    if not waypoint_done:
        return False
    if demo.waypoint_settled_at is None:
        demo.waypoint_settled_at = time.perf_counter()
        return False
    if time.perf_counter() - demo.waypoint_settled_at < WAYPOINT_SETTLE_S:
        return False
    demo.waypoint_index += 1
    demo.motion_started = False
    demo.waypoint_settled_at = None
    return False

def run(duration_s: float | None, show_viewer: bool, object_count: int | tuple[int, int], seed: int | None, test_actions: bool) -> None:
    if test_actions:
        demo = _create_action_demo(seed)
        scene = demo.scene
        advance_once = lambda: _advance_action_demo(demo)
        initial_display_s = 1.0 if show_viewer else 0.0
    else:
        scene = create_clutter_environment(object_count, rng=np.random.default_rng(seed))
        initial_display_s = 1.0 if show_viewer else 0.0
        started_for_clutter = False

        def advance_once() -> bool:
            nonlocal started_for_clutter
            if time.perf_counter() - started < initial_display_s:
                scene.step()
                return False
            started_for_clutter = True
            return advance_clutter(scene)

    started = time.perf_counter()

    def tick() -> bool:
        if time.perf_counter() - started < initial_display_s:
            scene.step()
            return False
        return advance_once()

    if not show_viewer:
        while duration_s is None or time.perf_counter() - started < duration_s:
            tick()
        return

    import mujoco.viewer
    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = (-0.333, -0.171, 0.745)
        viewer.cam.distance = 2.75
        viewer.cam.azimuth = -91.49
        viewer.cam.elevation = -35.17
        while viewer.is_running() and (duration_s is None or time.perf_counter() - started < duration_s):
            complete = tick()
            viewer.sync()
            time.sleep(0.01 if not complete else scene.model.opt.timestep)


def _parse_object_count(value: str) -> int | tuple[int, int]:
    if "-" not in value:
        count = int(value)
        if count < 0:
            raise argparse.ArgumentTypeError("count must be non-negative")
        return count
    low, high = (int(part) for part in value.split("-", maxsplit=1))
    if low < 0 or low > high:
        raise argparse.ArgumentTypeError("interval must satisfy 0 <= MIN <= MAX")
    return low, high


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch clutter or the 0--7 2F-85 push-action demo")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--object-count", type=_parse_object_count, default=3, help="clutter obstacle count: N or MIN-MAX")
    parser.add_argument("--test-actions", action="store_true", help="run push actions 0--7 in order")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.duration is not None and args.duration < 0:
        parser.error("--duration must be non-negative")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.duration, not arguments.no_viewer, arguments.object_count, arguments.seed, arguments.test_actions)
