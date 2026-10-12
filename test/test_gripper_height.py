"""Visual calibration: closed gripper bottom height above the workspace table.

This is a kinematic calibration, not a contact-force test. --z 0 aligns
the lowest gripper geometry with the workspace surface at world Z=0.754 m.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scene import WORKSPACE_REFERENCE_Z_M, create_fixed_scene


def bottom_position(scene) -> np.ndarray:
    """Lowest point of rendered/collision gripper geometry in world coordinates."""
    model, data = scene.model, scene.data
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_base")
    if root < 0:
        raise RuntimeError("Missing robotiq_base")
    points = []
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        while body > 0 and body != root:
            body = int(model.body_parentid[body])
        if body != root:
            continue
        rotation = data.geom_xmat[geom].reshape(3, 3)
        kind = model.geom_type[geom]
        size = model.geom_size[geom]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start = model.mesh_vertadr[mesh]
            local = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            local = np.array([(x,y,z) for x in (-size[0],size[0])
                              for y in (-size[1],size[1]) for z in (-size[2],size[2])])
        else:
            raise RuntimeError(f"Unsupported gripper geometry type: {kind}")
        world = local @ rotation.T + data.geom_xpos[geom]
        points.append(world[np.argmin(world[:, 2])])
    if not points:
        raise RuntimeError("No gripper geometry found")
    return min(points, key=lambda point: point[2]).copy()


def calibrate(z: float):
    scene = create_fixed_scene()
    initial = scene.robot.joint_angles_deg(scene.data).copy()
    for _ in range(int(8 / scene.model.opt.timestep)):
        scene.gripper.close(scene.data)
        scene.step()
        if scene.data.time >= 2 and scene.gripper.is_closed(scene.data):
            break
    else:
        raise RuntimeError("Gripper did not fully close")
    scene.robot.set_joint_angles_deg(scene.data, initial)
    rotation = np.diag((-1., 1., -1.))
    target = np.array((-.5, 0., WORKSPACE_REFERENCE_Z_M + z))
    # The pinch site is not necessarily the physical finger bottom.
    # Correct its target using the actual closed geometry, rather than
    # assuming that pinch Z equals the contact height.
    for _ in range(5):
        result = scene.robot.solve_ik_position(
            scene.data, target, target_rotation_matrix=rotation,
            initial_angles_deg=scene.robot.joint_angles_deg(scene.data),
            position_tolerance_m=1e-6, orientation_tolerance_rad=1e-6,
            max_iterations=1000, damping=.01,
        )
        if not result.success:
            raise RuntimeError("Height calibration IK failed")
        scene.robot.set_joint_angles_deg(scene.data, result.joint_angles_deg)
        error = WORKSPACE_REFERENCE_Z_M + z - bottom_position(scene)[2]
        if abs(error) < 1e-5:
            break
        target[2] += error
    else:
        raise RuntimeError("Could not align gripper bottom with requested height")
    scene.data.qvel[:] = 0
    bottom = bottom_position(scene)
    pinch = scene.robot.end_effector_pose(scene.data).position_m
    print(f"Table world Z: {WORKSPACE_REFERENCE_Z_M:.6f} m")
    print(f"Requested bottom height: {z:.6f} m; actual: {bottom[2]-WORKSPACE_REFERENCE_Z_M:.6f} m")
    print(f"Pinch world XYZ: {pinch}; bottom world XYZ: {bottom}")
    print(f"Pinch-to-bottom Z offset: {pinch[2]-bottom[2]:.6f} m")
    return scene


def run(z: float, no_viewer: bool, duration: float | None):
    scene = calibrate(z)
    if no_viewer:
        return
    import mujoco.viewer
    started = time.perf_counter()
    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
        viewer.cam.lookat[:] = (-.5, 0, WORKSPACE_REFERENCE_Z_M + .05)
        viewer.cam.distance = .65
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15
        with viewer.lock():
            bottom = bottom_position(scene)
            pinch = scene.robot.end_effector_pose(scene.data).position_m
            zero = bottom.copy()
            zero[2] = WORKSPACE_REFERENCE_Z_M
            markers = [(bottom, (0,1,0,1), f"BOTTOM z={z:.3f} m"),
                       (pinch, (1,0,1,1), "PINCH reference"),
                       (zero + (.035,0,0), (0,0.6,1,1), "TABLE z=0")]
            viewer.user_scn.ngeom = 0
            for pos, color, label in markers:
                geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                   np.full(3,.003), pos, np.eye(3).ravel(), np.array(color,dtype=float))
                geom.label = label
                viewer.user_scn.ngeom += 1
            viewer.opt.label = mujoco.mjtLabel.mjLABEL_NONE
        while viewer.is_running() and (duration is None or time.perf_counter()-started < duration):
            viewer.sync()
            time.sleep(.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z", type=float, default=0., help="Closed gripper bottom height above table, metres")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--duration", type=float)
    args = parser.parse_args()
    if not np.isfinite(args.z) or args.z < 0:
        parser.error("--z must be finite and non-negative")
    run(args.z, args.no_viewer, args.duration)
