"""Fixed-orientation bottom-centre pushes with actual MuJoCo object contacts."""
from collections import deque
import mujoco
import numpy as np

from scene import DEFAULT_JOINT_ANGLES_DEG, WORKSPACE_REFERENCE_Z_M

DEFAULT_DESCENT_SPEED_M_S = 0.08
DEFAULT_PUSH_SPEED_M_S = 0.08
DEFAULT_LIFT_SPEED_M_S = 0.04


class MotionError(RuntimeError):
    pass


class PushController:
    """Kinematic arm tracking; gripper and objects evolve under mj_step.

    Arm poses are prescribed each physics tick, as in the existing demos.
    Movable objects are never repositioned to fabricate the result of a push.
    """

    def __init__(self, scene, *, push_speed_m_s=DEFAULT_PUSH_SPEED_M_S,
                 descent_speed_m_s=DEFAULT_DESCENT_SPEED_M_S,
                 lift_speed_m_s=DEFAULT_LIFT_SPEED_M_S, on_step=None):
        speeds = (push_speed_m_s, descent_speed_m_s, lift_speed_m_s)
        if not np.isfinite(speeds).all() or min(speeds) <= 0:
            raise ValueError("Motion speeds must be finite and positive")
        self.scene = scene
        self.push_speed_m_s = float(push_speed_m_s)
        self.descent_speed_m_s = float(descent_speed_m_s)
        self.lift_speed_m_s = float(lift_speed_m_s)
        self.on_step = on_step
        self.arm_qpos = np.array(scene.robot._qpos_addresses)
        self.arm_dofs = np.array(scene.robot._dof_addresses)
        self.max_orientation_error_rad = 0.
        self.max_position_error_m = 0.

    def _step(self, angles, velocity=None):
        s = self.scene
        s.robot.set_joint_angles_deg(s.data, angles)
        s.data.qvel[self.arm_dofs] = 0 if velocity is None else velocity
        s.gripper.close(s.data)
        mujoco.mj_step(s.model, s.data)
        # Remove arm tracking error, without altering gripper or object joints.
        s.robot.set_joint_angles_deg(s.data, angles)
        if not np.isfinite(s.data.qpos).all() or not np.isfinite(s.data.qvel).all():
            raise MotionError("Non-finite simulation state")
        if self.on_step is not None:
            self.on_step()

    def hold(self, seconds):
        angles = self.scene.robot.joint_angles_deg(self.scene.data).copy()
        for _ in range(max(1, round(seconds/self.scene.model.opt.timestep))):
            self._step(angles)

    def close_and_settle(self):
        s = self.scene
        window_steps = int(np.ceil(s.gripper.CLOSED_STABLE_TIME_S/s.model.opt.timestep))+1
        samples = deque(maxlen=window_steps)
        addresses = [s.model.jnt_qposadr[j] for j in s.gripper._linkage_joint_ids]
        for _ in range(round(8/s.model.opt.timestep)):
            self.hold(s.model.opt.timestep)
            if s.gripper.is_closed(s.data, settled=False):
                samples.append(s.data.qpos[addresses].copy())
                if len(samples) == window_steps:
                    positions = np.asarray(samples)
                    # Contact solvers can produce isolated velocity spikes
                    # despite micron-scale displacement. Check the entire
                    # measured position window and its average drift instead.
                    if np.max(np.ptp(positions, axis=0)) < s.gripper.CLOSED_POSITION_TOLERANCE_RAD:
                        drift_speed = np.max(abs(positions[-1]-positions[0])) / ((window_steps-1)*s.model.opt.timestep)
                        if drift_speed < .01:
                            return
            else:
                samples.clear()
        raise MotionError("Gripper did not fully close with parallel, settled pads")

    def _solve(self, data, position, rotation):
        s = self.scene
        result = s.robot.solve_ik_bottom_position(
            data, position, target_rotation_matrix=rotation,
            initial_angles_deg=s.robot.joint_angles_deg(data),
            position_tolerance_m=1e-5, orientation_tolerance_rad=1e-5,
            max_iterations=600, damping=.01,
        )
        if not result.success:
            raise MotionError(f"IK unreachable: position={result.position_error_m:.5f}m orientation={result.orientation_error_rad:.5f}rad")
        return result.joint_angles_deg

    def _joint_move(self, target, seconds=1.):
        s = self.scene
        start = s.robot.joint_angles_deg(s.data).copy()
        # Equivalent 2pi wrist wraps avoid unnecessary full revolutions.
        target = np.asarray(target).copy()
        for k, joint in enumerate(s.robot._joint_ids):
            candidates = target[k] + 360*np.arange(-2, 3)
            if s.model.jnt_limited[joint]:
                low, high = np.rad2deg(s.model.jnt_range[joint])
                candidates = candidates[(candidates >= low) & (candidates <= high)]
            if len(candidates):
                target[k] = candidates[np.argmin(abs(candidates-start[k]))]
        steps = max(1, round(seconds/s.model.opt.timestep))
        previous = start
        for i in range(1, steps+1):
            t = i/steps
            angles = start+(target-start)*(3*t*t-2*t*t*t)
            self._step(angles, np.deg2rad(angles-previous)/s.model.opt.timestep)
            previous = angles
        self.hold(.1)

    def _segment_steps(self, start, end, speed_m_s):
        distance = np.linalg.norm(np.asarray(end)-np.asarray(start))
        return max(1, int(np.ceil(distance/(speed_m_s*self.scene.model.opt.timestep))))

    def push(self, primitive):
        s = self.scene
        self.close_and_settle()
        direction = np.array((np.cos(primitive.push_direction_rad), np.sin(primitive.push_direction_rad), 0.))
        z = np.array((0., 0., -1.))
        # Local Y (side face) points in the push direction, local Z down.
        rotation = np.column_stack((np.cross(direction, z), direction, z))
        above_start = primitive.start_position_m + (0, 0, .12)
        above_end = primitive.end_position_m + (0, 0, .12)
        planning = mujoco.MjData(s.model)
        planning.qpos[:] = s.data.qpos
        planning.ctrl[:] = s.data.ctrl
        try:
            # Continue from the previous action's lifted pose, not home.
            above_angles = self._solve(planning, above_start, rotation)
        except MotionError:
            # A neutral seed is only an offline IK fallback. The real arm is
            # never sent home; _joint_move starts at its actual current pose.
            s.robot.set_joint_angles_deg(planning, DEFAULT_JOINT_ANGLES_DEG)
            above_angles = self._solve(planning, above_start, rotation)
        s.robot.set_joint_angles_deg(planning, above_angles)
        # Preflight each entire Cartesian segment. Runtime uses continuous IK
        # from the preceding pose, so changing action cannot switch IK branches.
        for start, end in ((above_start, primitive.start_position_m),
                           (primitive.start_position_m, primitive.end_position_m),
                           (primitive.end_position_m, above_end)):
            for t in np.linspace(0, 1, 21):
                angles = self._solve(planning, start+(end-start)*t, rotation)
                s.robot.set_joint_angles_deg(planning, angles)
        self._joint_move(above_angles)
        self.close_and_settle()
        for end, speed in ((primitive.start_position_m, self.descent_speed_m_s),
                           (primitive.end_position_m, self.push_speed_m_s),
                           (above_end, self.lift_speed_m_s)):
            start = s.robot.bottom_center_pose(s.data).position_m.copy()
            steps = self._segment_steps(start, end, speed)
            previous = s.robot.joint_angles_deg(s.data).copy()
            for i in range(1, steps+1):
                # Per-physics-tick Cartesian tracking keeps vertical/yaw fixed
                # between endpoints, unlike joint-space interpolation.
                position = start+(end-start)*(i/steps)
                angles = self._solve(s.data, position, rotation)
                self._step(angles, np.deg2rad(angles-previous)/s.model.opt.timestep)
                previous = angles
                pose = s.robot.bottom_center_pose(s.data)
                angle_error = float(np.linalg.norm(s.robot._orientation_error(pose.rotation_matrix, rotation)))
                position_error = float(np.linalg.norm(pose.position_m-position))
                self.max_orientation_error_rad = max(self.max_orientation_error_rad, angle_error)
                self.max_position_error_m = max(self.max_position_error_m, position_error)
                if angle_error > .001 or position_error > .001:
                    raise MotionError(f"Push tracking failed: rotation={angle_error:.6f}rad position={position_error:.6f}m")
                if not s.gripper.is_closed(s.data, settled=False):
                    # Contact can briefly deflect the linkage. Pause at the
                    # current arm pose and re-close before advancing the path.
                    self.close_and_settle()
            self.hold(.15)
        # Stay at above_end. Camera-only robot exclusion permits the next
        # observation here, then direct transit to the next action's approach.
