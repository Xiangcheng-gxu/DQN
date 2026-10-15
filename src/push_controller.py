"""动作执行器：完全闭合 -> 到落点上方 -> 下降 -> 水平推 -> 抬起。

运动控制点是夹爪底部中心。每次推动时固定朝向，局部 Z 向下、侧面朝推向。
机械臂按 IK 逐步设置关节位置，夹爪和物体的接触由 MuJoCo 物理步计算。
"""
from collections import deque
import mujoco
import numpy as np

from scene import DEFAULT_JOINT_ANGLES_DEG

# 以下单位均为“米/仿真秒”，不是米/现实秒；无窗口时仿真可快于现实。
DEFAULT_DESCENT_SPEED_M_S = 0.48
DEFAULT_PUSH_SPEED_M_S = 0.12
DEFAULT_LIFT_SPEED_M_S = 0.48


class MotionError(RuntimeError):
    pass


class PushController:
    """逐物理步跟踪机械臂轨迹，不直接改物体位置来伪造推动效果。"""

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
        self.arm_dofs = np.array(scene.robot._dof_addresses)
        self.max_orientation_error_rad = 0.
        self.max_position_error_m = 0.

    def _step(self, angles, velocity=None):
        """执行一个物理步：机械臂就位、保持闭合、计算碰撞、按需刷新窗口。"""
        s = self.scene
        s.robot.set_joint_angles_deg(s.data, angles)
        s.data.qvel[self.arm_dofs] = 0 if velocity is None else velocity
        s.gripper.close(s.data)
        mujoco.mj_step(s.model, s.data)
        # 消除机械臂跟踪偏差，不修改夹爪关节或物体位置。
        s.robot.set_joint_angles_deg(s.data, angles)
        if not np.isfinite(s.data.qpos).all() or not np.isfinite(s.data.qvel).all():
            raise MotionError("Non-finite simulation state")
        if self.on_step is not None:
            self.on_step()

    def hold(self, seconds):
        """保持机械臂姿态一段仿真时间，物理仍在运行，不是 sleep 等现实时间。"""
        angles = self.scene.robot.joint_angles_deg(self.scene.data).copy()
        for _ in range(max(1, round(seconds/self.scene.model.opt.timestep))):
            self._step(angles)

    def close_and_settle(self):
        """确认两侧指垫平行且稳定后才继续；不能只发送255就立即下降。"""
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
                    # 约束求解偶尔产生瞬时速度尖峰；结合整个时间窗的位置变化判断稳定。
                    if np.max(np.ptp(positions, axis=0)) < s.gripper.CLOSED_POSITION_TOLERANCE_RAD:
                        drift_speed = np.max(abs(positions[-1]-positions[0])) / ((window_steps-1)*s.model.opt.timestep)
                        if drift_speed < .01:
                            return
            else:
                samples.clear()
        raise MotionError("Gripper did not fully close with parallel, settled pads")

    def _solve(self, data, position, rotation):
        """求底部中心 IK，未达到位置/朝向精度就报告动作失败。"""
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
        """在安全抬起位置之间转移，平滑改变关节角并避免多绕整圈。"""
        s = self.scene
        start = s.robot.joint_angles_deg(s.data).copy()
        # 从等效角度中挑离当前位置最近的一组，避免腕关节多绕一整圈。
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
        """路程除以速度和物理步长，得到这段路径要执行多少个仿真步。"""
        distance = np.linalg.norm(np.asarray(end)-np.asarray(start))
        return max(1, int(np.ceil(distance/(speed_m_s*self.scene.model.opt.timestep))))

    def push(self, primitive):
        """先离线检查轨迹可达，再逐步执行；推完停在终点上方，不回原点。"""
        s = self.scene
        self.close_and_settle()
        direction = np.array((np.cos(primitive.push_direction_rad), np.sin(primitive.push_direction_rad), 0.))
        z = np.array((0., 0., -1.))
        # 局部Y指向推动方向、局部Z向下，保证夹爪侧面正对推向且保持垂直。
        rotation = np.column_stack((np.cross(direction, z), direction, z))
        above_start = primitive.start_position_m + (0, 0, .12)
        above_end = primitive.end_position_m + (0, 0, .12)
        planning = mujoco.MjData(s.model)
        planning.qpos[:] = s.data.qpos
        planning.ctrl[:] = s.data.ctrl
        try:
            # 从上一个动作抬起后的真实姿态继续求解，而不是从原点重新起步。
            above_angles = self._solve(planning, above_start, rotation)
        except MotionError:
            # 默认姿态仅用于离线试算的备用初值；不会把真实机械臂送回原点。
            s.robot.set_joint_angles_deg(planning, DEFAULT_JOINT_ANGLES_DEG)
            above_angles = self._solve(planning, above_start, rotation)
        s.robot.set_joint_angles_deg(planning, above_angles)
        # 先离线采样检查下降、推动、抬起三段是否可达，再开始真实运动。
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
                # 每个物理步都重新求解笛卡尔路径，保持垂直与旋转角，不只检查端点。
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
                    # 接触可能使夹爪连杆短暂偏转，原地暂停并重新闭合后再推进。
                    self.close_and_settle()
            self.hold(.15)
        # 停在终点上方即可取图，相机会忽略机械臂；下一步直接前往新落点上方。
