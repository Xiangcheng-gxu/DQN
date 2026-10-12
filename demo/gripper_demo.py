"""安装在 UR5e 末端的 2F-85 夹爪交互式开合演示。"""

# =============================================================================
# 教程提示：2F-85 夹爪按“闭合→张开→闭合”的节奏循环。
# 【动手实验】修改 CYCLE_SECONDS 或 opening_command() 中各阶段的时长。
# 可调变量：ARM_POSE（六关节弧度）、CYCLE_SECONDS（循环秒数）。
# 夹爪命令范围为 0（闭合）到 255（完全张开）。
# =============================================================================


from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gripper import Gripper2F85  # noqa: E402


MODEL_PATH = PROJECT_ROOT / "models" / "scene.xml"
# 可编辑：机械臂保持姿态，六个值的单位为弧度。
ARM_POSE = (0.0, -1.57079633, 1.57079633, -1.57079633, -1.57079633, 0.0)
# 可编辑：夹爪完整开合循环的总时长，单位为秒。
CYCLE_SECONDS = 6.0


def opening_command(time_s: float) -> int:
    """返回一次闭合、张开、再闭合循环中对应的 0~255 开合命令。"""
    #可修改，阶段性的开合操作
    phase = time_s % CYCLE_SECONDS
    if phase < 1.0:
        return 0  # 保持闭合。
    if phase < 2.5:
        return round(255 * (phase - 1.0) / 1.5)  # 张开。
    if phase < 3.5:
        return 255  # 保持张开。
    if phase < 5.0:
        return round(255 * (1.0 - (phase - 3.5) / 1.5))  # 闭合。
    return 0  # 保持闭合。
    #

def step(model: mujoco.MjModel, data: mujoco.MjData, gripper: Gripper2F85) -> int:
    """保持机械臂姿态，并推进夹爪演示一个仿真步。"""
    command = opening_command(data.time)
    data.ctrl[:6] = ARM_POSE
    # 该演示保留连续开合轨迹；不属于 Gripper2F85 的最小 open/close 接口。
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
    low, high = model.actuator_ctrlrange[actuator_id]
    data.ctrl[actuator_id] = high - (high - low) * command / 255.0
    mujoco.mj_step(model, data)
    return command


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    gripper = Gripper2F85(model)

    data.qpos[:6] = ARM_POSE
    data.ctrl[:6] = ARM_POSE
    gripper.close(data)
    mujoco.mj_forward(model, data)

    if "--headless" in sys.argv:
        commands: list[int] = []
        for _ in range(int(CYCLE_SECONDS / model.opt.timestep)):
            commands.append(step(model, data, gripper))
        print(
            f"Gripper demo passed: command range={min(commands)}..{max(commands)}, "
            f"t={data.time:.2f}s"
        )
        return

    from mujoco import viewer

    with viewer.launch_passive(model, data) as simulation_viewer:
                # 以指定姿态启动自由相机，同时保留鼠标旋转、平移和缩放功能。
        simulation_viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        simulation_viewer.cam.lookat[:] = (-0.333, -0.171, 0.745)
        simulation_viewer.cam.distance = 2.75
        simulation_viewer.cam.azimuth = -91.49
        simulation_viewer.cam.elevation = -35.17
        while simulation_viewer.is_running():
            started = time.perf_counter()
            step(model, data, gripper)
            simulation_viewer.sync()
            remaining = model.opt.timestep - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()

