"""UR5e and Robotiq 2F-85 base scene with manual joint control."""

# =============================================================================
# 教程提示：在 MuJoCo 查看器中手动调节关节并观察 UR5e 与夹爪。
# 【动手实验】修改 ARM_POSE（弧度）或 GRIPPER_COMMAND（0 为闭合）。
# 可调变量：ARM_POSE、GRIPPER_COMMAND。
# =============================================================================


from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco

#可修改初始机械臂位姿
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "scene.xml"
# 可编辑：启动时的六关节姿态，单位为弧度。
ARM_POSE = (0.0, -1.57079633 , 1.57079633  , 0, 1.57079633, -1.57)
# 可编辑：夹爪开合命令；0 通常表示闭合。
GRIPPER_COMMAND = 0.0
#

def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """推进一帧，并把当前 Joint 面板姿态作为新的保持目标。"""
    data.ctrl[:6] = data.qpos[:6]
    data.ctrl[6] = GRIPPER_COMMAND
    mujoco.mj_step(model, data)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[:6] = ARM_POSE
    data.ctrl[:6] = ARM_POSE
    data.ctrl[6] = GRIPPER_COMMAND
    mujoco.mj_forward(model, data)

    if "--headless" in sys.argv:
        print("Scene demo passed: interactive joint control is not used in headless mode.")
        return

    from mujoco import viewer

    with viewer.launch_passive(model, data) as simulation_viewer:
        simulation_viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        simulation_viewer.cam.lookat[:] = (-0.333, -0.171, 0.745)
        simulation_viewer.cam.distance = 2.75
        simulation_viewer.cam.azimuth = -91.49
        simulation_viewer.cam.elevation = -35.17

        while simulation_viewer.is_running():
            started = time.perf_counter()
            step(model, data)
            simulation_viewer.sync()
            remaining = model.opt.timestep - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
