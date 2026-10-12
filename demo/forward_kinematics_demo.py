"""可交互调节的 UR5e 正运动学演示。"""

# =============================================================================
# 教程提示：正运动学的输入是关节角，输出是机械臂各部分与末端的空间位姿。
# 【动手实验】修改 POSES_DEG 中任一关节角度（单位为度）后重新运行，观察变化。
# 可调变量：POSES_DEG（关节轨迹）、POSE_HOLD_SECONDS（停留秒数）、
#           POSE_TRANSITION_SECONDS（过渡秒数）。六关节必须各提供一个数值。
# =============================================================================


from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot import UR5eRobot  # noqa: E402


SCENE_PATH = PROJECT_ROOT / "models" / "scene.xml"
#可修改，设定正运动学运动轨迹
# 可编辑：每行是一个六关节姿态，单位为度。
POSES_DEG = (
    (0.0, -90.0, 90.0, -90.0, -90.0, 0.0),
    (-45.0, -75.0, 105.0, -120.0, -90.0, 0.0),
    (45.0, -105.0, 105.0, -90.0, -90.0, 45.0),
    (0.0, -75.0, 105.0, -120.0, -90.0, 0.0),
)

# 可编辑：到达姿态后保持的时间，单位为秒。
POSE_HOLD_SECONDS = 1.0#机械臂停留时间
# 可编辑：姿态之间的平滑过渡时间，单位为秒。
POSE_TRANSITION_SECONDS = 2.0#机械臂移动时间
#

def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    robot = UR5eRobot(model)
    robot.start_pose_cycle_deg(
        data,
        POSES_DEG,
        hold_s=POSE_HOLD_SECONDS,
        transition_s=POSE_TRANSITION_SECONDS,
    )

    from mujoco import viewer

    with viewer.launch_passive(model, data) as simulation_viewer:
        simulation_viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        simulation_viewer.cam.lookat[:] = (-0.333, -0.171, 0.745)
        simulation_viewer.cam.distance = 2.75
        simulation_viewer.cam.azimuth = -91.49
        simulation_viewer.cam.elevation = -35.17

        while simulation_viewer.is_running():
            robot.update_pose_cycle(data)
            mujoco.mj_forward(model, data)
            simulation_viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
