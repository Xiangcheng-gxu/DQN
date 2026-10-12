"""保持末端姿态约束的 UR5e 交互式逆运动学演示。"""

# =============================================================================
# 教程提示：逆运动学的输入是末端目标位置，程序会求出相应的关节角。
# 【动手实验】逐次微调 TARGET_POSITIONS_M 内的 x、y 或 z（单位：米）。
# 可调变量：INITIAL_JOINTS_DEG、TARGET_POSITIONS_M、HOLD_SECONDS、TRANSITION_SECONDS。
# 本示例从竖直工具姿态出发，因此末端在运动中保持该方向约束。
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
#可修改初始正运动学位姿、目标点位、停顿时间、移动时间
# 可编辑：逆解的初始关节姿态，单位为度。
INITIAL_JOINTS_DEG = (0.0, -90.0, 90.0, -90.0, -90.0, 0.0)
# 可编辑：末端巡航目标 (x, y, z)，单位为米。
TARGET_POSITIONS_M = ((-0.25, -0.25, 1.0), (-0.25, 0.25, 1.0), (-0.75, -0.25, 1.0), (-0.75, 0.25, 1.0), (-0.50, 0.00, 1.0))
HOLD_SECONDS = 1.0
TRANSITION_SECONDS = 2.0
#
def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH)); data = mujoco.MjData(model); robot = UR5eRobot(model)
    robot.set_joint_angles_deg(data, INITIAL_JOINTS_DEG)
    robot.start_ik_position_cycle(data, TARGET_POSITIONS_M, hold_s=HOLD_SECONDS, transition_s=TRANSITION_SECONDS)
    from mujoco import viewer
    with viewer.launch_passive(model, data) as simulation_viewer:
        simulation_viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        simulation_viewer.cam.lookat[:] = (-0.333, -0.171, 0.745)
        simulation_viewer.cam.distance = 2.75; simulation_viewer.cam.azimuth = -91.49; simulation_viewer.cam.elevation = -35.17
        while simulation_viewer.is_running():
            robot.update_ik_position_cycle(data)
            mujoco.mj_forward(model, data)
            simulation_viewer.sync()
            time.sleep(0.01)
if __name__ == "__main__": main()
