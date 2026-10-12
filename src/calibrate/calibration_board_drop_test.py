"""实体标定板掉落测试。

场景模型不做修改，只在运行时关闭标定板夹持约束；因此带自由关节的标定板
应当在重力作用下落到桌面上，用于验证其质量、体积和碰撞属性。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_PATH = PROJECT_ROOT / "models" / "scene.xml"
DROP_POSITION = (-0.5204, 0.0343, 1.25)  # 桌面上方 0.50 m。


def prepare_drop() -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    """加载场景并将标定板释放在半空中。"""
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    weld_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, "calibration_board_grasp"
    )
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "calibration_board_freejoint"
    )
    if weld_id < 0 or joint_id < 0:
        raise RuntimeError("Calibration-board weld or freejoint was not found")

    data.eq_active[weld_id] = 0
    qpos_adr = model.jnt_qposadr[joint_id]
    qvel_adr = model.jnt_dofadr[joint_id]
    data.qpos[qpos_adr : qpos_adr + 7] = (*DROP_POSITION, 1.0, 0.0, 0.0, 0.0)
    data.qvel[qvel_adr : qvel_adr + 6] = 0.0
    mujoco.mj_forward(model, data)

    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "calibration_board"
    )
    return model, data, body_id


def main() -> None:
    model, data, board_id = prepare_drop()
    start_z = data.xpos[board_id, 2]

    if "--headless" in sys.argv:
        for _ in range(int(2.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        print(f"掉落测试：标定板 z 坐标 {start_z:.3f} m -> {data.xpos[board_id, 2]:.3f} m")
        return

    from mujoco import viewer

    with viewer.launch_passive(model, data) as simulation_viewer:
        while simulation_viewer.is_running():
            started = time.perf_counter()
            mujoco.mj_step(model, data)
            simulation_viewer.sync()
            remaining = model.opt.timestep - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
