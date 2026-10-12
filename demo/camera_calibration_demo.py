"""无界面快速相机标定入口。"""

# =============================================================================
# 教程提示：相机标定会让机械臂依次访问标定板上方的采集点。
# 【动手实验】修改 X/Y 坐标列表或高度列表，改变采样网格密度和覆盖范围。
# 可调变量：TABLETOP_HEIGHT_M、CALIBRATION_X_M、CALIBRATION_Y_M、CALIBRATION_HEIGHT_ABOVE_TABLE_M。
# CALIBRATION_POINTS_M 由三组坐标自动组合生成，无需手工逐个填写。
# =============================================================================


from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from calibrate.camera_calibration import (  # noqa: E402
    MuJoCoCameraCalibrator,
    RealUR5CalibrationConfig,
)


# 标定点配置：修改下列数值可改变快速标定的 45 点轨迹。
# 可编辑：桌面绝对高度，单位为米。
TABLETOP_HEIGHT_M = 0.75
# 可编辑：标定采样点的 X 坐标列表，单位为米。
CALIBRATION_X_M = (-0.55, -0.50, -0.45)
# 可编辑：标定采样点的 Y 坐标列表，单位为米。
CALIBRATION_Y_M = (-0.05, 0.00, 0.05, 0.10, 0.15)
CALIBRATION_HEIGHT_ABOVE_TABLE_M = (0.10, 0.15, 0.20)

# 轨迹为 3 × 5 × 3，共 45 个标定点；可增减三个坐标元组以编辑轨迹。
CALIBRATION_POINTS_M = tuple(
    (x_m, y_m, TABLETOP_HEIGHT_M + height_m)
    for x_m in CALIBRATION_X_M
    for y_m in CALIBRATION_Y_M
    for height_m in CALIBRATION_HEIGHT_ABOVE_TABLE_M
)

CALIBRATION_CONFIG = RealUR5CalibrationConfig(
    tabletop_height_m=TABLETOP_HEIGHT_M,
    tool_positions_m=CALIBRATION_POINTS_M,
)


def main() -> None:
    result = MuJoCoCameraCalibrator(
        scene_path=PROJECT_ROOT / "models" / "scene.xml",
        board_path=PROJECT_ROOT / "models" / "calibration_board.xml",
        output_dir=PROJECT_ROOT / "results",
        config=CALIBRATION_CONFIG,
    ).run()
    print(f"Valid calibration views: {result.valid_views}/{result.total_views}")
    print(f"RGB-D registration RMSE (m): {result.rmse_m:.9f}")
    print("Saved: camera_calibration")


if __name__ == "__main__":
    main()
