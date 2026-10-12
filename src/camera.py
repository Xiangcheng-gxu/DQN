"""``models/scene.xml`` 中固定 RealSense D435i 的 RGB-D 接口。"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
from rendering import RobotHiddenRenderer
# 场景中的 D435i 网格模型只负责显示真实相机在工作区中的安装位置与朝向。
# 实际采集 RGB 图像和深度图的是 scene.xml 中名为 topdown_depth 的 MuJoCo 视角相机。
# #添加相机的方式就是修改xml，让代码读取xml场景文件的时候认为这个地方应该有一个相机。
# 二者应保持同一位置和朝向：外观模型用于观察，MuJoCo 相机用于渲染与标定。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = PROJECT_ROOT / "models" / "scene.xml"
DEFAULT_CAMERA = "topdown_depth"


def _write_png(path: Path, image: np.ndarray) -> None:
    """不依赖额外图像包，写入 8 位 RGB/灰度或 16 位灰度 PNG。"""
    if image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8:
        height, width, _ = image.shape
        color_type, bit_depth = 2, 8
        rows = image
    elif image.ndim == 2 and image.dtype == np.uint8:
        height, width = image.shape
        color_type, bit_depth = 0, 8
        rows = image
    elif image.ndim == 2 and image.dtype == np.uint16:
        height, width = image.shape
        color_type, bit_depth = 0, 16
        rows = image.astype(">u2", copy=False)
    else:
        raise ValueError("Expected uint8 RGB/grayscale or uint16 grayscale image")

    raw = b"".join(b"\x00" + row.tobytes() for row in rows)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


class D435iCamera:
    """从场景顶视 D435i 渲染 RGB 图像和米单位深度图。"""

    def __init__(
        self,
        scene_path: str | Path = DEFAULT_SCENE,
        width: int = 640,
        height: int = 480,
        camera_name: str = DEFAULT_CAMERA,
    ) -> None:
        self.scene_path = Path(scene_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.camera_name = camera_name

        camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
        )
        if camera_id < 0:
            raise ValueError(f"Camera {self.camera_name!r} is not in {self.scene_path}")

        self.width = width
        self.height = height
        self._renderer = RobotHiddenRenderer(self.model, height=height, width=width)

    def capture(self, data: Optional[mujoco.MjData] = None) -> tuple[np.ndarray, np.ndarray]:
        """返回 D435i 的 ``(rgb, depth_m)``。

        ``rgb`` 的形状为 ``(height, width, 3)``，数据类型为 ``uint8``；
        ``depth_m`` 的形状为 ``(height, width)``，数据类型为 ``float32``，
        数值表示相机视线方向上的米单位距离。
        """
        render_data = self.data if data is None else data
        mujoco.mj_forward(self.model, render_data)
        self._renderer.update_scene(render_data, camera=self.camera_name)
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        try:
            depth_m = self._renderer.render().copy()
        finally:
            self._renderer.disable_depth_rendering()
        return rgb, depth_m

    def capture_and_save(
        self,
        output_dir: str | Path = PROJECT_ROOT / "output",
        data: Optional[mujoco.MjData] = None,
    ) -> tuple[Path, Path, Path]:
        """采集并保存 RGB PNG、可视化深度 PNG 与米单位深度 NPY。

        深度 PNG 会进行对比度拉伸，距离较近的像素更亮；NPY 文件保留原始
        ``float32`` 米单位深度，供算法直接使用。
        """
        rgb, depth_m = self.capture(data)
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        rgb_path = output_path / "d435i_rgb.png"
        depth_png_path = output_path / "d435i_depth.png"
        depth_npy_path = output_path / "d435i_depth.npy"

        valid = np.isfinite(depth_m) & (depth_m > 0)
        depth_preview = np.zeros(depth_m.shape, dtype=np.uint8)
        if np.any(valid):
            nearest = float(depth_m[valid].min())
            farthest = float(depth_m[valid].max())
            if farthest > nearest:
                scaled = (depth_m[valid] - nearest) / (farthest - nearest)
                depth_preview[valid] = np.rint(255.0 * (1.0 - scaled)).astype(np.uint8)
            else:
                depth_preview[valid] = 255

        _write_png(rgb_path, rgb)
        _write_png(depth_png_path, depth_preview)
        np.save(depth_npy_path, depth_m)
        return rgb_path, depth_png_path, depth_npy_path

    def close(self) -> None:
        """释放 OpenGL 渲染上下文。"""
        self._renderer.close()

    def __enter__(self) -> "D435iCamera":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


if __name__ == "__main__":
    with D435iCamera() as camera:
        rgb, depth_m = camera.capture()
        rgb_path, depth_png_path, depth_npy_path = camera.capture_and_save()
    print(f"RGB: shape={rgb.shape}, dtype={rgb.dtype}")
    print(
        "Depth (m): "
        f"shape={depth_m.shape}, dtype={depth_m.dtype}, "
        f"min={depth_m.min():.3f}, max={depth_m.max():.3f}"
    )
    print(f"Saved: {rgb_path}, {depth_png_path}, {depth_npy_path}")


