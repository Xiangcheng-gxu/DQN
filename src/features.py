"""视觉状态：顶视RGB-D -> 完整工作区 -> 250方形高度图 -> 262维特征。

高度图数值是“高出桌面多少米”，不是相机到物体的距离。
目标掩码用红色阈值获得；输入网络的特征不读取仿真物体的真实位置。
"""
from dataclasses import dataclass

import mujoco
import numpy as np
import cv2

from push_points import extract_target_contour
from rendering import RobotHiddenRenderer
from scene import WORKSPACE_LIMITS, WORKSPACE_REFERENCE_Z_M

STATE_DIM = 262  # 4个目标高度 + 4×8×8个周围高度 + 2个目标位置。
HEIGHTMAP_RESOLUTION_M = 0.002  # 50 cm工作区 / 250像素 = 2 mm/像素。


def extract_features(heightmap: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """把一张250×250高度图压缩成网络状态，而不是直接输入整张图片。

先把目标移到图中心，取目标四个区域的平均高度；再在上、右、下、左
各取64×64窗口，分成8×8格并求平均。最后拼接目标原始归一化XY位置。
前260项按本帧最大值归一化；边界外补零，保持输出长度不变。
"""
    pixels = np.argwhere(mask)
    if not len(pixels):
        return np.zeros(STATE_DIM, dtype=np.float32)
    h, w = heightmap.shape
    tx, ty = int(pixels[:, 1].mean()), int(pixels[:, 0].mean())
    cx, cy = w // 2, h // 2
    dx, dy = cx - int(tx), cy - int(ty)
    # 把目标平移到图中心，让“目标周围”在不同场景中有相同的特征位置。
    shifted = np.zeros_like(heightmap)
    ys, xs = np.nonzero(heightmap)
    xx, yy = xs + dx, ys + dy
    valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
    shifted[yy[valid], xx[valid]] = heightmap[ys[valid], xs[valid]]
    bx = max(1, int(np.ptp(pixels[:, 1]) / 2))
    by = max(1, int(np.ptp(pixels[:, 0]) / 2))
    padded = np.pad(shifted, 128)

    def crop(x, y, width, height):
        return padded[y + 128:y + 128 + height, x + 128:x + 128 + width]

    values = [float(crop(x, y, bx, by).mean())
              for x, y in ((cx-bx, cy-by), (cx, cy-by), (cx-bx, cy), (cx, cy))]
    # 四个周围窗口的顺序固定；训练后不能随意换顺序，否则输入含义会变。
    for x, y in ((cx-32, cy-by-64), (cx+bx, cy-32),
                 (cx-32, cy+by), (cx-bx-64, cy-32)):
        region = crop(x, y, 64, 64)
        pooled = region.reshape(8, 8, 8, 8).mean(axis=(1, 3))
        values.extend(pooled.ravel())
    local = np.asarray(values, dtype=np.float32)
    maximum = float(local.max())
    if maximum > 0:
        local /= maximum
    return np.concatenate((local, ((tx-cx)/(w/2), (ty-cy)/(h/2)))).astype(np.float32)


@dataclass
class Observation:
    """同一帧的图像、目标信息和状态；训练/推点/预览共用，避免各算各的。"""
    heightmap: np.ndarray
    target_mask: np.ndarray
    state: np.ndarray
    target_center_xy: np.ndarray | None
    target_height_m: float | None
    target_contour_px: np.ndarray | None
    rgb: np.ndarray | None = None  # 未裁剪的相机RGB原图。
    workspace_crop_rgb: np.ndarray | None = None
    workspace_rgb: np.ndarray | None = None  # 校正并缩放到250×250的工作区RGB。


class HeightmapCamera:
    """唯一的训练相机接口：先裁完整工作区，再校正投影，最后缩到250像素。

图像列增加对应世界+X，行增加对应世界+Y；不等同于viewer屏幕的上下左右。
机械臂和相机外壳只在该相机取图时隐藏，不从物理场景中删除。
"""

    def __init__(self, scene):
        self.scene = scene
        self.width, self.height = 640, 480
        self.renderer = RobotHiddenRenderer(scene.model, height=self.height, width=self.width)
        self.camera_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, "topdown_depth")
        fy = self.height / (2 * np.tan(np.deg2rad(scene.model.cam_fovy[self.camera_id]) / 2))
        u, v = np.meshgrid(np.arange(self.width)+.5, np.arange(self.height)+.5)
        self.rays = np.stack(((u-self.width/2)/fy, -(v-self.height/2)/fy, -np.ones_like(u)), axis=-1)
        self.shape = tuple(np.rint(np.ptp(WORKSPACE_LIMITS[:2], axis=1) / HEIGHTMAP_RESOLUTION_M).astype(int)[::-1])

    def workspace_crop_bounds(self):
        """把桌面工作区四角投影到图像，动态求裁剪边界，右/下边界不包含。

相机高度变化后不能沿用旧像素框；若看不完整工作区就报错，不悄悄裁掉。
"""
        data, model = self.scene.data, self.scene.model
        rotation = data.cam_xmat[self.camera_id].reshape(3, 3)
        corners = np.array([(x, y, WORKSPACE_REFERENCE_Z_M)
                            for x in WORKSPACE_LIMITS[0] for y in WORKSPACE_LIMITS[1]])
        local = (corners-data.cam_xpos[self.camera_id]) @ rotation
        if np.any(local[:, 2] >= 0):
            raise ValueError("Workspace is behind camera")
        focal = self.height / (2*np.tan(np.deg2rad(model.cam_fovy[self.camera_id])/2))
        pixels = np.column_stack((self.width/2+focal*local[:, 0]/(-local[:, 2]),
                                  self.height/2-focal*local[:, 1]/(-local[:, 2])))
        lower = np.floor(pixels.min(axis=0)).astype(int)
        upper = np.ceil(pixels.max(axis=0)).astype(int)
        if np.any(lower < 0) or np.any(upper > (self.width, self.height)):
            raise ValueError("Camera cannot see the complete workspace; increase height or FOV")
        return int(lower[0]), int(lower[1]), int(upper[0]), int(upper[1])

    def capture(self) -> Observation:
        """一次取得RGB和深度，经世界坐标投影得到高度图，再识别目标。"""
        model, data = self.scene.model, self.scene.data
        self.renderer.update_scene(data, camera=self.camera_id)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        try:
            depth = self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()
        x0, y0, x1, y1 = self.workspace_crop_bounds()
        # 第一步：只保留完整工作区的图像范围，不先resize整张相机图。
        crop_rgb = rgb[y0:y1, x0:x1].copy()
        rotation = data.cam_xmat[self.camera_id].reshape(3, 3)
        xyz = (self.rays[y0:y1, x0:x1] * depth[y0:y1, x0:x1, None]) @ rotation.T + data.cam_xpos[self.camera_id]
        xyz, colors = xyz.reshape(-1, 3), crop_rgb.reshape(-1, 3)
        # 第二步：深度反投影到世界坐标，再过滤工作区外或无效的采样点。
        valid = np.isfinite(xyz).all(axis=1)
        valid &= np.all(xyz[:, :2] >= WORKSPACE_LIMITS[:2, 0], axis=1)
        valid &= np.all(xyz[:, :2] < WORKSPACE_LIMITS[:2, 1], axis=1)
        valid &= (xyz[:, 2] >= WORKSPACE_REFERENCE_Z_M-.002) & (xyz[:, 2] <= WORKSPACE_REFERENCE_Z_M+.4)
        xyz, colors = xyz[valid], colors[valid]
        # 第三步：按世界XY重新栅格化，减小透视导致的高物体位置偏移。
        native_shape = (y1-y0, x1-x0)
        native_step = np.ptp(WORKSPACE_LIMITS[:2], axis=1)/np.array(native_shape[::-1])
        cells = np.floor((xyz[:, :2]-WORKSPACE_LIMITS[:2, 0])/native_step).astype(int)
        flat = cells[:, 1]*native_shape[1] + cells[:, 0]
        # 每个格子保留最高的可见点，并保留这个点对应的RGB，避免颜色和高度错配。
        order = np.argsort(xyz[:, 2], kind="stable")
        _, last = np.unique(flat[order][::-1], return_index=True)
        chosen = order[::-1][last]
        heightmap = np.zeros(native_shape, dtype=np.float32)
        color_map = np.zeros((*native_shape, 3), dtype=np.uint8)
        heightmap.ravel()[flat[chosen]] = np.maximum(0, xyz[chosen, 2]-WORKSPACE_REFERENCE_Z_M)
        color_map.reshape(-1, 3)[flat[chosen]] = colors[chosen]
        # 栅格化可能留下很细的采样缝；只补1.5像素以内的小洞，不补被遮挡的表面。
        occupied = np.zeros(native_shape, dtype=bool)
        occupied.ravel()[flat[chosen]] = True
        if occupied.any() and not occupied.all():
            distance, labels = cv2.distanceTransformWithLabels(
                (~occupied).astype(np.uint8), cv2.DIST_L2, 5,
                labelType=cv2.DIST_LABEL_PIXEL)
            fill = (~occupied) & (distance <= 1.5)
            height_lookup = np.concatenate(([0.], heightmap[occupied]))
            color_lookup = np.vstack((np.zeros((1, 3), dtype=np.uint8), color_map[occupied]))
            heightmap[fill] = height_lookup[labels[fill]]
            color_map[fill] = color_lookup[labels[fill]]
        size = self.shape[::-1]
        # 第四步：RGB与高度图使用完全相同的最近邻缩放，保持像素对齐。
        heightmap = cv2.resize(heightmap, size, interpolation=cv2.INTER_NEAREST_EXACT)
        color_map = cv2.resize(color_map, size, interpolation=cv2.INTER_NEAREST_EXACT)
        r, g, b = color_map.astype(float).transpose(2, 0, 1)
        # 第五步：红色明显高于绿/蓝且高于桌面5mm的像素，作为目标掩码。
        mask = (r > 60) & (r > 1.5*g) & (r > 1.5*b) & (heightmap > .005)
        contour = extract_target_contour(mask)
        center, target_height = None, None
        if contour is not None:
            # 用目标像素高度的中位数估计顶面，不混入邻近障碍物的高度。
            target_height = float(np.median(heightmap[mask]))
            center = WORKSPACE_LIMITS[:2, 0] + (contour.center_pixel_xy+.5)*HEIGHTMAP_RESOLUTION_M
        # 先保存轮廓供定位使用；只有a0-a3真正需要推点时才搜索四条边缘。
        return Observation(heightmap, mask, extract_features(heightmap, mask), center,
                           target_height, None if contour is None else contour.points, rgb=rgb,
                           workspace_crop_rgb=crop_rgb, workspace_rgb=color_map)

    def close(self):
        self.renderer.close()
