"""Reference-policy 262-dimensional, target-centred heightmap features."""
from dataclasses import dataclass

import mujoco
import numpy as np
import cv2

from push_points import extract_target_contour
from rendering import RobotHiddenRenderer
from scene import WORKSPACE_LIMITS, WORKSPACE_REFERENCE_Z_M

STATE_DIM = 262
HEIGHTMAP_RESOLUTION_M = 0.002


def extract_features(heightmap: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """4 target means + 4 x 8 x 8 surrounding means + normalized XY.

    Region order, cell order, and per-observation max normalization follow
    robot.extract_features/state_representation in the reference project.
    Zero padding also makes crops at the workspace boundary well defined.
    """
    pixels = np.argwhere(mask)
    if not len(pixels):
        return np.zeros(STATE_DIM, dtype=np.float32)
    h, w = heightmap.shape
    tx, ty = int(pixels[:, 1].mean()), int(pixels[:, 0].mean())
    cx, cy = w // 2, h // 2
    dx, dy = cx - int(tx), cy - int(ty)
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
    heightmap: np.ndarray
    target_mask: np.ndarray
    state: np.ndarray
    target_center_xy: np.ndarray | None
    target_height_m: float | None
    target_contour_px: np.ndarray | None
    rgb: np.ndarray | None = None  # Uncropped camera RGB, before heightmap projection.
    workspace_crop_rgb: np.ndarray | None = None
    workspace_rgb: np.ndarray | None = None  # Rectified, resized 250x250 workspace.


class HeightmapCamera:
    """Crop the whole workspace, rectify RGB-D, then resize to a 2 mm grid.

    No simulator object positions enter the network. The known red target
    supplies the mask instead of the reference's external segmentation model.
    Rows increase with world Y, columns with world X (reference convention).
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
        """Project table-plane workspace corners; return exclusive XY bounds.

        Recomputed from the active camera pose/FOV, never from a stale pixel box.
        Refuse clipped workspaces instead of silently resizing incomplete data.
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
        crop_rgb = rgb[y0:y1, x0:x1].copy()
        rotation = data.cam_xmat[self.camera_id].reshape(3, 3)
        xyz = (self.rays[y0:y1, x0:x1] * depth[y0:y1, x0:x1, None]) @ rotation.T + data.cam_xpos[self.camera_id]
        xyz, colors = xyz.reshape(-1, 3), crop_rgb.reshape(-1, 3)
        valid = np.isfinite(xyz).all(axis=1)
        valid &= np.all(xyz[:, :2] >= WORKSPACE_LIMITS[:2, 0], axis=1)
        valid &= np.all(xyz[:, :2] < WORKSPACE_LIMITS[:2, 1], axis=1)
        valid &= (xyz[:, 2] >= WORKSPACE_REFERENCE_Z_M-.002) & (xyz[:, 2] <= WORKSPACE_REFERENCE_Z_M+.4)
        xyz, colors = xyz[valid], colors[valid]
        # Rectify at the crop's native sampling density before the final resize.
        # Direct resizing of perspective depth would shift elevated objects' XY.
        native_shape = (y1-y0, x1-x0)
        native_step = np.ptp(WORKSPACE_LIMITS[:2], axis=1)/np.array(native_shape[::-1])
        cells = np.floor((xyz[:, :2]-WORKSPACE_LIMITS[:2, 0])/native_step).astype(int)
        flat = cells[:, 1]*native_shape[1] + cells[:, 0]
        # Keep the highest visible point per cell, with its corresponding RGB.
        order = np.argsort(xyz[:, 2], kind="stable")
        _, last = np.unique(flat[order][::-1], return_index=True)
        chosen = order[::-1][last]
        heightmap = np.zeros(native_shape, dtype=np.float32)
        color_map = np.zeros((*native_shape, 3), dtype=np.uint8)
        heightmap.ravel()[flat[chosen]] = np.maximum(0, xyz[chosen, 2]-WORKSPACE_REFERENCE_Z_M)
        color_map.reshape(-1, 3)[flat[chosen]] = colors[chosen]
        # Rasterisation can leave isolated one-pixel sampling seams. Copy the
        # nearest measured sample only across tiny holes, not hidden surfaces.
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
        # Identical nearest-neighbour sampling preserves colour/depth alignment
        # and avoids inventing intermediate object heights or mixed red edges.
        size = self.shape[::-1]
        heightmap = cv2.resize(heightmap, size, interpolation=cv2.INTER_NEAREST_EXACT)
        color_map = cv2.resize(color_map, size, interpolation=cv2.INTER_NEAREST_EXACT)
        r, g, b = color_map.astype(float).transpose(2, 0, 1)
        mask = (r > 60) & (r > 1.5*g) & (r > 1.5*b) & (heightmap > .005)
        contour = extract_target_contour(mask)
        center, target_height = None, None
        if contour is not None:
            # Robust top height from target pixels, not neighbouring obstacles.
            target_height = float(np.median(heightmap[mask]))
            center = WORKSPACE_LIMITS[:2, 0] + (contour.center_pixel_xy+.5)*HEIGHTMAP_RESOLUTION_M
        # Capture the contour for the centre, but sample its four edges only
        # when an inward action 0--3 actually requests a push point.
        return Observation(heightmap, mask, extract_features(heightmap, mask), center,
                           target_height, None if contour is None else contour.points, rgb=rgb,
                           workspace_crop_rgb=crop_rgb, workspace_rgb=color_map)

    def close(self):
        self.renderer.close()
