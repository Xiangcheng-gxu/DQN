"""Camera-only robot exclusion, without modifying physics/viewer models."""
from copy import copy

import mujoco
import numpy as np


class RobotHiddenRenderer(mujoco.Renderer):
    """Render RGB/depth with robot, gripper and camera housing omitted.

    Use camera-local MjvOption.geomgroup to omit robot visual/collision groups
    before rendering. Share the original model and data without copying or
    modifying them. RGB and depth both see geometry behind the robot, unlike
    transparent RGBA which can still occlude depth rendering.
    """

    ROBOT_RENDER_GROUPS = (2, 3)
    HIDDEN_RENDER_GROUPS = (1, 2, 3)

    def __init__(self, model, *, height=480, width=640, max_geom=10000):
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if root < 0:
            raise ValueError("Cannot hide robot: UR5e root body 'base' is missing")
        robot_bodies = np.zeros(model.nbody, dtype=bool)
        robot_bodies[root] = True
        # MuJoCo body indices are parent-before-child.
        for body in range(1, model.nbody):
            robot_bodies[body] |= robot_bodies[model.body_parentid[body]]
        robot_geoms = robot_bodies[model.geom_bodyid]
        self.robot_geom_ids = np.flatnonzero(robot_geoms)
        if (not np.isin(model.geom_group[robot_geoms], self.ROBOT_RENDER_GROUPS).all()
                or np.isin(model.geom_group[~robot_geoms], self.ROBOT_RENDER_GROUPS).any()):
            raise ValueError("Render groups 2/3 must be reserved for robot visual/collision geometry")
        housing_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "d435i_topdown_mount")
        housing_bodies = np.zeros(model.nbody, dtype=bool)
        if housing_root >= 0:
            housing_bodies[housing_root] = True
        for body in range(1, model.nbody):
            housing_bodies[body] |= housing_bodies[model.body_parentid[body]]
        housing_geoms = housing_bodies[model.geom_bodyid]
        self.housing_geom_ids = np.flatnonzero(housing_geoms)
        if (np.any(model.geom_group[housing_geoms] != 1)
                or np.any(model.geom_group[~housing_geoms] == 1)):
            raise ValueError("Render group 1 must be reserved for camera housing geometry")
        self._robot_hidden_option = mujoco.MjvOption()
        self._robot_hidden_option.geomgroup[list(self.HIDDEN_RENDER_GROUPS)] = 0
        super().__init__(model, height=height, width=width, max_geom=max_geom)

    def update_scene(self, data, camera=-1, scene_option=None):
        option = self._robot_hidden_option if scene_option is None else copy(scene_option)
        option.geomgroup[list(self.HIDDEN_RENDER_GROUPS)] = 0
        super().update_scene(data, camera=camera, scene_option=option)
