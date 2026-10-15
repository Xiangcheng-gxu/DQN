"""只在训练RGB/深度相机中隐藏机械臂、夹爪和相机外壳，不修改物理模型。"""
from copy import copy

import mujoco
import numpy as np


class RobotHiddenRenderer(mujoco.Renderer):
    """用相机自己的几何组开关隐藏物体，RGB和深度都能看到机械臂后方。

    group1是相机外壳，group2/3是机械臂显示/碰撞体；桌面物体不能用这三组。
    viewer仍显示完整场景，机械臂的碰撞仍存在，不是把模型设透明或删除。
    """

    ROBOT_RENDER_GROUPS = (2, 3)
    HIDDEN_RENDER_GROUPS = (1, 2, 3)

    def __init__(self, model, *, height=480, width=640, max_geom=10000):
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if root < 0:
            raise ValueError("Cannot hide robot: UR5e root body 'base' is missing")
        robot_bodies = np.zeros(model.nbody, dtype=bool)
        robot_bodies[root] = True
        # MuJoCo父物体编号在子物体之前，顺序遍历即可找到机械臂的全部后代。
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
