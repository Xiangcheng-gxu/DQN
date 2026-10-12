"""Robot-free camera RGB/depth; unchanged live model and viewer rendering."""
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import mujoco
import numpy as np

from camera import D435iCamera
from features import HeightmapCamera
from push_controller import PushController
from robot import UR5eRobot
from scene import (DEFAULT_JOINT_ANGLES_DEG, TARGET_COLOR, OBSTACLE_COLOR,
                   WORKSPACE_REFERENCE_Z_M, SceneObject, create_fixed_scene)


class CameraVisibilityTests(unittest.TestCase):
    def test_camera_centred_at_60cm_and_housing_excluded(self):
        with D435iCamera() as camera:
            rgb, depth = camera.capture()
            cid = mujoco.mj_name2id(camera.model, mujoco.mjtObj.mjOBJ_CAMERA, 'topdown_depth')
            bid = mujoco.mj_name2id(camera.model, mujoco.mjtObj.mjOBJ_BODY, 'd435i_topdown_mount')
            expected = (-.5, 0, WORKSPACE_REFERENCE_Z_M+.6)
            np.testing.assert_allclose(camera.data.cam_xpos[cid], expected, atol=1e-12)
            np.testing.assert_allclose(camera.data.xpos[bid], expected, atol=1e-12)
            np.testing.assert_allclose(-camera.data.cam_xmat[cid].reshape(3, 3)[:, 2], (0, 0, -1))
            renderer = camera._renderer
            ids = {g.objid for g in renderer.scene.geoms[:renderer.scene.ngeom]
                   if g.objtype == mujoco.mjtObj.mjOBJ_GEOM}
            self.assertEqual(len(renderer.housing_geom_ids), 9)
            self.assertTrue(ids.isdisjoint(renderer.housing_geom_ids))
            normal = mujoco.Renderer(camera.model)
            try:
                normal.update_scene(camera.data, camera='overview')
                normal_ids = {g.objid for g in normal.scene.geoms[:normal.scene.ngeom]
                              if g.objtype == mujoco.mjtObj.mjOBJ_GEOM}
                self.assertTrue(set(renderer.housing_geom_ids).issubset(normal_ids))
            finally:
                normal.close()
            # Moving the housing into the optical path still cannot affect either image.
            camera.model.body_pos[bid, 2] -= .2
            rgb_after, depth_after = camera.capture()
            np.testing.assert_array_equal(rgb, rgb_after)
            np.testing.assert_array_equal(depth, depth_after)

    def test_action_demo_preplans_all_directions_without_home_waypoints(self):
        from test_scene import APPROACH_HEIGHT_M, _create_action_demo
        with redirect_stdout(StringIO()):
            demo = _create_action_demo(7)
        self.assertEqual(len(demo.joint_waypoints_deg), 32)
        self.assertEqual(len(demo.cartesian_waypoints), 32)
        for action in range(8):
            end = demo.cartesian_waypoints[action*4+2][0]
            lifted = demo.cartesian_waypoints[action*4+3][0]
            np.testing.assert_allclose(lifted, end+(0, 0, APPROACH_HEIGHT_M))

    def test_occluding_robot_changes_neither_rgb_depth_nor_heightmap(self):
        objects = (SceneObject('target', TARGET_COLOR, np.array((-.5, 0)), .04, .03),
                   SceneObject('obstacle', OBSTACLE_COLOR, np.array((-.44, .04)), .04, .04))
        scene = create_fixed_scene(objects=objects)
        PushController(scene).close_and_settle()
        camera = HeightmapCamera(scene)
        try:
            self.assertIs(camera.renderer._model, scene.model)  # no model copy
            before = camera.capture()
            rgb_before = camera.renderer.render().copy()
            camera.renderer.enable_depth_rendering()
            try:
                depth_before = camera.renderer.render().copy()
            finally:
                camera.renderer.disable_depth_rendering()
            # Put a physically present closed gripper above the red target.
            result = scene.robot.solve_ik_bottom_position(
                scene.data, (-.5, 0, WORKSPACE_REFERENCE_Z_M+.06),
                target_rotation_matrix=np.diag((-1., 1., -1.)),
                position_tolerance_m=1e-6, orientation_tolerance_rad=1e-6,
                max_iterations=1000, damping=.01)
            self.assertTrue(result.success)
            scene.robot.set_joint_angles_deg(scene.data, result.joint_angles_deg)
            mujoco.mj_forward(scene.model, scene.data)
            groups = scene.model.geom_group.copy()
            rgba = scene.model.geom_rgba.copy()
            collisions = scene.model.geom_contype.copy()
            qpos, ctrl = scene.data.qpos.copy(), scene.data.ctrl.copy()
            after = camera.capture()
            rgb_after = camera.renderer.render().copy()
            camera.renderer.enable_depth_rendering()
            try:
                depth_after = camera.renderer.render().copy()
            finally:
                camera.renderer.disable_depth_rendering()
            np.testing.assert_array_equal(rgb_before, rgb_after)
            np.testing.assert_array_equal(depth_before, depth_after)
            np.testing.assert_array_equal(before.heightmap, after.heightmap)
            np.testing.assert_array_equal(before.target_mask, after.target_mask)
            self.assertGreater(after.target_mask.sum(), 100)
            for a, b in ((groups, scene.model.geom_group), (rgba, scene.model.geom_rgba),
                         (collisions, scene.model.geom_contype), (qpos, scene.data.qpos),
                         (ctrl, scene.data.ctrl)):
                np.testing.assert_array_equal(a, b)
            rendered_ids = {g.objid for g in camera.renderer.scene.geoms[:camera.renderer.scene.ngeom]
                            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM}
            self.assertTrue(rendered_ids.isdisjoint(camera.renderer.robot_geom_ids))
            # A separate normal viewer/renderer still sees and is occluded by
            # the robot; hiding is confined to task-camera options.
            normal = mujoco.Renderer(scene.model, height=480, width=640)
            try:
                normal.update_scene(scene.data, camera='topdown_depth')
                normal.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                normal_rgb = normal.render().copy()
                normal.enable_depth_rendering()
                normal_depth = normal.render().copy()
                normal_ids = {g.objid for g in normal.scene.geoms[:normal.scene.ngeom]
                              if g.objtype == mujoco.mjtObj.mjOBJ_GEOM}
                self.assertFalse(normal_ids.isdisjoint(camera.renderer.robot_geom_ids))
                self.assertGreater(np.count_nonzero(np.any(normal_rgb != rgb_after, axis=2)), 100)
                self.assertGreater(np.count_nonzero(abs(normal_depth-depth_after) > .001), 100)
            finally:
                normal.close()
        finally:
            camera.close()

    def test_d435i_rgb_and_depth_also_ignore_robot_pose(self):
        with D435iCamera() as camera:
            robot = UR5eRobot(camera.model)
            robot.set_joint_angles_deg(camera.data, DEFAULT_JOINT_ANGLES_DEG)
            first_rgb, first_depth = camera.capture()
            robot.set_joint_angles_deg(camera.data, DEFAULT_JOINT_ANGLES_DEG+(45, 10, 0, 10, 20, 0))
            second_rgb, second_depth = camera.capture()
            np.testing.assert_array_equal(first_rgb, second_rgb)
            np.testing.assert_array_equal(first_depth, second_depth)
            self.assertIs(camera._renderer._model, camera.model)


if __name__ == '__main__':
    unittest.main()
