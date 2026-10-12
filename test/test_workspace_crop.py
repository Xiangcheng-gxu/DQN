"""Workspace coverage, dynamic crop and resize-before-feature regressions."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mujoco
import numpy as np
from features import HeightmapCamera, extract_features
from scene import SceneObject, TARGET_COLOR, OBSTACLE_COLOR, create_fixed_scene


class WorkspaceCropTests(unittest.TestCase):
    def test_full_workspace_corners_and_resized_feature_input(self):
        xy = [(-.725, -.225), (-.275, -.225), (-.725, .225), (-.275, .225)]
        objects = tuple(SceneObject("target" if i == 0 else "obstacle",
                                    TARGET_COLOR if i == 0 else OBSTACLE_COLOR,
                                    np.array(point), .03, .02)
                        for i, point in enumerate(xy))
        scene = create_fixed_scene(objects=objects)
        camera = HeightmapCamera(scene)
        try:
            self.assertEqual(camera.workspace_crop_bounds(), (146, 66, 494, 414))
            with patch("features.extract_features", wraps=extract_features) as extractor:
                observation = camera.capture()
                heights, mask = extractor.call_args.args
                self.assertEqual(heights.shape, (250, 250))
                self.assertEqual(mask.shape, (250, 250))
            self.assertEqual(observation.workspace_crop_rgb.shape, (348, 348, 3))
            self.assertEqual(observation.workspace_rgb.shape, (250, 250, 3))
            self.assertEqual(observation.state.shape, (262,))
            for x, y in xy:
                col, row = np.floor((np.array((x, y))-(-.75, -.25))/.002).astype(int)
                self.assertAlmostEqual(float(observation.heightmap[row, col]), .02, places=5)
            np.testing.assert_allclose(observation.target_center_xy, xy[0], atol=.003)
            # Changing height must change the crop, not reuse the old 348px box.
            scene.model.cam_pos[camera.camera_id, 2] += .2
            mujoco.mj_forward(scene.model, scene.data)
            bounds = camera.workspace_crop_bounds()
            self.assertLess(bounds[2]-bounds[0], 348)
            # An incomplete field of view is rejected rather than silently clipped.
            scene.model.cam_pos[camera.camera_id, 2] = .954
            mujoco.mj_forward(scene.model, scene.data)
            with self.assertRaisesRegex(ValueError, "complete workspace"):
                camera.capture()
        finally:
            camera.close()


if __name__ == "__main__":
    unittest.main()
