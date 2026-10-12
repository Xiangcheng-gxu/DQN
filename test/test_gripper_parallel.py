"""Physical regression for parallel pads, actuation and bottom calibration."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import mujoco
import numpy as np

from scene import create_fixed_scene, DEFAULT_JOINT_ANGLES_DEG
from push_controller import PushController
from test_gripper_height import bottom_position


class GripperParallelTests(unittest.TestCase):
    def test_parallel_closure_open_close_and_bottom_reference(self):
        scene = create_fixed_scene()
        controller = PushController(scene)
        controller.close_and_settle()
        pads = scene.gripper._pad_body_ids
        closed_width = np.linalg.norm(scene.data.xpos[pads[0]]-scene.data.xpos[pads[1]])
        # Verify persistence, not just the first frame that reaches the command.
        max_angle = 0.
        for _ in range(500):
            controller.hold(scene.model.opt.timestep)
            max_angle = max(max_angle, *scene.gripper.pad_alignment_errors_rad(scene.data))
        self.assertLess(np.rad2deg(max_angle), .01)
        # Verify the nominal footprint constants against closed collision
        # boxes, rather than confusing the 85 mm opening with tip thickness.
        model, data = scene.model, scene.data
        base = scene.gripper._base_body_id
        rotation = data.xmat[base].reshape(3, 3)
        vertices = []
        for name in ('left_pad1', 'left_pad2', 'right_pad1', 'right_pad2'):
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            size = model.geom_size[geom]
            corners = np.array([(x,y,z) for x in (-size[0],size[0])
                                for y in (-size[1],size[1]) for z in (-size[2],size[2])])
            world = corners@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]
            vertices.extend((world-data.xpos[base])@rotation)
        dimensions = np.ptp(vertices, axis=0)
        self.assertAlmostEqual(dimensions[0], scene.gripper.CLOSED_TIP_ACROSS_PUSH_M, delta=.0001)
        self.assertAlmostEqual(dimensions[1], scene.gripper.CLOSED_TIP_ALONG_PUSH_M, delta=.0001)
        pose = scene.robot.bottom_center_pose(scene.data)
        self.assertLess(abs(pose.position_m[2]-bottom_position(scene)[2]), .00002)
        for _ in range(1500):
            scene.robot.set_joint_angles_deg(scene.data, DEFAULT_JOINT_ANGLES_DEG)
            scene.data.qvel[list(scene.robot._dof_addresses)] = 0
            scene.gripper.open(scene.data)
            mujoco.mj_step(scene.model, scene.data)
        open_width = np.linalg.norm(scene.data.xpos[pads[0]]-scene.data.xpos[pads[1]])
        self.assertGreater(open_width-closed_width, .07)
        controller.close_and_settle()
        self.assertTrue(scene.gripper.is_closed(scene.data, settled=False))
        self.assertLess(np.rad2deg(max(scene.gripper.pad_alignment_errors_rad(scene.data))), .01)


if __name__ == "__main__":
    unittest.main()
