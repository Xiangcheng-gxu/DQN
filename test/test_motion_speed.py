"""Independent descent/push speeds, timing and checkpoint overrides."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from environment import EnvironmentConfig
from push_controller import PushController
from train_dqn import parse_args


class MotionSpeedTests(unittest.TestCase):
    def test_push_stays_at_lifted_endpoint_without_home_move(self):
        from action_primitives import PushPrimitive
        from scene import create_fixed_scene
        scene = create_fixed_scene()
        controller = PushController(scene)
        primitive = PushPrimitive(4, np.array((-.5, 0, .779)),
                                  np.array((-.42, 0, .779)), 0.)
        with patch.object(controller, '_joint_move', wraps=controller._joint_move) as move:
            controller.push(primitive)
        self.assertEqual(move.call_count, 1)  # Only transit to approach, no return.
        np.testing.assert_allclose(scene.robot.bottom_center_pose(scene.data).position_m,
                                   primitive.end_position_m+(0, 0, .12), atol=1e-5)
        self.assertTrue(scene.gripper.is_closed(scene.data, settled=False))

    def test_descent_and_push_double_speed_but_lift_unchanged(self):
        scene = SimpleNamespace(robot=SimpleNamespace(_qpos_addresses=(), _dof_addresses=()),
                                model=SimpleNamespace(opt=SimpleNamespace(timestep=.002)))
        controller = PushController(scene)
        self.assertEqual(controller.descent_speed_m_s, .08)
        self.assertEqual(controller.push_speed_m_s, .08)
        self.assertEqual(controller.lift_speed_m_s, .04)
        start = np.zeros(3)
        self.assertEqual(controller._segment_steps(start, (0, 0, -.12), controller.descent_speed_m_s), 750)
        self.assertEqual(controller._segment_steps(start, (.08, 0, 0), controller.push_speed_m_s), 500)
        self.assertEqual(controller._segment_steps(start, (0, 0, .12), controller.lift_speed_m_s), 1500)
        self.assertEqual(controller._segment_steps(start, start, .08), 1)
        self.assertEqual(EnvironmentConfig().push_speed_m_s, .08)
        self.assertEqual(EnvironmentConfig().descent_speed_m_s, .08)

    def test_motion_speeds_must_be_finite_and_positive(self):
        for name in ('push_speed_m_s', 'descent_speed_m_s', 'lift_speed_m_s'):
            for value in (0, -.01, np.inf, np.nan):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        PushController(None, **{name: value})
        for name in ('push_speed_m_s', 'descent_speed_m_s'):
            with self.assertRaises(ValueError):
                EnvironmentConfig(**{name: np.nan})

    def test_checkpoint_retains_explicit_speed_and_defaults_missing_descent(self):
        config = EnvironmentConfig(push_speed_m_s=.06, descent_speed_m_s=.09)
        restored = EnvironmentConfig.from_checkpoint(vars(config))
        self.assertEqual(restored, config)
        old_config = dict(vars(config))
        del old_config['descent_speed_m_s']
        old_config['push_speed_m_s'] = .04
        restored = EnvironmentConfig.from_checkpoint(old_config)
        self.assertEqual(restored.push_speed_m_s, .04)
        self.assertEqual(restored.descent_speed_m_s, .08)

    def test_resume_allows_explicit_motion_speed_overrides(self):
        with patch.object(sys, 'argv', ['train_dqn.py', '--resume', 'example.pt',
                                       '--descent-speed', '.08', '--push-speed', '.08']):
            args = parse_args()
        self.assertEqual(args.descent_speed, .08)
        self.assertEqual(args.push_speed, .08)


if __name__ == '__main__':
    unittest.main()
