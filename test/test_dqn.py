"""Learning regressions; run with python -m unittest discover -s test -p test_dqn.py."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import torch

from dqn import DQN, DQNConfig
from features import extract_features
from memory import ReplayBuffer, Transition
from robot import UR5eRobot
from action_primitives import PushPrimitiveSet
from environment import EnvironmentConfig


class DQNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_target_relative_push_heights_and_center_starts(self):
        planner = PushPrimitiveSet(.754)
        center = np.array((-.5, 0.))
        boundary = np.array(((-.48, 0), (-.5, .02), (-.52, 0), (-.5, -.02)))
        for height in (.01, .02, .03, .04):
            for action in range(8):
                with self.subTest(height=height, action=action):
                    primitive = planner.plan(action, center, boundary, target_height_m=height)
                    expected = .754+height+(-.005 if action < 4 else .005)
                    self.assertAlmostEqual(primitive.start_position_m[2], expected)
                    self.assertAlmostEqual(primitive.end_position_m[2], expected)
                    self.assertAlmostEqual(np.linalg.norm(primitive.end_position_m[:2]-primitive.start_position_m[:2]), .08)
                    if action >= 4:
                        np.testing.assert_array_equal(primitive.start_position_m[:2], center)

    def test_old_fixed_height_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "different action-height convention"):
            EnvironmentConfig.from_checkpoint({"push_height_m": .002})
        with self.assertRaisesRegex(ValueError, "different action-height convention"):
            EnvironmentConfig.from_checkpoint({"action_height_mode": "target_relative_v1"})
        config = EnvironmentConfig()
        self.assertEqual(EnvironmentConfig.from_checkpoint(vars(config)), config)

    def test_feature_layout_and_boundaries(self):
        h = np.zeros((250, 250), dtype=np.float32)
        mask = np.zeros_like(h, dtype=bool)
        mask[115:136, 115:136] = True
        h[mask] = .04
        h[51:115, 93:157] = .02
        state = extract_features(h, mask)
        self.assertEqual(state.shape, (262,))
        np.testing.assert_allclose(state[:4], 1)
        np.testing.assert_allclose(state[4:68], .5, atol=1e-7)
        np.testing.assert_allclose(state[-2:], 0)
        edge_mask = np.zeros_like(mask)
        edge_mask[:20, :20] = True
        self.assertTrue(np.isfinite(extract_features(h, edge_mask)).all())
        np.testing.assert_array_equal(extract_features(h, np.zeros_like(mask)), np.zeros(262))

    def test_replay_copies_and_evicts(self):
        replay = ReplayBuffer(2, 7)
        state = np.ones(262, dtype=np.float32)
        replay.store(Transition(state, 0, -1, state+1, False))
        state[:] = 9
        self.assertEqual(replay.buffer[0].state[0], 1)
        replay.store(Transition(state, 1, 1, state, False))
        replay.store(Transition(state, 2, 2, state, True))
        self.assertEqual([t.action for t in replay.buffer], [1, 2])

    def test_bellman_targets_dqn_double_and_terminal(self):
        for double, terminal in ((False, False), (True, False), (True, True)):
            agent = DQN(DQNConfig(batch_size=1, init_replay_buffer_size=1, double_dqn=double))
            with torch.no_grad():
                for parameter in agent.network.parameters():
                    parameter.zero_()
                for parameter in agent.target_network.parameters():
                    parameter.zero_()
                agent.network.out.bias[:2] = torch.tensor((4., 3.))
                agent.target_network.out.bias[:2] = torch.tensor((1., 8.))
            next_q = (1.003 if double else 7.995)
            expected = (2+(0 if terminal else .9*next_q)-4)**2
            loss = agent.learn(Transition(np.zeros(262,dtype=np.float32), 0, 2,
                                          np.ones(262,dtype=np.float32), terminal))
            self.assertAlmostEqual(loss, expected, places=4)
            self.assertTrue(all(p.grad is None for p in agent.target_network.parameters()))

    def test_updates_and_resume(self):
        agent = DQN(DQNConfig(batch_size=2, init_replay_buffer_size=2), seed=11)
        original = [p.detach().clone() for p in agent.network.parameters()]
        rng = np.random.default_rng(9)
        transitions = [Transition(rng.random(262,dtype=np.float32), i%8, float(i-1),
                                  rng.random(262,dtype=np.float32), i==2) for i in range(3)]
        for transition in transitions:
            agent.learn(transition)
        self.assertEqual(agent.learn_steps, 2)
        self.assertTrue(any(not torch.equal(p, q) for p,q in zip(original, agent.network.parameters())))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.pt"
            agent.save(path, metadata={"episode": 1})
            restored, metadata = DQN.load(path)
            self.assertEqual(metadata["episode"], 1)
            self.assertEqual(len(restored.replay), len(agent.replay))
            state = transitions[0].state
            self.assertEqual(agent.select_action(state), restored.select_action(state))
            self.assertAlmostEqual(agent.learn(transitions[0]), restored.learn(transitions[0]), places=7)
            np.testing.assert_array_equal(agent.q_values(state), restored.q_values(state))

    def test_180_degree_orientation_is_not_success(self):
        error = UR5eRobot._orientation_error(np.eye(3), np.diag((-1., -1., 1.)))
        self.assertAlmostEqual(np.linalg.norm(error), np.pi)


if __name__ == "__main__":
    unittest.main()
