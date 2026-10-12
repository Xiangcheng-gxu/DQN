"""Concise Chinese training results."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_dqn import step_result_text


class TrainingOutputTests(unittest.TestCase):
    def test_result_labels(self):
        expected = {
            "separated": "成功分离", "pushed": "有效推动", "empty_push": "空推",
            "out_of_workspace": "物体越界", "target_lost": "目标丢失",
            "motion_failed": "动作失败", "push_point_failed": "推点获取失败",
        }
        for reason, label in expected.items():
            with self.subTest(reason=reason):
                self.assertEqual(step_result_text(reason, 0), label)

    def test_step_limit_preserves_action_result(self):
        self.assertEqual(step_result_text("step_limit", -5), "空推，达到步数上限")
        self.assertEqual(step_result_text("step_limit", -1), "有效推动，达到步数上限")


if __name__ == "__main__":
    unittest.main()
