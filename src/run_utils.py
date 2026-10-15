"""训练和评估共用的小工具：中文输出、成功统计、覆盖保存最新相机图像。"""
import cv2
import numpy as np
import csv
from dataclasses import dataclass


@dataclass
class SuccessStatistics:
    """只累计成功回合的数量和步数，失败回合不能混入平均值。"""
    count: int = 0
    total_steps: int = 0

    def record(self, success, steps):
        if success:
            self.count += 1
            self.total_steps += steps

    def format(self):
        average = f"{self.total_steps/self.count:.2f}步" if self.count else "暂无"
        return (f"    成功回合数: {self.count} | 成功回合总步数: {self.total_steps}\n"
                f"    成功回合平均步数: {average}")

    @classmethod
    def restore(cls, metadata, csv_path):
        if "success_statistics" in metadata:
            return cls(**metadata["success_statistics"])
        statistics = cls()
        # 老模型若没有累计统计，就从对应训练日志恢复；不读取检查点之后的回合。
        if csv_path.is_file():
            with csv_path.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    if int(row["episode"]) <= metadata["episode"]:
                        statistics.record(int(row["success"]), int(row["steps"]))
        return statistics


def step_result_text(reason, reward):
    """内部英文结果用于 CSV，终端则翻译成简洁中文。"""
    labels = {
        "separated": "成功分离", "pushed": "有效推动", "empty_push": "空推",
        "out_of_workspace": "物体越界", "target_lost": "目标丢失",
        "motion_failed": "动作失败", "push_point_failed": "空推",
        "no_effective_actions": "所有动作均无效, 结束回合",
    }
    if reason == "step_limit":
        return ("空推" if reward == -5 else "有效推动") + ", 达到步数上限"
    return labels.get(reason, "未知结果")


def format_step_output(step, action, reward, reason, epsilon, selection_mode):
    """逐步打印动作、探索或利用、奖励、结果和本回合探索率。"""
    mode = {"explore": "探索", "exploit": "利用"}[selection_mode]
    return (f"\n        ---- Step {step} ----\n"
            f"            动作: a{action}\n"
            f"            选择方式: {mode}\n"
            f"            奖励: {reward:+g}\n"
            f"            结果: {step_result_text(reason, reward)}\n"
            f"            探索率: {epsilon:.4f} ({epsilon:.2%})")


def save_latest_images(observation, directory):
    """每次覆盖五张预览图；先写临时文件，再替换，避免看到半张图。"""
    directory.mkdir(parents=True, exist_ok=True)
    images = {
        "heightmap_250.png": np.rint(np.clip(observation.heightmap / .04, 0, 1) * 255).astype(np.uint8),
        "target_mask_250.png": observation.target_mask.astype(np.uint8) * 255,
    }
    for attribute, name in (("rgb", "rgb_original.png"),
                            ("workspace_crop_rgb", "workspace_crop.png"),
                            ("workspace_rgb", "workspace_rgb_250.png")):
        value = getattr(observation, attribute, None)
        if value is not None:
            images[name] = value
    for name, image in images.items():
        destination = directory / name
        temporary = directory / (name + ".tmp")
        # 相机返回 RGB，但 OpenCV 编码要求 BGR；灰度图不需要转换。
        pixels = image[..., ::-1] if image.ndim == 3 else image
        ok, encoded = cv2.imencode(".png", pixels)
        if not ok:
            raise RuntimeError(f"Cannot encode preview: {name}")
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(destination)
