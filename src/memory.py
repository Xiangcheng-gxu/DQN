"""经验回放池：保存过去的尝试，随机抽取一批重新学习。"""
from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass
class Transition:
    """一条经验 = 当前状态、动作、奖励、下一状态、是否真正结束。"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminal: bool


class ReplayBuffer:
    """先进先出存储、均匀随机采样；容量满时自动移除最旧的一条。"""
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.buffer)

    def store(self, transition: Transition):
        # 必须复制状态数组，否则下一帧改动数组时，过去的经验也会被改掉。
        self.buffer.append(Transition(
            np.array(transition.state, dtype=np.float32, copy=True), int(transition.action),
            float(transition.reward), np.array(transition.next_state, dtype=np.float32, copy=True),
            bool(transition.terminal),
        ))

    def sample(self, batch_size: int):
        """无放回抽样，再把多条经验拼成网络可一次处理的批次数组。"""
        items = self.rng.sample(list(self.buffer), batch_size)
        return (np.stack([t.state for t in items]), np.array([t.action for t in items]),
                np.array([t.reward for t in items], dtype=np.float32),
                np.stack([t.next_state for t in items]),
                np.array([t.terminal for t in items], dtype=np.float32))

    def state_dict(self):
        """随机采样状态也要保存，恢复训练时才能延续随机序列。"""
        return {"capacity": self.capacity, "rng": self.rng.getstate(),
                "transitions": [vars(t) for t in self.buffer]}

    def load_state_dict(self, state):
        self.capacity = state["capacity"]
        self.buffer = deque((Transition(**t) for t in state["transitions"]), maxlen=self.capacity)
        self.rng.setstate(state["rng"])
