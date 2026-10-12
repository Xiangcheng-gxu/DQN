"""Uniform FIFO replay, with copied transitions and reproducible sampling."""
from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminal: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.buffer)

    def store(self, transition: Transition):
        self.buffer.append(Transition(
            np.array(transition.state, dtype=np.float32, copy=True), int(transition.action),
            float(transition.reward), np.array(transition.next_state, dtype=np.float32, copy=True),
            bool(transition.terminal),
        ))

    def sample(self, batch_size: int):
        items = self.rng.sample(list(self.buffer), batch_size)
        return (np.stack([t.state for t in items]), np.array([t.action for t in items]),
                np.array([t.reward for t in items], dtype=np.float32),
                np.stack([t.next_state for t in items]),
                np.array([t.terminal for t in items], dtype=np.float32))

    def state_dict(self):
        return {"capacity": self.capacity, "rng": self.rng.getstate(),
                "transitions": [vars(t) for t in self.buffer]}

    def load_state_dict(self, state):
        self.capacity = state["capacity"]
        self.buffer = deque((Transition(**t) for t in state["transitions"]), maxlen=self.capacity)
        self.rng.setstate(state["rng"])
