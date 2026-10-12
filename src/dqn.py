"""Single-teacher DQN reproduced from the reference dqn.py."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from features import STATE_DIM
from memory import ReplayBuffer, Transition


@dataclass
class DQNConfig:
    hidden_units: tuple[int, ...] = (300, 200)
    learning_rate: float = .001
    discount: float = .9
    tau: float = .999  # Reference convention: retention of OLD target weights.
    epsilon_start: float = .9
    epsilon_end: float = .05
    epsilon_decay: int = 20000
    replay_buffer_size: int = 10000
    init_replay_buffer_size: int = 500
    batch_size: int = 32
    double_dqn: bool = False

    def __post_init__(self):
        if not self.hidden_units or any(n <= 0 for n in self.hidden_units):
            raise ValueError("hidden_units must be positive")
        if not 1 <= self.batch_size <= self.init_replay_buffer_size <= self.replay_buffer_size:
            raise ValueError("Require batch <= warmup <= replay capacity")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1 or self.epsilon_decay <= 0:
            raise ValueError("Invalid epsilon schedule")
        if not 0 <= self.discount <= 1 or not 0 <= self.tau < 1 or self.learning_rate <= 0:
            raise ValueError("Invalid learning parameters")


class QNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=8, hidden_units=(300, 200)):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        previous = state_dim
        for index, units in enumerate(hidden_units):
            layer = nn.Linear(previous, units)
            if index > 0:
                nn.init.uniform_(layer.weight, -.003, .003)
                nn.init.uniform_(layer.bias, -.003, .003)
            self.hidden_layers.append(layer)
            previous = units
        self.out = nn.Linear(previous, action_dim)
        nn.init.uniform_(self.out.weight, -.003, .003)
        nn.init.uniform_(self.out.bias, -.003, .003)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return self.out(x)


class DQN:
    def __init__(self, config=None, *, device="cpu", seed=1):
        self.config = config or DQNConfig()
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.network = QNetwork(hidden_units=self.config.hidden_units).to(self.device)
        self.target_network = QNetwork(hidden_units=self.config.hidden_units).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        self.replay = ReplayBuffer(self.config.replay_buffer_size, seed)
        self.learn_steps = 0
        self.env_steps = 0

    @property
    def epsilon(self):
        c = self.config
        return c.epsilon_end + (c.epsilon_start-c.epsilon_end)*math.exp(-self.learn_steps/c.epsilon_decay)

    @torch.no_grad()
    def q_values(self, state):
        return self.network(torch.as_tensor(state, dtype=torch.float32, device=self.device)).cpu().numpy()

    def select_action(self, state, *, training=True):
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(8))
        return int(np.argmax(self.q_values(state)))

    def learn(self, transition: Transition):
        self.replay.store(transition)
        self.env_steps += 1
        c = self.config
        if len(self.replay) < c.init_replay_buffer_size:
            return None
        state, action, reward, next_state, terminal = (
            torch.as_tensor(a, device=self.device) for a in self.replay.sample(c.batch_size))
        # Reference soft update is performed before calculating the TD target.
        with torch.no_grad():
            for target, online in zip(self.target_network.parameters(), self.network.parameters()):
                target.mul_(c.tau).add_(online, alpha=1-c.tau)
            if c.double_dqn:
                best = self.network(next_state).argmax(1, keepdim=True)
                next_q = self.target_network(next_state).gather(1, best).squeeze(1)
            else:
                next_q = self.target_network(next_state).max(1).values
            target_q = reward + (1-terminal)*c.discount*next_q
        q = self.network(state).gather(1, action.long()[:, None]).squeeze(1)
        loss = nn.functional.mse_loss(q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.learn_steps += 1
        return float(loss.detach())

    def save(self, path, *, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": 1, "config": asdict(self.config), "network": self.network.state_dict(),
                   "target": self.target_network.state_dict(), "optimizer": self.optimizer.state_dict(),
                   "replay": self.replay.state_dict(), "learn_steps": self.learn_steps,
                   "env_steps": self.env_steps, "rng": self.rng.bit_generator.state,
                   "torch_rng": torch.get_rng_state(), "metadata": metadata or {}}
        if torch.cuda.is_available():
            payload["cuda_rng"] = torch.cuda.get_rng_state_all()
        temporary = path.with_suffix(path.suffix+".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, *, device="cpu"):
        # Full replay/optimizer checkpoint; only load your own trusted files.
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("format") != 1:
            raise ValueError("Unsupported checkpoint format")
        agent = cls(DQNConfig(**payload["config"]), device=device)
        agent.network.load_state_dict(payload["network"])
        agent.target_network.load_state_dict(payload["target"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent.replay.load_state_dict(payload["replay"])
        agent.learn_steps, agent.env_steps = payload["learn_steps"], payload["env_steps"]
        agent.rng.bit_generator.state = payload["rng"]
        torch.set_rng_state(payload["torch_rng"].cpu())
        if "cuda_rng" in payload and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in payload["cuda_rng"]])
        return agent, payload["metadata"]
