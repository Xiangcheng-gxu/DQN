"""DQN 学习核心：262维状态 -> 8个动作Q值 -> 经验回放 -> TD误差更新。

Q值表示从当前状态执行某动作后，预计能得到的折扣累计奖励，不是成功概率。
新训练的实际参数由 train_dqn.py 的 parse_args 传入；此处也保留独立使用的默认值。
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from features import STATE_DIM
from memory import ReplayBuffer, Transition


@dataclass
class DQNConfig:
    """学习参数容器。保存到模型中，以便续训时完整恢复。"""

    hidden_units: tuple[int, ...] = (300, 200)  # 隐藏层神经元数量。
    learning_rate: float = .001  # 每次更新网络参数的步幅。
    discount: float = .9  # gamma：未来奖励的折扣，越大越重视后续结果。
    tau: float = .999  # 这里是“旧目标网络保留比例”，不是新权重比例。
    epsilon_start: float = 1.0
    epsilon_end: float = .005
    epsilon_decay: int = 1000  # 探索率在线性衰减中到达最终值的回合编号。
    replay_buffer_size: int = 10000  # 最多保存多少条经验，满了丢弃最旧经验。
    init_replay_buffer_size: int = 500  # warmup：达到多少条经验才开始更新。
    batch_size: int = 32  # 每次随机抽取多少条经验学习。
    double_dqn: bool = False  # 可选：在线网络选动作，目标网络给该动作估值。

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
    """全连接网络。输入一条262维特征，输出a0到a7的8个Q值。"""
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
        # 隐藏层使用 ReLU；输出层不加 softmax，因为 Q 值可正可负且不是概率。
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return self.out(x)


class DQN:
    def __init__(self, config=None, *, seed=1):
        self.config = config or DQNConfig()
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        # 在线网络负责选动作和学习；目标网络慢慢跟随它，减轻学习目标的抖动。
        # 教学项目统一使用CPU，不做显卡检测或设备切换。
        self.network = QNetwork(hidden_units=self.config.hidden_units).cpu()
        self.target_network = QNetwork(hidden_units=self.config.hidden_units).cpu()
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        self.replay = ReplayBuffer(self.config.replay_buffer_size, seed)
        self.learn_steps = 0
        self.env_steps = 0
        self.current_episode = 0
        self.last_action_mode = None

    @property
    def epsilon(self):
        """第1回合用初始探索率，第N回合到最终值；同一回合内保持不变。"""
        c = self.config
        progress = (1.0 if c.epsilon_decay == 1 else
                    min(max(self.current_episode-1, 0)/(c.epsilon_decay-1), 1.0))
        if progress >= 1:
            return c.epsilon_end
        return c.epsilon_start + (c.epsilon_end-c.epsilon_start)*progress

    @torch.no_grad()
    def q_values(self, state):
        """只做推理，不建立梯度图；返回8个动作的估值。"""
        # NumPy float32数组直接转CPU张量；输出仍为NumPy，便于选动作和打印。
        return self.network(torch.as_tensor(state, dtype=torch.float32, device="cpu")).numpy()

    def select_action(self, state, *, training=True, action_mask=None):
        """训练按epsilon随机探索，否则取最大Q值；mask只约束本次选动作。"""
        allowed = np.ones(8, dtype=bool) if action_mask is None else np.asarray(action_mask, dtype=bool)
        if allowed.shape != (8,) or not allowed.any():
            raise ValueError("Action mask must have shape (8,) with at least one allowed action")
        candidates = np.flatnonzero(allowed)
        if training and self.rng.random() < self.epsilon:
            self.last_action_mode = "explore"
            return int(self.rng.choice(candidates))
        self.last_action_mode = "exploit"
        return int(np.argmax(np.where(allowed, self.q_values(state), -np.inf)))

    def learn(self, transition: Transition):
        """每个交互步调用一次：先存经验，预热完成后随机抽一批并更新。"""
        self.replay.store(transition)
        self.env_steps += 1
        c = self.config
        if len(self.replay) < c.init_replay_buffer_size:
            # 预热不是不收集经验，而是暂时不改神经网络参数。
            return None
        state, action, reward, next_state, terminal = (
            torch.as_tensor(a, device="cpu") for a in self.replay.sample(c.batch_size))
        # 先软更新目标网络：保留 tau 的旧参数，加上 (1-tau) 的在线参数。
        with torch.no_grad():
            for target, online in zip(self.target_network.parameters(), self.network.parameters()):
                target.mul_(c.tau).add_(online, alpha=1-c.tau)
            if c.double_dqn:
                best = self.network(next_state).argmax(1, keepdim=True)
                next_q = self.target_network(next_state).gather(1, best).squeeze(1)
            else:
                next_q = self.target_network(next_state).max(1).values
            # 真正终止时没有未来奖励；仅达到步数上限不作为 terminal。
            target_q = reward + (1-terminal)*c.discount*next_q
        q = self.network(state).gather(1, action.long()[:, None]).squeeze(1)
        # 只更新这批经验中“实际选过的动作”的Q值，损失是预测与TD目标的均方差。
        loss = nn.functional.mse_loss(q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.learn_steps += 1
        return float(loss.detach())

    def save(self, path, *, metadata=None):
        """保存的不只是网络，还包括优化器、经验池和随机状态，支持继续训练。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": 1, "config": asdict(self.config), "network": self.network.state_dict(),
                   "target": self.target_network.state_dict(), "optimizer": self.optimizer.state_dict(),
                   "replay": self.replay.state_dict(), "learn_steps": self.learn_steps,
                   "env_steps": self.env_steps, "rng": self.rng.bit_generator.state,
                   "current_episode": self.current_episode, "epsilon_schedule": "linear_episode",
                   "torch_rng": torch.get_rng_state(), "metadata": metadata or {}}
        temporary = path.with_suffix(path.suffix+".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        """恢复完整检查点。只加载自己或教师提供的可信pt文件。"""
        # weights_only=False 可以恢复经验池，也意味着不能加载来源不明的文件。
        # 检查点中的网络和优化器张量都映射到CPU，兼容先前保存的模型。
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != 1:
            raise ValueError("Unsupported checkpoint format")
        agent = cls(DQNConfig(**payload["config"]))
        agent.network.load_state_dict(payload["network"])
        agent.target_network.load_state_dict(payload["target"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent.replay.load_state_dict(payload["replay"])
        agent.learn_steps, agent.env_steps = payload["learn_steps"], payload["env_steps"]
        agent.current_episode = payload.get("current_episode", payload.get("metadata", {}).get("episode", 0))
        agent.rng.bit_generator.state = payload["rng"]
        torch.set_rng_state(payload["torch_rng"].cpu())
        return agent, payload["metadata"]
