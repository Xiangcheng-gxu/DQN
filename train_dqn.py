"""训练入口：读取参数 -> 创建环境 -> 选动作 -> 推动物体 -> 学习 -> 保存。

学生先看 parse_args 中的 default，再顺着 train 的回合循环阅读。
命令行参数优先于 default；续训则使用模型中保存的配置，不重新覆盖它。
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import numpy as np
import torch

from dqn import DQN, DQNConfig
from environment import ClutterPushEnv, EnvironmentConfig
from memory import Transition
from run_utils import format_step_output, save_latest_images
from run_utils import SuccessStatistics


def learning_overrides(args):
    """把命令行学习参数装入 DQNConfig；warmup 对应预热经验条数。"""
    overrides = {name: getattr(args, name) for name in DQNConfig.__dataclass_fields__
                 if getattr(args, name, None) is not None}
    if args.warmup is not None:
        overrides["init_replay_buffer_size"] = args.warmup
    return overrides


def train(args):
    """一个 episode 是一局场景；一个 step 是选择并尝试执行一次推动。"""
    # 网络很小，固定单线程，避免CPU多线程调度开销；NumPy/OpenCV处理视觉数据。
    torch.set_num_threads(1)
    start_episode = 0
    success_statistics = SuccessStatistics()
    # 第一步：续训恢复网络、经验池、优化器、随机状态；新训练从头初始化。
    if args.resume:
        agent, metadata = DQN.load(args.resume)
        config = EnvironmentConfig.from_checkpoint(metadata["environment"]["config"])
        env = ClutterPushEnv(config, viewer=args.viewer)
        env.rng.bit_generator.state = metadata["environment"]["rng"]
        start_episode = metadata["episode"]
        success_statistics = SuccessStatistics.restore(metadata, Path(args.resume).parent/"training.csv")
    else:
        agent = DQN(DQNConfig(**learning_overrides(args)), seed=args.seed)
        env = ClutterPushEnv(EnvironmentConfig(object_count=args.object_count, max_steps=args.max_steps),
                             seed=args.seed, viewer=args.viewer)
    output = Path(args.output) if args.output else (Path(args.resume).parent if args.resume else Path("results/dqn"))
    output.mkdir(parents=True, exist_ok=True)
    # 第二步：准备输出。新训练覆盖同名日志/模型；续训追加日志。
    # config.json 只是本次运行的记录，不是输入配置文件。
    csv_path = output / "training.csv"
    (output/"config.json").write_text(json.dumps({"agent": vars(agent.config), "environment": vars(env.config)}, indent=2), encoding="utf-8")
    print("\n---------------- 训练开始 ----------------", flush=True)
    print(f"    回合数: {args.episodes} | 每回合最多: {env.config.max_steps}步 | 设备: CPU", flush=True)
    preview_dir = Path(__file__).resolve().parent / "temp"
    print(f"    图片目录: {preview_dir}\n", flush=True)
    fieldnames = ["episode", "steps", "reward", "success", "reason", "epsilon", "loss", "buffer", "updates", "seconds"]
    last_completed = start_episode
    try:
        with csv_path.open("a" if args.resume else "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            if stream.tell() == 0:
                writer.writeheader()
            for episode in range(start_episode+1, start_episode+args.episodes+1):
                # 第三步：创建新杂乱场景，聚拢完成、夹爪闭合后获取第一帧。
                agent.current_episode = episode
                started = time.perf_counter()
                print(f"\n    ------------ Episode {episode} ------------", flush=True)
                state, _ = env.reset()
                save_latest_images(env.observation, preview_dir)
                total_reward, losses = 0., []
                for step in range(1, env.config.max_steps+1):
                    # 第四步：epsilon 概率随机探索，其余按 Q 值利用；训练启用 IAS。
                    action = agent.select_action(state, action_mask=env.action_mask)
                    next_state, reward, terminal, truncated, info = env.step(action)
                    save_latest_images(env.observation, preview_dir)
                    # 第五步：存一条经验，达到 warmup 后做一次梯度更新。
                    # 只把 terminal 存为真正终止；单纯达到步数上限仍允许 TD 自举。
                    loss = agent.learn(Transition(state, action, reward, next_state, terminal))
                    if loss is not None:
                        losses.append(loss)
                    total_reward += reward
                    print(format_step_output(step, action, reward, info['reason'], agent.epsilon,
                                             agent.last_action_mode), flush=True)
                    state = next_state
                    if terminal or truncated:
                        break
                # 第六步：每局写一行 CSV。平均成功步数只统计成功回合。
                success_statistics.record(info["success"], step)
                last_completed = episode
                mean_loss = float(np.mean(losses)) if losses else None
                row = dict(episode=episode, steps=step, reward=total_reward, success=int(info["success"]),
                           reason=info["reason"], epsilon=agent.epsilon, loss=mean_loss, buffer=len(agent.replay),
                           updates=agent.learn_steps, seconds=round(time.perf_counter()-started, 2))
                writer.writerow(row)
                stream.flush()
                print(f"\n    回合小结: {step}步 | 总奖励 {total_reward:+g} | "
                      f"{'成功' if info['success'] else '未成功'}", flush=True)
                print(success_statistics.format(), flush=True)
                print("    ----------------------------------------\n", flush=True)
                if episode % args.save_every == 0:
                    # 定期检查点便于回看模型；last.pt 用于下次恢复训练。
                    meta = {"episode": episode, "environment": env.state_dict(),
                            "success_statistics": vars(success_statistics).copy()}
                    agent.save(output/f"episode_{episode:06d}.pt", metadata=meta)
                    agent.save(output/"last.pt", metadata=meta)
    except KeyboardInterrupt:
        print("训练已中断, 正在保存模型和经验.", flush=True)
    finally:
        # 正常结束或 Ctrl+C 后都保存，避免只剩几百回合以前的检查点。
        agent.save(output/"last.pt", metadata={"episode": last_completed, "environment": env.state_dict(),
                                              "success_statistics": vars(success_statistics).copy()})
        env.close()
    print(f"\n---------------- 训练结束 ----------------\n    模型已保存: {output/'last.pt'}\n", flush=True)


def parse_args():
    """教学调参入口：可改各行 default，也可在命令后写 --参数 值。"""
    parser = argparse.ArgumentParser(description="DQN训练", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 1. 运行设置: 固定CPU，只需设置随机种子
    setup = parser.add_argument_group("运行设置")
    setup.add_argument("--seed", type=int, default=1, help="随机种子")

    # 2. 任务设置: 训练回合, 障碍物数量和步数上限
    task = parser.add_argument_group("任务设置")
    task.add_argument("--episodes", type=int, default=300, help="本次训练回合数, 续训时为追加回合数")
    task.add_argument("--object-count", type=int, default=1, help="障碍物数量")
    task.add_argument("--max-steps", type=int, default=8, help="每回合最多步数(1至8)")

    # 3. 网络与学习: 网络结构, 学习率和目标网络更新
    learning = parser.add_argument_group("网络与学习")
    learning.add_argument("--hidden-units", "--hidden_units", nargs="+", type=int, default=[300, 200], help="隐藏层, 例如: 300 200")
    learning.add_argument("--learning-rate", "--learning_rate", type=float, default=0.001, help="学习率")
    learning.add_argument("--discount", type=float, default=0.99, help="奖励折扣系数")
    learning.add_argument("--tau", type=float, default=0.999, help="软更新时旧目标网络权重保留系数")
    learning.add_argument("--double-dqn", "--double_dqn", action=argparse.BooleanOptionalAction, default=False, help="启用/关闭Double DQN")

    # 4. 探索策略: 按回合线性衰减, 第epsilon_decay回合达到最终探索率
    exploration = parser.add_argument_group("探索策略")
    exploration.add_argument("--epsilon-start", "--epsilon_start", type=float, default=1.0, help="初始探索率")
    exploration.add_argument("--epsilon-end", "--epsilon_end", type=float, default=0.005, help="最终探索率")
    exploration.add_argument("--epsilon-decay", "--epsilon_decay", type=int, default=200, help="线性衰减终点回合, 第几回合达到最终探索率")

    # 5. 经验回放: 容量, 预热经验数和采样批大小
    replay = parser.add_argument_group("经验回放")
    replay.add_argument("--replay-buffer-size", "--replay_buffer_size", type=int, default=10000, help="经验池容量")
    replay.add_argument("--warmup", "--init-replay-buffer-size", "--init_replay_buffer_size", type=int, default=80, help="开始学习前的经验数")
    replay.add_argument("--batch-size", "--batch_size", type=int, default=32, help="学习批大小")

    # 6. 保存与续训: 新训练覆盖同名结果, 续训使用检查点配置
    saving = parser.add_argument_group("保存与续训")
    saving.add_argument("--output", default=None, help="结果目录, 新训练使用results/dqn, 续训使用检查点目录")
    saving.add_argument("--save-every", type=int, default=100, help="每隔多少回合保存检查点")
    saving.add_argument("--resume", default=None, help="从可信检查点续训, 沿用其学习和环境配置")

    # 7. 可视化: 默认逐步中文输出
    display = parser.add_argument_group("可视化")
    display.add_argument("--viewer", action="store_true", default=False, help="显示仿真窗口")
    args = parser.parse_args()
    if args.episodes < 1 or args.save_every < 1:
        parser.error("回合数和保存间隔必须为正数")
    if args.max_steps is not None and not 1 <= args.max_steps <= 8:
        parser.error("每回合步数必须为1至8")
    # 默认值不覆盖续训配置, 仅拒绝明确传入的学习和任务修改参数
    fixed_groups = (learning, exploration, replay)
    fixed_options = {option for group in fixed_groups for action in group._group_actions
                     for option in action.option_strings} | {"--object-count", "--max-steps"}
    if args.resume and any(token.split("=", 1)[0] in fixed_options for token in sys.argv[1:]):
        parser.error("续训沿用检查点配置, 修改学习或任务参数请开始新训练")
    if not args.resume:
        try:
            DQNConfig(**learning_overrides(args))
        except (ValueError, TypeError) as error:
            parser.error(f"学习配置无效: {error}")
    return args


if __name__ == "__main__":
    train(parse_args())
