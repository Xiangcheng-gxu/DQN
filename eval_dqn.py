"""评估入口：加载训练好的模型，固定采用最大 Q 值动作，不探索也不学习。

评估不使用 IAS；任何空推都立即判为本回合失败，然后开始下一回合。
评估日志另存为 evaluation.csv，不修改训练模型或 training.csv。
"""
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent/"src"))
import torch

from dqn import DQN
from environment import ClutterPushEnv, EnvironmentConfig
from run_utils import format_step_output, save_latest_images
from run_utils import SuccessStatistics


def evaluate(args):
    """先恢复模型与环境配置，再逐回合验证模型是否能把目标分离。"""
    # 与训练保持一致：小网络在CPU上单线程推理，不检测显卡。
    torch.set_num_threads(1)
    agent, metadata = DQN.load(args.checkpoint)
    config = EnvironmentConfig.from_checkpoint(metadata["environment"]["config"])
    if args.object_count is not None:
        config.object_count = args.object_count
    if getattr(args, "max_steps", None) is not None:
        config.max_steps = args.max_steps
    config.__post_init__()
    # eval() 切换神经网络推理模式；不代表给环境开启 IAS 或更新参数。
    agent.network.eval()
    if args.output:
        output_path = Path(args.output).resolve()
        if output_path.suffix.lower() != ".csv" or output_path == Path(args.checkpoint).resolve().parent/"training.csv":
            raise ValueError("Evaluation output must be a separate CSV, not the training log")
    env = ClutterPushEnv(config, seed=args.seed, viewer=args.viewer, ias_enabled=False)
    success, total_steps, success_steps = 0, 0, 0
    records = []
    success_statistics = SuccessStatistics()
    preview_dir = Path(__file__).resolve().parent/"temp"
    print("\n---------------- 评估开始 ----------------", flush=True)
    print(f"    模型: {args.checkpoint}\n    回合数: {args.episodes} | 障碍物数: {config.object_count} | 每回合最多: {config.max_steps}步", flush=True)
    print("    策略: 仅利用, IAS关闭, 空推即本回合失败, 不学习\n", flush=True)
    try:
        for episode in range(1, args.episodes+1):
            print(f"\n    ------------ Episode {episode} ------------", flush=True)
            state, _ = env.reset()
            save_latest_images(env.observation, preview_dir)
            reward_sum = 0
            for step in range(1, config.max_steps+1):
                action = agent.select_action(state, training=False)
                state, reward, terminal, truncated, info = env.step(action)
                # 只在评估入口提前结束空推，训练环境的空推处理不受影响。
                # 最后一步空推可能被环境标为 step_limit，也要按空推失败统计。
                if (info["reason"] in ("empty_push", "push_point_failed")
                        or (info["reason"] == "step_limit" and reward == -5.)):
                    info = {**info, "success": False, "reason": "empty_push"}
                    terminal, truncated = True, False
                save_latest_images(env.observation, preview_dir)
                print(format_step_output(step, action, reward, info["reason"], 0., agent.last_action_mode), flush=True)
                reward_sum += reward
                if terminal or truncated:
                    break
            success += int(info["success"])
            total_steps += step
            if info["success"]:
                success_steps += step
            success_statistics.record(info["success"], step)
            records.append({"episode": episode, "steps": step, "reward": reward_sum,
                            "success": int(info["success"]), "reason": info["reason"]})
            print(f"\n    回合小结: {step}步 | 总奖励 {reward_sum:+g} | {'成功' if info['success'] else '未成功'} | 累计成功率 {success/episode:.1%}", flush=True)
            print(success_statistics.format(), flush=True)
            print("    ----------------------------------------\n", flush=True)
    finally:
        env.close()
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    efficiency = f"{success_steps/success:.2f}" if success else "n/a"
    print("\n---------------- 评估结束 ----------------", flush=True)
    print(f"    成功: {success}/{args.episodes} | 成功率: {success/args.episodes:.1%}\n"
          f"    平均步数: {total_steps/args.episodes:.2f} | 成功回合平均步数: {efficiency}", flush=True)
    if args.output:
        print(f"    评估记录: {args.output}\n", flush=True)
    return {"success_rate": success/args.episodes, "episodes": args.episodes, "records": records}


def parse_args():
    """评估参数独立于训练。--episodes 是评估次数，不是追加训练次数。"""
    parser = argparse.ArgumentParser(description="DQN模型评估", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 1. 模型: 训练和推理都固定使用CPU
    model = parser.add_argument_group("模型")
    model.add_argument("--checkpoint", "--model", default="results/dqn_2000/last.pt", help="可信模型检查点")
    # 2. 评估任务: 与训练参数独立, 未指定障碍物数时沿用模型配置
    task = parser.add_argument_group("评估任务")
    task.add_argument("--episodes", type=int, default=100, help="评估回合数")
    task.add_argument("--seed", type=int, default=10001, help="评估随机种子")
    task.add_argument("--object-count", type=int, default=5, help="障碍物数量, 未指定则沿用模型配置")
    task.add_argument("--max-steps", type=int, default=8, help="每回合最多步数(1至8)")
    # 3. 结果与可视化
    output = parser.add_argument_group("结果与可视化")
    output.add_argument("--viewer", action="store_true", default=True, help="显示仿真窗口")
    output.add_argument("--output", default="results/evaluation.csv", help="独立评估CSV路径")
    args = parser.parse_args()
    if args.episodes < 1 or not 1 <= args.max_steps <= 8:
        parser.error("评估回合数必须为正数, 步数必须为1至8")
    if args.object_count is not None and args.object_count < 1:
        parser.error("障碍物数量必须至少为1")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
