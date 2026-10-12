"""Greedy evaluation of a saved DQN without exploration or learning."""
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent/"src"))
import torch

from dqn import DQN
from environment import ClutterPushEnv, EnvironmentConfig
from train_dqn import device_name


def evaluate(args):
    torch.set_num_threads(1)
    agent, metadata = DQN.load(args.checkpoint, device=device_name(args.device))
    config = EnvironmentConfig.from_checkpoint(metadata["environment"]["config"])
    if args.object_count is not None:
        config.object_count = args.object_count
    if args.push_speed is not None:
        config.push_speed_m_s = args.push_speed
    if args.descent_speed is not None:
        config.descent_speed_m_s = args.descent_speed
    config.__post_init__()
    env = ClutterPushEnv(config, seed=args.seed, viewer=args.viewer)
    success, total_steps, success_steps = 0, 0, 0
    records = []
    print(f"EVAL | episodes={args.episodes} | obstacles={config.object_count} | greedy | checkpoint={args.checkpoint}", flush=True)
    print(f"speed | descend={config.descent_speed_m_s:.2f}m/s push={config.push_speed_m_s:.2f}m/s", flush=True)
    try:
        for episode in range(1, args.episodes+1):
            state, _ = env.reset()
            reward_sum = 0
            for step in range(1, config.max_steps+1):
                action = agent.select_action(state, training=False)
                state, reward, terminal, truncated, info = env.step(action)
                reward_sum += reward
                if terminal or truncated:
                    break
            success += int(info["success"])
            total_steps += step
            if info["success"]:
                success_steps += step
            records.append({"episode": episode, "steps": step, "reward": reward_sum,
                            "success": int(info["success"]), "reason": info["reason"]})
            print(f"ep={episode} steps={step} reward={reward_sum:+.0f} {info['reason']} success={success/episode:.1%}", flush=True)
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
    print(f"result | success={success}/{args.episodes} ({success/args.episodes:.1%}) "
          f"mean_steps={total_steps/args.episodes:.2f} success_steps={efficiency}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="results/dqn/last.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--object-count", type=int)
    parser.add_argument("--push-speed", type=float, help="override horizontal speed, m/s")
    parser.add_argument("--descent-speed", type=float, help="override descent speed, m/s")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--output", help="optional evaluation CSV")
    arguments = parser.parse_args()
    if arguments.episodes < 1:
        parser.error("episodes must be positive")
    evaluate(arguments)
