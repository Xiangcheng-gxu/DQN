"""Train or resume a single DQN in the current MuJoCo clutter scene."""
import argparse
from collections import deque
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
from camera import _write_png


def step_result_text(reason, reward):
    labels = {
        "separated": "成功分离", "pushed": "有效推动", "empty_push": "空推",
        "out_of_workspace": "物体越界", "target_lost": "目标丢失",
        "motion_failed": "动作失败", "push_point_failed": "推点获取失败",
    }
    if reason == "step_limit":
        return ("空推" if reward == -5 else "有效推动") + "，达到步数上限"
    return labels.get(reason, "未知结果")


def save_latest_images(observation, directory):
    """Overwrite camera-derived previews without exposing half-written PNGs."""
    directory.mkdir(parents=True, exist_ok=True)
    images = {
        "heightmap_250.png": np.rint(np.clip(observation.heightmap / .04, 0, 1) * 255).astype(np.uint8),
        "target_mask_250.png": observation.target_mask.astype(np.uint8) * 255,
    }
    if observation.rgb is not None:
        images["rgb_original.png"] = observation.rgb
    if observation.workspace_crop_rgb is not None:
        images["workspace_crop.png"] = observation.workspace_crop_rgb
    if observation.workspace_rgb is not None:
        images["workspace_rgb_250.png"] = observation.workspace_rgb
    for name, image in images.items():
        destination = directory / name
        temporary = directory / (name + ".tmp")
        _write_png(temporary, image)
        temporary.replace(destination)


def device_name(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --device cpu or install CUDA PyTorch")
    return requested


def train(args):
    torch.set_num_threads(1)
    device = device_name(args.device)
    settings = json.loads(Path(args.config).read_text(encoding="utf-8"))
    history = deque(maxlen=100)
    start_episode = 0
    if args.resume:
        agent, metadata = DQN.load(args.resume, device=device)
        config = EnvironmentConfig.from_checkpoint(metadata["environment"]["config"])
        if args.push_speed is not None:
            config.push_speed_m_s = args.push_speed
        if args.descent_speed is not None:
            config.descent_speed_m_s = args.descent_speed
        config.__post_init__()
        env = ClutterPushEnv(config, viewer=args.viewer)
        env.rng.bit_generator.state = metadata["environment"]["rng"]
        history.extend(metadata.get("recent_success", []))
        start_episode = metadata["episode"]
    else:
        if args.object_count is not None:
            settings["environment"]["object_count"] = args.object_count
        if args.max_steps is not None:
            settings["environment"]["max_steps"] = args.max_steps
        if args.push_speed is not None:
            settings["environment"]["push_speed_m_s"] = args.push_speed
        if args.descent_speed is not None:
            settings["environment"]["descent_speed_m_s"] = args.descent_speed
        if args.warmup is not None:
            settings["agent"]["init_replay_buffer_size"] = args.warmup
        if args.batch_size is not None:
            settings["agent"]["batch_size"] = args.batch_size
        agent = DQN(DQNConfig(**settings["agent"]), device=device, seed=args.seed)
        env = ClutterPushEnv(EnvironmentConfig(**settings["environment"]), seed=args.seed, viewer=args.viewer)
    output = Path(args.output) if args.output else (Path(args.resume).parent if args.resume else Path("results/dqn"))
    output.mkdir(parents=True, exist_ok=True)
    # Do not accidentally overwrite an existing run.
    csv_path = output / "training.csv"
    if not args.resume and (csv_path.exists() or (output/"last.pt").exists()):
        raise FileExistsError(f"Run already exists: {output}; use --resume or another --output")
    (output/"config.json").write_text(json.dumps({"agent": vars(agent.config), "environment": vars(env.config)}, indent=2), encoding="utf-8")
    print(f"开始训练：{args.episodes}回合，每回合最多{env.config.max_steps}步，设备={device}", flush=True)
    preview_dir = Path(__file__).resolve().parent / "temp"
    print(f"图片实时覆盖：{preview_dir}", flush=True)
    fieldnames = ["episode", "steps", "reward", "success", "reason", "epsilon", "loss", "buffer", "updates", "seconds"]
    last_completed = start_episode
    try:
        with csv_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            if stream.tell() == 0:
                writer.writeheader()
            for episode in range(start_episode+1, start_episode+args.episodes+1):
                started = time.perf_counter()
                print(f"—— 第{episode}回合 ——", flush=True)
                state, reset_info = env.reset()
                save_latest_images(env.observation, preview_dir)
                total_reward, losses = 0., []
                for step in range(1, env.config.max_steps+1):
                    action = agent.select_action(state)
                    next_state, reward, terminal, truncated, info = env.step(action)
                    save_latest_images(env.observation, preview_dir)
                    loss = agent.learn(Transition(state, action, reward, next_state, terminal))
                    if loss is not None:
                        losses.append(loss)
                    total_reward += reward
                    print(f"回合{episode} | 第{step}步 | 动作a{action} | 奖励{reward:+g} | "
                          f"{step_result_text(info['reason'], reward)}", flush=True)
                    if args.verbose:
                        print(f"  调试：最大位移{info['max_displacement_m']*1000:.1f}毫米，"
                              f"姿态误差{info['orientation_error_rad']:.4f}弧度", flush=True)
                        if info["motion_error"]:
                            print(f"  错误详情：{info['motion_error']}", flush=True)
                    state = next_state
                    if terminal or truncated:
                        break
                history.append(int(info["success"]))
                last_completed = episode
                mean_loss = float(np.mean(losses)) if losses else None
                row = dict(episode=episode, steps=step, reward=total_reward, success=int(info["success"]),
                           reason=info["reason"], epsilon=agent.epsilon, loss=mean_loss, buffer=len(agent.replay),
                           updates=agent.learn_steps, seconds=round(time.perf_counter()-started, 2))
                writer.writerow(row)
                stream.flush()
                print(f"回合{episode}结束 | 共{step}步 | 总奖励{total_reward:+g} | "
                      f"{'成功' if info['success'] else '未成功'}", flush=True)
                if args.verbose:
                    text_loss = "预热中" if mean_loss is None else f"{mean_loss:.4f}"
                    print(f"  调试：近100回合成功率{np.mean(history):.0%}，损失={text_loss}，"
                          f"经验数{len(agent.replay)}，更新{agent.learn_steps}次", flush=True)
                if episode % args.save_every == 0:
                    meta = {"episode": episode, "environment": env.state_dict(), "recent_success": list(history)}
                    agent.save(output/f"episode_{episode:06d}.pt", metadata=meta)
                    agent.save(output/"last.pt", metadata=meta)
    except KeyboardInterrupt:
        print("训练已中断，正在保存模型和经验。", flush=True)
    finally:
        agent.save(output/"last.pt", metadata={"episode": last_completed, "environment": env.state_dict(),
                                              "recent_success": list(history)})
        env.close()
    print(f"模型已保存：{output/'last.pt'}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent/"configs"/"dqn.json"))
    parser.add_argument("--episodes", type=int, default=10000, help="additional episodes, also on resume")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", help="default: results/dqn; on resume: checkpoint directory")
    parser.add_argument("--resume", help="trusted full checkpoint")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--object-count", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--push-speed", type=float, help="horizontal m/s; allowed on resume")
    parser.add_argument("--descent-speed", type=float, help="descent m/s; allowed on resume")
    parser.add_argument("--warmup", type=int, help="override for short smoke runs")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="额外显示位移、误差和学习统计")
    args = parser.parse_args()
    if args.episodes < 1 or args.save_every < 1:
        parser.error("episodes and save-every must be positive")
    if any(x is not None and (not np.isfinite(x) or x <= 0) for x in (args.push_speed, args.descent_speed)):
        parser.error("motion speeds must be finite and positive")
    if args.resume and any(x is not None for x in (args.object_count, args.max_steps, args.warmup, args.batch_size)):
        parser.error("resume uses checkpoint configuration; only motion-speed overrides are allowed")
    return args


if __name__ == "__main__":
    train(parse_args())
