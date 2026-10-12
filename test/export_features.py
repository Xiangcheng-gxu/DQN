"""Export one real clutter observation for visual inspection."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from camera import _write_png
from environment import ClutterPushEnv


def export(seed: int, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    env = ClutterPushEnv(seed=seed)
    try:
        env.reset()
        obs = env.observation
        assert obs.heightmap.shape == (250, 250)
        np.save(output / "heightmap_250.npy", obs.heightmap)
        np.save(output / "state_262.npy", obs.state)
        _write_png(output / "heightmap_250.png",
                   np.rint(np.clip(obs.heightmap / .04, 0, 1) * 255).astype(np.uint8))
        _write_png(output / "target_mask_250.png", obs.target_mask.astype(np.uint8) * 255)
        metadata = {
            "seed": seed,
            "shape": list(obs.heightmap.shape),
            "height_unit": "metres above workspace table (world z = 0.754 m)",
            "pixel_resolution_m": .002,
            "preview": "black = 0 m, white = 0.04 m; no resizing or flipping",
            "axes": "columns increase with world X; rows increase with world Y",
            "target_mask": "white = target, black = other",
            "network_state_dimension": int(obs.state.size),
            "robot_visible_in_camera": False,
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved to {output.resolve()}")
        print(f"heightmap: {obs.heightmap.shape}, {obs.heightmap.dtype}, "
              f"range {obs.heightmap.min():.6f}..{obs.heightmap.max():.6f} m")
        print(f"target pixels: {np.count_nonzero(obs.target_mask)}; state: {obs.state.shape}")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=ROOT / "temp" / "features_250_seed7")
    args = parser.parse_args()
    export(args.seed, args.output)
