"""Plot MoE expert stats from JSONL."""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_stats(path: str) -> list[dict]:
    stats = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            stats.append(json.loads(line))
    return stats


def main() -> None:
    path = os.path.join(ROOT_DIR, "weights", "moe_stats.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"MoE stats not found: {path}")

    stats = load_stats(path)
    if not stats:
        raise SystemExit("No MoE stats to plot.")

    last = stats[-1]
    p_last = last.get("p", [])
    if not p_last:
        raise SystemExit("Latest MoE stats missing 'p'.")

    out_dir = os.path.join(ROOT_DIR, "assets")
    os.makedirs(out_dir, exist_ok=True)

    # Histogram (last step)
    plt.figure(figsize=(6, 4))
    x = np.arange(len(p_last))
    plt.bar(x, p_last)
    plt.xticks(x, [f"E{i}" for i in range(len(p_last))])
    plt.xlabel("Expert")
    plt.ylabel("Share")
    plt.title("MoE Expert Share (Last Step)")
    plt.tight_layout()
    hist_path = os.path.join(out_dir, "moe_expert_hist.png")
    plt.savefig(hist_path, dpi=150)

    # Heatmap over steps
    p_matrix = np.array([row.get("p", []) for row in stats], dtype=float)
    plt.figure(figsize=(8, 4))
    plt.imshow(p_matrix, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Share")
    plt.xlabel("Expert")
    plt.ylabel("Step index")
    plt.title("MoE Expert Share Heatmap")
    plt.tight_layout()
    heat_path = os.path.join(out_dir, "moe_expert_heatmap.png")
    plt.savefig(heat_path, dpi=150)

    print(f"Saved {hist_path}")
    print(f"Saved {heat_path}")


if __name__ == "__main__":
    main()
