"""Plot SFT loss curve from a CSV log."""

from __future__ import annotations

import csv
import os

import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    csv_path = os.path.join(ROOT_DIR, "weights", "sft_log.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"Log CSV not found: {csv_path}")

    steps = []
    losses = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))
            except (KeyError, ValueError):
                continue

    if not steps:
        raise SystemExit("No data in log CSV.")

    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, linewidth=1.5)
    plt.title("SFT Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    out_dir = os.path.join(ROOT_DIR, "assets")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sft_loss.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
