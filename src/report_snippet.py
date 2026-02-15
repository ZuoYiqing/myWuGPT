"""Generate report snippets for SFT + MoE analysis."""

from __future__ import annotations

import csv
import json
import os
import sys

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src import model as model_module  # noqa: E402


def load_losses(path: str) -> list[tuple[int, float]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                step = int(row["step"])
                loss = float(row["loss"])
            except (KeyError, ValueError):
                continue
            rows.append((step, loss))
    return rows


def load_moe_stats(path: str) -> list[dict]:
    stats = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            stats.append(json.loads(line))
    return stats


def load_checkpoint(path: str) -> dict:
    try:
        from torch.serialization import safe_globals

        with safe_globals([model_module.GPTConfig]):
            return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return torch.load(path, map_location="cpu")


def count_params(ckpt_path: str) -> int:
    ckpt = load_checkpoint(ckpt_path)
    state = ckpt.get("model_state_dict", {})
    return sum(t.numel() for t in state.values())


def detect_collapse(stats: list[dict], threshold: float = 0.01, ratio: float = 0.8) -> bool:
    if not stats:
        return False
    p_matrix = [row.get("p", []) for row in stats if row.get("p")]
    if not p_matrix:
        return False
    num_steps = len(p_matrix)
    num_experts = len(p_matrix[0])
    for exp_idx in range(num_experts):
        low_count = sum(1 for row in p_matrix if row[exp_idx] <= threshold)
        if low_count / max(num_steps, 1) >= ratio:
            return True
    return False


def main() -> None:
    log_path = os.path.join(ROOT_DIR, "weights", "sft_log.csv")
    moe_path = os.path.join(ROOT_DIR, "weights", "moe_stats.jsonl")
    ckpt_path = os.path.join(ROOT_DIR, "weights", "sft_all_data_continued.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ROOT_DIR, "weights", "sft.pt")

    losses = load_losses(log_path)
    if not losses:
        raise SystemExit(f"No loss data found in {log_path}")

    steps = [s for s, _ in losses]
    loss_vals = [v for _, v in losses]
    first_loss = loss_vals[0]
    last_loss = loss_vals[-1]
    min_loss = min(loss_vals)
    total_steps = steps[-1]

    param_count = count_params(ckpt_path)

    stats = load_moe_stats(moe_path)
    last = stats[-1] if stats else {}
    p = last.get("p", [])
    max_share = last.get("max_share")
    entropy = last.get("entropy")
    e_count = len(p) if p else 0
    collapsed = detect_collapse(stats)

    sft_paragraph = (
        f"SFT：模型参数量约 {param_count:,}；训练步数 {total_steps}。"
        f"Loss 从 {first_loss:.4f} 降到 {last_loss:.4f}，最低 {min_loss:.4f}。"
    )
    moe_paragraph = (
        f"MoE：Top-2 路由，E={e_count}。"
        f"当前 max_share={max_share:.4f}，entropy={entropy:.4f}。"
        f"坍缩判断：{'有坍缩迹象' if collapsed else '未见明显坍缩'}。"
    )

    print(sft_paragraph)
    print(moe_paragraph)


if __name__ == "__main__":
    main()
