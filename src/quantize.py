"""Dynamic quantization for GPT checkpoints."""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from torch import nn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src import model as model_module  # noqa: E402


def build_model(config):
    gpt_config_cls = getattr(model_module, "GPTConfig", None)
    if gpt_config_cls is not None and not isinstance(config, gpt_config_cls):
        config = gpt_config_cls(**config)
    return model_module.GPT(config)


def load_checkpoint(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config")
    if config is None:
        raise ValueError("Checkpoint missing config")
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


def encode_prompt(prompt: str) -> list[int]:
    return list(prompt.encode("utf-8"))


def decode_tokens(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8", errors="replace")


def generate(model, idx, max_new_tokens):
    model.eval()
    block_size = model.config.block_size
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
    return idx


def human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description="Quantize GPT checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(ROOT_DIR, "weights", "best_sft_model.pth"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(ROOT_DIR, "weights", "quantized_model.pt"),
    )
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    args = parser.parse_args()

    device = torch.device("cpu")
    model = load_checkpoint(args.checkpoint, device)

    original_size = os.path.getsize(args.checkpoint)

    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({"model_state_dict": quantized_model.state_dict(), "config": model.config}, args.output)
    quantized_size = os.path.getsize(args.output)

    print(
        "file size | original: "
        f"{human_size(original_size)} | quantized: {human_size(quantized_size)}"
    )

    prompt_ids = encode_prompt(args.prompt)
    if not prompt_ids:
        prompt_ids = [0]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    start = time.perf_counter()
    out = generate(quantized_model, idx, max_new_tokens=32)
    elapsed = time.perf_counter() - start
    text = decode_tokens(out[0].tolist())
    print("sample:", text)
    print(f"latency: {elapsed:.4f}s (CPU)")


if __name__ == "__main__":
    main()
