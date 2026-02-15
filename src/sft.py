"""Supervised fine-tuning (SFT) on instruction JSONL data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from typing import Iterable, Iterator

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src import model as model_module  # noqa: E402


def build_model(config):
    gpt_config_cls = getattr(model_module, "GPTConfig", None)
    if gpt_config_cls is not None and not isinstance(config, gpt_config_cls):
        config = gpt_config_cls(**config)
    return model_module.GPT(config)


def load_tokenizer(tokenizer_info: dict | None):
    if tokenizer_info is None:
        raise RuntimeError("Tokenizer info missing in checkpoint.")
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "tiktoken is required. Install with `pip install tiktoken`."
        ) from exc
    encoding_name = tokenizer_info.get("name") or tokenizer_info.get("encoding", "gpt2")
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding


def load_checkpoint(path: str, device: torch.device):
    # Use safe_globals to allowlist local classes (e.g., GPTConfig) when loading
    try:
        from torch.serialization import safe_globals
        with safe_globals([model_module.GPTConfig]):
            checkpoint = torch.load(path, map_location=device, weights_only=False)
    except Exception:
        checkpoint = torch.load(path, map_location=device)

    config = checkpoint.get("config")
    if config is None:
        raise ValueError("Checkpoint missing config")
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    tokenizer_info = checkpoint.get("tokenizer")
    return model, tokenizer_info


def normalize_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def load_jsonl_file(path: str) -> tuple[list[dict], int]:
    data: list[dict] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(example, dict):
                skipped += 1
                continue
            if not all(key in example for key in ("instruction", "input", "output")):
                skipped += 1
                continue
            data.append(example)
    return data, skipped


def load_json_file(path: str) -> tuple[list[dict], int]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError(f"JSON root must be list or dict: {path}")
    data = []
    skipped = 0
    for example in payload:
        if not isinstance(example, dict):
            skipped += 1
            continue
        if not all(key in example for key in ("instruction", "input", "output")):
            skipped += 1
            continue
        data.append(example)
    return data, skipped


def collect_data_files(path: str) -> list[str]:
    if os.path.isdir(path):
        files = []
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if not os.path.isfile(full_path):
                continue
            ext = os.path.splitext(entry)[1].lower()
            if ext in {".jsonl", ".json"}:
                files.append(full_path)
        return sorted(files)
    return [path]


def load_data(path: str) -> list[dict]:
    files = collect_data_files(path)
    if not files:
        raise ValueError(f"No JSON/JSONL data files found at {path}")

    data: list[dict] = []
    total_skipped = 0
    for file_path in files:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".jsonl":
            batch, skipped = load_jsonl_file(file_path)
        elif ext == ".json":
            batch, skipped = load_json_file(file_path)
        else:
            continue
        if skipped:
            print(f"Warning: skipped {skipped} invalid records in {file_path}")
        data.extend(batch)
        total_skipped += skipped

    if not data:
        raise ValueError("No valid training examples loaded.")

    # Deduplicate by normalized (instruction, input, output).
    deduped: list[dict] = []
    seen = set()
    for example in data:
        key = (
            normalize_value(example.get("instruction")).strip(),
            normalize_value(example.get("input")).strip(),
            normalize_value(example.get("output")).strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)

    print(f"Loaded {len(files)} data files")
    print(f"Examples: {len(data)} (deduped to {len(deduped)})")
    if total_skipped:
        print(f"Total skipped invalid records: {total_skipped}")
    return deduped


def format_prompt(example: dict) -> tuple[str, str]:
    instruction = normalize_value(example.get("instruction")).strip()
    user_input = normalize_value(example.get("input")).strip()
    output = normalize_value(example.get("output")).strip()

    if user_input:
        prompt = (
            "### User\n"
            f"Instruction: {instruction}\n"
            f"Input: {user_input}\n"
            "\n### Assistant\n"
        )
    else:
        prompt = (
            "### User\n"
            f"Instruction: {instruction}\n"
            "\n### Assistant\n"
        )
    return prompt, output


def build_batch(
    batch: list[dict],
    encoding,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids_list = []
    labels_list = []
    eot_token = encoding.eot_token if encoding.eot_token is not None else 0
    for example in batch:
        prompt, response = format_prompt(example)
        prompt_ids = encoding.encode(prompt)
        response_ids = encoding.encode(response)
        full_ids = prompt_ids + response_ids
        if len(full_ids) < 2:
            continue
        labels = [-100] * len(prompt_ids) + response_ids
        full_ids = full_ids[: block_size + 1]
        labels = labels[: block_size + 1]
        if all(label == -100 for label in labels[1:]):
            continue
        if len(full_ids) < block_size + 1:
            pad_len = block_size + 1 - len(full_ids)
            full_ids.extend([eot_token] * pad_len)
            labels.extend([-100] * pad_len)
        input_ids_list.append(full_ids[:-1])
        labels_list.append(labels[1:])

    if not input_ids_list:
        raise ValueError("Batch has no valid examples after tokenization.")

    x = torch.tensor(input_ids_list, dtype=torch.long, device=device)
    y = torch.tensor(labels_list, dtype=torch.long, device=device)
    return x, y


def batch_iterator(data: list[dict], batch_size: int) -> list[list[dict]]:
    return [data[idx : idx + batch_size] for idx in range(0, len(data), batch_size)]


def batch_stream(data: list[dict], batch_size: int, seed: int) -> Iterator[list[dict]]:
    rng = random.Random(seed)
    data = list(data)
    while True:
        rng.shuffle(data)
        for idx in range(0, len(data), batch_size):
            yield data[idx : idx + batch_size]


def split_data(data: list[dict], seed: int, eval_ratio: float = 0.1) -> tuple[list[dict], list[dict]]:
    if len(data) < 2:
        return data, []
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    eval_size = max(1, int(len(data) * eval_ratio))
    eval_size = min(eval_size, len(data) - 1)
    eval_indices = set(indices[:eval_size])
    eval_data = [data[i] for i in indices if i in eval_indices]
    train_data = [data[i] for i in indices if i not in eval_indices]
    return train_data, eval_data


def evaluate(
    model: torch.nn.Module,
    data: list[dict],
    encoding,
    block_size: int,
    device: torch.device,
    batch_size: int,
) -> float | None:
    if not data:
        return None
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in batch_iterator(data, batch_size):
            try:
                x, y = build_batch(batch, encoding, block_size, device)
            except ValueError:
                continue
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-100,
            )
            total_loss += loss.item()
            steps += 1
    model.train()
    if steps == 0:
        return None
    return total_loss / steps


def compute_grad_norm(parameters) -> float:
    total_sq = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        param_norm = param.grad.data.norm(2).item()
        total_sq += param_norm * param_norm
    return total_sq**0.5


def collect_moe_stats(model) -> dict | None:
    counts = None
    for block in getattr(model, "blocks", []):
        mlp = getattr(block, "mlp", None)
        stats = getattr(mlp, "last_router_stats", None)
        if not stats or "token_counts" not in stats:
            continue
        block_counts = stats["token_counts"].detach().to("cpu")
        if counts is None:
            counts = block_counts.clone().to(torch.float64)
        else:
            counts = counts + block_counts
    if counts is None:
        return None
    total_selected = counts.sum().item()
    if total_selected <= 0:
        return None
    p = (counts / total_selected).tolist()
    max_share = max(p)
    entropy = -sum(pi * math.log(pi + 1e-12) for pi in p)
    return {
        "counts": [int(v) for v in counts.tolist()],
        "p": [float(v) for v in p],
        "max_share": float(max_share),
        "entropy": float(entropy),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT on instruction JSONL")
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(ROOT_DIR, "data"),
        help="Path to instruction JSON/JSONL or a directory of data files",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(ROOT_DIR, "weights", "pretrain.pt"),
        help="Path to pretrain checkpoint",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(ROOT_DIR, "weights", "sft.pt"),
        help="Output checkpoint path",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument(
        "--log-csv",
        type=str,
        default=os.path.join(ROOT_DIR, "weights", "sft_log.csv"),
        help="CSV path for training logs",
    )
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--moe-stats-jsonl",
        type=str,
        default=os.path.join(ROOT_DIR, "weights", "moe_stats.jsonl"),
        help="JSONL path for MoE router stats",
    )
    parser.add_argument("--moe-log-interval", type=int, default=50)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume SFT from a previous SFT checkpoint")
    parser.add_argument("--resume-ckpt", type=str, default=None, help="Path to SFT checkpoint to resume (defaults to --out)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer_info = load_checkpoint(args.checkpoint, device)
    if tokenizer_info is None:
        raise RuntimeError("Tokenizer info missing in pretrain checkpoint.")
    encoding = load_tokenizer(tokenizer_info)
    model.train()

    data = load_data(args.data)
    train_data, eval_data = split_data(data, args.seed, eval_ratio=0.1)
    train_batches = batch_stream(train_data, args.batch_size, args.seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    log_interval = max(args.log_interval, 1)
    log_path = args.log_csv
    if not os.path.isabs(log_path):
        log_path = os.path.join(ROOT_DIR, log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "loss", "lr", "grad_norm", "tokens"])
    moe_log_interval = max(args.moe_log_interval, 1)
    moe_path = args.moe_stats_jsonl
    if not os.path.isabs(moe_path):
        moe_path = os.path.join(ROOT_DIR, moe_path)
    os.makedirs(os.path.dirname(moe_path), exist_ok=True)
    moe_file = open(moe_path, "a", encoding="utf-8")

    # Resume support
    start_step = 0
    if args.resume:
        resume_path = args.resume_ckpt if args.resume_ckpt is not None else args.out
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(ROOT_DIR, resume_path)
        if os.path.exists(resume_path):
            try:
                from torch.serialization import safe_globals
                with safe_globals([model_module.GPTConfig]):
                    ck = torch.load(resume_path, map_location=device, weights_only=False)
            except Exception:
                ck = torch.load(resume_path, map_location=device)
            model.load_state_dict(ck["model_state_dict"])
            if "optimizer_state_dict" in ck:
                try:
                    optimizer.load_state_dict(ck["optimizer_state_dict"])
                except Exception as e:
                    print(f"Warning: failed to load optimizer state: {e}")
            start_step = ck.get("step", 0) or 0
            print(f"Resumed SFT from checkpoint {resume_path} at step {start_step}")
        else:
            print(f"Warning: resume requested but checkpoint not found at {resume_path}. Starting from scratch.")

    # iterate from the next step after start_step up to max_steps
    for step in range(start_step + 1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        micro_steps = 0
        tokens_in_step = 0
        for _ in range(args.grad_accum):
            batch = next(train_batches)
            try:
                x, y = build_batch(batch, encoding, args.block_size, device)
            except ValueError:
                continue
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-100,
            )
            (loss / args.grad_accum).backward()
            accum_loss += loss.item()
            micro_steps += 1
            tokens_in_step += int((y != -100).sum().item())

        if micro_steps == 0:
            continue

        grad_norm = compute_grad_norm(model.parameters())
        lr = optimizer.param_groups[0].get("lr", args.learning_rate)
        optimizer.step()
        step_loss = accum_loss / micro_steps
        if step % log_interval == 0:
            print(f"step {step:05d} | loss {step_loss:.6f}")
            log_writer.writerow(
                [step, f"{step_loss:.6f}", f"{lr:.8f}", f"{grad_norm:.6f}", tokens_in_step]
            )
            log_file.flush()
        if step % moe_log_interval == 0:
            moe_stats = collect_moe_stats(model)
            if moe_stats is not None:
                moe_stats["step"] = step
                moe_file.write(json.dumps(moe_stats, ensure_ascii=False) + "\n")
                moe_file.flush()

        if args.eval_interval > 0 and step % args.eval_interval == 0:
            eval_loss = evaluate(
                model,
                eval_data,
                encoding,
                args.block_size,
                device,
                args.batch_size,
            )
            if eval_loss is None:
                print(f"eval step {step:05d} | loss None")
            else:
                print(f"eval step {step:05d} | loss {eval_loss:.6f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tokenizer_payload = dict(tokenizer_info)
    if "name" not in tokenizer_payload:
        tokenizer_payload["name"] = tokenizer_payload.get("encoding", "gpt2")
    final_step = globals().get('step', None)
    if final_step is None:
        final_step = locals().get('step', None)
    if final_step is None:
        final_step = start_step if 'start_step' in locals() else 0

    log_file.flush()
    log_file.close()
    moe_file.flush()
    moe_file.close()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": model.config if hasattr(model, "config") else None,
            "tokenizer": tokenizer_payload,
            "train_args": vars(args),
            "step": final_step,
        },
        args.out,
    )
    print(f"saved checkpoint to {args.out} (step {final_step})")


if __name__ == "__main__":
    main()
