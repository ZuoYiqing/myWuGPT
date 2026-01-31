"""Pretrain GPT on TinyStories with tiktoken tokenization."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

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


def load_tokenizer(encoding_name: str):
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "tiktoken is required. Install with `pip install tiktoken`."
        ) from exc

    encoding = tiktoken.get_encoding(encoding_name)
    return encoding


def get_dataset(split: str, streaming: bool, seed: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required. Install with `pip install datasets`."
        ) from exc

    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=streaming)
    if streaming:
        dataset = dataset.shuffle(buffer_size=10_000, seed=seed)
    else:
        dataset = dataset.shuffle(seed=seed)
    return dataset


def token_batcher(dataset_iter, encoding, batch_size, block_size, device):
    buffer: list[int] = []
    tokens_per_batch = batch_size * (block_size + 1)
    eot_token = encoding.eot_token

    while True:
        while len(buffer) < tokens_per_batch:
            try:
                sample = next(dataset_iter)
            except StopIteration:
                return
            text = sample.get("text", "")
            if text:
                buffer.extend(encoding.encode(text))
            buffer.append(eot_token)

        batch_tokens = buffer[:tokens_per_batch]
        buffer = buffer[tokens_per_batch:]
        batch = torch.tensor(batch_tokens, dtype=torch.long).view(batch_size, block_size + 1)
        x = batch[:, :-1].contiguous().to(device)
        y = batch[:, 1:].contiguous().to(device)
        yield x, y


def main():
    parser = argparse.ArgumentParser(description="Train GPT on TinyStories")
    parser.add_argument("--encoding", type=str, default="gpt2")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--streaming", action="store_true", help="Use streaming dataset mode.")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.set_defaults(streaming=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--log-loss-csv", type=str, default="logs/pretrain_loss.csv")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    encoding = load_tokenizer(args.encoding)
    dataset = get_dataset(args.split, args.streaming, args.seed)

    if args.download_only:
        print("Dataset downloaded and cached.")
        return

    vocab_size = encoding.n_vocab
    config = dict(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = build_model(config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    loss_writer = None
    loss_file = None
    if args.log_loss_csv:
        loss_path = args.log_loss_csv
        if not os.path.isabs(loss_path):
            loss_path = os.path.join(ROOT_DIR, loss_path)
        os.makedirs(os.path.dirname(loss_path), exist_ok=True)
        file_exists = os.path.exists(loss_path)
        loss_file = open(loss_path, "a", newline="", encoding="utf-8")
        loss_writer = csv.writer(loss_file)
        if not file_exists:
            loss_writer.writerow(["step", "loss", "lr"])

    dataset_iter = iter(dataset)
    batch_iter = token_batcher(dataset_iter, encoding, args.batch_size, args.block_size, device)
    max_iters = args.max_steps if args.max_iters is None else args.max_iters
    max_iters = max(max_iters, 1)
    warmup_iters = max(args.warmup_iters, 0)

    def get_lr(step_idx: int) -> float:
        if warmup_iters > 0 and step_idx < warmup_iters:
            return args.learning_rate * (step_idx + 1) / warmup_iters
        if max_iters <= warmup_iters:
            progress = 1.0
        else:
            progress = (step_idx - warmup_iters) / (max_iters - warmup_iters)
        progress = min(max(progress, 0.0), 1.0)
        return args.min_lr + 0.5 * (args.learning_rate - args.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    for step in range(1, args.max_steps + 1):
        lr = get_lr(step - 1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                x, y = next(batch_iter)
            except StopIteration:
                dataset_iter = iter(dataset)
                batch_iter = token_batcher(
                    dataset_iter, encoding, args.batch_size, args.block_size, device
                )
                x, y = next(batch_iter)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            (loss / args.grad_accum).backward()
            step_loss += loss.item()

        optimizer.step()

        avg_loss = step_loss / args.grad_accum
        if step % args.log_interval == 0:
            print(f"step {step:05d} | loss {avg_loss:.6f} | lr {lr:.6e}")
        if loss_writer is not None:
            loss_writer.writerow([step, f"{avg_loss:.6f}", f"{lr:.6e}"])

    if loss_file is not None:
        loss_file.close()

    os.makedirs(os.path.join(ROOT_DIR, "weights"), exist_ok=True)
    ckpt_path = os.path.join(ROOT_DIR, "weights", "pretrain.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": model.config if hasattr(model, "config") else config,
            "tokenizer": {
                "encoding": args.encoding,
                "vocab_size": vocab_size,
                "eot_token": encoding.eot_token,
            },
            "train_args": vars(args),
        },
        ckpt_path,
    )
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
