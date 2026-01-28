"""Train GPT on TinyStories with next-token prediction."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Iterable, Iterator

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src import model as model_module  # noqa: E402


def build_model(config_kwargs):
    gpt_config_cls = getattr(model_module, "GPTConfig", None)
    if gpt_config_cls is None:
        return model_module.GPT(**config_kwargs)
    return model_module.GPT(gpt_config_cls(**config_kwargs))


def build_tokenizer(name: str):
    import tiktoken

    return tiktoken.get_encoding(name)


def load_tinystories(split: str, streaming: bool):
    from datasets import load_dataset

    return load_dataset("roneneldan/TinyStories", split=split, streaming=streaming)


def token_stream(dataset: Iterable[dict], encoding) -> Iterator[list[int]]:
    eot_token = getattr(encoding, "eot_token", None)
    for sample in dataset:
        text = sample.get("text", "")
        if not text:
            continue
        tokens = encoding.encode(text)
        if eot_token is not None:
            tokens.append(eot_token)
        if tokens:
            yield tokens


def batch_iterator(
    token_iter: Iterator[list[int]],
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    buffer: list[int] = []
    target_len = batch_size * (block_size + 1)
    for tokens in token_iter:
        buffer.extend(tokens)
        while len(buffer) >= target_len:
            chunk = buffer[:target_len]
            buffer = buffer[target_len:]
            data = torch.tensor(chunk, dtype=torch.long)
            data = data.view(batch_size, block_size + 1)
            x = data[:, :-1].to(device)
            y = data[:, 1:].to(device)
            yield x, y


def infinite_token_stream(dataset: Iterable[dict], encoding) -> Iterator[list[int]]:
    while True:
        for tokens in token_stream(dataset, encoding):
            yield tokens


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    encoding = build_tokenizer(args.tokenizer)
    vocab_size = encoding.n_vocab

    model = build_model(
        dict(
            vocab_size=vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
        )
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    dataset = load_tinystories(args.split, args.streaming)
    tokens = infinite_token_stream(dataset, encoding)
    batches = batch_iterator(tokens, args.batch_size, args.block_size, device)

    csv_writer = None
    csv_file = None
    if args.log_loss_csv:
        os.makedirs(os.path.dirname(args.log_loss_csv), exist_ok=True)
        csv_file = open(args.log_loss_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["step", "loss", "timestamp"])

    start_time = time.time()
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.grad_accum_steps):
            x, y = next(batches)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            (loss / args.grad_accum_steps).backward()
            total_loss += loss.item()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        avg_loss = total_loss / args.grad_accum_steps
        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"step {step:05d} | loss {avg_loss:.6f} | {elapsed:.1f}s")
        if csv_writer is not None:
            csv_writer.writerow([step, f"{avg_loss:.6f}", f"{time.time():.3f}"])

    if csv_file is not None:
        csv_file.close()

    os.makedirs(os.path.join(ROOT_DIR, "weights"), exist_ok=True)
    ckpt_path = os.path.join(ROOT_DIR, "weights", "pretrain.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": model.config,
            "tokenizer": {"name": args.tokenizer},
            "step": args.max_steps,
        },
        ckpt_path,
    )
    print(f"saved checkpoint to {ckpt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT on TinyStories")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--streaming", action="store_true", help="Use streaming dataset")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.set_defaults(streaming=True)

    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--log-loss-csv", type=str, default="logs/pretrain_loss.csv")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
