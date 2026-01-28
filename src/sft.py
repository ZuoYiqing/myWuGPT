"""Supervised fine-tuning (SFT) on Alpaca-style data."""

from __future__ import annotations

import argparse
import json
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


def load_tokenizer(tokenizer_info: dict | None):
    if tokenizer_info is None:
        raise RuntimeError("Tokenizer info missing in checkpoint.")
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "tiktoken is required. Install with `pip install tiktoken`."
        ) from exc
    encoding_name = tokenizer_info.get("encoding", "gpt2")
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding


def load_checkpoint(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config")
    if config is None:
        raise ValueError("Checkpoint missing config")
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    tokenizer_info = checkpoint.get("tokenizer")
    return model, tokenizer_info


def load_alpaca_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Expected a list of Alpaca examples.")
    return data


def format_prompt(example: dict) -> tuple[str, str]:
    instruction = (example.get("instruction") or "").strip()
    user_input = (example.get("input") or "").strip()
    output = (example.get("output") or "").strip()

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
        if len(full_ids) < block_size + 1:
            pad_len = block_size + 1 - len(full_ids)
            full_ids.extend([encoding.eot_token] * pad_len)
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


def main():
    parser = argparse.ArgumentParser(description="SFT on Alpaca-style JSON")
    parser.add_argument("--data", type=str, required=True, help="Path to Alpaca JSON")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(ROOT_DIR, "weights", "pretrain.pt"),
        help="Path to pretrain checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer_info = load_checkpoint(args.checkpoint, device)
    encoding = load_tokenizer(tokenizer_info)
    model.train()

    data = load_alpaca_json(args.data)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_loss = None
    global_step = 0

    for epoch in range(1, args.max_epochs + 1):
        torch.random.manual_seed(args.seed + epoch)
        perm = torch.randperm(len(data)).tolist()
        shuffled = [data[i] for i in perm]
        batches = batch_iterator(shuffled, args.batch_size)
        epoch_loss = 0.0
        step_in_epoch = 0
        batch_idx = 0

        while batch_idx < len(batches):
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(args.grad_accum):
                if batch_idx >= len(batches):
                    break
                batch = batches[batch_idx]
                batch_idx += 1
                if not batch:
                    continue
                x, y = build_batch(batch, encoding, args.block_size, device)
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-100,
                )
                (loss / args.grad_accum).backward()
                accum_loss += loss.item()
            optimizer.step()
            global_step += 1
            step_in_epoch += 1
            step_loss = accum_loss / max(args.grad_accum, 1)
            epoch_loss += step_loss
            print(f"epoch {epoch} | step {global_step:05d} | loss {step_loss:.6f}")

        avg_epoch_loss = epoch_loss / max(step_in_epoch, 1)
        if best_loss is None or avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            os.makedirs(os.path.join(ROOT_DIR, "weights"), exist_ok=True)
            save_path = os.path.join(ROOT_DIR, "weights", "best_sft_model.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": model.config if hasattr(model, "config") else None,
                    "tokenizer": tokenizer_info,
                    "epoch": epoch,
                    "loss": avg_epoch_loss,
                },
                save_path,
            )
            print(f"saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
