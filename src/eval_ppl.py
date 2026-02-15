"""Evaluate next-token loss and perplexity on TinyStories."""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src import model as model_module  # noqa: E402
from src import train_tinystories as ts  # noqa: E402


def load_checkpoint(path: str, device: torch.device):
    try:
        from torch.serialization import safe_globals

        with safe_globals([model_module.GPTConfig]):
            checkpoint = torch.load(path, map_location=device, weights_only=False)
    except Exception:
        checkpoint = torch.load(path, map_location=device)
    return checkpoint


def is_quantized_state(state_dict: dict) -> bool:
    for key in state_dict.keys():
        if "_packed_params" in key or "scale" in key or "zero_point" in key:
            return True
    return False


def has_quantized_embedding(state_dict: dict) -> bool:
    return any("embedding._packed_params" in key for key in state_dict.keys())


def build_eval_model(checkpoint: dict, device: torch.device) -> tuple[torch.nn.Module, dict | None]:
    config = checkpoint.get("config")
    if config is None:
        raise ValueError("Checkpoint missing config")

    model = ts.build_model(config).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint missing model_state_dict")

    if is_quantized_state(state_dict):
        qconfig_spec = {nn.Linear: torch.quantization.default_dynamic_qconfig}
        if has_quantized_embedding(state_dict):
            embedding_qconfig = getattr(
                torch.quantization, "float_qparams_weight_only_qconfig", None
            )
            if embedding_qconfig is None:
                raise RuntimeError(
                    "Quantized embeddings detected but this PyTorch build does not support "
                    "float_qparams_weight_only_qconfig. Re-export with --no-quantize-embedding."
                )
            qconfig_spec[nn.Embedding] = embedding_qconfig
        model = torch.quantization.quantize_dynamic(
            model, qconfig_spec=qconfig_spec, dtype=torch.qint8
        )
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    return model, checkpoint.get("tokenizer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPL on TinyStories")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(ROOT_DIR, ckpt_path)

    checkpoint = load_checkpoint(ckpt_path, torch.device("cpu"))
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint missing model_state_dict")
    is_quantized = is_quantized_state(state_dict)
    if is_quantized:
        device = torch.device("cpu")
        if args.device and args.device != "cpu":
            print("Info: quantized models run on CPU; overriding --device to cpu.")
    else:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer_info = build_eval_model(checkpoint, device)
    model.eval()

    encoding_name = "gpt2"
    if isinstance(tokenizer_info, dict):
        encoding_name = tokenizer_info.get("encoding", encoding_name)
    encoding = ts.load_tokenizer(encoding_name)

    split = args.split
    dataset = ts.get_dataset(split, streaming=True, seed=1337)
    dataset_iter = iter(dataset)

    config = checkpoint.get("config")
    block_size = config.block_size if hasattr(config, "block_size") else getattr(config, "block_size", 256)
    batch_size = 8
    vocab_size = encoding.n_vocab

    batch_iter = ts.token_batcher(dataset_iter, encoding, batch_size, block_size, device)
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        while batches < args.max_batches:
            try:
                x, y = next(batch_iter)
            except StopIteration:
                break
            logits = model(x)
            loss = F.cross_entropy(logits.float().view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
            batches += 1

    if batches == 0:
        raise SystemExit("No batches were evaluated.")

    avg_loss = total_loss / batches
    ppl = math.exp(avg_loss)
    print(f"avg_loss={avg_loss:.6f}")
    print(f"ppl={ppl:.6f}")


if __name__ == "__main__":
    main()
