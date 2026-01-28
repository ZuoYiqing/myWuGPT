"""Minimal inference script for GPT text generation."""

import argparse
import os
import sys

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src import model as model_module  # noqa: E402


def build_model(config):
    gpt_config_cls = getattr(model_module, "GPTConfig", None)
    if gpt_config_cls is not None and not isinstance(config, gpt_config_cls):
        config = gpt_config_cls(**config)
    return model_module.GPT(config)


def get_tokenizer(tokenizer_info: dict | None):
    if not tokenizer_info:
        return (
            lambda prompt: list(prompt.encode("utf-8")),
            lambda tokens: bytes(tokens).decode("utf-8", errors="replace"),
        )
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "tiktoken is required to use this checkpoint. Install with `pip install tiktoken`."
        ) from exc
    encoding_name = tokenizer_info.get("encoding", "gpt2")
    encoding = tiktoken.get_encoding(encoding_name)
    return (
        lambda prompt: encoding.encode(prompt),
        lambda tokens: encoding.decode(tokens),
    )


def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()
    block_size = model.config.block_size
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                top_k = max(top_k, 1)
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
    return idx


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config")
    if config is None:
        raise ValueError("Checkpoint missing config")
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    tokenizer_info = checkpoint.get("tokenizer")
    return model, tokenizer_info


def main():
    parser = argparse.ArgumentParser(description="Minimal GPT inference")
    parser.add_argument("--prompt", type=str, default="Hello", help="Input prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(ROOT_DIR, "weights", "min_ckpt.pt"),
        help="Path to checkpoint",
    )
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer_info = load_checkpoint(args.checkpoint, device)
    encode_prompt, decode_tokens = get_tokenizer(tokenizer_info)

    prompt = args.prompt
    token_ids = encode_prompt(prompt)
    if not token_ids:
        token_ids = [0]
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)

    out = generate(
        model,
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    output_text = decode_tokens(out[0].tolist())
    print(output_text)


if __name__ == "__main__":
    main()
