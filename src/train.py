"""Minimal training script for GPT on random data."""

import os
import sys

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src import model as model_module  # noqa: E402


def build_model():
    config_kwargs = dict(
        vocab_size=256,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=128,
    )
    gpt_config_cls = getattr(model_module, "GPTConfig", None)
    if gpt_config_cls is not None:
        return model_module.GPT(gpt_config_cls(**config_kwargs))
    return model_module.GPT(**config_kwargs)


def get_batch_random(batch_size, seq_len, vocab_size, device):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = x[:, 1:].contiguous()
    return x, y


def main():
    torch.manual_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1337)

    model = build_model().to(device)
    model.train()

    batch_size = 16
    seq_len = 64
    vocab_size = 256

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(1, 201):
        x, y = get_batch_random(batch_size, seq_len, vocab_size, device)
        logits = model(x)
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, vocab_size),
            y.view(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"step {step:03d} | loss {loss.item():.6f}")

    os.makedirs(os.path.join(ROOT_DIR, "weights"), exist_ok=True)
    ckpt_path = os.path.join(ROOT_DIR, "weights", "min_ckpt.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": model.config,
            "step": 200,
        },
        ckpt_path,
    )
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
