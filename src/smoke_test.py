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


def main():
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1337)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = build_model().to(device)
    model.train()

    batch_size = 2
    seq_len = 32
    vocab_size = 256

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    logits = model(x)

    loss = F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, vocab_size),
        x[:, 1:].contiguous().view(-1),
    )
    loss.backward()

    total_params = sum(p.numel() for p in model.parameters())
    grad_norm = None
    for param in model.parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            break

    print(f"loss: {loss.item():.6f}")
    print(f"params: {total_params}")
    print(f"grad_norm: {grad_norm:.6f}" if grad_norm is not None else "grad_norm: None")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
