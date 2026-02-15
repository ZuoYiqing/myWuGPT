"""Export a small GPT model to ONNX and a text graph summary."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src import model as model_module  # noqa: E402


class ExportRMSNorm(torch.nn.Module):
    """RMSNorm implementation compatible with ONNX export."""

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight.detach().clone())
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.mean(x * x, dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


def replace_rmsnorm(module: torch.nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, torch.nn.RMSNorm):
            replacement = ExportRMSNorm(child.weight, eps=child.eps)
            setattr(module, name, replacement)
        else:
            replace_rmsnorm(child)


def build_small_model() -> torch.nn.Module:
    config = model_module.GPTConfig(
        vocab_size=256,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=128,
        dropout=0.0,
        use_pos_emb=True,
    )
    model = model_module.GPT(config)
    model.eval()
    return model


def write_model_graph(model: torch.nn.Module, path: Path) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    lines = [f"Total parameters: {total_params}"]

    def walk(module: torch.nn.Module, prefix: str = "") -> None:
        for name, child in module.named_children():
            params = sum(p.numel() for p in child.parameters())
            lines.append(f"{prefix}{name}: {child.__class__.__name__} (params={params})")
            walk(child, prefix + "  ")

    walk(model)
    path.write_text("\n".join(lines), encoding="utf-8")


def export_assets(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"
    graph_path = output_dir / "model_graph.txt"

    model = build_small_model()
    if hasattr(torch.nn, "RMSNorm"):
        replace_rmsnorm(model)
    dummy = torch.zeros((1, 32), dtype=torch.long)
    torch.onnx.export(
        model,
        dummy,
        onnx_path.as_posix(),
        input_names=["x"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"x": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
        do_constant_folding=True,
    )
    write_model_graph(model, graph_path)
    return onnx_path, graph_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ONNX + model graph text")
    parser.add_argument("--out-dir", type=str, default=os.path.join(ROOT_DIR, "assets"))
    args = parser.parse_args()
    export_assets(Path(args.out_dir))


if __name__ == "__main__":
    main()
