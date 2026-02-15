"""Print report-ready architecture snippets in Chinese."""

from __future__ import annotations


def main() -> None:
    moe_snippet = (
        "MoE 路由逻辑图解：模型在 model.py 的 MLP 类（注释“MoE with SwiGLU experts”）中实现 MoE，"
        "router = nn.Linear(n_embd, num_experts) 产生路由 logits，num_experts 默认值为 4。"
        "在 MLP.forward 中对 logits 做 softmax，随后使用 topk(k=2) 选择两个专家（Top-2 gating），"
        "将 token dispatch 到两个专家 FFN，再按 gate 权重进行加权 combine。"
        "为避免专家坍缩，报告中关注 expert share / entropy / max_share 等统计指标。"
    )
    arch_snippet = (
        "模型结构图解：注意力模块在 CausalSelfAttention 中使用 RoPE（旋转位置编码）并配合 RMSNorm 做预归一化；"
        "MLP 使用 SwiGLU 专家（SwiGLUExpert）替换传统 FFN。"
        "每个 TransformerBlock 采用残差结构：x + attn、x + mlp，整体为 decoder-only GPT。"
    )
    print(moe_snippet)
    print()
    print(arch_snippet)


if __name__ == "__main__":
    main()
