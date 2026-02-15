"""
把 Word(.docx) 语料变成“模型可训练的数据”，本质就两步：
1) 从 docx 抽取纯文本（段落 + 表格）
2) 按训练方式组织成数据格式：
   - 预训练(continual pretrain)：一个或多个 .txt（纯文本拼接即可）
   - 指令微调(SFT)：.jsonl，每行一个样本 {"messages":[...]} 或 {"prompt":..., "response":...}

下面这个脚本会：
- 读取 /mnt/data/训练部分语料.docx（你也可以换成本地路径）
- 抽取段落+表格文本
- 做基础清洗
- 生成：
   out/corpus.txt                 # 适合预训练/继续预训练
   out/sft_messages.jsonl         # 适合 ChatML/messages 格式 SFT（若能解析出Q/A）
   out/sft_prompt_response.jsonl  # 适合 prompt/response 格式 SFT（若能解析出Q/A）

只需要改两处：
- DOCX_PATH：你的word路径
- detect_qa_pairs()：根据你语料里“问答分隔符”的实际写法，微调规则
"""

import os
import re
import json
from typing import List, Tuple, Optional

from docx import Document


# ========= 1) 配置 & CLI =========
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Convert .docx into pretrain / SFT datasets")
    p.add_argument("--docx", type=str, default=os.path.join("data", "部分训练语料.docx"), help="Path to .docx file (default: data/部分训练语料.docx)")
    p.add_argument("--out-dir", type=str, default="out", help="Output directory (default: out)")
    p.add_argument("--min-qa-length", type=int, default=1, help="Minimum length for detected Q/A content")
    p.add_argument("--no-pretrain", action="store_true", help="Don't write pretrain corpus.txt")
    p.add_argument("--no-sft", action="store_true", help="Don't write SFT jsonl files even if QA pairs found")

    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--redact-violence", dest="redact_violence", action="store_true", help="Redact samples that contain violent intents (default)")
    grp.add_argument("--no-redact-violence", dest="redact_violence", action="store_false", help="Don't redact violent samples")
    p.set_defaults(redact_violence=True)

    return p.parse_args()


# ========= 2) 抽取 docx 文本：段落 + 表格 =========
def extract_docx_text(docx_path: str) -> List[str]:
    doc = Document(docx_path)
    lines: List[str] = []

    # 段落
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            lines.append(t)

    # 表格（按行拼接）
    for table in doc.tables:
        for row in table.rows:
            cells = [re.sub(r"\s+", " ", (c.text or "").strip()) for c in row.cells]
            row_text = " | ".join([c for c in cells if c])
            if row_text:
                lines.append(row_text)

    return lines


# ========= 3) 清洗 =========
def normalize_lines(lines: List[str]) -> List[str]:
    out = []
    for s in lines:
        s = s.replace("\u00a0", " ")            # nbsp
        s = re.sub(r"[ \t]+", " ", s)           # 多空格
        s = re.sub(r"\n+", "\n", s)             # 多换行
        s = s.strip()
        if s:
            out.append(s)
    return out


# ========= 4.5) 从连续文本中解析 JSON 对象（有时 Word 把多个 JSON 连在一起） =========

def extract_json_objects_from_text(text: str) -> List[dict]:
    """从一段长文本中解析出多个顶层 JSON 对象（处理字符串内的括号/转义）。
    返回解析成功的 dict 列表。
    """
    objs: List[dict] = []
    start = None
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if ch == '"' and not esc:
            in_str = not in_str
        if in_str and ch == '\\' and not esc:
            esc = True
            continue
        else:
            esc = False
        if not in_str:
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    chunk = text[start:i + 1]
                    try:
                        obj = json.loads(chunk)
                        objs.append(obj)
                    except Exception:
                        # ignore invalid JSON chunk
                        pass
                    start = None
    return objs


# ========= 4) 组织成“预训练文本” =========
def save_pretrain_txt(lines: List[str], out_path: str) -> None:
    # 用空行分隔段落，有利于一些 tokenizer/训练管线
    with open(out_path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + "\n\n")


# ========= 5) 尝试从文本里解析 Q/A -> SFT =========
def detect_qa_pairs(lines: List[str]) -> List[Tuple[str, str]]:
    """更鲁棒的 Q/A 识别规则：
    - 支持同一行中的 "问题：... 答案：..." 或 "Q: ... A: ..."
    - 支持标注式的下一行答案（行是 "问题：..." 下一行是 "答案：..."）
    - 支持问句（以 ? 或 ？ 结尾）后紧跟一行作为答案
    - 支持编号的问答（1. 问...  1. 答...）

    这些规则不会覆盖所有情况，但对常见的 Word QA 格式效果较好。
    """
    qa: List[Tuple[str, str]] = []
    i = 0
    n = len(lines)

    # 常用模式
    same_line_pat = re.compile(r"^\s*(?:Q|Question|问题)[:：]\s*(.+?)(?:\s*(?:A|Answer|答案)[:：]\s*(.+))?$")
    label_a_pat = re.compile(r"^\s*(?:A|Answer|答案)[:：]\s*(.+)$", flags=re.IGNORECASE)
    label_q_pat = re.compile(r"^\s*(?:Q|Question|问题)[:：]\s*(.+)$", flags=re.IGNORECASE)
    bracket_q_pat = re.compile(r"^\s*【问】\s*(.+)$")
    bracket_a_pat = re.compile(r"^\s*【答】\s*(.+)$")
    # numbered patterns like: 1. 问: ... 或 1) 问...
    num_q_pat = re.compile(r"^\s*\d+[\.)]\s*(?:问|问题|Q)[:：]?\s*(.+)")
    num_a_pat = re.compile(r"^\s*\d+[\.)]\s*(?:答|答案|A)[:：]?\s*(.+)")

    while i < n:
        line = lines[i].strip()

        # 1) 同行 Q/A
        m = same_line_pat.match(line)
        if m:
            q = m.group(1).strip()
            a = m.group(2)
            if a:
                qa.append((q, a.strip()))
                i += 1
                continue
            # 如果没有同一行的答案，检查下一行是否是标注为答案
            if i + 1 < n:
                mnext_a = label_a_pat.match(lines[i + 1]) or bracket_a_pat.match(lines[i + 1]) or num_a_pat.match(lines[i + 1])
                if mnext_a:
                    qa.append((q, mnext_a.group(1).strip()))
                    i += 2
                    continue

        # 2) 标注式 Q followed by A
        m_q = label_q_pat.match(line) or bracket_q_pat.match(line) or num_q_pat.match(line)
        if m_q and i + 1 < n:
            q = m_q.group(1).strip()
            # next line might be labelled A
            m_a = label_a_pat.match(lines[i + 1]) or bracket_a_pat.match(lines[i + 1]) or num_a_pat.match(lines[i + 1])
            if m_a:
                qa.append((q, m_a.group(1).strip()))
                i += 2
                continue
            # or next line is short and looks like an answer
            nxt = lines[i + 1].strip()
            if 0 < len(nxt) <= 800 and not nxt.endswith("?") and not nxt.endswith("？"):
                qa.append((q, nxt))
                i += 2
                continue

        # 3) 问句 + 下一行为答案的简单启发式
        if line.endswith("?") or line.endswith("？"):
            if i + 1 < n:
                nxt = lines[i + 1].strip()
                if len(nxt) <= 800 and not nxt.endswith("?") and not nxt.endswith("？"):
                    qa.append((line, nxt))
                    i += 2
                    continue

        # 4) 同行包含 "问题... 答案..."
        inline_q_a = re.search(r"(?:问题|Q)[:：](.+?)\s+(?:答案|A)[:：](.+)$", line)
        if inline_q_a:
            qa.append((inline_q_a.group(1).strip(), inline_q_a.group(2).strip()))
            i += 1
            continue

        i += 1

    # 去重并过滤过短内容
    cleaned: List[Tuple[str, str]] = []
    seen = set()
    for q, a in qa:
        q_norm = q.strip()
        a_norm = a.strip()
        if len(q_norm) < 2 or len(a_norm) < 1:
            continue
        key = (q_norm, a_norm)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append((q_norm, a_norm))

    return cleaned


def save_sft_jsonl_messages(qa_pairs: List[Tuple[str, str]], out_path: str) -> None:
    """
    ChatML/messages 常用格式：
    {"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
    """
    system_prompt = "你是一个严谨、清晰的中文助教，回答要结构化、可执行。"

    with open(out_path, "w", encoding="utf-8") as f:
        for q, a in qa_pairs:
            obj = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_sft_jsonl_prompt_response(qa_pairs: List[Tuple[str, str]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for q, a in qa_pairs:
            obj = {"prompt": q, "response": a}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ========= 6) 主流程 =========
def main():
    args = parse_args()
    docx_path = args.docx
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading docx: {docx_path}")
    lines = extract_docx_text(docx_path)
    lines = normalize_lines(lines)

    # Debug: 写出原始行到调试文件并打印前若干行，方便确认文档结构
    debug_path = os.path.join(out_dir, "raw_lines_debug.txt")
    with open(debug_path, "w", encoding="utf-8") as df:
        for i, ln in enumerate(lines):
            df.write(f"{i+1:04d}: {ln}\n")
    print(f"Wrote debug raw lines to: {debug_path}")
    print("First few lines:")
    for i, ln in enumerate(lines[:40]):
        print(f"{i+1:03d}: {ln}")

    # 处理文档里连着的 JSON 对象（常见于把多个 jsonl 粘在一起的情况）
    objs = []
    if len(lines) == 1 and ('{"instruction"' in lines[0] or lines[0].strip().startswith('{')):
        objs = extract_json_objects_from_text(lines[0])
        if objs:
            print(f"Detected {len(objs)} JSON objects embedded in the document; converting to samples.")
            samples = []
            qa_pairs_from_json: List[Tuple[str, str]] = []
            redacted = 0
            for obj in objs:
                inst = (obj.get("instruction") or "").strip()
                inp = (obj.get("input") or "").strip()
                outp = obj.get("output")

                # 简单的暴力意图检测 & 可选红线过滤
                intent_text = ""
                if isinstance(outp, dict):
                    intent_text = str(outp.get("intent", ""))
                combined_check = inst + " " + intent_text + " " + json.dumps(outp, ensure_ascii=False)
                should_redact = False
                if getattr(args, "redact_violence", True):
                    for kw in ["打击", "攻击", "投弹", "发射导弹", "摧毁", "发射", "炸毁"]:
                        if kw in combined_check:
                            should_redact = True
                            break
                if should_redact:
                    redacted += 1
                    out_str = "[REDACTED]"
                else:
                    out_str = json.dumps(outp, ensure_ascii=False)

                prompt = inst + ("\n\n输入: " + inp if inp else "")
                qa_pairs_from_json.append((prompt, out_str))
                samples.append(json.dumps(obj, ensure_ascii=False))

            # 覆盖 lines 与 qa_pairs 以继续后续的保存逻辑
            lines = samples
            qa_pairs = qa_pairs_from_json
            print(f"Converted to {len(samples)} samples, redacted {redacted} items.")

    # 6.1 预训练纯文本（可通过 --no-pretrain 关闭）
    pretrain_path = os.path.join(out_dir, "corpus.txt")
    if not args.no_pretrain:
        save_pretrain_txt(lines, pretrain_path)

    # 6.2 SFT：尝试解析 Q/A（可通过 --no-sft 关闭）
    if 'qa_pairs' not in locals():
        qa_pairs = detect_qa_pairs(lines)

    if qa_pairs and not args.no_sft:
        msg_path = os.path.join(out_dir, "sft_messages.jsonl")
        pr_path = os.path.join(out_dir, "sft_prompt_response.jsonl")
        save_sft_jsonl_messages(qa_pairs, msg_path)
        save_sft_jsonl_prompt_response(qa_pairs, pr_path)

    # 6.3 打印一些统计，方便你确认
    print("=== DONE ===")
    print(f"Total lines extracted: {len(lines)}")
    if not args.no_pretrain:
        print(f"Pretrain text saved to: {pretrain_path}")
    print(f"Detected QA pairs: {len(qa_pairs)}")
    if qa_pairs and not args.no_sft:
        print(f"SFT(messages) saved to: {os.path.join(out_dir, 'sft_messages.jsonl')}")
        print(f"SFT(prompt/response) saved to: {os.path.join(out_dir, 'sft_prompt_response.jsonl')}")
        print("Sample QA:")
        for i, (q, a) in enumerate(qa_pairs[:10]):
            print(f"\n--- QA {i+1} ---\nQ: {q}\nA: {a}")


if __name__ == "__main__":
    main()
