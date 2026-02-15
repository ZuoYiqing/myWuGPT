"""Filter SFT JSONL into sensitive and nonviolent files.

Usage: python src/filter_sft_sensitive.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "out", "sft_prompt_response.jsonl")
OUT_NONVIOLENT = os.path.join(ROOT, "out", "sft_prompt_response_nonviolent.jsonl")
OUT_SENSITIVE = os.path.join(ROOT, "out", "sensitive.jsonl")

# Conservative sensitive keyword list (Chinese)
SENSITIVE_KEYWORDS = [
    "摧毁",
    "发射导弹",
    "投弹",
    "击毁",
    "打击",
    "炸",
    "炸毁",
    "沉重打击",
    "击毁目标",
    "摧毁敌",
    "攻击",
    "诱敌",
    "诱饵",
    "击毁",
    "投放炸弹",
]

sensitive_count = 0
nonviolent_count = 0
lines_total = 0

os.makedirs(os.path.dirname(OUT_NONVIOLENT), exist_ok=True)

with open(IN, 'r', encoding='utf-8') as fin, \
     open(OUT_NONVIOLENT, 'w', encoding='utf-8') as fout_safe, \
     open(OUT_SENSITIVE, 'w', encoding='utf-8') as fout_sensitive:
    for line_no, line in enumerate(fin, 1):
        line = line.strip()
        if not line:
            continue
        lines_total += 1
        try:
            obj = json.loads(line)
        except Exception as e:
            # treat unparsable lines as sensitive
            fout_sensitive.write(line + '\n')
            sensitive_count += 1
            continue
        prompt = (obj.get('prompt', '') or '')
        response = (obj.get('response', '') or '')
        text = (prompt + '\n' + response).lower()

        # redacted responses are considered sensitive
        if response.strip() == "[REDACTED]":
            fout_sensitive.write(json.dumps(obj, ensure_ascii=False) + '\n')
            sensitive_count += 1
            continue

        matched = False
        for kw in SENSITIVE_KEYWORDS:
            if kw in text:
                fout_sensitive.write(json.dumps(obj, ensure_ascii=False) + '\n')
                sensitive_count += 1
                matched = True
                break
        if matched:
            continue

        # otherwise safe
        fout_safe.write(json.dumps(obj, ensure_ascii=False) + '\n')
        nonviolent_count += 1

print(f"Filtered {lines_total} lines: {nonviolent_count} non-violent, {sensitive_count} sensitive.")
print(f"Wrote safe -> {OUT_NONVIOLENT}")
print(f"Wrote sensitive -> {OUT_SENSITIVE}")
