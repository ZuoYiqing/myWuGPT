"""Convert prompt/response JSONL to instruction/input/output JSONL for sft.py

Writes to data/sft_nonviolent.jsonl
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "out", "sft_prompt_response_nonviolent.jsonl")
OUT = os.path.join(ROOT, "data", "sft_nonviolent.jsonl")

sep_re = re.compile(r"\n\n输入[:：]\s*", flags=re.IGNORECASE)

count_in = 0
count_out = 0
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(IN, 'r', encoding='utf-8') as fin, open(OUT, 'w', encoding='utf-8') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        count_in += 1
        try:
            obj = json.loads(line)
        except Exception:
            continue
        prompt = obj.get('prompt', '') or ''
        response = obj.get('response', '') or ''
        # split prompt into instruction and input
        parts = sep_re.split(prompt, maxsplit=1)
        if len(parts) == 2:
            instruction = parts[0].strip()
            user_input = parts[1].strip()
        else:
            # fallback: no explicit input part
            instruction = prompt.strip()
            user_input = ""

        out_obj = {
            'instruction': instruction,
            'input': user_input,
            'output': response,
        }
        fout.write(json.dumps(out_obj, ensure_ascii=False) + '\n')
        count_out += 1

print(f"Converted {count_in} entries -> {count_out} written to {OUT}")