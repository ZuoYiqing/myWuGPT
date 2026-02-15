# myWuGPT

## UAV SFT demo (JSONL)
Pretrain checkpoint expected at `weights/pretrain.pt`.

```bash
python src/build_sft_data.py
python src/sft.py --data data/sft_uav_game.jsonl --out weights/sft.pt --max-steps 1000 --eval-interval 200
python src/inference.py --ckpt weights/sft.pt --prompt "UAV swarm task: secure corridor G3-G7, avoid Zone Q, report as ActionPlan JSON." --max-new-tokens 200 --temperature 0.2 --top-k 40
```

## RAG ingest
Dependencies:
- sentence-transformers
- chromadb
- PyPDF2
- python-docx
- ebooklib
- beautifulsoup4

SFT logging/plotting:
- matplotlib
- numpy

Minimal demo:
```bash
python src/rag.py --ingest --kb-dir knowledge_base --persist-dir .chroma
python src/inference.py --use-rag 1 --prompt "Ask a question about your KB"
```
