#!/usr/bin/env python3
"""
Local semantic search with citations.
Usage:
  python scripts/query.py "clinical question"
"""
import sys, json, pathlib
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path(__file__).resolve().parents[1]
IDX = ROOT / "data" / "index"

EMB = np.load(IDX / "embeddings.npy")
meta_lines = [json.loads(l) for l in open(IDX / "meta.jsonl", encoding="utf-8")]
MODEL_NAME = open(IDX / "model.txt").read().strip()
model = SentenceTransformer(MODEL_NAME)

def top_k(query, k=5):
    q = model.encode([query], normalize_embeddings=True).astype(np.float32)[0]
    sims = EMB @ q
    idx = np.argsort(-sims)[:k]
    return [(float(sims[i]), meta_lines[i]) for i in idx]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/query.py 'question'")
        exit()
    q = sys.argv[1]
    results = top_k(q)
    print("\nQuery:", q)
    print("\nTop matches:")
    for score, r in results:
        print("-" * 80)
        print(f"Score: {score:.3f} | {r['file']} p.{r['page']}")
        txt = r["text"].strip().replace("\n", " ")
        print(txt[:500] + ("..." if len(txt) > 500 else ""))
