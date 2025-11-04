#!/usr/bin/env python3
"""
Extract text + pages from PDFs in data/raw, chunk them, build embeddings, and
save an index under data/index/
"""
import os, json, pathlib
from typing import List, Dict
import numpy as np
import fitz  # PyMuPDF
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
IDX = ROOT / "data" / "index"
PROC.mkdir(parents=True, exist_ok=True)
IDX.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

CHUNK_SIZE = 1200
OVERLAP = 200

def extract_pdf(pdf_path: pathlib.Path):
    doc = fitz.open(pdf_path)
    rows = []
    for pno in range(len(doc)):
        page = doc[pno]
        text = page.get_text("text") or ""
        if text.strip():
            rows.append({"file": pdf_path.name, "page": pno + 1, "text": text})
    return rows

def chunk_text(rows):
    chunks = []
    for r in rows:
        t = r["text"].strip()
        if not t:
            continue
        start = 0
        while start < len(t):
            end = min(len(t), start + CHUNK_SIZE)
            chunk = t[start:end]
            chunks.append({"file": r["file"], "page": r["page"], "text": chunk})
            start = end - OVERLAP
            if start < 0:
                start = 0
    return chunks

def embed_texts(texts):
    vecs = []
    B = 64
    for i in tqdm(range(0, len(texts), B), desc="Embedding"):
        batch = texts[i:i+B]
        emb = model.encode(batch, normalize_embeddings=True)
        vecs.append(emb.astype(np.float32))
    return np.vstack(vecs)

def main():
    pdfs = list(RAW.glob("*.pdf"))
    if not pdfs:
        print("No PDFs in data/raw.")
        return

    all_chunks = []
    for pdf in pdfs:
        print(f"[extract] {pdf.name}")
        rows = extract_pdf(pdf)
        chunks = chunk_text(rows)
        all_chunks.extend(chunks)

    meta_path = IDX / "meta.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for r in all_chunks:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    texts = [r["text"] for r in all_chunks]
    print(f"[index] {len(texts)} chunks")
    X = embed_texts(texts)

    np.save(IDX / "embeddings.npy", X)
    with open(IDX / "model.txt", "w") as f:
        f.write(MODEL_NAME)

    print("✅ Index built → data/index/")

if __name__ == "__main__":
    main()
