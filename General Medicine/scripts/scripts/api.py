#!/usr/bin/env python3
# FastAPI wrapper around your local embeddings index.
# Start with:
#   uvicorn scripts.api:app --host 0.0.0.0 --port 8000 --reload

import os, json, pathlib
from typing import List, Dict, Any
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

ROOT = pathlib.Path(__file__).resolve().parents[1]
IDX = ROOT / "data" / "index"

# --- Simple token auth ---
API_TOKEN = os.getenv("API_TOKEN", "").strip()

def auth(authorization: str = Header(default="")):
    if not API_TOKEN:
        # If no token set, allow (dev mode)
        return
    # Expect "Bearer <token>"
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1] == API_TOKEN:
        return
    raise HTTPException(status_code=401, detail="Unauthorized")

# --- Load index once at startup ---
if not (IDX / "embeddings.npy").exists():
    raise RuntimeError("No index found. Run scripts/build_index.py first.")

EMB = np.load(IDX / "embeddings.npy")  # [N, D]
meta_lines = [json.loads(l) for l in open(IDX / "meta.jsonl", "r", encoding="utf-8")]
MODEL_NAME = open(IDX / "model.txt", "r").read().strip()
model = SentenceTransformer(MODEL_NAME)

def top_k(query: str, k: int = 5) -> List[Dict[str, Any]]:
    q = model.encode([query], normalize_embeddings=True).astype(np.float32)[0]
    sims = EMB @ q
    idx = np.argsort(-sims)[:k]
    out = []
    for i in idx:
        r = dict(meta_lines[i])  # file, page, text
        r["score"] = float(sims[i])
        # short snippet for UI
        snippet = (r["text"].strip().replace("\n", " "))
        if len(snippet) > 600:
            snippet = snippet[:600] + " …"
        r["snippet"] = snippet
        del r["text"]
        out.append(r)
    return out

# --- FastAPI app ---
app = FastAPI(title="Personal Assistant Medical Search API")

# CORS for your website (replace with your domain when deployed)
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchIn(BaseModel):
    query: str
    k: int = 5

class SearchOut(BaseModel):
    query: str
    results: List[Dict[str, Any]]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/search", response_model=SearchOut, dependencies=[Depends(auth)])
def api_search(body: SearchIn):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    results = top_k(body.query.strip(), k=max(1, min(body.k, 20)))
    return {"query": body.query, "results": results}
