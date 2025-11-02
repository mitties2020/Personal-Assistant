#!/usr/bin/env python3
import os, re, sys, io, hashlib, tempfile, shutil, requests, yaml
from datetime import datetime
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, DATETIME
from pypdf import PdfReader

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
INDEX_DIR = os.path.join(DATA_DIR, "indexdir")
DL_DIR = os.path.join(tempfile.gettempdir(), "ffm_dl")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(DL_DIR, exist_ok=True)

schema = Schema(
    chunk_id=ID(stored=True, unique=True),
    title=TEXT(stored=True),
    org=TEXT(stored=True),
    published=DATETIME(stored=True),
    text=TEXT(stored=True),
)

def open_or_create_index(path):
    from whoosh import index as windex
    if os.path.isdir(path) and os.listdir(path):
        return windex.open_dir(path)
    os.makedirs(path, exist_ok=True)
    return windex.create_in(path, schema)

def chunk_text(txt, max_chars=1600):
    sents = re.split(r"(?<=[\\.!?])\\s+", txt.strip())
    out, cur, cur_len = [], [], 0
    for s in sents:
        L = len(s)
        if cur and cur_len + L > max_chars:
            out.append(" ".join(cur))
            cur, cur_len = [s], L
        else:
            cur.append(s); cur_len += L
    if cur:
        out.append(" ".join(cur))
    return out

def download(url, to_dir):
    name = url.split("/")[-1] or "file.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    path = os.path.join(to_dir, name)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(65536):
                if chunk:
                    f.write(chunk)
    return path

def pdf_text(path):
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        t = p.extract_text() or ""
        t = re.sub(r"[ \\t]+", " ", t)
        pages.append(t)
    return "\\n".join(pages)

def ingest(ix, path, title, org="", published=""):
    try:
        dt = datetime.fromisoformat(published) if published else datetime.utcnow()
    except Exception:
        dt = datetime.utcnow()
    txt = pdf_text(path)
    chunks = chunk_text(txt, 2000)
    basehash = hashlib.sha1(title.encode("utf-8")).hexdigest()[:8]
    w = ix.writer()
    for i, c in enumerate(chunks):
        cid = f"{basehash}:{i}"
        w.add_document(chunk_id=cid, title=title, org=org, published=dt, text=c)
    w.commit()
    return len(chunks)

def main():
    src_file = os.path.join(REPO_ROOT, "sources.yml")
    if not os.path.exists(src_file):
        print("[!] sources.yml not found at repo root"); sys.exit(1)

    with open(src_file, "r", encoding="utf-8") as f:
        sources = yaml.safe_load(f) or []

    if not sources:
        print("[!] No sources defined in sources.yml"); sys.exit(1)

    # fresh index build: wipe old indexdir to avoid stale segments
    if os.path.isdir(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    os.makedirs(INDEX_DIR, exist_ok=True)

    ix = open_or_create_index(INDEX_DIR)
    total_chunks = 0

    for s in sources:
        url = s.get("url"); title = s.get("title") or url
        org = s.get("org") or ""; published = s.get("published") or ""
        if not url:
            print("[!] Skipping an entry without url"); continue
        print(f"[+] {title} — {org} ({published})")
        path = download(url, DL_DIR)
        n = ingest(ix, path, title, org, published)
        total_chunks += n
        print(f"    ✓ {n} chunks")

    print(f"[OK] Built index at data/indexdir with {total_chunks} chunks")

if __name__ == "__main__":
    main()
