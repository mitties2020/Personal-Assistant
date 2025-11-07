from flask import Flask, request, jsonify
from flask_cors import CORS
import os, io, zipfile, chardet
from pathlib import Path

# Optional LLM (DeepSeek only; will stay up even if no key/balance)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
client = OpenAI(base_url="https://api.deepseek.com", api_key=DEEPSEEK_API_KEY) if (OpenAI and DEEPSEEK_API_KEY) else None

# Optional: allow OpenAI key fallback (only if you add it)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
oa_client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY and not client) else None

APP = Flask(__name__)
CORS(APP, resources={r"/answer": {"origins": "*"}, r"/health": {"origins": "*"}, r"/reindex": {"origins": "*"}})

# -------- Ingestion helpers --------
DATA_DIR = Path("clinical_data")
UNZIP_DIR = DATA_DIR / "_unzipped"
TEXT_CACHE = DATA_DIR / "_text_cache"  # where .pdf/.docx are converted to .txt
CORPUS = {}  # {filepath: text}

def ensure_dirs():
    UNZIP_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)

def _read_text_file(p: Path) -> str:
    raw = p.read_bytes()
    # try to detect encoding for random txt
    enc = chardet.detect(raw).get("encoding") or "utf-8"
    try:
        return raw.decode(enc, errors="ignore")
    except Exception:
        return raw.decode("utf-8", errors="ignore")

def _pdf_to_txt(src: Path) -> Path:
    from pdfminer.high_level import extract_text
    out = TEXT_CACHE / (src.stem + ".txt")
    try:
        text = extract_text(str(src)) or ""
    except Exception:
        text = ""
    out.write_text(text, encoding="utf-8", errors="ignore")
    return out

def _docx_to_txt(src: Path) -> Path:
    from docx import Document
    out = TEXT_CACHE / (src.stem + ".txt")
    try:
        doc = Document(str(src))
        text = "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        text = ""
    out.write_text(text, encoding="utf-8", errors="ignore")
    return out

def _maybe_unzip(p: Path):
    """Unzip into UNZIP_DIR/<zipname>/ and return that folder path."""
    try:
        tgt = UNZIP_DIR / p.stem
        if tgt.exists():
            return tgt
        tgt.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(tgt)
        return tgt
    except Exception:
        return None

def _collect_texts(root: Path):
    """Yield (file_path, text) for supported files under root (recursive)."""
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        ext = path.suffix.lower()
        if ext in (".md", ".txt"):
            yield (path, _read_text_file(path))
        elif ext == ".pdf":
            t = _pdf_to_txt(path)
            yield (path, _read_text_file(t))
        elif ext == ".docx":
            t = _docx_to_txt(path)
            yield (path, _read_text_file(t))
        elif ext == ".zip":
            unzip = _maybe_unzip(path)
            if unzip:
                # recurse into unzipped content
                yield from _collect_texts(unzip)

def build_corpus():
    """Rebuild in-memory corpus from clinical_data/ (incl. zips, pdf, docx)."""
    ensure_dirs()
    CORPUS.clear()
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    # scan main folder
    for fp, text in _collect_texts(DATA_DIR):
        if text and text.strip():
            CORPUS[str(fp)] = text
    # also scan _unzipped and cache again (in case of direct edits)
    for fp, text in _collect_texts(UNZIP_DIR):
        if text and text.strip():
            CORPUS[str(fp)] = text

# Initial load at startup
build_corpus()

# -------- Mini UI --------
@APP.route("/", methods=["GET"])
def root():
    html = """<!doctype html><meta charset="utf-8"><title>Personal Assistant</title>
<style>body{font-family:system-ui,Arial,sans-serif;max-width:680px;margin:24px auto;padding:0 16px}
textarea{width:100%;height:120px}button{width:100%;padding:12px;margin-top:8px;font-size:16px}
#ans{white-space:pre-wrap;background:#f6f8fa;padding:14px;border-radius:10px;margin-top:16px}</style>
<h2>Personal Assistant</h2>
<textarea id="q" placeholder="Ask anything (you can upload .zip/.pdf/.docx into clinical_data/ and hit /reindex)"></textarea>
<button id="ask">Ask</button><div id="ans"></div>
<script>
document.getElementById('ask').onclick = async () => {
  const q = document.getElementById('q').value.trim(); if(!q) return;
  document.getElementById('ans').textContent = 'Working…';
  const r = await fetch('/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
  const j = await r.json();
  document.getElementById('ans').textContent = j.answer || j.error || 'No response';
};
</script>"""
    return html

@APP.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "files_indexed": len(CORPUS)})

@APP.route("/reindex", methods=["POST", "GET"])
def reindex():
    build_corpus()
    return jsonify({"ok": True, "files_indexed": len(CORPUS)})

def build_prompt(question: str) -> str:
    # Create a small context window from your corpus (very simple keyword match)
    qlow = question.lower()
    hits = []
    for path, text in CORPUS.items():
        if any(k in text.lower() for k in qlow.split()[:5]):
            hits.append((path, text[:4000]))  # clip per file
        if len(hits) >= 6:
            break
    context = "\n\n---\n\n".join(f"[[{p}]]\n{text}" for p, text in hits) or "(no local context matched)"
    return f"""You are a concise clinical assistant for an Australian ED doctor.

Use the CONTEXT if relevant, otherwise answer from general knowledge.
Keep the answer ~180 words with sections:
1) What it is & criteria
2) Common causes & complications
3) Immediate management (doses in adult units)
4) Ongoing care / monitoring

CONTEXT:
{context}

QUESTION: {question}
"""

def call_deepseek(prompt: str) -> str:
    if not client:
        raise RuntimeError("DeepSeek not configured. Set DEEPSEEK_API_KEY.")
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return (r.choices[0].message.content or "").strip()

def call_openai(prompt: str) -> str:
    if not oa_client:
        raise RuntimeError("OpenAI not configured.")
    r = oa_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return (r.choices[0].message.content or "").strip()

@APP.route("/answer", methods=["POST"])
def answer():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "Missing 'question'"}), 400

    prompt = build_prompt(q)

    # Prefer DeepSeek if configured, else OpenAI; otherwise still return friendly note
    try:
        if client:
            text = call_deepseek(prompt)
            return jsonify({"ok": True, "provider": "deepseek", "answer": text})
        elif oa_client:
            text = call_openai(prompt)
            return jsonify({"ok": True, "provider": "openai", "answer": text})
        else:
            return jsonify({"ok": True, "provider": "none",
                            "answer": "Backend is up. Add DEEPSEEK_API_KEY (or OPENAI_API_KEY) and redeploy."})
    except Exception as e:
        # don’t crash if quota/balance problems
        return jsonify({"ok": False, "error": f"AI unavailable: {e}"}), 200

if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
