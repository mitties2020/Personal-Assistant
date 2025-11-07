from flask import Flask, request, jsonify
from flask_cors import CORS
import os, zipfile, chardet, re
from pathlib import Path

# ---- DeepSeek (OpenAI-compatible client) ----
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
ds_client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=DEEPSEEK_API_KEY
) if (OpenAI and DEEPSEEK_API_KEY) else None

app = Flask(__name__)
CORS(app, resources={
    r"/answer": {"origins": "*"},
    r"/health": {"origins": "*"},
    r"/reindex": {"origins": "*"}
})

# ---------- Corpus (clinical_data/) ----------
DATA_DIR = Path("clinical_data")
UNZIP_DIR = DATA_DIR / "_unzipped"
TEXT_CACHE = DATA_DIR / "_text_cache"
CORPUS = {}  # {path: text}


def ensure_dirs():
    for d in (DATA_DIR, UNZIP_DIR, TEXT_CACHE):
        if d.exists() and not d.is_dir():
            d.unlink()
        d.mkdir(parents=True, exist_ok=True)


def _read_text_file(p: Path) -> str:
    raw = p.read_bytes()
    enc = chardet.detect(raw).get("encoding") or "utf-8"
    try:
        return raw.decode(enc, errors="ignore")
    except Exception:
        return raw.decode("utf-8", errors="ignore")


def _pdf_to_txt(src: Path) -> Path:
    try:
        from pdfminer.high_level import extract_text
    except Exception:
        return src
    out = TEXT_CACHE / (src.stem + ".txt")
    try:
        text = extract_text(str(src)) or ""
    except Exception:
        text = ""
    out.write_text(text, encoding="utf-8", errors="ignore")
    return out


def _docx_to_txt(src: Path) -> Path:
    try:
        from docx import Document
    except Exception:
        return src
    out = TEXT_CACHE / (src.stem + ".txt")
    try:
        doc = Document(str(src))
        text = "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        text = ""
    out.write_text(text, encoding="utf-8", errors="ignore")
    return out


def _maybe_unzip(p: Path):
    try:
        tgt = UNZIP_DIR / p.stem
        if tgt.exists() and not tgt.is_dir():
            tgt.unlink()
        if not tgt.exists():
            tgt.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(tgt)
        return tgt
    except Exception:
        return None


def _collect_texts(root: Path):
    if not root.exists() or not root.is_dir():
        return
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        ext = path.suffix.lower()
        if ext in (".md", ".txt"):
            yield (path, _read_text_file(path))
        elif ext == ".pdf":
            yield (path, _read_text_file(_pdf_to_txt(path)))
        elif ext == ".docx":
            yield (path, _read_text_file(_docx_to_txt(path)))
        elif ext == ".zip":
            zdir = _maybe_unzip(path)
            if zdir:
                yield from _collect_texts(zdir)


def build_corpus():
    ensure_dirs()
    CORPUS.clear()
    for fp, text in _collect_texts(DATA_DIR):
        if text and text.strip():
            CORPUS[str(fp)] = text


def _score_doc(q_tokens, text):
    if not text:
        return 0
    tl = text.lower()
    score = sum(tl.count(t) * 3 for t in q_tokens if len(t) > 2)
    if any(w in tl for w in ("dose", " mg", "criteria", "guideline", "iv ", "im ")):
        score += 5
    return score


def build_prompt(question: str) -> str:
    q = question.strip()
    q_tokens = re.findall(r"[a-z0-9]+", q.lower())

    scored = [(s, p, text[:4000])
              for p, text in CORPUS.items()
              if (s := _score_doc(q_tokens, text[:8000])) > 0]
    scored.sort(reverse=True)
    top = scored[:5]

    if top:
        ctx_blocks = [f"[Source: {Path(p).name}]\n{text}" for (s, p, text) in top]
        context = "\n\n---\n\n".join(ctx_blocks)
    else:
        context = "(no local context matched — answer from emergency medicine standards.)"

    return f"""You are an Emergency Department clinical assistant.

Use CONTEXT if relevant. Be concise (~200 words) and use this exact structure:

### What it is & criteria
- ...

### Common causes & complications
- ...

### Immediate management
- Adult doses in mg or mcg; focus on first 5–10 minutes.

### Ongoing care / monitoring
- Escalation, disposition, red flags.

CONTEXT:
{context}

QUESTION:
{q}
"""


def call_deepseek(prompt: str) -> str:
    if not ds_client:
        raise RuntimeError("DeepSeek not configured.")
    r = ds_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.15,
    )
    return (r.choices[0].message.content or "").strip()


# Load corpus at startup
build_corpus()


# ---------- Routes ----------
@app.route("/", methods=["GET"])
def ui():
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Clinical Q&A Engine</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {
  font-family: system-ui, -apple-system, sans-serif;
  background: #020817;
  color: #e5e7eb;
  margin: 0; padding: 40px;
}
.container {
  max-width: 880px;
  margin: auto;
  background: rgba(15,23,42,0.95);
  border-radius: 24px;
  padding: 24px 28px;
  box-shadow: 0 0 80px rgba(0,0,0,0.4);
}
h1 {
  font-size: 28px;
  color: #38bdf8;
  margin-bottom: 10px;
}
textarea {
  width: 100%;
  min-height: 100px;
  border-radius: 14px;
  background: #0b1221;
  color: #f1f5f9;
  border: 1px solid #334155;
  padding: 12px;
  font-size: 14px;
}
button {
  margin-top: 10px;
  width: 100%;
  padding: 10px;
  border: none;
  border-radius: 14px;
  background: linear-gradient(90deg, #38bdf8, #6366f1);
  color: #020617;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
}
button:hover { opacity: 0.9; }
#ans {
  white-space: pre-wrap;
  background: #0b1221;
  border-radius: 14px;
  padding: 16px;
  margin-top: 18px;
  font-size: 15px;
  color: #f8fafc;
}
</style>
</head>
<body>
<div class="container">
  <h1>Clinical Q&A Engine</h1>
  <textarea id="q" placeholder="Ask your clinical question here..."></textarea>
  <button id="ask">Generate Answer</button>
  <div id="ans">Answer will appear here.</div>
</div>
<script>
document.getElementById('ask').onclick = async () => {
  const q = document.getElementById('q').value.trim();
  const ans = document.getElementById('ans');
  if(!q){ ans.textContent='Enter a question first.'; return; }
  ans.textContent='Thinking...';
  const res = await fetch('/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
  const j = await res.json();
  ans.innerHTML = j.answer ? j.answer.replace(/\\n/g,'<br>') : (j.error || 'Error.');
};
</script>
</body>
</html>"""


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "files_indexed": len(CORPUS),
        "deepseek_configured": bool(ds_client)
    })


@app.route("/reindex", methods=["POST", "GET"])
def reindex():
    build_corpus()
    return jsonify({"ok": True, "files_indexed": len(CORPUS)})


@app.route("/answer", methods=["POST"])
def answer():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "Missing 'question'"}), 400

    prompt = build_prompt(q)
    if not ds_client:
        return jsonify({
            "ok": True,
            "provider": "none",
            "answer": "DeepSeek not configured. Add DEEPSEEK_API_KEY in Render Environment."
        })

    try:
        text = call_deepseek(prompt)
        return jsonify({"ok": True, "provider": "deepseek", "answer": text})
    except Exception as e:
        return jsonify({"ok": False, "error": f"AI unavailable: {e}"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
