import os, json, hashlib, datetime as dt, re
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Optional OpenAI (only used if key is set) ---
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

APP = Flask(__name__)
CORS(APP, resources={r"/answer": {"origins": "*"}, r"/ingest": {"origins": "*"}, r"/health": {"origins": "*"}})

# --------- CONFIG ---------
DATA_DIR = Path(os.getenv("INDEX_DIR", "/data/indexdir")).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_JSON = DATA_DIR / "index.json"

ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*").strip()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
oa_client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None

for p in (DATA_DIR, UPLOAD_DIR):
    p.mkdir(parents=True, exist_ok=True)
if not INDEX_JSON.exists():
    INDEX_JSON.write_text("[]", encoding="utf-8")

# --------- UTILITIES ---------
def load_index():
    try:
        return json.loads(INDEX_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_index(items):
    INDEX_JSON.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def ext(path: str) -> str:
    return Path(path).suffix.lower()

def extract_text_from_bytes(filename: str, content: bytes) -> str:
    # PDF -> text via pdfminer; otherwise treat as UTF-8 text
    if ext(filename) == ".pdf":
        from pdfminer.high_level import extract_text
        tmp = UPLOAD_DIR / ("tmp_" + sha256_bytes(content) + ".pdf")
        tmp.write_bytes(content)
        try:
            text = extract_text(str(tmp)) or ""
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
        return text
    # text-ish
    try:
        return content.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def simple_score(query: str, text: str) -> int:
    """Very small keyword scorer: counts query tokens in text."""
    q_terms = [t for t in re.split(r"[^\w%]+", query.lower()) if t]
    if not q_terms:
        return 0
    txt = text.lower()
    return sum(txt.count(t) for t in q_terms)

def top_snippets(query: str, text: str, max_chars=700, windows=3):
    # return a few surrounding snippets around matches
    txt = text
    q = query.split()[0:5]
    anchors = []
    for w in q:
        for m in re.finditer(re.escape(w), txt, flags=re.IGNORECASE):
            anchors.append(m.start())
    anchors = sorted(set(anchors))[:windows]
    snips = []
    for a in anchors:
        start = max(0, a - 220)
        end = min(len(txt), a + 220)
        snips.append(txt[start:end].strip())
    combined = "\n...\n".join(snips)
    if not combined:
        combined = (txt[:max_chars] + "…") if len(txt) > max_chars else txt
    return combined[:max_chars]

def build_prompt(question: str, hits: list):
    # Compose a grounded prompt with up to 3 doc snippets
    src_blocks = []
    for h in hits[:3]:
        src_blocks.append(
            f"[{h['title']} | {h['org']} | {h['published']}]\n{h['snippet']}"
        )
    sources_section = "\n\n".join(src_blocks) if src_blocks else "No local sources."

    return f"""
You are a clinical assistant for an Australian ED doctor.

Use the SOURCES below to answer the QUESTION in ~180–220 words with exactly these sections:

1) What it is & criteria
2) Common causes & complications
3) Immediate management (first-line actions & doses)
4) Ongoing care / monitoring

Be specific with adult doses/units/routes (e.g., adrenaline 0.5 mg IM; Ca gluconate 10% 10 mL IV over 2–5 min).
Prefer Australian guidance where equivalent.

SOURCES:
{sources_section}

QUESTION: {question}
"""

# --------- ROUTES ---------
@APP.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@APP.route("/ingest", methods=["POST"])
def ingest():
    """
    Form-data:
      - file: (pdf or txt)
      - title: optional
      - org: optional (e.g., 'NSW Health')
      - published: optional ISO date (YYYY-MM-DD)
      - url: optional
    """
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file"}), 400

    f = request.files["file"]
    raw = f.read()
    if not raw:
        return jsonify({"ok": False, "error": "Empty file"}), 400

    digest = sha256_bytes(raw)
    out = UPLOAD_DIR / (digest + (ext(f.filename) or ".bin"))
    out.write_bytes(raw)

    text = extract_text_from_bytes(f.filename, raw)
    if not text.strip():
        return jsonify({"ok": False, "error": "No text extracted"}), 400

    meta = {
        "id": digest,
        "title": request.form.get("title") or Path(f.filename).stem,
        "org": request.form.get("org") or "",
        "published": request.form.get("published") or "",
        "url": request.form.get("url") or "",
        "path": str(out),
        "chars": len(text),
    }

    # Store extracted text next to file
    (UPLOAD_DIR / f"{digest}.txt").write_text(text, encoding="utf-8")

    idx = load_index()
    # replace if already exists
    idx = [d for d in idx if d.get("id") != digest]
    idx.append(meta)
    save_index(idx)

    return jsonify({"ok": True, "doc": meta})

@APP.route("/answer", methods=["POST"])
def answer():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "Missing 'question'"}), 400

    # Rank local docs
    idx = load_index()
    docs_scored = []
    for d in idx:
        try:
            txt = (UPLOAD_DIR / f"{d['id']}.txt").read_text(encoding="utf-8")
            score = simple_score(q, txt)
            if score > 0:
                docs_scored.append({
                    **d,
                    "score": score,
                    "snippet": top_snippets(q, txt)
                })
        except Exception:
            pass

    docs_scored.sort(key=lambda x: x["score"], reverse=True)

    if oa_client:
        try:
            prompt = build_prompt(q, docs_scored)
            resp = oa_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            text = (resp.choices[0].message.content or "").strip()
            # Return thin source list (no big payloads)
            sources = [
                {k: d.get(k) for k in ("title", "org", "published", "url")}
                for d in docs_scored[:3]
            ]
            return jsonify({"ok": True, "answer": text, "sources": sources})
        except Exception as e:
            return jsonify({"ok": False, "error": f"OpenAI error: {e}"}), 500

    # Fallback when no key set
    return jsonify({
        "ok": True,
        "answer": "Backend is up, but no model is configured. Set OPENAI_API_KEY in Render → Environment and redeploy.",
        "sources": []
    })

# --------- ENTRYPOINT ---------
if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
