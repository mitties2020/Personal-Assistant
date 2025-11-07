from flask import Flask, request, jsonify
from flask_cors import CORS
import os, io, zipfile, chardet
from pathlib import Path

# Optional LLM (DeepSeek via OpenAI client-style; safe if not configured)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
client = OpenAI(base_url="https://api.deepseek.com",
                api_key=DEEPSEEK_API_KEY) if (OpenAI and DEEPSEEK_API_KEY) else None

# Optional fallback: OpenAI key (only used if DeepSeek not set)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
oa_client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY and not client) else None

APP = Flask(__name__)
CORS(APP, resources={
    r"/answer": {"origins": "*"},
    r"/health": {"origins": "*"},
    r"/reindex": {"origins": "*"},
})

# ---------- Local corpus (clinical_data) ----------

DATA_DIR = Path("clinical_data")
UNZIP_DIR = DATA_DIR / "_unzipped"
TEXT_CACHE = DATA_DIR / "_text_cache"
CORPUS = {}  # {filepath: text}


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # if something weird exists with these names as files, ignore failure
    for d in (UNZIP_DIR, TEXT_CACHE):
        try:
            if d.exists() and not d.is_dir():
                d.unlink()
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


def _read_text_file(p: Path) -> str:
    raw = p.read_bytes()
    enc = (chardet.detect(raw).get("encoding") or "utf-8")
    try:
        return raw.decode(enc, errors="ignore")
    except Exception:
        return raw.decode("utf-8", errors="ignore")


def _pdf_to_txt(src: Path) -> Path:
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(str(src)) or ""
    except Exception:
        text = ""
    out = TEXT_CACHE / (src.stem + ".txt")
    out.write_text(text, encoding="utf-8", errors="ignore")
    return out


def _docx_to_txt(src: Path) -> Path:
    try:
        from docx import Document
        doc = Document(str(src))
        text = "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        text = ""
    out = TEXT_CACHE / (src.stem + ".txt")
    out.write_text(text, encoding="utf-8", errors="ignore")
    return out


def _maybe_unzip(p: Path):
    try:
        tgt = UNZIP_DIR / p.stem
        if not tgt.exists():
            tgt.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(tgt)
        return tgt
    except Exception:
        return None


def _collect_texts(root: Path):
    if not root.exists():
        return
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
                yield from _collect_texts(unzip)


def build_corpus():
    ensure_dirs()
    CORPUS.clear()
    for fp, text in _collect_texts(DATA_DIR) or []:
        if text and text.strip():
            CORPUS[str(fp)] = text


# initial load
build_corpus()

# ---------- UI ----------


@APP.route("/", methods=["GET"])
def root():
    # Single-page minimal UI
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FFM Clinical Assistant</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    :root {
      --bg:#030712;
      --card:#0f172a;
      --accent:#38bdf8;
      --accent-soft:rgba(56,189,248,0.12);
      --text:#e5e7eb;
      --muted:#9ca3af;
      --radius:18px;
      --font:system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;
    }
    *{box-sizing:border-box;}
    body{
      margin:0;
      min-height:100vh;
      font-family:var(--font);
      background:radial-gradient(circle at top,var(--accent-soft),var(--bg));
      color:var(--text);
      display:flex;
      align-items:center;
      justify-content:center;
      padding:18px;
    }
    .wrap{
      width:100%;
      max-width:720px;
      background:linear-gradient(to bottom right,rgba(148,163,253,0.06),rgba(15,23,42,0.98));
      border-radius:24px;
      padding:20px 20px 18px;
      box-shadow:0 18px 55px rgba(15,23,42,0.9);
      border:1px solid rgba(148,163,253,0.15);
      backdrop-filter:blur(14px);
    }
    .header{
      display:flex;
      align-items:center;
      gap:10px;
      margin-bottom:8px;
    }
    .pill{
      padding:3px 10px;
      border-radius:999px;
      font-size:10px;
      color:var(--accent);
      background:var(--accent-soft);
      border:1px solid rgba(56,189,248,0.35);
    }
    h1{
      font-size:20px;
      margin:0;
      font-weight:600;
      letter-spacing:0.02em;
    }
    p.sub{
      margin:2px 0 14px;
      font-size:12px;
      color:var(--muted);
    }
    label{
      font-size:11px;
      color:var(--muted);
      display:block;
      margin-bottom:4px;
      letter-spacing:0.06em;
      text-transform:uppercase;
    }
    textarea{
      width:100%;
      min-height:80px;
      max-height:180px;
      resize:vertical;
      padding:10px 11px;
      border-radius:var(--radius);
      border:1px solid rgba(148,163,253,0.25);
      background:rgba(2,6,23,0.9);
      color:var(--text);
      font-size:13px;
      outline:none;
      box-shadow:inset 0 0 0 1px transparent;
    }
    textarea::placeholder{color:#6b7280;}
    textarea:focus{
      border-color:var(--accent);
      box-shadow:0 0 0 1px rgba(56,189,248,0.35);
    }
    button{
      margin-top:8px;
      width:100%;
      padding:10px 12px;
      border-radius:var(--radius);
      border:none;
      font-weight:500;
      font-size:13px;
      letter-spacing:0.04em;
      text-transform:uppercase;
      background:linear-gradient(to right,var(--accent),#6366f1);
      color:#020817;
      cursor:pointer;
      display:flex;
      align-items:center;
      justify-content:center;
      gap:8px;
      box-shadow:0 10px 30px rgba(37,99,235,0.45);
    }
    button span.icon{
      display:inline-block;
      transform:translateY(1px);
    }
    button:active{
      transform:translateY(1px);
      box-shadow:0 4px 16px rgba(15,23,42,0.9);
    }
    .answer{
      margin-top:10px;
      padding:9px 10px;
      border-radius:var(--radius);
      background:rgba(2,6,23,0.96);
      border:1px solid rgba(75,85,99,0.6);
      font-size:12px;
      line-height:1.5;
      white-space:pre-wrap;
      max-height:260px;
      overflow-y:auto;
    }
    .answer.empty{
      color:#6b7280;
      border-style:dashed;
      text-align:left;
    }
    .meta{
      display:flex;
      justify-content:space-between;
      align-items:center;
      margin-top:4px;
      font-size:9px;
      color:var(--muted);
    }
    .status-dot{
      width:7px;height:7px;
      border-radius:999px;
      background:#22c55e;
      box-shadow:0 0 8px #22c55e;
      display:inline-block;
      margin-right:4px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <div class="pill">FFM • Clinical Q&A Engine</div>
    </div>
    <h1>Name your clinical question</h1>
    <p class="sub">
      One-line question in → structured, high-yield answer out. 
      Built for ED workflow. Always confirm with local guidelines.
    </p>
    <label for="q">Clinical question</label>
    <textarea id="q" placeholder="e.g. Hyperkalaemia with broad QRS: immediate management"></textarea>
    <button id="ask">
      <span class="icon">➜</span>
      <span>Get answer</span>
    </button>
    <div id="ans" class="answer empty">
      Response will appear here in 4 clear sections:
      1) What it is & criteria · 2) Causes & complications · 3) Immediate management · 4) Ongoing care.
    </div>
    <div class="meta">
      <div><span class="status-dot"></span>Backend online</div>
      <div id="provider">Model: local data / API hybrid</div>
    </div>
  </div>
  <script>
    const qEl = document.getElementById('q');
    const ansEl = document.getElementById('ans');
    const providerEl = document.getElementById('provider');
    const btn = document.getElementById('ask');

    async function ask() {
      const q = qEl.value.trim();
      if (!q) return;
      ansEl.classList.remove('empty');
      ansEl.textContent = "Thinking...";
      providerEl.textContent = "Model: resolving…";
      try {
        const res = await fetch('/answer', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({question: q})
        });
        const j = await res.json();
        if (j.ok && j.answer) {
          ansEl.textContent = j.answer;
        } else {
          ansEl.textContent = j.error || "No response from engine.";
        }
        providerEl.textContent = "Model: " + (j.provider || 'none');
      } catch (e) {
        ansEl.textContent = "Network or server error. Please try again.";
        providerEl.textContent = "Model: error";
      }
    }

    btn.addEventListener('click', ask);
    qEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        ask();
      }
    });
  </script>
</body>
</html>
"""


@APP.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "files_indexed": len(CORPUS)})


@APP.route("/reindex", methods=["POST", "GET"])
def reindex():
    build_corpus()
    return jsonify({"ok": True, "files_indexed": len(CORPUS)})


def build_prompt(question: str) -> str:
    # naive keyword-based context from CORPUS
    qlow = question.lower()
    hits = []
    for path, text in CORPUS.items():
        if any(k in text.lower() for k in qlow.split()[:5]):
            hits.append((path, text[:4000]))
        if len(hits) >= 6:
            break
    context = "\n\n---\n\n".join(f"[[{p}]]\n{text}" for p, text in hits) or "(no local context matched)"
    return f"""You are a concise clinical assistant for an Australian ED doctor.
Use the CONTEXT if relevant, otherwise answer from your usual knowledge.
Keep the answer around 180 words with EXACT sections:
1) What it is & criteria
2) Common causes & complications
3) Immediate management (adult doses, units, routes)
4) Ongoing care / monitoring

CONTEXT:
{context}

QUESTION: {question}
"""


def call_deepseek(prompt: str) -> str:
    if not client:
        raise RuntimeError("DeepSeek not configured.")
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

    try:
        if client:
            text = call_deepseek(prompt)
            return jsonify({"ok": True, "provider": "deepseek", "answer": text})
        elif oa_client:
            text = call_openai(prompt)
            return jsonify({"ok": True, "provider": "openai", "answer": text})
        else:
            # Still respond so the UI doesn’t look broken
            return jsonify({
                "ok": True,
                "provider": "none",
                "answer": (
                    "Backend is running.\n\n"
                    "- To enable AI answers: set DEEPSEEK_API_KEY (or OPENAI_API_KEY) "
                    "in your Render Environment and redeploy.\n"
                    "- You can already attach local files into clinical_data/ and reindex."
                )
            })
    except Exception as e:
        # Quota / balance / API errors show to user but don’t kill service
        return jsonify({"ok": False, "error": f"AI unavailable: {e}"}), 200


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
