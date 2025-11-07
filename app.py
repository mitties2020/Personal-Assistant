from flask import Flask, request, jsonify
from flask_cors import CORS
import os, zipfile, chardet, re
from pathlib import Path

# ---- DeepSeek client (OpenAI-compatible SDK) ----
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

# ---------- local corpus (clinical_data/) ----------
DATA_DIR = Path("clinical_data")
UNZIP_DIR = DATA_DIR / "_unzipped"
TEXT_CACHE = DATA_DIR / "_text_cache"
CORPUS = {}  # {path: text}


def ensure_dirs():
    # Make sure our folders exist and are real dirs (not leftover files)
    for d in (DATA_DIR, UNZIP_DIR, TEXT_CACHE):
        if d.exists() and not d.is_dir():
            d.unlink()
        d.mkdir(parents=True, exist_ok=True)


def _read_text_file(p: Path) -> str:
    raw = p.read_bytes()
    enc = (chardet.detect(raw).get("encoding") or "utf-8")
    try:
        return raw.decode(enc, errors="ignore")
    except Exception:
        return raw.decode("utf-8", errors="ignore")


def _pdf_to_txt(src: Path) -> Path:
    # Optional: requires pdfminer.six (already in requirements if you set it)
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
    # Optional: requires python-docx (only used if installed)
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
    """Unzip into UNZIP_DIR/<zipname>/ and return that folder path."""
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
    """Yield (file_path, text) for supported files under root (recursive)."""
    if not root.exists() or not root.is_dir():
        return
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        ext = path.suffix.lower()
        if ext in (".md", ".txt"):
            yield (path, _read_text_file(path))
        elif ext == ".pdf":
            tpath = _pdf_to_txt(path)
            if tpath.suffix.lower() == ".txt":
                yield (path, _read_text_file(tpath))
        elif ext == ".docx":
            tpath = _docx_to_txt(path)
            if tpath.suffix.lower() == ".txt":
                yield (path, _read_text_file(tpath))
        elif ext == ".zip":
            zdir = _maybe_unzip(path)
            if zdir:
                yield from _collect_texts(zdir)


def build_corpus():
    """Rebuild in-memory corpus from clinical_data/."""
    ensure_dirs()
    CORPUS.clear()
    for fp, text in _collect_texts(DATA_DIR):
        if text and text.strip():
            CORPUS[str(fp)] = text


def _score_doc(q_tokens, text):
    """Tiny relevance score: keyword hits + bonus for guideline-ish content."""
    if not text:
        return 0
    tl = text.lower()
    score = 0
    for t in q_tokens:
        if len(t) < 3:
            continue
        hits = tl.count(t)
        score += hits * 3
    if any(w in tl for w in (
        "dose", " mg", "mcg", " iv", "im ", "neb", "bolus",
        "criteria", "indication", "contraindication", "guideline"
    )):
        score += 5
    return score


def build_prompt(question: str) -> str:
    q = question.strip()
    q_tokens = re.findall(r"[a-z0-9]+", q.lower())

    # Rank docs by rough relevance
    scored = []
    for path, text in CORPUS.items():
        s = _score_doc(q_tokens, text[:8000])
        if s > 0:
            scored.append((s, path, text[:4000]))
    scored.sort(reverse=True)
    top = scored[:5]

    if top:
        ctx_blocks = [
            f"[Source: {Path(p).name}]\n{text}"
            for (s, p, text) in top
        ]
        context = "\n\n---\n\n".join(ctx_blocks)
    else:
        context = (
            "(no local documents matched closely; "
            "answer from standard emergency practice, state uncertainty if unsure)."
        )

    return f"""You are an Emergency Department clinical assistant.

Use the CONTEXT when relevant. If CONTEXT conflicts with generic knowledge,
prioritise CONTEXT but clearly state assumptions. If something is guideline- or
specialist-dependent, say so and recommend checking local policy.

Answer in EXACTLY this Markdown structure, concise (~180–220 words):

### What it is & criteria
- ...

### Common causes & complications
- ...

### Immediate management
- Bullet, stepwise, adult doses in standard units.
- Focus on first 5–10 minutes and life threats.

### Ongoing care / monitoring
- Disposition options, monitoring, escalation red flags.

CONTEXT:
{context}

QUESTION:
{q}
"""


def call_deepseek(prompt: str) -> str:
    if not ds_client:
        raise RuntimeError("DeepSeek not configured. Set DEEPSEEK_API_KEY.")
    r = ds_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.15,
    )
    return (r.choices[0].message.content or "").strip()


# Build index on startup
build_corpus()


# ---------- UI ----------
@app.route("/", methods=["GET"])
def ui():
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Clinical Q&A Engine</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {
  --bg: #020817;
  --card: #020817;
  --accent: #38bdf8;
  --accent-soft: rgba(56,189,248,0.10);
  --text-main: #e5e7eb;
  --text-soft: #9ca3af;
  --radius-xl: 22px;
  --shadow-soft: 0 18px 60px rgba(15,23,42,0.7);
  --font: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  font-family: var(--font);
  background:
    radial-gradient(circle at top left, rgba(56,189,248,0.08), transparent 55%),
    radial-gradient(circle at top right, rgba(129,140,248,0.06), transparent 55%),
    #020817;
  color: var(--text-main);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 32px 16px;
}
.app-shell {
  width: 100%;
  max-width: 980px;
  background:
    radial-gradient(circle at top, rgba(56,189,248,0.06), transparent 70%)
    var(--card);
  border-radius: 32px;
  padding: 26px 22px 22px;
  box-shadow: var(--shadow-soft);
  border: 1px solid rgba(148,163,253,0.18);
  backdrop-filter: blur(18px);
}
.header {
  margin-bottom: 18px;
}
.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 11px;
  border-radius: 999px;
  border: 1px solid rgba(148,163,253,0.35);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--accent);
  background: rgba(2,6,23,0.98);
}
.badge span.dot {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: var(--accent);
  box-shadow: 0 0 10px var(--accent);
}
.header h1 {
  margin: 6px 0 4px;
  font-size: 30px;
  font-weight: 650;
  letter-spacing: 0.01em;
}
.header h2 {
  margin: 0;
  font-size: 14px;
  font-weight: 400;
  color: var(--text-soft);
}
.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 6px;
}
.chip {
  font-size: 9px;
  padding: 3px 8px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
}
.main {
  display: grid;
  grid-template-columns: minmax(0, 1.3fr) minmax(0, 1.7fr);
  gap: 14px;
  margin-top: 10px;
}
@media (max-width: 780px) {
  .app-shell { padding: 18px 14px 16px; }
  .main { grid-template-columns: 1fr; }
}
.card {
  background: radial-gradient(circle at top, rgba(15,23,42,0.9), rgba(6,8,18,0.98));
  border-radius: var(--radius-xl);
  padding: 12px 11px 11px;
  border: 1px solid rgba(75,85,99,0.55);
}
.label {
  font-size: 10px;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.14em;
  margin-bottom: 4px;
}
.question-label {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-main);
  margin-bottom: 6px;
}
textarea {
  width: 100%;
  resize: vertical;
  min-height: 90px;
  max-height: 220px;
  padding: 10px 10px;
  border-radius: 16px;
  border: 1px solid rgba(75,85,99,0.9);
  background: rgba(2,6,23,0.99);
  color: var(--text-main);
  font-size: 13px;
  line-height: 1.5;
  outline: none;
  transition: all 0.18s ease;
}
textarea::placeholder {
  color: rgba(148,163,253,0.42);
}
textarea:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px rgba(56,189,248,0.12);
}
.button-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 7px;
}
button {
  flex: 1;
  padding: 9px 10px;
  border-radius: 18px;
  border: none;
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  letter-spacing: 0.03em;
  background: linear-gradient(90deg, var(--accent), #4f46e5);
  color: #020817;
  box-shadow: 0 10px 28px rgba(15,23,42,0.65);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  transition: all 0.18s ease;
}
button span.icon {
  font-size: 14px;
}
button:hover {
  transform: translateY(-1px);
  box-shadow: 0 14px 34px rgba(15,23,42,0.85);
}
button:active {
  transform: translateY(0);
  box-shadow: 0 6px 18px rgba(15,23,42,0.7);
}
.hint {
  font-size: 9px;
  color: var(--text-soft);
}
.answer-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 6px;
  margin-bottom: 4px;
}
.answer-title {
  font-size: 12px;
  font-weight: 500;
  color: var(--text-main);
}
.answer-meta {
  font-size: 8px;
  color: var(--text-soft);
}
#ans {
  font-size: 12px;
  line-height: 1.6;
  color: var(--text-main);
  white-space: pre-wrap;
}
#ans h3, #ans h4, #ans strong {
  color: var(--accent);
}
.status-pill {
  padding: 2px 6px;
  border-radius: 999px;
  border: 1px solid rgba(75,85,99,0.9);
  font-size: 8px;
  color: var(--text-soft);
}
</style>
</head>
<body>
<div class="app-shell">
  <div class="header">
    <div class="badge"><span class="dot"></span> Clinical Q&A Engine</div>
    <h1>Name your clinical question</h1>
    <h2>One line in → structured, high-yield ED answer out. Always confirm with local guidelines.</h2>
    <div class="chip-row">
      <div class="chip">Adult ED</div>
      <div class="chip">First 10 minutes</div>
      <div class="chip">Guideline-aware</div>
    </div>
  </div>
  <div class="main">
    <div class="card">
      <div class="label">Prompt</div>
      <div class="question-label">Clinical question</div>
      <textarea id="q" placeholder="e.g. Hyperkalaemia with broad QRS: immediate and ongoing management"></textarea>
      <div class="button-row">
        <button id="ask"><span class="icon">⚡</span><span>Generate answer</span></button>
        <div class="hint">Draws from your uploaded clinical_data plus DeepSeek.</div>
      </div>
    </div>
    <div class="card">
      <div class="answer-header">
        <div class="answer-title">Answer</div>
        <div class="answer-meta">
          <span id="providerLabel" class="status-pill">engine: idle</span>
        </div>
      </div>
      <div id="ans">Ask a question to see a structured summary here.</div>
    </div>
  </div>
</div>
<script>
async function ask() {
  const box = document.getElementById('q');
  const ans = document.getElementById('ans');
  const providerLabel = document.getElementById('providerLabel');
  const question = (box.value || '').trim();
  if (!question) {
    box.focus();
    return;
  }
  ans.textContent = 'Thinking through your question…';
  providerLabel.textContent = 'engine: contacting';
  try {
    const res = await fetch('/answer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    const data = await res.json();
    if (data.ok && data.answer) {
      providerLabel.textContent = data.provider
        ? ('engine: ' + data.provider)
        : 'engine: ready';
      // allow Markdown-ish line breaks
      ans.innerHTML = data.answer
        .replace(/\\n\\n/g, '<br><br>')
        .replace(/\\n/g, '<br>');
    } else {
      providerLabel.textContent = 'engine: error';
      ans.textContent = data.error || 'No response. Please try again.';
    }
  } catch (e) {
    providerLabel.textContent = 'engine: error';
    ans.textContent = 'Connection error. Please try again.';
  }
}
document.getElementById('ask').addEventListener('click', ask);
document.getElementById('q').addEventListener('keydown', function(e) {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') ask();
});
</script>
</body>
</html>
"""


# ---------- API ----------
@app.route("/health", methods=["GET"])
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
        # Keep UI alive if no key / no balance.
        return jsonify({
            "ok": True,
            "provider": "none",
            "answer":
                "Engine online, but no working DeepSeek key/balance is configured.\n"
                "Add DEEPSEEK_API_KEY in Render → Environment and top up your DeepSeek account."
        })

    try:
        text = call_deepseek(prompt)
        return jsonify({"ok": True, "provider": "deepseek", "answer": text})
    except Exception as e:
        # Do not kill the service on quota/network errors.
        return jsonify({"ok": False, "error": f"AI unavailable: {e}"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
