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
    if any(w in tl for w in ("dose", "criteria", "guideline",
                             "treatment", "monitoring", "diagnosis")):
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
        context = "(no local context matched — generating answer from general clinical knowledge.)"

    return f"""You are a clinical reasoning assistant.

Use the CONTEXT if it is relevant. Provide a concise, structured explanation suitable for a clinician.

Format your response under these headings:

### Definition & diagnostic criteria
- ...

### Aetiology / contributing factors
- ...

### Management approach
- Include medications (with adult dosages in mg/mcg where relevant), investigations, and monitoring.

### Follow-up / long-term care
- Outline key steps for follow-up, escalation, and inter-specialty coordination.

Avoid local policy details unless clearly implied by context.

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
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #010a16;
  --card: #050816;
  --accent: #38bdf8;
  --accent-soft: rgba(56,189,248,0.18);
  --accent-grad: linear-gradient(90deg, #38bdf8, #818cf8);
  --text: #e5e7eb;
  --text-soft: #9ca3af;
  --font: 'Inter', system-ui, -apple-system, sans-serif;
  --radius-xl: 22px;
  --radius-md: 16px;
  --shadow: 0 0 80px rgba(15,23,42,0.65);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 40px 16px;
  background:
    radial-gradient(circle at top left, rgba(56,189,248,0.08), transparent),
    radial-gradient(circle at top right, rgba(129,140,248,0.06), transparent),
    #020817;
  color: var(--text);
  font-family: var(--font);
  display: flex;
  justify-content: center;
  min-height: 100vh;
}
.container {
  width: 100%;
  max-width: 1040px;
  padding: 28px 26px 26px;
  background: radial-gradient(circle at top, rgba(15,23,42,0.9), #020817);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow);
  border: 1px solid rgba(148,163,253,0.18);
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(0, 1.3fr);
  gap: 22px;
}
header {
  grid-column: 1 / -1;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
}
.brand {
  font-size: 11px;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--accent);
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.brand-dot {
  width: 7px; height: 7px;
  border-radius: 999px;
  background: var(--accent-grad);
}
h1 {
  margin: 4px 0 0;
  font-size: 28px;
  font-weight: 700;
  letter-spacing: 0.01em;
}
.subtitle {
  font-size: 12px;
  color: var(--text-soft);
  margin-top: 4px;
}
.label {
  font-size: 12px;
  color: var(--accent);
  margin-bottom: 5px;
  letter-spacing: .12em;
  text-transform: uppercase;
}
textarea {
  width: 100%;
  min-height: 130px;
  border-radius: var(--radius-md);
  border: 1px solid #1f2937;
  background: rgba(5,10,25,0.98);
  color: var(--text);
  padding: 12px 12px 12px;
  font-size: 15px;
  line-height: 1.6;
  font-family: var(--font);
  resize: vertical;
  transition: 0.22s ease;
}
textarea::placeholder {
  color: #6b7280;
}
textarea:focus {
  border-color: var(--accent);
  box-shadow: 0 0 16px rgba(56,189,248,0.35);
  outline: none;
  background: #020817;
}
.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}
button.primary {
  flex: 1 1 auto;
  padding: 11px 16px;
  border-radius: var(--radius-md);
  border: none;
  background: var(--accent-grad);
  color: #020817;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  box-shadow: 0 8px 24px rgba(56,189,248,0.35);
  transition: transform .1s ease, box-shadow .2s ease, filter .15s ease;
}
button.primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 12px 32px rgba(56,189,248,0.45);
  filter: saturate(1.08);
}
button.primary:active {
  transform: translateY(1px);
  box-shadow: 0 3px 10px rgba(15,23,42,0.55);
}
button.secondary {
  flex: 0 0 auto;
  padding: 9px 14px;
  border-radius: 999px;
  border: 1px solid rgba(148,163,253,0.4);
  background: rgba(10,16,30,0.98);
  color: var(--text-soft);
  font-size: 11px;
  font-weight: 500;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  transition: all .16s ease;
}
button.secondary:hover {
  color: var(--accent);
  border-color: var(--accent);
  box-shadow: 0 0 14px rgba(56,189,248,0.24);
}
button.secondary:active {
  transform: translateY(1px);
  box-shadow: none;
}
#ans-wrap {
  padding: 12px 14px 14px;
  border-radius: var(--radius-md);
  background: radial-gradient(circle at top left, rgba(15,23,42,1), #020817);
  border: 1px solid rgba(31,41,55,0.98);
  box-shadow: inset 0 0 0 1px rgba(15,23,42,0.9);
  max-height: 480px;
  overflow-y: auto;
}
#ans-label {
  font-size: 11px;
  color: var(--text-soft);
  margin-bottom: 4px;
  text-transform: uppercase;
  letter-spacing: .14em;
}
#ans {
  font-size: 15px;
  line-height: 1.7;
  color: var(--text);
  white-space: pre-wrap;
}
#ans h3 {
  font-size: 16px;
  margin: 10px 0 4px;
  color: var(--accent);
}
#ans ul {
  padding-left: 18px;
}
.badge {
  font-size: 9px;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid var(--accent-soft);
  color: var(--accent);
}
.toast {
  position: fixed;
  bottom: 18px;
  right: 18px;
  background: rgba(15,23,42,0.98);
  color: var(--accent);
  padding: 8px 14px;
  font-size: 11px;
  border-radius: 999px;
  border: 1px solid rgba(56,189,248,0.6);
  box-shadow: 0 12px 30px rgba(15,23,42,0.9);
  opacity: 0;
  transform: translateY(6px);
  pointer-events: none;
  transition: all .18s ease;
}
.toast.show {
  opacity: 1;
  transform: translateY(0);
}
@media (max-width: 840px) {
  .container {
    grid-template-columns: 1fr;
    gap: 18px;
  }
  #ans-wrap {
    max-height: none;
  }
}
</style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <div class="brand"><span class="brand-dot"></span>Clinical Q&A Engine</div>
      <h1>Name your clinical question</h1>
      <div class="subtitle">One line in → structured, high-yield answer out. Always interpret within your local practice standards.</div>
    </div>
    <div class="badge">DeepSeek-enabled</div>
  </header>

  <section>
    <div class="label">Clinical question</div>
    <textarea id="q" placeholder="e.g. Management of lithium toxicity, evaluation of first episode psychosis, anticoagulation in AF with CKD..."></textarea>
    <div class="actions">
      <button class="primary" id="ask">
        <span>⚡ Generate answer</span>
      </button>
      <button class="secondary" id="copyBtn">
        📋 Copy answer
      </button>
      <button class="secondary" id="downloadBtn">
        ⬇️ Export as .txt
      </button>
    </div>
  </section>

  <section id="ans-wrap">
    <div id="ans-label">Answer</div>
    <div id="ans">Your structured answer will appear here.</div>
  </section>
</div>

<div id="toast" class="toast">Copied to clipboard</div>

<script>
const ansEl = document.getElementById('ans');
const toast = document.getElementById('toast');

function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 1500);
}

document.getElementById('ask').onclick = async () => {
  const q = document.getElementById('q').value.trim();
  if (!q) {
    ansEl.textContent = 'Please enter a clinical question.';
    return;
  }
  ansEl.textContent = 'Generating...';
  const res = await fetch('/answer', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ question: q })
  });
  const j = await res.json();
  if (j.answer) {
    ansEl.innerHTML = j.answer.replace(/\\n/g, '<br>');
  } else {
    ansEl.textContent = j.error || 'Error generating answer.';
  }
};

document.getElementById('copyBtn').onclick = async () => {
  const text = ansEl.innerText.trim();
  if (!text) {
    showToast('No answer to copy yet.');
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    showToast('Answer copied to clipboard.');
  } catch (e) {
    showToast('Clipboard unavailable in this browser.');
  }
};

document.getElementById('downloadBtn').onclick = () => {
  const text = ansEl.innerText.trim();
  if (!text) {
    showToast('No answer to export yet.');
    return;
  }
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'clinical_answer.txt';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  showToast('.txt file downloaded.');
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

    if not ds_client:
        return jsonify({
            "ok": True,
            "provider": "none",
            "answer": "DeepSeek not configured. Add DEEPSEEK_API_KEY in Render Environment."
        })

    prompt = build_prompt(q)
    try:
        text = call_deepseek(prompt)
        return jsonify({"ok": True, "provider": "deepseek", "answer": text})
    except Exception as e:
        return jsonify({"ok": False, "error": f"AI unavailable: {e}"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
