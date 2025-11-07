from flask import Flask, request, jsonify
from flask_cors import CORS
import os, chardet, json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# LLM client (DeepSeek via OpenAI SDK style)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

APP = Flask(__name__)
CORS(APP, resources={
    r"/answer": {"origins": "*"},
    r"/health": {"origins": "*"},
    r"/reindex": {"origins": "*"}
})

# ---------- AI SETUP ----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
deepseek_client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=DEEPSEEK_API_KEY
) if (OpenAI and DEEPSEEK_API_KEY) else None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
openai_client = OpenAI(
    api_key=OPENAI_API_KEY
) if (OpenAI and OPENAI_API_KEY and not deepseek_client) else None

# ---------- CORPUS (TXT ONLY) ----------
DATA_DIR = Path("clinical_data")
CHUNKS, META, VECTORIZER, MATRIX = [], [], None, None


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    gk = DATA_DIR / ".gitkeep"
    if not gk.exists():
        gk.touch()


def _read_text_file(p: Path) -> str:
    raw = p.read_bytes()
    enc = chardet.detect(raw).get("encoding") or "utf-8"
    try:
        return raw.decode(enc, errors="ignore")
    except Exception:
        return raw.decode("utf-8", errors="ignore")


def build_corpus():
    """Index all .txt files into TF-IDF matrix."""
    global CHUNKS, META, VECTORIZER, MATRIX
    ensure_dirs()
    CHUNKS, META = [], []

    for path in DATA_DIR.rglob("*.txt"):
        if path.name.startswith("."):
            continue
        txt = _read_text_file(path)
        if not txt.strip():
            continue
        i = 0
        while i < len(txt):
            chunk = txt[i:i + 1200].strip()
            if chunk:
                CHUNKS.append(chunk)
                META.append({"path": str(path), "offset": i})
            i += 900

    if not CHUNKS:
        VECTORIZER, MATRIX = None, None
        return

    VECTORIZER = TfidfVectorizer(stop_words="english", max_features=40000)
    MATRIX = VECTORIZER.fit_transform(CHUNKS)


def semantic_context(query: str, k: int = 5) -> str:
    if not (VECTORIZER and MATRIX is not None and CHUNKS):
        return "(no local context indexed)"
    qv = VECTORIZER.transform([query])
    sims = cosine_similarity(qv, MATRIX)[0]
    top = sims.argsort()[::-1][:k]
    pieces = []
    for i in top:
        score = sims[i]
        meta = META[i]
        pieces.append(f"[{meta['path']} | {score:.2f}]\n{CHUNKS[i]}")
    return "\n\n---\n\n".join(pieces)


# Build once at startup
build_corpus()

# ---------- PROMPT & PARSING ----------

def build_prompt(question: str) -> str:
    ctx = semantic_context(question)
    return f"""
You are a concise, evidence-based ED clinical assistant in Australia.

Use the CONTEXT if relevant; if not, use safe standard practice.
Respond ONLY as strict JSON with these keys:
- "what_it_is_and_criteria"
- "common_causes_and_complications"
- "immediate_management"
- "ongoing_care_monitoring"

Each value should be either a short paragraph or bullet-style text.
Keep it ~180 words total, practical for a senior ED doctor. No extra prose.

CONTEXT:
{ctx}

QUESTION: {question}
""".strip()


def call_llm(prompt: str):
    if deepseek_client:
        r = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
        )
        return (r.choices[0].message.content or "").strip(), "deepseek"

    if openai_client:
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
        )
        return (r.choices[0].message.content or "").strip(), "openai"

    raise RuntimeError("No AI configured (set DEEPSEEK_API_KEY or OPENAI_API_KEY).")


def safe_parse_structured(raw: str):
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return {
                "what_it_is_and_criteria": obj.get("what_it_is_and_criteria") or obj.get("what_it_is"),
                "common_causes_and_complications": obj.get("common_causes_and_complications"),
                "immediate_management": obj.get("immediate_management"),
                "ongoing_care_monitoring": obj.get("ongoing_care_monitoring"),
            }
    except Exception:
        pass
    return None


# ---------- ROUTES ----------

@APP.route("/", methods=["GET"])
def home():
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Personal Assistant — Doctor View</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{
  --bg:#020817;
  --card:#0f172a;
  --accent:#38bdf8;
  --accent-soft:rgba(56,189,248,0.12);
  --text:#e5e7eb;
  --muted:#9ca3af;
  --danger:#fca5a5;
  --radius:18px;
  --shadow:0 18px 40px rgba(15,23,42,0.55);
  --font:system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;
}
*{box-sizing:border-box;}
body{
  margin:0;
  min-height:100vh;
  font-family:var(--font);
  background:
    radial-gradient(circle at top, rgba(56,189,248,0.14), transparent 55%),
    radial-gradient(circle at bottom, rgba(14,165,233,0.04), transparent 55%),
    var(--bg);
  color:var(--text);
  display:flex;
  justify-content:center;
}
.wrap{
  width:100%;
  max-width:780px;
  padding:26px 16px 32px;
}
.header{
  display:flex;
  align-items:center;
  gap:10px;
  margin-bottom:4px;
}
.icon{
  width:30px;
  height:30px;
  border-radius:50%;
  background:var(--accent-soft);
  display:flex;
  align-items:center;
  justify-content:center;
  font-size:17px;
  color:var(--accent);
}
h1{
  font-size:22px;
  margin:0;
  font-weight:600;
}
.sub{
  font-size:12px;
  color:var(--muted);
  margin-bottom:18px;
}
.label{
  font-size:12px;
  color:var(--accent);
  margin-bottom:4px;
  letter-spacing:0.02em;
  text-transform:uppercase;
}
textarea{
  width:100%;
  min-height:70px;
  padding:10px 12px;
  border-radius:12px;
  border:1px solid rgba(148,163,253,0.28);
  background:rgba(2,6,23,0.98);
  color:var(--text);
  font-size:13px;
  resize:vertical;
  outline:none;
}
textarea::placeholder{
  color:#6b7280;
}
button{
  margin-top:8px;
  width:100%;
  padding:11px 14px;
  background:linear-gradient(to right,#38bdf8,#22c55e);
  color:#020817;
  border:none;
  border-radius:999px;
  font-size:14px;
  font-weight:600;
  cursor:pointer;
  box-shadow:0 10px 30px rgba(15,23,42,0.75);
  display:flex;
  align-items:center;
  justify-content:center;
  gap:6px;
}
button span.icon-s{
  font-size:16px;
}
button:hover{
  transform:translateY(-1px);
}
#card{
  margin-top:16px;
  background:var(--card);
  border-radius:var(--radius);
  padding:14px 14px 10px;
  box-shadow:var(--shadow);
  display:none;
}
.sec-title{
  font-weight:600;
  font-size:11px;
  margin:8px 0 2px;
  color:var(--accent);
  text-transform:uppercase;
  letter-spacing:0.05em;
}
ul{
  margin:0 0 4px 14px;
  padding:0;
  font-size:12px;
  color:var(--text);
}
li{
  margin:0 0 2px;
}
.meta{
  font-size:10px;
  color:var(--muted);
  margin-top:6px;
}
.err{
  color:var(--danger);
  font-size:10px;
  margin-top:4px;
  white-space:pre-wrap;
}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div class="icon">⚕️</div>
    <div>
      <h1>Personal Assistant — Doctor View</h1>
      <div class="sub">One-line clinical question → crisp 4-part ED summary (adult-focused).</div>
    </div>
  </div>

  <div class="label">Name your clinical question</div>
  <textarea id="q" placeholder="e.g. Anaphylaxis in adults: definition, immediate management, and observation plan."></textarea>
  <button id="ask"><span class="icon-s">⚡</span><span>Generate answer</span></button>

  <div id="card">
    <div id="what"></div>
    <div id="causes"></div>
    <div id="immed"></div>
    <div id="ongoing"></div>
    <div id="meta" class="meta"></div>
    <div id="err" class="err"></div>
  </div>
</div>

<script>
async function ask(){
  const q = document.getElementById('q').value.trim();
  const card = document.getElementById('card');
  const what = document.getElementById('what');
  const causes = document.getElementById('causes');
  const immed = document.getElementById('immed');
  const ongoing = document.getElementById('ongoing');
  const meta = document.getElementById('meta');
  const err = document.getElementById('err');

  if(!q) return;

  card.style.display = 'block';
  what.innerHTML = '<div class="sec-title">What it is & criteria</div><ul><li>Working…</li></ul>';
  causes.innerHTML = '';
  immed.innerHTML = '';
  ongoing.innerHTML = '';
  meta.textContent = '';
  err.textContent = '';

  try{
    const r = await fetch('/answer',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q})
    });
    const j = await r.json();

    if(!j.ok){
      err.textContent = j.error || 'Error generating answer.';
      meta.textContent = j.provider ? ('Provider: '+j.provider) : '';
      return;
    }

    const s = j.structured || {};

    function asList(val){
      if(!val) return '';
      if(Array.isArray(val)){
        if(!val.length) return '';
        return '<ul>'+val.map(v => '<li>'+String(v)+'</li>').join('')+'</ul>';
      }
      if(typeof val === 'string'){
        const parts = val.split(/[\n•\-]+/).map(t=>t.trim()).filter(Boolean);
        if(parts.length <= 1) return '<ul><li>'+val+'</li></ul>';
        return '<ul>'+parts.map(p=>'<li>'+p+'</li>').join('')+'</ul>';
      }
      return '<ul><li>'+String(val)+'</li></ul>';
    }

    what.innerHTML =
      '<div class="sec-title">What it is & criteria</div>' +
      (asList(s.what_it_is_and_criteria) || '<ul><li>No data.</li></ul>');

    causes.innerHTML =
      '<div class="sec-title">Common causes & complications</div>' +
      (asList(s.common_causes_and_complications) || '<ul><li>No data.</li></ul>');

    immed.innerHTML =
      '<div class="sec-title">Immediate management</div>' +
      (asList(s.immediate_management) || '<ul><li>No data.</li></ul>');

    ongoing.innerHTML =
      '<div class="sec-title">Ongoing care / monitoring</div>' +
      (asList(s.ongoing_care_monitoring) || '<ul><li>No data.</li></ul>');

    meta.textContent =
      (j.provider ? ('Provider: '+j.provider+'. ') : '') +
      (j.context_used ? 'Local clinical_data context applied.' : '');

    err.textContent = '';

  }catch(e){
    err.textContent = 'Network or server error: '+e.message;
    meta.textContent = '';
  }
}

document.getElementById('ask').addEventListener('click', ask);
</script>
</body>
</html>
"""


@APP.route("/answer", methods=["POST"])
def answer():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "Missing 'question'"}), 400

    prompt = build_prompt(q)
    try:
        raw, provider = call_llm(prompt)
    except Exception as e:
        return jsonify({"ok": False, "error": f"{e}", "provider": "none"}), 200

    structured = safe_parse_structured(raw)

    return jsonify({
        "ok": True,
        "provider": provider,
        "structured": structured,
        "answer": raw,
        "context_used": True
    })


@APP.route("/reindex", methods=["POST", "GET"])
def reindex():
    build_corpus()
    return jsonify({"ok": True, "files_indexed": len(CHUNKS)})


@APP.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "files_indexed": len(CHUNKS)})


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
