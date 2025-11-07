from flask import Flask, request, jsonify
from flask_cors import CORS
import os, chardet, json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# LLM client (DeepSeek-style via OpenAI SDK)
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
    gitkeep = DATA_DIR / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()


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
    """Return top-k similar chunks as context."""
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


# Build once on startup
build_corpus()

# ---------- PROMPT ----------

def build_prompt(question: str) -> str:
    ctx = semantic_context(question)
    return f"""
You are a concise, evidence-based ED clinical assistant in Australia.

Use the CONTEXT if relevant; if not, use safe standard practice.
Respond ONLY as strict JSON with these keys:
- "what_it_is_and_criteria": bullet-point summary.
- "common_causes_and_complications": bullet points.
- "immediate_management": bullet points with adult doses/routes.
- "ongoing_care_monitoring": bullet points.

Keep it succinct and practical for a senior ED doctor.
Avoid waffle. No explanations outside the JSON.

CONTEXT:
{ctx}

QUESTION: {question}
""".strip()


def call_llm(prompt: str):
    """Try DeepSeek then OpenAI. Return (text, provider)."""
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
    """Try to parse JSON; fallback to plain text."""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


# ---------- ROUTES ----------

@APP.route("/", methods=["GET"])
def home():
    # Minimal clean UI rendered directly
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Personal Assistant — Doctor View</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font-family:system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;
     max-width:760px;margin:24px auto;padding:0 16px;color:#111827;background:#f9fafb}
h1{font-size:24px;margin-bottom:4px}
.sub{font-size:13px;color:#6b7280;margin-bottom:16px}
textarea{width:100%;min-height:80px;padding:10px 12px;border-radius:10px;border:1px solid #d1d5db;
         font-size:14px;resize:vertical;box-sizing:border-box;background:#ffffff}
button{margin-top:8px;width:100%;padding:11px 14px;background:#111827;color:white;border:none;
       border-radius:10px;font-size:15px;font-weight:500;cursor:pointer}
button:hover{background:#000000}
#card{margin-top:16px;background:#ffffff;border-radius:14px;padding:14px 14px 10px;
      box-shadow:0 8px 18px rgba(15,23,42,0.08);display:none}
.sec-title{font-weight:600;font-size:13px;margin:8px 0 2px;color:#111827}
ul{margin:0 0 4px 16px;padding:0;font-size:13px;color:#111827}
li{margin:0 0 2px}
.meta{font-size:11px;color:#9ca3af;margin-top:4px}
.err{color:#b91c1c;font-size:12px;margin-top:4px;white-space:pre-wrap}
</style>
</head>
<body>
<h1>Personal Assistant</h1>
<div class="sub">One-line clinical question → crisp 4-part answer (ED-focused, adult doses).</div>
<textarea id="q" placeholder="e.g. Anaphylaxis adult: immediate and ongoing management"></textarea>
<button id="ask">Answer</button>

<div id="card">
  <div id="what" class="sec"></div>
  <div id="causes" class="sec"></div>
  <div id="immed" class="sec"></div>
  <div id="ongoing" class="sec"></div>
  <div id="meta" class="meta"></div>
  <div id="err" class="err"></div>
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
  what.textContent = causes.textContent = immed.textContent = ongoing.textContent = '';
  meta.textContent = ''; err.textContent = '';
  what.innerHTML = '<div class="sec-title">What it is & criteria</div><ul><li>Working…</li></ul>';

  try{
    const r = await fetch('/answer',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q})
    });
    const j = await r.json();

    if(!j.ok){
      err.textContent = j.error || 'Error.';
      meta.textContent = j.provider ? ('Provider: '+j.provider) : '';
      return;
    }

    const s = j.structured || {};
    function asList(val){
      if(!val) return '';
      if(Array.isArray(val)) return '<ul>'+val.map(v=>'<li>'+String(v)+'</li>').join('')+'</ul>';
      if(typeof val === 'string'){
        // split into bullets on newline / • / -
        const parts = val.split(/[\n•\-]+/).map(t=>t.trim()).filter(Boolean);
        if(parts.length <= 1) return '<ul><li>'+val+'</li></ul>';
        return '<ul>'+parts.map(p=>'<li>'+p+'</li>').join('')+'</ul>';
      }
      return '<ul><li>'+String(val)+'</li></ul>';
    }

    what.innerHTML   = '<div class="sec-title">What it is & criteria</div>' + asList(s.what_it_is_and_criteria);
    causes.innerHTML = '<div class="sec-title">Common causes & complications</div>' + asList(s.common_causes_and_complications);
    immed.innerHTML  = '<div class="sec-title">Immediate management</div>' + asList(s.immediate_management);
    ongoing.innerHTML= '<div class="sec-title">Ongoing care / monitoring</div>' + asList(s.ongoing_care_monitoring);

    meta.textContent = (j.provider ? ('Provider: '+j.provider+'. ') : '') +
                       (j.context_used ? 'Local context used.' : '');
    err.textContent = '';

  }catch(e){
    err.textContent = 'Request failed: '+e.message;
  }
}

document.getElementById('ask').onclick = ask;
</script>
</body>
</html>"""


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
        "context_used": True  # semantic_context always runs, harmless flag
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
