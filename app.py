from flask import Flask, request, jsonify
from flask_cors import CORS
import os, chardet
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional DeepSeek/OpenAI client
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

APP = Flask(__name__)
CORS(APP, resources={r"/answer": {"origins": "*"}, r"/health": {"origins": "*"}, r"/reindex": {"origins": "*"}})

# --- AI setup ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
deepseek_client = OpenAI(base_url="https://api.deepseek.com", api_key=DEEPSEEK_API_KEY) if (OpenAI and DEEPSEEK_API_KEY) else None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
openai_client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY and not deepseek_client) else None

# --- Text-only corpus setup ---
DATA_DIR = Path("clinical_data")
CHUNKS, META, VECTORIZER, MATRIX = [], [], None, None


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # ensure at least .gitkeep exists
    if not any(DATA_DIR.iterdir()):
        (DATA_DIR / ".gitkeep").touch()


def _read_text_file(p: Path) -> str:
    raw = p.read_bytes()
    enc = chardet.detect(raw).get("encoding") or "utf-8"
    try:
        return raw.decode(enc, errors="ignore")
    except Exception:
        return raw.decode("utf-8", errors="ignore")


def build_corpus():
    global CHUNKS, META, VECTORIZER, MATRIX
    ensure_dirs()
    CHUNKS, META = [], []

    for path in DATA_DIR.rglob("*.txt"):
        txt = _read_text_file(path)
        if not txt.strip():
            continue
        # Break long docs into 1k-char chunks
        i = 0
        while i < len(txt):
            chunk = txt[i:i + 1000].strip()
            if chunk:
                CHUNKS.append(chunk)
                META.append({"path": str(path), "offset": i})
            i += 800

    if not CHUNKS:
        return

    VECTORIZER = TfidfVectorizer(stop_words="english", max_features=50000)
    MATRIX = VECTORIZER.fit_transform(CHUNKS)


def semantic_context(query: str, k=5) -> str:
    if not (VECTORIZER and MATRIX is not None and CHUNKS):
        return "(no local context indexed)"
    qv = VECTORIZER.transform([query])
    sims = cosine_similarity(qv, MATRIX)[0]
    top = sims.argsort()[::-1][:k]
    pieces = []
    for i in top:
        meta = META[i]
        score = sims[i]
        pieces.append(f"[{meta['path']} | {score:.2f}]\n{CHUNKS[i]}")
    return "\n\n---\n\n".join(pieces)


# Build on startup
build_corpus()


def build_prompt(question: str) -> str:
    ctx = semantic_context(question)
    return f"""
You are a concise, evidence-based clinical assistant for an Australian ED doctor.
Answer the QUESTION using CONTEXT if relevant, otherwise general knowledge.
Respond strictly in JSON with the following keys:
"what_it_is_and_criteria", "common_causes_and_complications", "immediate_management", "ongoing_care_monitoring".
Each field should contain short, factual bullet points.
Ensure adult doses use standard units (e.g., Adrenaline 0.5 mg IM 1:1000 lateral thigh).

CONTEXT:
{ctx}

QUESTION: {question}
""".strip()


def call_ai(prompt):
    if deepseek_client:
        resp = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
        )
        return resp.choices[0].message.content.strip(), "deepseek"
    elif openai_client:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
        )
        return resp.choices[0].message.content.strip(), "openai"
    else:
        raise RuntimeError("No AI configured (set DEEPSEEK_API_KEY or OPENAI_API_KEY)")


@APP.route("/", methods=["GET"])
def root():
    html = """<!doctype html><meta charset="utf-8">
<title>Personal Assistant</title>
<style>
body{font-family:system-ui,Arial,sans-serif;max-width:700px;margin:24px auto;padding:0 16px}
textarea{width:100%;height:100px;padding:8px;border-radius:8px;border:1px solid #ccc}
button{margin-top:8px;width:100%;padding:10px;background:#222;color:#fff;border:none;border-radius:8px;font-size:16px}
#out{margin-top:16px;white-space:pre-wrap;background:#f9fafb;padding:10px;border-radius:10px;font-size:14px}
</style>
<h2>Personal Assistant</h2>
<textarea id="q" placeholder="Ask e.g. 'Anaphylaxis adult: immediate and ongoing management'"></textarea>
<button id="ask">Ask</button>
<div id="out"></div>
<script>
async function ask(){
  const q=document.getElementById('q').value.trim();
  const out=document.getElementById('out');
  if(!q)return;
  out.textContent='Working...';
  const r=await fetch('/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
  const j=await r.json();
  out.textContent = j.structured ? 
    JSON.stringify(j.structured,null,2) : (j.answer||j.error||'No response');
}
document.getElementById('ask').onclick=ask;
</script>"""
    return html


@APP.route("/answer", methods=["POST"])
def answer():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "Missing 'question'"}), 400

    prompt = build_prompt(q)
    try:
        raw, provider = call_ai(prompt)
        return jsonify({"ok": True, "provider": provider, "structured": raw})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 200


@APP.route("/reindex", methods=["POST", "GET"])
def reindex():
    build_corpus()
    return jsonify({"ok": True, "files_indexed": len(CHUNKS)})


@APP.route("/health")
def health():
    return jsonify({"ok": True, "files_indexed": len(CHUNKS)})


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
