from flask import Flask, request, jsonify
from flask_cors import CORS
import os, io, zipfile, chardet
from pathlib import Path

# Optional LLM (DeepSeek via OpenAI SDK; safe if not configured)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
client = OpenAI(base_url="https://api.deepseek.com",
                api_key=DEEPSEEK_API_KEY) if (OpenAI and DEEPSEEK_API_KEY) else None

# Optional fallback (OpenAI)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
oa_client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY and not client) else None

APP = Flask(__name__)
CORS(APP, resources={
    r"/answer": {"origins": "*"},
    r"/health": {"origins": "*"},
    r"/reindex": {"origins": "*"},
})

# ---------- Corpus setup ----------
DATA_DIR = Path("clinical_data")
UNZIP_DIR = DATA_DIR / "_unzipped"
TEXT_CACHE = DATA_DIR / "_text_cache"
CORPUS = {}

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for d in (UNZIP_DIR, TEXT_CACHE):
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
    if not root.exists(): return
    for path in root.rglob("*"):
        if path.is_dir(): continue
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
        if text.strip():
            CORPUS[str(fp)] = text

build_corpus()

# ---------- Modern UI ----------
@APP.route("/", methods=["GET"])
def root():
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Clinical Q&A Engine</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root {
  --bg:#030712;
  --accent:#38bdf8;
  --accent-soft:rgba(56,189,248,0.12);
  --text:#e5e7eb;
  --muted:#9ca3af;
  --radius:18px;
  --font:system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;
}
body {
  margin:0; min-height:100vh;
  display:flex; align-items:center; justify-content:center;
  background:radial-gradient(circle at top,var(--accent-soft),var(--bg));
  font-family:var(--font); color:var(--text);
  padding:20px;
}
.wrap {
  width:100%; max-width:720px;
  background:linear-gradient(to bottom right,rgba(15,23,42,0.98),rgba(2,6,23,1));
  border-radius:24px; padding:26px 24px 22px;
  box-shadow:0 18px 55px rgba(15,23,42,0.9);
  border:1px solid rgba(56,189,248,0.22);
  backdrop-filter:blur(14px);
}
.pill {
  padding:5px 12px; border-radius:999px;
  font-size:11px; font-weight:500; text-transform:uppercase;
  color:var(--accent); background:var(--accent-soft);
  border:1px solid rgba(56,189,248,0.4);
  letter-spacing:0.12em;
  display:inline-block;
}
h1 {
  font-size:32px; font-weight:700;
  margin:16px 0 6px; letter-spacing:0.01em;
  color:var(--text);
}
label {
  font-size:12px; color:var(--muted);
  text-transform:uppercase; letter-spacing:0.08em;
  margin-top:12px; display:block;
}
textarea {
  width:100%; min-height:90px; resize:vertical;
  padding:12px 14px; border-radius:var(--radius);
  border:1px solid rgba(148,163,253,0.25);
  background:rgba(2,6,23,0.96); color:var(--text);
  font-size:14px; outline:none;
}
textarea:focus {
  border-color:var(--accent);
  box-shadow:0 0 0 1px rgba(56,189,248,0.35);
}
button {
  margin-top:10px; width:100%; padding:12px;
  border:none; border-radius:var(--radius);
  background:linear-gradient(to right,var(--accent),#6366f1);
  color:#020817; font-weight:600; font-size:13px;
  letter-spacing:0.16em; text-transform:uppercase;
  cursor:pointer; box-shadow:0 12px 32px rgba(37,99,235,0.5);
}
button:active { transform:translateY(1px); }
.answer {
  margin-top:12px; padding:14px 16px;
  border-radius:var(--radius);
  background:rgba(2,6,23,0.98);
  border:1px solid rgba(75,85,99,0.7);
  font-size:15px; line-height:1.65; white-space:pre-wrap;
  max-height:300px; overflow-y:auto;
  font-family:'Segoe UI',Roboto,system-ui,sans-serif;
}
.answer.empty { color:#6b7280; border-style:dashed; }
.meta {
  margin-top:8px; font-size:11px; color:var(--muted);
  display:flex; justify-content:space-between;
}
.status-dot {
  width:7px; height:7px; border-radius:50%;
  background:#22c55e; box-shadow:0 0 8px #22c55e;
  display:inline-block; margin-right:4px;
}
</style>
</head>
<body>
  <div class="wrap">
    <div class="pill">Clinical Q&A Engine</div>
    <h1>Name your clinical question</h1>
    <label for="q">Clinical question</label>
    <textarea id="q" placeholder="e.g. Hyperkalaemia with broad QRS: immediate management"></textarea>
    <button id="ask">Get Answer</button>
    <div id="ans" class="answer empty">Your structured answer will appear here.</div>
    <div class="meta">
      <div><span class="status-dot"></span>Backend online</div>
      <div id="provider">Model: local data / API hybrid</div>
    </div>
  </div>
<script>
const qEl=document.getElementById('q');
const ansEl=document.getElementById('ans');
const providerEl=document.getElementById('provider');
const btn=document.getElementById('ask');
async function ask(){
 const q=qEl.value.trim(); if(!q)return;
 ansEl.classList.remove('empty');
 ansEl.textContent="Thinking...";
 providerEl.textContent="Model: resolving…";
 try{
   const res=await fetch('/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
   const j=await res.json();
   if(j.ok&&j.answer){ansEl.textContent=j.answer;}else{ansEl.textContent=j.error||"No response.";}
   providerEl.textContent="Model: "+(j.provider||'none');
 }catch(e){
   ansEl.textContent="Network or server error.";
   providerEl.textContent="Model: error";
 }}
btn.addEventListener('click',ask);
qEl.addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.metaKey||e.ctrlKey)){e.preventDefault();ask();}});
</script>
</body>
</html>
"""

# ---------- Backend ----------
@APP.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "files_indexed": len(CORPUS)})

@APP.route("/reindex", methods=["POST", "GET"])
def reindex():
    build_corpus()
    return jsonify({"ok": True, "files_indexed": len(CORPUS)})

def build_prompt(q: str) -> str:
    qlow = q.lower()
    hits = []
    for path, text in CORPUS.items():
        if any(k in text.lower() for k in qlow.split()[:5]):
            hits.append((path, text[:4000]))
        if len(hits) >= 6: break
    context = "\n\n---\n\n".join(f"[[{p}]]\n{text}" for p, text in hits) or "(no local context matched)"
    return f"""You are a concise clinical assistant for an Australian ED doctor.
Use the CONTEXT if relevant, otherwise answer from your usual knowledge.
Keep it ~180 words with these sections:
1) What it is & criteria
2) Causes & complications
3) Immediate management (adult doses, units, routes)
4) Ongoing care / monitoring

CONTEXT:
{context}

QUESTION: {q}
"""

def call_deepseek(prompt: str) -> str:
    if not client: raise RuntimeError("DeepSeek not configured.")
    r = client.chat.completions.create(model="deepseek-chat",
                                       messages=[{"role":"user","content":prompt}],
                                       temperature=0.2)
    return (r.choices[0].message.content or "").strip()

def call_openai(prompt: str) -> str:
    if not oa_client: raise RuntimeError("OpenAI not configured.")
    r = oa_client.chat.completions.create(model="gpt-4o-mini",
                                          messages=[{"role":"user","content":prompt}],
                                          temperature=0.2)
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
            return jsonify({"ok": True, "provider": "none",
                            "answer": "Backend running.\nSet DEEPSEEK_API_KEY or OPENAI_API_KEY in Render → Environment."})
    except Exception as e:
        return jsonify({"ok": False, "error": f"AI unavailable: {e}"}), 200

if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
