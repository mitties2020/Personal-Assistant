import os, re, json, uuid, shutil
from flask import Flask, request, jsonify, Response
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID, DATETIME
from whoosh.qparser import QueryParser
from datetime import datetime
import pypdf

APP = Flask(__name__)

# ✅ Allow your website to call this server
try:
    from flask_cors import CORS
    CORS(APP, resources={r"/*": {"origins": "*"}})
except:
    pass

# ✅ Where we store indexed medical guidelines
DATA_DIR = "data/indexdir"
os.makedirs(DATA_DIR, exist_ok=True)

schema = Schema(
    id=ID(stored=True, unique=True),
    title=TEXT(stored=True),
    org=TEXT(stored=True),
    published=DATETIME(stored=True),
    chunk=TEXT(stored=True)
)

if not os.listdir(DATA_DIR):
    ix = create_in(DATA_DIR, schema)
else:
    ix = open_dir(DATA_DIR)

ADMIN_KEY = os.environ.get("ADMIN_KEY", "changeme")

def pdf_to_chunks(file):
    reader = pypdf.PdfReader(file)
    chunks = []
    for page in reader.pages:
        text = page.extract_text() or ""
        for chunk in text.split("\n"):
            if len(chunk.strip()) > 30:
                chunks.append(chunk.strip())
    return chunks

def score_sentence(s):
    s2 = s.lower()
    score = 0
    if re.search(r"(immediate|urgent|stat|airway|breathing|circulation|iv|mg|mcg|dose|ampule|adrenaline|insulin|calcium|hyperkalaemia|anaphylaxis)", s2):
        score += 3
    return score

def format_answer(chunks):
    if not chunks:
        return "<p>No matches found in your local index.</p>"
    chunks.sort(key=lambda x: score_sentence(x), reverse=True)
    best = chunks[:10]
    html = "<ul>"
    for c in best:
        html += f"<li>{c}</li>"
    html += "</ul>"
    return html

@APP.route("/health")
def health():
    return jsonify({"ok": True})

# ✅ UI FOR DOCTORS
@APP.route("/ui")
def ui():
    html = """<!doctype html><meta charset="utf-8">
<title>Clinical Assistant</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{font-family:system-ui}
body{margin:20px;max-width:900px}
textarea{width:100%;min-height:110px;padding:12px}
button{padding:10px 14px;background:#0ea5e9;color:#fff;border:0;border-radius:8px;margin-top:10px}
.card{margin-top:12px;padding:12px;border:1px solid #ddd;border-radius:8px}
.src{font-size:12px;color:#444}
</style>
<h2>Clinical Assistant</h2>
<p>Fast guideline answer generator</p>
<textarea id="q" placeholder="Hyperkalaemia: immediate management"></textarea>
<button id="ask">Run</button>
<div id="ans" class="card" style="display:none">
  <h3>Answer</h3>
  <div id="answer"></div>
  <div id="srcs"></div>
</div>
<script>
const $=s=>document.querySelector(s);
$("#ask").onclick=async()=>{
  const q=$("#q").value.trim(); if(!q)return;
  $("#ans").style.display="block";
  $("#answer").textContent="Working...";
  const r=await fetch("/answer",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:q})});
  const j=await r.json();
  $("#answer").innerHTML=j.answer;
  const s=(j.sources||[]).map(x=>`<div class='src'>• ${x.title} (${x.org})</div>`).join("")
  $("#srcs").innerHTML=s;
}
</script>"""
    return Response(html, mimetype="text/html")

# ✅ Ingest new medical PDFs
@APP.route("/ingest", methods=["POST"])
def ingest():
    key = request.headers.get("X-ADMIN-KEY")
    if key != ADMIN_KEY:
        return jsonify({"error":"unauthorized"}), 403
    
    file = request.files.get("file")
    if not file:
        return jsonify({"error":"No file"}), 400

    title = request.form.get("title", "Unknown")
    org = request.form.get("org", "Unknown")
    pub = request.form.get("published", "2025-01-01")
    published_dt = datetime.fromisoformat(pub)

    chunks = pdf_to_chunks(file)
    writer = ix.writer()
    for c in chunks:
        writer.add_document(
            id=str(uuid.uuid4()),
            title=title,
            org=org,
            published=published_dt,
            chunk=c
        )
    writer.commit()

    return jsonify({"ok":True,"chunks":len(chunks)})

# ✅ Answer questions from guideline database
@APP.route("/answer", methods=["POST"])
def answer():
    q = request.json.get("question","")
    s = ix.searcher()
    qp = QueryParser("chunk", ix.schema)
    q2 = qp.parse(q)
    results = s.search(q2, limit=50)
    chunks = [r["chunk"] for r in results]
    answer_html = format_answer(chunks)

    sources = []
    for r in results[:5]:
        sources.append({"title":r["title"],"org":r["org"],"published":str(r["published"])})

    return jsonify({"ok":True,"answer":answer_html,"sources":sources})

if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=8000)
