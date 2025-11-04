# ffm_flask2.py  — minimal API + UI
from flask import Flask, request, jsonify
import os, json

APP = Flask(__name__)

ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")

@APP.after_request
def _cors(r):
    r.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    r.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return r

# --- HEALTH ---
@APP.route("/health", methods=["GET"])
def health():
    return jsonify(ok=True)

# --- ANSWER (dummy example that echoes the question; your real logic can live here) ---
@APP.route("/answer", methods=["POST"])
def answer():
    data = request.get_json(force=True) or {}
    q = (data.get("question") or "").strip()
    # TODO: replace with your search/summarise function
    if not q:
        return jsonify(ok=True, answer="<p>Please enter a question.</p>", sources=[])
    html = f"""
      <h3 style='margin:6px 0 2px'>Definition / criteria</h3>
      <ul style='margin:6px 0 10px 18px'><li>Stub answer for: {q}</li></ul>
      <h3 style='margin:6px 0 2px'>Causes / complications</h3>
      <ul style='margin:6px 0 10px 18px'><li>Example cause</li></ul>
      <h3 style='margin:6px 0 2px'>Immediate management</h3>
      <ul style='margin:6px 0 10px 18px'><li>Example first-line step</li></ul>
      <h3 style='margin:6px 0 2px'>Ongoing care / details</h3>
      <ul style='margin:6px 0 10px 18px'><li>Example monitoring</li></ul>
    """
    return jsonify(ok=True, answer=html, sources=[])

# --- UI PAGE ---
@APP.route("/", methods=["GET"])
def home():
    return """
<!doctype html><meta charset="utf-8">
<title>Clinical Assistant</title>
<div style="max-width:760px;margin:auto;padding:20px;border:1px solid #e5e7eb;border-radius:14px;background:#fafafa;font-family:system-ui">
  <h2 style="margin:0 0 8px">Clinical Assistant</h2>
  <p style="margin:0 0 12px;color:#6b7280">Ask a focused question (e.g. “Hyperkalaemia — criteria + immediate management”).</p>
  <textarea id="q" style="width:100%;height:96px;padding:12px;border-radius:10px;border:1px solid #d1d5db;font-size:15px"></textarea>
  <button id="ask" style="margin-top:10px;padding:10px 16px;border:0;border-radius:10px;background:#2563eb;color:#fff;font-size:15px;cursor:pointer">Get answer</button>
  <div id="ans" style="display:none;margin-top:16px">
    <div id="answer" style="font-size:16px;line-height:1.55"></div>
    <div id="srcs" style="margin-top:10px;color:#6b7280;font-size:13px"></div>
  </div>
</div>
<script>
async function go(){
  const q = document.getElementById("q").value.trim();
  if(!q) return;
  const box = document.getElementById("ans");
  const body = document.getElementById("answer");
  const srcs = document.getElementById("srcs");
  box.style.display="block"; body.textContent="Working…"; srcs.innerHTML="";
  try {
    const r = await fetch("/answer", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({question:q, k:12})});
    const j = await r.json();
    body.innerHTML = j.answer || "No matches.";
    srcs.innerHTML = (j.sources||[]).map(s =>
      `• ${s.title||''} — ${s.org||''} ${s.published?('('+s.published.slice(0,10)+')'):''}`
    ).join("<br>");
  } catch(e){ body.textContent = "Error: "+e.message; }
}
document.getElementById("ask").onclick = go;
</script>
""", 200, {"Content-Type":"text/html; charset=utf-8"}

if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
