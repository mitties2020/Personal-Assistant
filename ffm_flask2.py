from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
import os
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

APP = Flask(__name__)
CORS(APP, resources={r"/answer": {"origins": "*"}, r"/health": {"origins": "*"}})

# --- Optional OpenAI (won’t crash if missing) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
oa_client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY) else None

@APP.route("/", methods=["GET"])
def root():
    # tiny inline UI so you always get *something* at /
    html = """<!doctype html><meta charset="utf-8"><title>Assistant</title>
<style>body{font-family:system-ui,Arial,sans-serif;max-width:680px;margin:24px auto;padding:0 16px}
textarea{width:100%;height:120px}button{width:100%;padding:12px;margin-top:8px;font-size:16px}
#ans{white-space:pre-wrap;background:#f6f8fa;padding:14px;border-radius:10px;margin-top:16px}</style>
<h2>ED Clinical Assistant</h2>
<textarea id="q" placeholder="e.g., Hyperkalaemia with broad QRS: immediate management"></textarea>
<button id="ask">Answer</button><div id="ans"></div>
<script>
document.getElementById('ask').onclick = async () => {
  const q = document.getElementById('q').value.trim(); if(!q) return;
  document.getElementById('ans').textContent = 'Working…';
  const r = await fetch('/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
  const j = await r.json();
  document.getElementById('ans').textContent = j.answer || j.error || 'No response';
};
</script>"""
    return html

@APP.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

def build_prompt(q: str) -> str:
    return f"""
You are a clinical assistant for an Australian ED doctor. Produce a succinct response (~180 words)
with exactly these sections:
1) What it is & criteria
2) Common causes & complications
3) Immediate management (first-line actions & doses)
4) Ongoing care / monitoring
Be specific with adult doses/units/routes (e.g., adrenaline 0.5 mg IM; Ca gluconate 10% 10 mL IV over 2–5 min).
Question: {q}
"""

@APP.route("/answer", methods=["POST"])
def answer():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "Missing 'question'"}), 400

    if oa_client:
        try:
            resp = oa_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": build_prompt(q)}],
                temperature=0.2,
            )
            text = (resp.choices[0].message.content or "").strip()
            return jsonify({"ok": True, "answer": text, "sources": []})
        except Exception as e:
            # You’ll see quota errors here but the server will still run
            return jsonify({"ok": False, "error": f"OpenAI error: {e}"}), 200

    # No key / client — still return something so UI works
    return jsonify({"ok": True, "answer": "Backend up. Add OPENAI_API_KEY in Render → Environment and redeploy.", "sources": []})

if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
