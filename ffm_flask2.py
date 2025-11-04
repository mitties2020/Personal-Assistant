# filename: ffm_flask2.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from openai import OpenAI  # SDK works for DeepSeek too

APP = Flask(__name__)
CORS(APP, resources={r"/answer": {"origins": "*"}, r"/health": {"origins": "*"}, r"/whoami": {"origins": "*"}})

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
ds_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com") if DEEPSEEK_API_KEY else None

@APP.route("/health")
def health():
    return jsonify({"ok": True})

@APP.route("/whoami")
def whoami():
    return jsonify({
        "provider": "deepseek" if ds_client else "none",
        "base_url": "https://api.deepseek.com" if ds_client else None,
        "has_OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY"))
    })

PROMPT = """You are an Australian ED clinical assistant. Answer succinctly (~180 words)
with sections:
1) What it is & criteria
2) Causes/complications
3) Immediate management (adult doses/routes/units)
4) Ongoing care / monitoring
Question: {q}
"""

@APP.route("/", methods=["GET"])
def root():
    return """<!doctype html><meta charset="utf-8"><title>Personal Assistant</title>
<style>body{font-family:system-ui,Arial,sans-serif;max-width:680px;margin:24px auto;padding:0 16px}
textarea{width:100%;height:120px}button{width:100%;padding:12px;margin-top:8px;font-size:16px}
#ans{white-space:pre-wrap;background:#f6f8fa;padding:14px;border-radius:10px;margin-top:16px}</style>
<h2>Personal Assistant</h2>
<textarea id="q" placeholder="e.g., Hyperkalaemia with broad QRS: immediate management"></textarea>
<button id="ask">Ask</button><div id="ans"></div>
<script>
document.getElementById('ask').onclick = async () => {
  const q = document.getElementById('q').value.trim(); if(!q) return;
  document.getElementById('ans').textContent = 'Working…';
  const r = await fetch('/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
  const j = await r.json();
  document.getElementById('ans').textContent = j.answer || j.error || 'No response';
};
</script>"""

@APP.route("/answer", methods=["POST"])
def answer():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "Missing 'question'"}), 400
    if not ds_client:
        return jsonify({"ok": False, "error": "DeepSeek not configured. Set DEEPSEEK_API_KEY and redeploy."}), 200
    try:
        r = ds_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": PROMPT.format(q=q)}],
            temperature=0.2,
        )
        text = (r.choices[0].message.content or "").strip()
        return jsonify({"ok": True, "answer": text, "sources": []})
    except Exception as e:
        return jsonify({"ok": False, "error": f"AI unavailable: {e}"}), 200

if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
