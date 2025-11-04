from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# ---- Try to load OpenAI, fail safe ----
try:
    from openai import OpenAI
except:
    OpenAI = None

APP = Flask(__name__)
CORS(APP, resources={r"/answer": {"origins": "*"}, r"/health": {"origins": "*"}})

# ---- Load API key safely ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY) else None

# ---- Basic UI ----
@APP.route("/", methods=["GET"])
def ui():
    return """
<!doctype html><meta charset="utf-8"><title>Personal Assistant</title>
<style>
body{font-family:system-ui,Arial,sans-serif;max-width:680px;margin:24px auto;padding:0 16px}
textarea{width:100%;height:120px}
button{width:100%;padding:12px;margin-top:8px;font-size:16px}
#ans{white-space:pre-wrap;background:#f6f8fa;padding:14px;border-radius:10px;margin-top:16px}
</style>
<h2>Personal Assistant</h2>
<textarea id="q" placeholder="Ask anything (clinical, admin, planning)…"></textarea>
<button id="ask">Ask</button>
<div id="ans"></div>
<script>
async function ask(){
  const q=document.getElementById('q').value.trim();
  if(!q) return;
  document.getElementById('ans').textContent='Thinking...';
  const r = await fetch('/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
  const j = await r.json();
  document.getElementById('ans').textContent = j.answer || j.error || "No reply";
}
document.getElementById('ask').onclick = ask;
</script>
"""

@APP.route("/health")
def health():
    return jsonify({"ok": True})

# ---- Prompt logic ----
def build_prompt(q: str):
    return f"You are a helpful Australian medical & life assistant.\nUser question: {q}"

# ---- Answer endpoint ----
@APP.route("/answer", methods=["POST"])
def answer():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()

    if not q:
        return jsonify({"ok": False, "error": "Missing question"}), 400

    # If OpenAI configured, call it
    if client:
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":build_prompt(q)}],
                temperature=0.2,
            )
            out = r.choices[0].message.content.strip()
            return jsonify({"ok": True, "answer": out})
        except Exception as e:
            # DO NOT crash on quota/timeout/key failure
            return jsonify({"ok": False, "error": f"AI unavailable: {e}"}), 200

    # No API key
    return jsonify({
        "ok": True,
        "answer": "✅ Backend online.\n❗ Add OPENAI_API_KEY in Render → Environment and redeploy."
    })

if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
