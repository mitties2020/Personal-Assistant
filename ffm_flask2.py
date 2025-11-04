from flask import Flask, request, jsonify
from flask_cors import CORS
import os, requests

APP = Flask(__name__)
CORS(APP, resources={r"/*": {"origins": "*"}})

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()

def call_deepseek(prompt: str):
    if not DEEPSEEK_API_KEY:
        return "⚠️ DeepSeek API key not set in Render environment."

    try:
        r = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            },
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=30
        )
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ DeepSeek API Error: {e}"

def build_prompt(q: str) -> str:
    return f"""
You are an Australian Emergency Medicine assistant.

Format the response in markdown with headings:

### What it is / Diagnostic criteria
### Causes & risks
### Immediate ED management (with adult doses)
### Ongoing care and monitoring

Give exact Australian ED doses (e.g., 0.5mg IM adrenaline, 10mL Ca gluconate IV).
Be concise, structured, and clinically safe.

Question: {q}
"""

@APP.route("/")
def home():
    return """
<!doctype html><meta charset="utf-8">
<title>Personal Assistant</title>
<style>
body{font-family:system-ui;max-width:650px;margin:40px auto;padding:0 16px}
textarea{width:100%;height:120px;padding:8px}
button{padding:10px;font-size:16px;width:100%;margin-top:8px}
#ans{white-space:pre-wrap;background:#f6f8fa;padding:14px;border-radius:10px;margin-top:16px}
</style>
<h2>ED Clinical Assistant</h2>
<textarea id="q" placeholder="e.g. Anaphylaxis adult immediate management"></textarea>
<button onclick="ask()">Answer</button>
<div id="ans"></div>
<script>
async function ask(){
  let q=document.getElementById('q').value.trim();
  if(!q) return;
  document.getElementById('ans').textContent='Working…';
  let r=await fetch('/answer',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question:q})});
  let j=await r.json();
  document.getElementById('ans').innerHTML = j.answer || j.error;
}
</script>
"""

@APP.route("/answer", methods=["POST"])
def answer():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "Missing question"}), 400

    prompt = build_prompt(q)
    response = call_deepseek(prompt)

    return jsonify({"ok": True, "answer": response})

@APP.route("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
