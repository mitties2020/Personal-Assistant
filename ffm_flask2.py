from flask import Flask, request, jsonify
from flask_cors import CORS
import os

try:
    from openai import OpenAI
except:
    OpenAI = None

APP = Flask(__name__)
CORS(APP, resources={r"/answer": {"origins": "*"}, r"/health": {"origins": "*"}})

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()

# DeepSeek client (using OpenAI SDK)
ds_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
) if (OpenAI and DEEPSEEK_API_KEY) else None

@APP.route("/", methods=["GET"])
def UI():
    return """
    <h2>Personal Assistant</h2>
    <textarea id='q' style='width:100%;height:100px'></textarea><br>
    <button onclick='ask()'>Ask</button>
    <pre id='a'></pre>
    <script>
    async function ask() {
        let q = document.getElementById('q').value;
        let r = await fetch('/answer', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question:q})});
        let j = await r.json();
        document.getElementById('a').textContent = j.answer || j.error;
    }
    </script>
    """

@APP.route("/health")
def health():
    return jsonify({"ok": True})

@APP.route("/answer", methods=["POST"])
def answer():
    data = request.get_json() or {}
    q = (data.get("question") or "").strip()

    if not q:
        return jsonify({"ok": False, "error": "Missing question"}), 400

    if not ds_client:
        return jsonify({"ok": False, "error": "DeepSeek not configured"}), 200

    try:
        res = ds_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a concise medical assistant for ED doctors in Australia."},
                {"role": "user", "content": q}
            ]
        )
        text = res.choices[0].message.content.strip()
        return jsonify({"ok": True, "answer": text})

    except Exception as e:
        return jsonify({"ok": False, "error": f"DeepSeek error: {e}"}), 200

if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
