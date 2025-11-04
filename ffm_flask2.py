from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# ---- DeepSeek client ----
try:
    from openai import OpenAI
except:
    OpenAI = None

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
client = None
if OpenAI and DEEPSEEK_API_KEY:
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# ---- Flask App ----
APP = Flask(__name__)
CORS(APP, resources={r"/answer": {"origins": "*"}, r"/health": {"origins": "*"}})

@APP.route("/")
def home():
    return """
    <h2>Aussie ED Assistant (DeepSeek)</h2>
    <p>POST to /answer with question</p>
    """

@APP.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

def build_prompt(q):
    return f"""
You are an Australian Emergency Medicine clinical assistant.

Produce ~150 words with these sections:

1) What it is & criteria
2) Causes / complications
3) Immediate management (with doses)
4) Ongoing care / monitoring

Use Australian guidelines if applicable.
Question: {q}
"""

@APP.route("/answer", methods=["POST"])
def answer():
    if not client:
        return jsonify({"ok": False, "error": "DeepSeek API key missing"}), 200

    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "Missing question"}), 400

    try:
        res = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": build_prompt(q)}],
            temperature=0.2,
        )
        answer = res.choices[0].message.content
        return jsonify({"ok": True, "answer": answer})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 200

if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
