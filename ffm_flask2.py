from flask import Flask, request, jsonify, redirect, send_from_directory
from flask_cors import CORS
import os
from openai import OpenAI

# --- Flask App ---
APP = Flask(__name__, static_folder="static")
CORS(APP)

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    print("⚠️  No OpenAI key loaded! Set OPENAI_API_KEY in Render.")
client = OpenAI(api_key=OPENAI_API_KEY)


# ✅ Health check
@APP.route("/health")
def health():
    return jsonify({"ok": True})


# ✅ Serve UI
@APP.route("/")
def root():
    return redirect("/ui")

@APP.route("/ui")
def ui():
    return send_from_directory("static", "index.html")


# ✅ Build prompt
def build_prompt(q):
    return f"""
You are a clinical assistant for an Australian ED doctor. Provide a concise emergency management answer with:

1) What it is + diagnostic criteria
2) Common causes & complications
3) Immediate ED management (doses/routes)
4) Ongoing care / monitoring / red flags

Be accurate and reference Australian practice (e.g., ASCIA, NSW Health, WA ED pathways) when equivalent.

Question: {q}
"""


# ✅ Main API endpoint
@APP.route("/answer", methods=["POST"])
def answer():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()

    if not q:
        return jsonify({"ok": False, "error": "Missing question"}), 400

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": build_prompt(q)}],
            temperature=0.2,
        )
        return jsonify({
            "ok": True,
            "answer": resp.choices[0].message.content.strip(),
            "sources": []
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ✅ Run
if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
