import os
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

if not API_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY in .env")

app = Flask(__name__, template_folder="templates")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/clinical-qa", methods=["POST"])
def clinical_qa():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "No clinical question received."}), 400

    system_prompt = (
        "You are a clinical question-to-answer engine for doctors in Australia. "
        "Return a single, structured, high-yield answer per query.\n\n"
        "Format your response with clear markdown headings:\n"
        "1. Overview\n"
        "2. Assessment\n"
        "3. Risk Stratification\n"
        "4. Management (stepwise, include first-line, second-line, doses if standard)\n"
        "5. Monitoring & Follow-up\n"
        "6. Red Flags / When to escalate\n"
        "7. Key References (guidelines / major trials only)\n\n"
        "Rules:\n"
        "- Base answers on accepted evidence & guidelines.\n"
        "- If uncertain, say so; do NOT fabricate data.\n"
        "- Assume adult patients unless specified.\n"
        "- Assume Australian context if relevant.\n"
        "- End every answer with: 'Always verify with local policies, product information, and senior review before acting.'"
    )

    try:
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "temperature": 0.3,
            "max_tokens": 900,
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=60)

        if resp.status_code != 200:
            return jsonify({
                "error": "DeepSeek API error",
                "status": resp.status_code,
                "body": resp.text,
            }), 500

        data = resp.json()
        answer = data["choices"][0]["message"]["content"].strip()
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": "Server exception", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
