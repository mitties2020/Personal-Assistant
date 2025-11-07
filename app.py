# app.py
import os
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv

# ---------------- CONFIG ----------------
# 1️⃣  Create a file called ".env" in the same folder as this app.py
#     and put this inside (replace with your own key):
#     OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
#
# 2️⃣  Then run:
#     pip install flask openai python-dotenv
# ----------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("❌ Missing OPENAI_API_KEY in .env file")

client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__, template_folder="templates")

# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/clinical-qa", methods=["POST"])
def clinical_qa():
    """
    Receives a JSON object: {"question": "text here"}
    Sends it to OpenAI/DeepSeek and returns structured answer JSON.
    """
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No clinical question received."}), 400

    system_prompt = (
        "You are a structured, evidence-based clinical Q&A assistant. "
        "Provide concise, high-yield answers following Australian clinical standards. "
        "Sections: Overview, Assessment, Risk Stratification, Management, Monitoring, Key References. "
        "Include a disclaimer: 'Always verify with local policies and senior review.'"
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # or deepseek-chat if using DeepSeek/OpenRouter
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
            max_tokens=900,
        )

        answer = completion.choices[0].message.content.strip()
        return jsonify({"answer": answer})

    except Exception as e:
        print("⚠️ Error:", e)
        return jsonify({"error": str(e)}), 500

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
