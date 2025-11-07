import os
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

"""
HOW TO CONFIGURE:

1. Install deps:
   pip install flask openai python-dotenv

2. Create a .env file (same folder as app.py) with:
   OPENAI_API_KEY=your_real_key_here

   If you're using OpenRouter or DeepSeek (OpenAI-compatible), also set:
   OPENAI_BASE_URL=https://api.openrouter.ai/v1        # or your DeepSeek/OpenRouter endpoint
   OPENAI_MODEL=deepseek-chat                          # or your chosen model

3. By default this uses OpenAI official endpoint + GPT-5-mini (or your chosen model).
"""

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1-mini")  # change if you like

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment/.env")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
)

app = Flask(__name__, template_folder="templates", static_folder="static")

# -----------------------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    # Renders templates/index.html (see instructions below)
    return render_template("index.html")


@app.route("/api/clinical-qa", methods=["POST"])
def clinical_qa():
    """
    Expects JSON: { "question": "your clinical question" }
    Returns JSON: { "answer": "structured text..." }
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Missing 'question' in request body."}), 400

    # System prompt keeps things clinical, structured, and safe.
    system_prompt = """
You are a clinical decision-support assistant for doctors.
Provide concise, high-yield answers using current evidence and major guidelines.
Always:
- Use clear headings.
- Cover: Overview, Assessment, Risk Stratification, Management, Monitoring, Red Flags.
- Highlight dose ranges, but never guess if uncertain.
- Assume practice in Australia unless otherwise stated.
- Include a short, high-yield reference list (guidelines / major trials).
- Add a strong disclaimer at the end that this is not a substitute for local policies or senior review.
Do NOT fabricate guidelines or references.
"""

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Clinical question: {question}",
                },
            ],
            temperature=0.2,
            max_tokens=900,
        )

        answer = completion.choices[0].message.content.strip()
        return jsonify({"answer": answer})

    except Exception as e:
        # Log server-side and return safe message to frontend
        print("Error in /api/clinical-qa:", repr(e), flush=True)
        return (
            jsonify(
                {
                    "error": "Failed to generate answer from model.",
                    "details": str(e),
                }
            ),
            500,
        )


# -----------------------------------------------------------------------------
# ENTRYPOINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
