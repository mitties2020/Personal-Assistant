import os
import json
import logging
from flask import Flask, request, jsonify
from openai import OpenAI

# Logging
logging.basicConfig(level=logging.INFO)

# OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Optional Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

SUPA = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        from supabase import create_client
        SUPA = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        logging.info("✅ Supabase connected.")
    except Exception as e:
        logging.warning(f"⚠️ Supabase failed: {e}")
        SUPA = None
else:
    logging.warning("⚠️ No Supabase ENV — running without DB.")

# Flask app
APP = Flask(__name__)

@APP.route("/health")
def health():
    return jsonify({"ok": True})

@APP.route("/")
def home():
    return """
    <h2>FFM API</h2>
    <p>POST your question to /answer</p>
    """

def query_supabase(question):
    if not SUPA:
        return []
    try:
        res = SUPA.table("evidence").select("*").ilike("content", f"%{question}%").limit(5).execute()
        return res.data or []
    except Exception as e:
        logging.warning(f"Supabase query fail: {e}")
        return []

@APP.route("/answer", methods=["POST"])
def answer():
    data = request.get_json(force=True)
    question = data.get("question","").strip()

    if not question:
        return jsonify({"error":"No question"}), 400

    # fetch supporting evidence if DB exists
    supabase_results = query_supabase(question)
    evidence_text = "\n\n".join(item.get("content","") for item in supabase_results)

    prompt = f"""
You are a concise clinical assistant. Format answers for doctors, with Australian medical verbiage.

Question: {question}

Relevant clinical sources:
{evidence_text or "No internal evidence found"}

Format:
1) Definition
2) Causes / Pathophysiology
3) Red flags / criteria
4) ED Management (stepwise + doses)
5) Sources
"""

    try:
        ai = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2
        )
        answer_text = ai.choices[0].message.content
    except Exception as e:
        logging.error(e)
        return jsonify({"error":"AI failure"}), 500

    return jsonify({
        "answer": answer_text,
        "sources": supabase_results
    })

if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT",8000)))
