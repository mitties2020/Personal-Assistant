from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Optional deps (safe if not configured)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from supabase import create_client  # not used yet; keeps future DB option
except Exception:
    create_client = None

APP = Flask(__name__)
CORS(APP, resources={r"/answer": {"origins": "*"}, r"/health": {"origins": "*"}})

# --- OpenAI (optional but recommended) ---
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
oa_client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY) else None

# --- Supabase (optional; not required to boot) ---
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip()
SUPABASE_SERVICE_KEY = (os.getenv("SUPABASE_SERVICE_KEY") or "").strip()
sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) if (create_client and SUPABASE_URL and SUPABASE_SERVICE_KEY) else None

@APP.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

def build_prompt(question: str) -> str:
    return f"""
You are a clinical assistant for an Australian ED doctor. Produce a succinct response (~180 words)
with exactly these sections:

1) What it is & criteria
2) Common causes & complications
3) Immediate management (first-line actions & doses)
4) Ongoing care / monitoring

Be specific with adult doses/units/routes (e.g., adrenaline 0.5 mg IM; calcium gluconate 10% 10 mL IV over 2–5 min).
Prefer Australian guidance when principles are equivalent.

Question: {question}
"""

@APP.route("/answer", methods=["POST"])
def answer():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "Missing 'question'"}), 400

    if oa_client:
        try:
            resp = oa_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": build_prompt(q)}],
                temperature=0.2,
            )
            text = (resp.choices[0].message.content or "").strip()
            return jsonify({"ok": True, "answer": text, "sources": []})
        except Exception as e:
            return jsonify({"ok": False, "error": f"OpenAI error: {e}"}), 500

    # No model configured: return a helpful note (no crash)
    return jsonify({
        "ok": True,
        "answer": "Backend is up, but no model is configured. Set OPENAI_API_KEY in Render → Environment and redeploy.",
        "sources": []
    })

# gunicorn entrypoint
if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
