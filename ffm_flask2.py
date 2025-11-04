from flask_cors import CORS

APP = Flask(__name__)
CORS(APP, resources={r"/answer": {"origins": "*"}, r"/health": {"origins": "*"}})
# ---- OpenAI (optional but recommended) ----
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
oa_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---- Supabase optional ----
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPA = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        from supabase import create_client
        SUPA = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        logging.info("Supabase connected.")
    except Exception as e:
        logging.warning(f"Supabase disabled (init error): {e}")
else:
    logging.info("Supabase not configured; proceeding without DB.")

APP = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@APP.route("/health")
def health():
    return jsonify({"ok": True})

@APP.route("/")
def root():
    return "FFM API is running. POST /answer"

def fetch_evidence(q: str):
    """Return a list of {title, content} from Supabase if configured, else []."""
    if not SUPA:
        return []
    try:
        # Adjust table/column names to your schema
        # Example table: evidence(title text, content text)
        res = SUPA.table("evidence").select("title,content").ilike("content", f"%{q}%").limit(5).execute()
        return res.data or []
    except Exception as e:
        logging.warning(f"Supabase query failed: {e}")
        return []

@APP.route("/answer", methods=["POST"])
def answer():
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    # Pull any internal snippets (optional)
    snippets = fetch_evidence(question)
    joined = "\n\n".join([f"- {s.get('title', 'Untitled')}: {s.get('content','')[:800]}" for s in snippets]) or "No internal evidence."

    # If OpenAI is available, have it compose a succinct clinical answer
    if oa_client:
        prompt = f"""
You are a concise Australian ED assistant. Provide a brief 4-part summary:

1) Definition
2) Causes / Criteria / Red flags
3) Immediate ED management (doses, routes)
4) Ongoing care / monitoring

Question: {question}

Internal notes:
{joined}
"""
        try:
            resp = oa_client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role":"user","content":prompt}],
                temperature=0.2
            )
            text = resp.choices[0].message.content
        except Exception as e:
            logging.exception("OpenAI error")
            return jsonify({"error":"AI failure", "detail": str(e)}), 500
    else:
        # Fallback if no OpenAI: return internal notes only
        text = f"(No OpenAI configured)\n\nInternal notes:\n{joined}"

    return jsonify({
        "answer": text,
        "sources": [{"title": s.get("title","Untitled")} for s in snippets]
    })

if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
