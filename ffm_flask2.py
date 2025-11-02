# ffm_flask2.py
import os, re
from datetime import datetime, timezone
from flask import Flask, request, jsonify, Response

APP = Flask(__name__)

# ---------- CORS + optional Basic Auth ----------
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
BASIC_USER = os.getenv("BASIC_USER")
BASIC_PASS = os.getenv("BASIC_PASS")

def _auth_ok(u, p): return BASIC_USER and BASIC_PASS and u == BASIC_USER and p == BASIC_PASS
def _need_auth(): return Response("Auth required", 401, {"WWW-Authenticate": 'Basic realm="FFM"'})

def requires_auth(f):
    from functools import wraps
    @wraps(f)
    def w(*a, **k):
        if not BASIC_USER:  # auth disabled
            return f(*a, **k)
        auth = request.authorization
        if not auth or not _auth_ok(auth.username, auth.password):
            return _need_auth()
        return f(*a, **k)
    return w

@APP.after_request
def add_cors(h):
    h.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    h.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    h.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return h

# ---------- Storage (Whoosh) ----------
BASE = os.path.abspath(os.path.dirname(__file__))
DOC_DIR = os.path.join(BASE, "docs")
IDX_DIR = os.path.join(BASE, "whoosh")
os.makedirs(DOC_DIR, exist_ok=True)
os.makedirs(IDX_DIR, exist_ok=True)

from whoosh import index
from whoosh.fields import Schema, TEXT, ID, DATETIME
from whoosh.qparser import MultifieldParser

SCHEMA = Schema(
    chunk_id=ID(stored=True, unique=True),
    title=TEXT(stored=True),
    org=TEXT(stored=True),
    published=DATETIME(stored=True),
    text=TEXT(stored=True),
)

if not os.listdir(IDX_DIR):
    index.create_in(IDX_DIR, SCHEMA)
IX = index.open_dir(IDX_DIR)

# ---------- Heuristics ----------
ORG_WEIGHTS = {
    "ASCIA": 4.0,
    "WA Health": 3.0,
    "NSW Health": 3.0,
    "NSW Health CEC": 3.0,
    "Queensland Health": 3.0,
    "SCCM/ESICM": 3.0,
}

def _doc_bonus(org, published_iso):
    """Org quality + freshness bonus."""
    bonus = ORG_WEIGHTS.get((org or "").strip(), 0.0)
    try:
        dt = datetime.fromisoformat(published_iso.replace("Z", "")) if published_iso else None
        if dt:
            now = datetime.now(dt.tzinfo or timezone.utc)
            age_years = max(0.0, (now - dt).days / 365.0)
            bonus += max(0.0, 2.0 - 0.5 * age_years)  # up to +2 if very recent
    except Exception:
        pass
    return bonus

def _score_sentence(s):
    s2 = (s or "").lower()
    score = 0
    if re.search(r'\b(immediate|first[-\s]?line|stat|urgent|resus|airway|breathing|circulation)\b', s2): score += 5
    if re.search(r'\b(\d+(\.\d+)?)\s?(mg|mcg|g|ml|mL|%)\b', s2): score += 3
    if re.search(r'\b(iv|im|po|neb|infus|bolus|repeat|monitor|ecg|titrate)\b', s2): score += 2
    if re.search(r'\b(adrenaline|epinephrine|calcium|insulin|dextrose|salbutamol|bicarbonate|potassium|magnesium|ceftriaxone|piperacillin|vancomycin)\b', s2): score += 2
    L = len(s)
    if L: score += min(L // 120, 2)
    if re.match(r'^\s*[-•\d\)]\s', s): score += 1
    return score

def _clean_line(t):
    if not t: return ""
    t = re.sub(r'(\d)\s+(\d)', r'\1\2', t)
    t = t.replace(" m g", " mg").replace(" m l", " mL").replace(" mc g", " mcg")
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def _summarise(question, hits):
    pools = {"definition/criteria": [], "causes/complications": [], "immediate management": [], "ongoing care / monitoring": []}
    def bucket(s):
        s2 = s.lower()
        if any(k in s2 for k in ["criteria","definition","diagnos","recognition"]): return "definition/criteria"
        if any(k in s2 for k in ["cause","trigger","risk","complication"]): return "causes/complications"
        if any(k in s2 for k in ["immediate","first-line","stat","airway","adrenaline","calcium","insulin","bolus","iv "]): return "immediate management"
        return "ongoing care / monitoring"

    for h in hits:
        org = h.get("org",""); pub = h.get("published"); pub_iso = pub.isoformat() if pub else ""
        bonus = _doc_bonus(org, pub_iso)
        for raw in (h.get("text") or "").split(". "):
            line = _clean_line(raw)
            if len(line) < 20: continue
            pools[bucket(line)].append((_score_sentence(line) + bonus, line))

    html = "<ol>"
    for head in ["definition/criteria", "causes/complications", "immediate management", "ongoing care / monitoring"]:
        items = sorted(pools[head], key=lambda x: x[0], reverse=True)[:5]
        if not items: continue
        html += f"<li><b>{head.title()}</b><ul>"
        for _, s in items:
            html += f"<li>{s}</li>"
        html += "</ul></li>"
    html += "</ol>"
    return html

# ---------- Routes ----------
@APP.route("/health")
def health(): return jsonify(ok=True)

@APP.route("/answer", methods=["POST"])
@requires_auth
def answer():
    try:
        data = request.get_json(force=True) or {}
        question = (data.get("question") or "").strip()
        k = int(data.get("k") or 12)
    except Exception:
        return jsonify(error="bad_request"), 400
    if not question:
        return jsonify(answer="Empty question."), 200

    with IX.searcher() as s:
        parser = MultifieldParser(["title","org","text"], schema=IX.schema)
        hits = s.search(parser.parse(question), limit=k)
        rows = []
        for h in hits:
            rows.append({
                "chunk_id": h.get("chunk_id"),
                "title": h.get("title"),
                "org": h.get("org"),
                "published": h.get("published").isoformat() if h.get("published") else "",
                "text": h.get("text")
            })

    if not rows:
        return jsonify(answer="<p>No matches found in your local index.</p>", sources=[]), 200

    html = _summarise(question, rows)
    srcs = [{"title":r["title"], "org":r["org"], "published":r["published"]} for r in rows[:6]]
    return jsonify(answer=html, sources=srcs), 200

@APP.route("/", methods=["GET"])
def home():
    return "FFM API is running. POST /answer"

if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
