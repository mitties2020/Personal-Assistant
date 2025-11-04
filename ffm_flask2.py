import os, re, hashlib, datetime, json, io
from flask import Flask, request, jsonify, Response
from pypdf import PdfReader
from supabase import create_client, Client
import httpx

APP = Flask(__name__)

# -------- env & supabase ----------
SB_URL = os.environ.get("SUPABASE_URL", "")
SB_ANON = os.environ.get("SUPABASE_ANON_KEY", "")
SB_SERVICE = os.environ.get("SUPABASE_SERVICE_ROLE", "")
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")

def sb() -> Client:
    key = SB_SERVICE or SB_ANON
    return create_client(SB_URL, key)

# --------- helpers ----------
def _clean(s: str) -> str:
    if not s: return ""
    s = re.sub(r"\s+", " ", s)
    s = s.replace("\uf0a7", "•")
    return s.strip()

def _score_sentence(s: str) -> int:
    s2 = s.lower()
    score = 0
    if re.search(r"\b(immediate|first[-\s]?line|stat|urgent|resus|airway|breathing|circulation|defibrill)\b", s2):
        score += 6
    if re.search(r"\b(\d+(\.\d+)?)\s?(mg|mcg|g|ml|mL|%)\b", s2): score += 3
    if re.search(r"\b(iv|im|neb|bolus|infus|repeat|titrate|monitor|ecg)\b", s2): score += 2
    if re.search(r"\b(adrenaline|epinephrine|calcium|insulin|dextrose|salbutamol|bicarbonate|potassium|magnesium|ceftriaxone|piperacillin|vancomycin)\b", s2):
        score += 2
    if re.match(r'^\s*[-•\d\)]\s', s): score += 1
    L = len(s); 
    if L: score += min(L//120, 2)
    return score

def _chunks(text: str, max_chars=2000):
    sents = re.split(r"(?<=[\.\?!])\s+", text.strip())
    out, cur = [], []
    for s in sents:
        if sum(len(x) for x in cur)+len(s) > max_chars and cur:
            out.append(" ".join(cur)); cur=[s]
        else:
            cur.append(s)
    if cur: out.append(" ".join(cur))
    return out

def _pdf_to_text(byts: bytes) -> str:
    buf = io.BytesIO(byts)
    try:
        reader = PdfReader(buf)
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        return "\n".join(pages)
    except Exception:
        return ""

def _fetch_text_from_url(url: str) -> tuple[str, bytes]:
    with httpx.Client(follow_redirects=True, timeout=60) as cx:
        r = cx.get(url)
        r.raise_for_status()
        ctype = r.headers.get("content-type","")
        data = r.content
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            txt = _pdf_to_text(data)
        else:
            # basic HTML strip
            body = r.text
            body = re.sub(r"(?is)<script.*?>.*?</script>", " ", body)
            body = re.sub(r"(?is)<style.*?>.*?</style>", " ", body)
            body = re.sub(r"(?is)<.*?>", " ", body)
            txt = body
        return (_clean(txt), data)

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _format_doctor_answer(defn, causes, immediate, ongoing) -> str:
    parts = []
    def block(title, items):
        if not items: return ""
        lis = "".join([f"<li>{_clean(x)}</li>" for x in items[:8]])
        return f"<h4 style='margin:8px 0 6px'>{title}</h4><ul style='margin:6px 0 10px 18px'>{lis}</ul>"
    parts.append(block("Definition / diagnostic criteria", defn))
    parts.append(block("Common causes / complications", causes))
    parts.append(block("Immediate management (first 60–90 minutes)", immediate))
    parts.append(block("Ongoing care / details", ongoing))
    return "".join([p for p in parts if p])

# --------- data access ----------
TABLE = "clinical_guidelines"

def _search_supabase(q: str, k: int = 18):
    # very simple token OR search on title/text via ilike
    tokens = [t for t in re.split(r"[^A-Za-z0-9%]+", q) if t]
    tokens = tokens[:6] or [q]
    client = sb()
    hits = {}
    for t in tokens:
        like = f"%{t}%"
        data = client.table(TABLE).select("id,title,org,published,text").ilike("text", like).limit(k).execute().data
        for row in data or []:
            rid = row["id"]
            if rid not in hits:
                hits[rid] = row
    # crude relevance: count token hits + presence of “immediate/dose”
    scored = []
    for r in hits.values():
        txt = (r.get("text") or "").lower()
        s = sum(txt.count(t.lower()) for t in tokens)
        if re.search(r"\b(immediate|stat|first[-\s]?line)\b", txt): s += 2
        if re.search(r"\b(\d+(\.\d+)?)\s?(mg|mcg|g|ml|%)\b", txt): s += 1
        scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:k]]

def _distill_sections(texts: list[str], question: str):
    # pick top sentences for each section
    all_sents = []
    for t in texts:
        for s in re.split(r"(?<=[\.\?!])\s+", t):
            ss = _clean(s)
            if 6 <= len(ss) <= 500:
                all_sents.append(ss)
    def pick(filter_regex=None, extra=None, n=6):
        cand = []
        for s in all_sents:
            if filter_regex and not re.search(filter_regex, s, re.I): 
                continue
            score = _score_sentence(s)
            if extra: score += extra(s)
            cand.append((score, s))
        cand.sort(key=lambda x:x[0], reverse=True)
        return [s for _, s in cand[:n]]

    defs = pick(r"\b(definition|diagnos|criteria|triad|signs|symptoms|presentation)\b", n=6)
    causes = pick(r"\b(cause|trigger|risk|aetiol|etiol|complication)\b", n=6)
    immediate = pick(r"\b(immediate|first[-\s]?line|stat|adrenaline|epinephrine|calcium|insulin|dextrose|bolus|infus|iv|im)\b", n=8)
    ongoing = pick(None, n=8)  # leftovers/most practical high-score lines
    return defs, causes, immediate, ongoing

# --------- routes ----------
@APP.route("/health")
def health():
    return jsonify({"ok": True})

# Minimal UI (unchanged)
@APP.route("/", methods=["GET"])
def home():
    html = """
<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Clinical Assistant — Doctor View</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{font-family:system-ui,sans-serif} body{margin:24px;max-width:900px}
h1{margin:0 0 10px} .small{color:#6b7280;font-size:12.5px}
textarea{width:100%;min-height:86px;padding:10px;border:1px solid #e5e7eb;border-radius:10px;font-size:15px}
button{padding:10px 14px;border:0;border-radius:10px;background:#111827;color:white;font-weight:600;cursor:pointer}
#answerText{margin-top:14px;border:1px solid #e5e7eb;border-radius:12px;padding:14px;font-size:15.5px;line-height:1.4;background:#fafafa}
#srcs .src{color:#374151;font-size:12.5px;margin:4px 0}
</style></head><body>
<h1>Clinical Assistant</h1>
<div class="small">One-line question in → succinct 4-part answer out.</div>
<textarea id="q" placeholder="e.g., Hyperkalaemia with broad QRS: definition, causes, immediate management"></textarea>
<div style="margin-top:8px"><button id="ask">Answer</button></div>
<div id="ans" style="display:none">
  <h3>Answer</h3>
  <div id="answerText"></div>
  <div id="srcs"></div>
</div>
<script>
const $ = s => document.querySelector(s);
$("#ask").onclick = async () => {
  const q = $("#q").value.trim(); if (!q) return;
  $("#ans").style.display = "block";
  $("#answerText").innerHTML = "Working…";
  $("#srcs").innerHTML = "";
  try {
    const r = await fetch("/answer", {method:"POST",headers:{"Content-Type":"application/json"},body: JSON.stringify({question:q,k:18})});
    const j = await r.json();
    $("#answerText").innerHTML = j.answer || "No matches.";
    const srcs = (j.sources||[]).map(s => `<div class="src">• ${s.title||'Untitled'} — ${s.org||''} ${s.published?('('+s.published.slice(0,10)+')'):''}</div>`).join("");
    $("#srcs").innerHTML = srcs ? `<div class="small" style="margin-top:6px">Sources</div>${srcs}` : "";
  } catch(e){ $("#answerText").textContent = "Error: " + e.message; }
};
</script></body></html>
"""
    return Response(html, mimetype="text/html")

@APP.route("/answer", methods=["POST"])
def answer():
    data = request.get_json(force=True) or {}
    q = (data.get("question") or "").strip()
    k = int(data.get("k") or 18)
    if not q:
        return jsonify({"ok": True, "answer": "<p>Ask a clinical question.</p>", "sources": []})
    rows = _search_supabase(q, k=k)
    if not rows:
        return jsonify({"ok": True, "answer": "<p>No matches found in your evidence set.</p>", "sources": []})
    texts = [r.get("text") or "" for r in rows]
    d, c, im, on = _distill_sections(texts, q)
    html = _format_doctor_answer(d, c, im, on)
    srcs = [{"title": r.get("title"), "org": r.get("org"), "published": r.get("published")} for r in rows[:6]]
    return jsonify({"ok": True, "answer": html, "sources": srcs})

# --------- ingestion (server-side only, use Admin-Key) ----------
def _require_admin():
    key = request.headers.get("Admin-Key") or request.headers.get("X-Admin-Key")
    if not ADMIN_KEY or key != ADMIN_KEY:
        return jsonify({"ok": False, "error": "unauthorised"}), 401

@APP.route("/ingest_link", methods=["POST"])
def ingest_link():
    auth = _require_admin()
    if auth: return auth
    title = request.form.get("title","").strip()
    org = request.form.get("org","").strip()
    published = request.form.get("published","").strip()
    url = request.form.get("url","").strip()
    if not url:
        return jsonify({"ok": False, "error":"url required"}), 400
    txt, byts = _fetch_text_from_url(url)
    if not txt.strip():
        return jsonify({"ok": False, "error":"no text extracted"}), 400
    digest = _sha256(byts)
    pub = None
    try:
        if published: pub = datetime.datetime.fromisoformat(published)
    except Exception:
        pub = None
    row = {
        "title": title or url,
        "org": org,
        "published": pub.isoformat() if pub else None,
        "url": url,
        "sha256": digest,
        "text": _clean(txt)
    }
    res = sb().table(TABLE).upsert(row, on_conflict="sha256").execute()
    return jsonify({"ok": True, "inserted": bool(res.data)})

@APP.route("/ingest_text", methods=["POST"])
def ingest_text():
    auth = _require_admin()
    if auth: return auth
    j = request.get_json(force=True) or {}
    title = (j.get("title") or "").strip()
    org = (j.get("org") or "").strip()
    published = (j.get("published") or "").strip()
    text = (j.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error":"text required"}), 400
    pub = None
    try:
        if published: pub = datetime.datetime.fromisoformat(published)
    except Exception:
        pub = None
    row = {
        "title": title or "Untitled",
        "org": org,
        "published": pub.isoformat() if pub else None,
        "url": None,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": _clean(text)
    }
    res = sb().table(TABLE).upsert(row, on_conflict="sha256").execute()
    return jsonify({"ok": True, "inserted": bool(res.data)})

if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=8000)
