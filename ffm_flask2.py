# ffm_flask2.py
# Minimal clinical assistant API + UI backed by Supabase
#
# ENV required on Render:
#   SUPABASE_URL=https://xxxxx.supabase.co
#   SUPABASE_SERVICE_KEY=eyJhbGciOi...
#
# TABLE (public.guidelines), suggested columns:
#   id uuid PK default gen_random_uuid()
#   title text
#   org text
#   url text
#   published timestamp with time zone (nullable)
#   text text
#   created_at timestamp default now()
#
# RLS: enable; add policy to ALLOW SELECT for all (using: true).
# (Keep INSERT restricted; we ingest via service key on the server side.)

from __future__ import annotations
import os, re, html
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, request, jsonify, Response
from supabase import create_client, Client

APP = Flask(__name__)

# -------------------- Supabase client --------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Set SUPABASE_URL and SUPABASE_SERVICE_KEY env vars.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# -------------------- Helpers --------------------
def _sentences(txt: str) -> List[str]:
    """Split into sentences & bullet lines, keep shortish chunks."""
    if not txt:
        return []
    # normalize bullets/newlines
    t = re.sub(r"\r\n?", "\n", txt)
    t = re.sub(r"[ \t]+", " ", t)
    lines = [l.strip() for l in t.split("\n") if l.strip()]
    # split paragraphs into sentences
    out: List[str] = []
    for ln in lines:
        if re.match(r"^[-•\d\)\(•]\s", ln):
            out.append(ln)
        else:
            parts = re.split(r"(?<=[\.\?\!])\s+", ln)
            out.extend([p.strip() for p in parts if p.strip()])
    # clip overly long pieces
    return [s[:500] for s in out if 10 <= len(s) <= 500]

_IMMEDIATE_RE = re.compile(
    r"\b(immediate|first[-\s]?line|stat|urgent|airway|breathing|circulation|"
    r"resus|abcde|adrenaline|epinephrine|calcium|insulin|dextrose|hyperton|3%|"
    r"magnesium|ceftriaxone|piperacillin|tazobactam|vancomycin|bolus|defibrill)\b",
    re.I,
)
_DOSE_RE = re.compile(r"\b(\d+(\.\d+)?)\s?(mcg|mg|g|mL|ml|%)\b", re.I)
_ROUTE_RE = re.compile(r"\b(iv|im|po|neb|infus|bolus)\b", re.I)
_CRITERIA_RE = re.compile(r"\b(definition|criteria|diagnos|meets|signs?|symptoms?)\b", re.I)
_CAUSES_RE = re.compile(r"\b(causes?|triggers?|aetiolog|etiolog|risk)\b", re.I)
_COMP_RE = re.compile(r"\b(complication|shock|arrhythm|arrest|oedema|edema)\b", re.I)
_FOLLOW_RE = re.compile(r"\b(monitor|observe|repeat|titrate|admit|review|"
                        r"escalate|red flags?|disposition|reassess|ecg)\b", re.I)

def _score_sentence(s: str) -> int:
    s2 = s.lower()
    score = 0
    if _IMMEDIATE_RE.search(s2): score += 8
    if _DOSE_RE.search(s2): score += 4
    if __ROUTE_RE.search(s2): score += 3
    if s.strip().startswith(("-", "•")): score += 1
    # prefer informative length
    L = len(s)
    score += 1 if 80 <= L <= 240 else 0
    return score

def _categorise(s: str) -> str:
    s2 = s.lower()
    if _IMMEDIATE_RE.search(s2) or _DOSE_RE.search(s2): return "Immediate"
    if _CRITERIA_RE.search(s2): return "Definition/Criteria"
    if _CAUSES_RE.search(s2) or _COMP_RE.search(s2): return "Causes/Complications"
    if _FOLLOW_RE.search(s2): return "Ongoing"
    # fallback: put in the first section that needs filling
    return "Definition/Criteria"

def _clean_for_html(s: str) -> str:
    # join hyphen linebreak splits, un-weird spacing, escape, keep bullets
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return html.escape(s)

def _render_answer(blocks: Dict[str, List[str]]) -> str:
    order = ["Definition/Criteria", "Causes/Complications", "Immediate", "Ongoing"]
    titles = {
        "Definition/Criteria": "What it is & how to recognise it",
        "Causes/Complications": "Common causes & complications",
        "Immediate": "Immediate management (first steps & doses)",
        "Ongoing": "Monitoring / follow-up",
    }
    html_parts = []
    for key in order:
        items = blocks.get(key, [])
        if not items: continue
        html_parts.append(f"<h4 style='margin:8px 0 6px'>{titles[key]}</h4>")
        lis = "".join(f"<li>{_clean_for_html(x)}</li>" for x in items)
        html_parts.append(f"<ul style='margin:6px 0 10px 18px'>{lis}</ul>")
    return "".join(html_parts) if html_parts else "<p>No matches found in your knowledge base.</p>"

def _pick_sentences(rows: List[Dict[str, Any]], q: str) -> (str, List[Dict[str, str]]):
    # collect candidate sentences with scores
    cand: List[Dict[str, Any]] = []
    for r in rows:
        sents = _sentences(r.get("text") or "")
        for s in sents:
            score = _score_sentence(s)
            # light boost if query tokens appear
            for tok in set(re.findall(r"[a-z0-9]{3,}", q.lower())):
                if tok in s.lower(): score += 1
            cand.append({"score": score, "sent": s, "row": r})

    # sort by score and keep top N sentences
    cand.sort(key=lambda x: x["score"], reverse=True)
    top = cand[:60]

    # bucket into 4 sections, keep 3/3/6/4 items respectively
    slots = {"Definition/Criteria": 3, "Causes/Complications": 3, "Immediate": 6, "Ongoing": 4}
    chosen: Dict[str, List[str]] = {k: [] for k in slots}
    srcs: List[Dict[str, str]] = []
    used_row_ids = set()

    for c in top:
        cat = _categorise(c["sent"])
        if len(chosen[cat]) < slots[cat]:
            chosen[cat].append(c["sent"])
            rid = c["row"].get("id")
            if rid and rid not in used_row_ids:
                used_row_ids.add(rid)
                srcs.append({
                    "title": c["row"].get("title") or "",
                    "org": c["row"].get("org") or "",
                    "published": (c["row"].get("published") or "") if isinstance(c["row"].get("published"), str)
                                 else (c["row"].get("published").isoformat() if c["row"].get("published") else ""),
                    "url": c["row"].get("url") or ""
                })

    return _render_answer(chosen), srcs[:8]

# -------------------- Search (Supabase) --------------------
def _supa_search(q: str, limit: int = 80) -> List[Dict[str, Any]]:
    """
    Simple OR/ILIKE search across title, org, text.
    """
    terms = re.findall(r"[A-Za-z0-9]{3,}", q)
    if not terms:
        res = supabase.table("guidelines").select("*").limit(20).execute()
        return res.data or []

    # Build PostgREST OR filter: title.ilike.%term%,text.ilike.%term%,org.ilike.%term%,...
    parts = []
    for t in set(terms):
        pat = f"%{t}%"
        parts.append(f"title.ilike.{pat}")
        parts.append(f"text.ilike.{pat}")
        parts.append(f"org.ilike.{pat}")
    filt = ",".join(parts)

    res = supabase.table("guidelines").select("*").or_(filt).limit(limit).execute()
    return res.data or []

# -------------------- Routes --------------------
@APP.route("/health")
def health():
    return jsonify({"ok": True})

@APP.route("/answer", methods=["POST"])
def answer():
    # Accept either JSON body or form field 'q'
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        q = (payload.get("question") or payload.get("q") or "").strip()
        k = int(payload.get("k") or 12)
    else:
        q = (request.form.get("q") or "").strip()
        k = int(request.form.get("k") or 12)
    if not q:
        return jsonify({"ok": False, "error": "Empty question"}), 400

    rows = _supa_search(q, limit=max(40, k * 6))
    if not rows:
        return jsonify({"ok": True, "answer": "<p>No matches found in your knowledge base.</p>", "sources": []})

    html_answer, srcs = _pick_sentences(rows, q)
    return jsonify({"ok": True, "answer": html_answer, "sources": srcs})

@APP.route("/ingest_text", methods=["POST"])
def ingest_text():
    """
    Ingest a single guideline paragraph/page into Supabase.
    Body: {"title": "...", "org":"...", "url":"...", "published":"2024-06-01", "text":"..."}
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "No text"}), 400
    title = data.get("title") or ""
    org = data.get("org") or ""
    url = data.get("url") or ""
    published = data.get("published")
    try:
        pub_dt = datetime.fromisoformat(published) if published else None
    except Exception:
        pub_dt = None

    ins = {
        "title": title, "org": org, "url": url, "text": text
    }
    if pub_dt:
        ins["published"] = pub_dt.isoformat()

    res = supabase.table("guidelines").insert(ins).execute()
    return jsonify({"ok": True, "inserted": res.data})

# -------------------- UI --------------------
@APP.route("/")
def home():
    return Response("""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>FFM — Doctor View</title>
<style>
  :root { color-scheme: light dark; --ink:#0f172a; --mut:#475569; --pill:#e2e8f0;}
  body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial; margin:24px; color:var(--ink); }
  h1 { font-size:22px; margin:0 0 6px; }
  .small { color:var(--mut); font-size:12.5px;}
  textarea { width:100%; height:88px; padding:12px; font: 14px/1.4 ui-sans-serif; border:1px solid #cbd5e1; border-radius:10px; outline:none }
  button { margin-top:10px; padding:10px 14px; border-radius:12px; border:1px solid #0ea5e9; background:#0ea5e9; color:white; font-weight:600; cursor:pointer }
  #ans { margin-top:16px; }
  #answerText { font-size:15px; line-height:1.45; white-space:normal }
  h4 { font-size:15px; margin:12px 0 8px }
  ul { margin:6px 0 12px 18px }
  li { margin:4px 0 }
  .pill { display:inline-block; background:var(--pill); padding:2px 8px; border-radius:999px; font-size:11.5px; margin-right:6px }
</style>
</head><body>
  <h1>FFM — Doctor View</h1>
  <div class="small">One-line question in → succinct 4-part answer out.</div>
  <textarea id="q" placeholder="e.g., Hyperkalaemia with wide QRS: definition, causes, immediate management"></textarea>
  <div><button id="askBtn">Answer</button></div>
  <div id="ans" style="display:none">
    <div id="answerText"></div>
    <div id="srcs" class="small"></div>
  </div>
<script>
const $ = s => document.querySelector(s);
async function ask(){
  const q = $("#q").value.trim(); if(!q) return;
  $("#ans").style.display="block";
  $("#answerText").innerHTML = "<div class='small'>Working…</div>";
  $("#srcs").innerHTML = "";
  try {
    const r = await fetch("/answer", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({question:q, k: 14})});
    const j = await r.json();
    $("#answerText").innerHTML = j.answer || "No matches.";
    const srcs = (j.sources||[]).map(s => {
      const bits = [s.org||"", s.title||"", s.published?("("+s.published.slice(0,10)+")"):""].filter(Boolean).join(" — ");
      const link = s.url ? `<a href="${s.url}" target="_blank" rel="noopener">link</a>` : "";
      return `<div class="small">• ${bits} ${link}</div>`;
    }).join("");
    $("#srcs").innerHTML = srcs ? `<div style="margin-top:8px"><span class="pill">Sources</span></div>${srcs}` : "";
  } catch(e){
    $("#answerText").textContent = "Error: " + (e && e.message || e);
  }
}
$("#askBtn").onclick = ask;
</script>
</body></html>""", mimetype="text/html")

# -------------------- Entrypoint --------------------
if __name__ == "__main__":
    # Local dev
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
