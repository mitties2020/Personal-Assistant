# ffm_flask2.py
import os, re, hashlib
from datetime import datetime, timezone
from flask import Flask, request, jsonify, Response

# ---------- Flask ----------
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
        if not BASIC_USER: return f(*a, **k)  # auth disabled
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
    bonus = ORG_WEIGHTS.get((org or "").strip(), 0.0)
    try:
        dt
