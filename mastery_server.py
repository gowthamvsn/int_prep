"""
Mastery Hub — a second, independent view over the same study files already
served by server.py (port 5000). Nothing here is moved or duplicated content:
every route reads the same .md / .html files on disk. This app only adds
ordering (easiest -> hardest), cross-topic links, a knowledge-graph map, and
per-topic mastery tracking (own DB, own port -- the original app is untouched).

Run:  python mastery_server.py   ->  http://localhost:5001
"""
import json
import os
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import markdown as md
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, request, send_from_directory, session, url_for

from mastery_curriculum import TOPICS, TOPICS_BY_ID, TIERS

load_dotenv()

ROOT = Path(__file__).resolve().parent
DOC_TEMPLATE = (ROOT / "doc_template.html").read_text(encoding="utf-8")
DB_FILE = ROOT / "mastery.db"

HTML_FILES = {
    "nca-genl": "NCA-GENL-study-guide.html",
    "bnsf-visual": "bnsf-technical-visual.html",
    "ds-fundamentals": "ds-fundamentals-visual.html",
}

AZURE_KEY = os.environ["AZURE_OPENAI_KEY"]
AZURE_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
AZURE_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")

# Set only when deployed publicly (see .env.example / deploy notes). When unset,
# auth is skipped entirely so local dev on localhost is unaffected.
APP_PASSWORD = os.environ.get("APP_PASSWORD")

app = Flask(__name__, static_folder=None)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.permanent_session_lifetime = timedelta(days=30)
# HTTPS-only cookies once a password is actually configured (i.e. once this is
# deployed, not running locally over plain http).
app.config["SESSION_COOKIE_SECURE"] = bool(APP_PASSWORD)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


LOGIN_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mastery Hub &mdash; sign in</title>
<style>
body{font-family:system-ui,sans-serif;background:#0e0e12;color:#eee;display:flex;
  min-height:100vh;align-items:center;justify-content:center;margin:0}
form{background:#1a1a20;padding:2rem 1.8rem;border-radius:.6rem;width:min(90vw,20rem)}
h1{font-size:1.1rem;margin:0 0 1rem}
input{width:100%;box-sizing:border-box;font-size:1rem;padding:.6rem .7rem;border-radius:.4rem;
  border:1px solid #333;background:#0e0e12;color:#eee;margin-bottom:.8rem}
button{width:100%;padding:.6rem;border-radius:.4rem;border:none;background:#5b8cff;
  color:#fff;font-size:1rem;cursor:pointer}
.err{color:#ff6b6b;font-size:.85rem;margin:-.4rem 0 .8rem}
</style></head><body>
<form method="post">
<h1>Mastery Hub</h1>
{{ERROR}}
<input type="password" name="password" placeholder="Password" autofocus required>
<button type="submit">Enter</button>
</form></body></html>"""


@app.before_request
def require_login():
    if not APP_PASSWORD:
        return None  # auth disabled — local dev
    if request.endpoint in ("login", "static"):
        return None
    if session.get("authed"):
        return None
    return redirect(url_for("login", next=request.path))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        if secrets.compare_digest(request.form.get("password", ""), APP_PASSWORD or ""):
            session.permanent = True
            session["authed"] = True
            return redirect(request.args.get("next") or url_for("index"))
        error = '<p class="err">Wrong password.</p>'
    return LOGIN_PAGE.replace("{{ERROR}}", error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS progress ("
        "topic_id TEXT PRIMARY KEY, status TEXT NOT NULL, updated_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS qa_history ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "source TEXT, topic TEXT, question TEXT, answer TEXT, created_at TEXT)"
    )
    conn.commit()
    conn.close()


init_db()


def topic_href(topic):
    if topic["kind"] == "html":
        return topic["route"]
    return f"/topic/{topic['id']}"


# ---------------------------------------------------------------- search ----

_ENTITIES = {
    "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&middot;": "\u00b7",
    "&rarr;": "->", "&larr;": "<-", "&#39;": "'", "&quot;": '"', "&hellip;": "...",
    "&mdash;": "\u2014", "&ndash;": "\u2013",
}


def _strip_html_for_search(html_text: str) -> str:
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", html_text, flags=re.S | re.I)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    for k, v in _ENTITIES.items():
        text = text.replace(k, v)
    return re.sub(r"[ \t]+", " ", text)


def _blocks_from_text(text: str, min_len: int = 25):
    return [b.strip() for b in re.split(r"\n\s*\n", text) if len(b.strip()) >= min_len]


_QSTR = r'"((?:[^"\\]|\\.)*)"'
_BANK_ENTRY_RE = re.compile(
    r"\{d:" + _QSTR + r",q:" + _QSTR + r",o:\[(.*?)\],a:\[[^\]]*\],e:" + _QSTR + r"\}", re.S
)


def _extract_bank_legacy(html_text: str):
    """The 56-question practice exam: a JS array literal with unquoted keys, not valid JSON."""
    m = re.search(r"const BANK=(\[.*?\]);", html_text, re.S)
    if not m:
        return []
    out = []
    for mm in _BANK_ENTRY_RE.finditer(m.group(1)):
        domain, question, opts_raw, explanation = mm.groups()
        opts = re.findall(_QSTR, opts_raw)
        out.append({
            "label": f"Practice exam (56-Q) \u2014 {domain}",
            "text": " ".join([question] + opts + [explanation]),
        })
    return out


def _extract_bank2(html_text: str):
    """The 110-question community bank: valid JSON, dumped straight from Python."""
    m = re.search(r"const BANK2=(\[.*?\]);", html_text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(1))
    except ValueError:
        return []
    out = []
    for item in data:
        out.append({
            "label": f"Community bank (110-Q) \u2014 {item.get('id', '')} \u00b7 {item.get('d', '')}",
            "text": " ".join([item.get("q", "")] + item.get("o", []) + [item.get("e", "")]),
        })
    return out


def _build_search_sources():
    """One entry per searchable unit: a topic's prose, or one quiz question. Built fresh
    per request (files are small and this is a personal local tool -- no index to keep in sync)."""
    sources = []
    for t in TOPICS:
        href = topic_href(t)
        try:
            if t["kind"] == "md":
                raw = (ROOT / t["file"]).read_text(encoding="utf-8")
                for block in _blocks_from_text(raw):
                    sources.append({"title": t["title"], "href": href, "label": None, "text": block})
            else:  # kind == "html"
                raw = (ROOT / HTML_FILES[t["id"]]).read_text(encoding="utf-8")
                prose = _strip_html_for_search(raw)
                for block in _blocks_from_text(prose):
                    sources.append({"title": t["title"], "href": href, "label": None, "text": block})
                if t["id"] == "nca-genl":
                    for entry in _extract_bank_legacy(raw):
                        sources.append({"title": t["title"], "href": href + "#quiz", "label": entry["label"], "text": entry["text"]})
                    for entry in _extract_bank2(raw):
                        sources.append({"title": t["title"], "href": href + "#quiz2", "label": entry["label"], "text": entry["text"]})
        except (FileNotFoundError, KeyError):
            continue
    return sources


def _make_snippet(block: str, tokens, width: int = 130) -> str:
    low = block.lower()
    positions = [low.find(t) for t in tokens if low.find(t) != -1]
    pos = min(positions) if positions else 0
    start = max(0, pos - width // 2)
    end = min(len(block), start + width * 2)
    snippet = block[start:end].strip()
    if start > 0:
        snippet = "\u2026" + snippet
    if end < len(block):
        snippet = snippet + "\u2026"
    return snippet


def _highlight(snippet: str, tokens) -> str:
    escaped = xml_escape(snippet)
    for t in sorted(set(tokens), key=len, reverse=True):
        escaped = re.sub("(" + re.escape(xml_escape(t)) + ")", r"<mark>\1</mark>", escaped, flags=re.I)
    return escaped


def render_doc_page(title: str, body_html: str, source: str) -> str:
    return DOC_TEMPLATE.replace("{{TITLE}}", title).replace("{{BODY}}", body_html).replace("{{SOURCE}}", source)


def render_nav() -> str:
    # The port-5000 sibling app only exists on the same machine — link to it
    # only when we're actually being viewed from that machine (local dev).
    is_local_host = request.host.split(":")[0] in ("localhost", "127.0.0.1")
    local_link = (
        '<span style="color:var(--line)">|</span>'
        '<a href="http://localhost:5000/" style="text-decoration:none;color:var(--muted)">Original Study Hub (5000)</a>'
        if is_local_host else ""
    )
    logout_link = (
        '<a href="/logout" style="text-decoration:none;color:var(--muted)">Log out</a>' if APP_PASSWORD else ""
    )
    return (
        '<nav style="display:flex;gap:1rem;align-items:center;margin-bottom:1.2rem;'
        'font-family:var(--mono);font-size:.8rem;flex-wrap:wrap">'
        '<a href="/" style="text-decoration:none;font-weight:700;color:var(--accent-hi)">&#127942; Mastery Hub</a>'
        '<a href="/map" style="text-decoration:none;color:var(--muted)">Knowledge Map</a>'
        f"{local_link}"
        '<form method="get" action="/search" style="margin-left:auto;display:flex;gap:.4rem">'
        '<input type="text" name="q" placeholder="&#128269; Search the hub…" '
        'style="font-family:var(--mono);font-size:.78rem;border:1px solid var(--line);border-radius:.35rem;'
        'padding:.3rem .55rem;background:var(--bg);color:var(--ink);width:12rem">'
        "</form>"
        f"{logout_link}"
        "</nav>"
    )

PROGRESS_JS = """
<script>
(function(){
"use strict";
const STATUSES=["new","learning","mastered"];
const LABEL={new:"Not started",learning:"Learning",mastered:"Mastered"};
async function loadProgress(){
  try{
    const res=await fetch("/api/progress");const data=await res.json();
    const map=data.progress||{};
    document.querySelectorAll("[data-progress-group]").forEach(g=>{
      const id=g.getAttribute("data-progress-group");
      setActive(g,map[id]||"new");
    });
    updateSummary(map);
  }catch(e){/* server not reachable yet */}
}
function setActive(group,status){
  group.querySelectorAll("button").forEach(b=>{
    b.setAttribute("aria-pressed", b.dataset.status===status ? "true":"false");
  });
}
function updateSummary(map){
  const bar=document.getElementById("progressSummary");
  if(!bar)return;
  const total=bar.dataset.total?parseInt(bar.dataset.total,10):0;
  const mastered=Object.values(map).filter(v=>v==="mastered").length;
  const learning=Object.values(map).filter(v=>v==="learning").length;
  bar.querySelector(".psFill").style.width=(total?(mastered/total*100):0)+"%";
  bar.querySelector(".psText").textContent=mastered+" / "+total+" mastered"+(learning?", "+learning+" in progress":"");
}
document.addEventListener("click", async e=>{
  const btn=e.target.closest("[data-status]");
  if(!btn)return;
  const group=btn.closest("[data-progress-group]");
  const id=group.getAttribute("data-progress-group");
  const status=btn.dataset.status;
  setActive(group,status);
  try{
    const res=await fetch("/api/progress",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({topic_id:id,status})});
    const data=await res.json();
    updateSummary(data.progress||{});
  }catch(err){/* ignore */}
});
document.addEventListener("DOMContentLoaded",loadProgress);
if(document.readyState!=="loading")loadProgress();
})();
</script>
"""


def progress_buttons(topic_id: str) -> str:
    btns = "".join(
        f'<button type="button" data-status="{s}" title="{l}" '
        f'style="font-family:var(--mono);font-size:.68rem;border:1px solid var(--line);'
        f'background:var(--panel);color:var(--muted);border-radius:.3rem;padding:.25rem .5rem;'
        f'cursor:pointer;margin-right:.3rem">{l}</button>'
        for s, l in [("new", "New"), ("learning", "Learning"), ("mastered", "Mastered")]
    )
    return f'<div class="progress-group" data-progress-group="{topic_id}" style="margin:.6rem 0">{btns}</div>'


PROGRESS_CSS = """
<style>
.wrap{max-width:56rem}
.progress-group button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:var(--bg);font-weight:700}
.topic-card{background:var(--panel);border:1px solid var(--line);border-radius:.55rem;padding:1rem 1.1rem;margin:.9rem 0}
.topic-card h3{margin:.1rem 0 .3rem}
.topic-card .blurb{color:var(--muted);margin:.2rem 0 .6rem}
.related-pills{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.5rem}
.related-pills a{font-family:var(--mono);font-size:.72rem;text-decoration:none;color:var(--accent-hi);
  background:var(--accent-soft);border-radius:1rem;padding:.2rem .65rem}
.role-badge{display:inline-block;font-family:var(--mono);font-size:.66rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.05em;color:var(--accent-hi);border:1px solid var(--accent);border-radius:.3rem;
  padding:.1rem .45rem;margin-left:.5rem;vertical-align:middle}
.tier-block{margin:2.2rem 0}
.tier-block h2{margin-bottom:.15rem}
.reference-block{background:var(--panel2);border:1px dashed var(--accent);border-radius:.6rem;padding:.2rem 1.1rem 1rem;margin-bottom:2.6rem}
.tier-sub{color:var(--muted);font-family:var(--mono);font-size:.82rem;margin:0 0 .8rem}
#progressSummary{background:var(--panel);border:1px solid var(--line);border-radius:.5rem;padding:.8rem 1rem;margin:1rem 0}
#progressSummary .psTrack{background:var(--panel2);border-radius:1rem;height:.6rem;overflow:hidden;margin-top:.4rem}
#progressSummary .psFill{background:var(--accent);height:100%;width:0%;transition:width .3s}
#progressSummary .psText{font-family:var(--mono);font-size:.78rem;color:var(--muted)}
.search-form{display:flex;gap:.5rem;margin:.8rem 0 1.2rem;max-width:40rem}
.search-form input{flex:1;font-family:var(--body);font-size:.95rem;border:1px solid var(--line);border-radius:.4rem;padding:.5rem .7rem;background:var(--bg);color:var(--ink)}
.search-group{margin:1.4rem 0;padding-bottom:1rem;border-bottom:1px solid var(--line)}
.search-group h3{margin:0 0 .5rem;font-size:1.05rem}
.search-count{font-family:var(--mono);font-size:.68rem;color:var(--muted);border:1px solid var(--line);border-radius:1rem;padding:.05rem .5rem;vertical-align:middle}
.search-hit{margin:.5rem 0}
.search-hit a{display:block;text-decoration:none;color:inherit;border:1px solid var(--line);border-radius:.4rem;padding:.55rem .75rem;background:var(--panel)}
.search-hit a:hover{border-color:var(--accent)}
.search-label{font-family:var(--mono);font-size:.68rem;color:var(--accent-hi);margin-bottom:.25rem}
.search-snippet{font-size:.9rem;color:var(--muted);line-height:1.5}
.search-snippet mark{background:var(--accent-soft);color:var(--ink);border-radius:.2rem;padding:0 .1rem}
.search-more{font-family:var(--mono);font-size:.72rem;color:var(--muted);margin-top:.3rem}
</style>
"""


@app.route("/search")
def search():
    query = (request.args.get("q") or "").strip()
    tokens = [t for t in query.lower().split() if t]

    body = (
        f"{render_nav()}"
        f"{PROGRESS_CSS}"
        '<a class="backlink" href="/">&larr; Mastery Hub</a>'
        "<h1>Search</h1>"
        '<form method="get" action="/search" class="search-form">'
        f'<input type="text" name="q" value="{xml_escape(query)}" placeholder="Search every topic and both practice banks…" autofocus>'
        '<button class="btn" type="submit">Search</button>'
        "</form>"
    )

    if not tokens:
        body += '<p class="tier-sub">Type one or more words — results need ALL of them to appear in the same paragraph or question.</p>'
        return render_doc_page("Search", body, "mastery-hub")

    hits_by_topic = {}
    for src in _build_search_sources():
        low = src["text"].lower()
        if all(t in low for t in tokens):
            hits_by_topic.setdefault((src["title"], src["href"].split("#")[0]), []).append(src)

    if not hits_by_topic:
        body += f'<p class="tier-sub">No matches for <strong>{xml_escape(query)}</strong> anywhere in the hub.</p>'
        return render_doc_page("Search", body, "mastery-hub")

    total_hits = sum(len(v) for v in hits_by_topic.values())
    body += f'<p class="tier-sub">{total_hits} match{"es" if total_hits != 1 else ""} across {len(hits_by_topic)} topic{"s" if len(hits_by_topic) != 1 else ""}.</p>'

    for (title, base_href), hits in sorted(hits_by_topic.items(), key=lambda kv: -len(kv[1])):
        # Prose and quiz-bank hits are capped SEPARATELY -- a topic with lots of prose
        # matches must never bury the (usually far fewer, easy-to-miss-otherwise)
        # quiz-question hits, which is the whole reason those get extracted at all.
        quiz_hits = [h for h in hits if h["label"]]
        prose_hits = [h for h in hits if not h["label"]]
        shown = quiz_hits[:12] + prose_hits[:5]
        hidden_count = (len(quiz_hits) - len(quiz_hits[:12])) + (len(prose_hits) - len(prose_hits[:5]))

        body += f'<div class="search-group"><h3><a href="{base_href}">{xml_escape(title)}</a> <span class="search-count">{len(hits)}</span></h3>'
        for h in shown:
            snippet = _highlight(_make_snippet(h["text"], tokens), tokens)
            label = f'<div class="search-label">{xml_escape(h["label"])}</div>' if h["label"] else ""
            body += f'<div class="search-hit"><a href="{h["href"]}">{label}<div class="search-snippet">{snippet}</div></a></div>'
        if hidden_count > 0:
            body += f'<div class="search-more">+ {hidden_count} more match{"es" if hidden_count != 1 else ""} in this topic — open it and use your browser\'s find (Ctrl/Cmd+F)</div>'
        body += "</div>"

    return render_doc_page(f"Search: {query}", body, "mastery-hub")


@app.route("/")
def index():
    total = len(TOPICS)
    sections = ""
    for tier in TIERS:
        topics = [t for t in TOPICS if t["tier"] == tier["id"]]
        cards = ""
        for t in topics:
            related = "".join(
                f'<a href="{topic_href(TOPICS_BY_ID[r])}">{TOPICS_BY_ID[r]["title"]}</a>'
                for r in t["related"]
            )
            role_badge = '<span class="role-badge">AI Engineer</span>' if t.get("role") == "ai_engineer" else ""
            cards += (
                f'<div class="topic-card">'
                f'<h3><a href="{topic_href(t)}" style="text-decoration:none;color:inherit">{t["title"]}</a>{role_badge}</h3>'
                f'<p class="blurb">{t["blurb"]}</p>'
                f"{progress_buttons(t['id'])}"
                f'<div class="related-pills">{related}</div>'
                f"</div>"
            )
        is_ref = tier.get("is_reference", False)
        heading = tier["name"] if is_ref else f'Tier {tier["id"]} &middot; {tier["name"]}'
        sections += (
            f'<div class="tier-block{" reference-block" if is_ref else ""}">'
            f'<h2>{"&#128278; " if is_ref else ""}{heading}</h2>'
            f'<p class="tier-sub">{tier["subtitle"]}</p>'
            f"{cards}</div>"
        )

    num_ladder_tiers = sum(1 for t in TIERS if not t.get("is_reference"))
    body = (
        f"{render_nav()}"
        f"{PROGRESS_CSS}"
        "<h1>Mastery Hub</h1>"
        f'<p class="lede">All {total} topics from the original Study Hub, reordered easiest &rarr; hardest into '
        f"{num_ladder_tiers} tiers, with related topics cross-linked across what used to be separate silos "
        "(NCA-GENL exam, BNSF interview prep, DS fundamentals, practice docs), plus a standalone "
        "Reference tier that sits outside the ladder. Track your progress per topic "
        'below, or see how everything connects on the <a href="/map">Knowledge Map</a>.</p>'
        f'<div id="progressSummary" data-total="{total}">'
        '<div class="psText">Loading progress&hellip;</div>'
        '<div class="psTrack"><div class="psFill"></div></div></div>'
        f"{sections}"
        f"{PROGRESS_JS}"
    )
    return render_doc_page("Mastery Hub", body, "mastery-hub")


@app.route("/topic/<topic_id>")
def topic(topic_id):
    t = TOPICS_BY_ID.get(topic_id)
    if not t:
        return "Not found", 404
    if t["kind"] == "html":
        return app.redirect(t["route"])

    text = (ROOT / t["file"]).read_text(encoding="utf-8")
    content_html = md.markdown(text, extensions=["fenced_code", "tables", "sane_lists"])
    related = "".join(
        f'<a href="{topic_href(TOPICS_BY_ID[r])}">{TOPICS_BY_ID[r]["title"]}</a>'
        for r in t["related"]
    )
    role_badge = '<span class="role-badge">AI Engineer</span>' if t.get("role") == "ai_engineer" else ""
    meta = (
        f"{render_nav()}"
        f"{PROGRESS_CSS}"
        '<a class="backlink" href="/">&larr; Mastery Hub</a>'
        f'<p class="tier-sub">Tier {t["tier"]} &middot; {t["tier_name"]}{role_badge}</p>'
        f"{progress_buttons(t['id'])}"
        f'<div class="related-pills" style="margin-bottom:1.2rem">{related}</div>'
    )
    body = meta + content_html
    return render_doc_page(t["title"], body, "mastery-hub") + PROGRESS_JS


@app.route("/m/<name>")
def html_guide(name):
    fname = HTML_FILES.get(name)
    if not fname:
        return "Not found", 404
    return send_from_directory(ROOT, fname)


# ---------------------------------------------------------------- map ----

TIER_X = 90
TIER_GAP = 230
NODE_W, NODE_H = 190, 58
ROW_GAP = 108
TOP_PAD = 70


def build_map_svg() -> str:
    pos = {}
    for col, tier in enumerate(TIERS):
        topics = [t for t in TOPICS if t["tier"] == tier["id"]]
        x = TIER_X + col * TIER_GAP
        for i, t in enumerate(topics):
            y = TOP_PAD + i * ROW_GAP
            pos[t["id"]] = (x, y)

    max_rows = max(sum(1 for t in TOPICS if t["tier"] == tier["id"]) for tier in TIERS)
    width = TIER_X + len(TIERS) * TIER_GAP + NODE_W
    height = TOP_PAD + max_rows * ROW_GAP + 40

    edges_seen = set()
    edge_svg = []
    for t in TOPICS:
        x1, y1 = pos[t["id"]]
        for r in t["related"]:
            key = tuple(sorted((t["id"], r)))
            if key in edges_seen:
                continue
            edges_seen.add(key)
            x2, y2 = pos[r]
            sx1, sy1 = x1 + NODE_W / 2, y1 + NODE_H / 2
            sx2, sy2 = x2 + NODE_W / 2, y2 + NODE_H / 2
            mx = (sx1 + sx2) / 2
            edge_svg.append(
                f'<path d="M{sx1:.0f},{sy1:.0f} C{mx:.0f},{sy1:.0f} {mx:.0f},{sy2:.0f} {sx2:.0f},{sy2:.0f}" '
                f'class="edge"/>'
            )

    node_svg = []
    header_svg = []
    for col, tier in enumerate(TIERS):
        hx = TIER_X + col * TIER_GAP
        is_ref = tier.get("is_reference", False)
        top_label = "Reference" if is_ref else f'Tier {tier["id"]}'
        bottom_label = "open anytime" if is_ref else tier["name"]
        header_svg.append(
            f'<text x="{hx + NODE_W/2:.0f}" y="30" text-anchor="middle" class="tierhead">'
            f'{xml_escape(top_label)}</text>'
            f'<text x="{hx + NODE_W/2:.0f}" y="46" text-anchor="middle" class="tierhead2">'
            f'{xml_escape(bottom_label)}</text>'
        )
    for t in TOPICS:
        x, y = pos[t["id"]]
        title = xml_escape(t["title"])
        # wrap title into up to 3 lines of ~24 chars
        words = title.split(" ")
        lines, cur = [], ""
        for w in words:
            if len(cur) + len(w) + 1 > 24:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            lines.append(cur)
        lines = lines[:3]
        text_lines = "".join(
            f'<tspan x="{x+10:.0f}" dy="{"0" if i==0 else "13"}">{ln}</tspan>'
            for i, ln in enumerate(lines)
        )
        href = topic_href(t)
        node_svg.append(
            f'<a href="{href}"><g class="node">'
            f'<rect x="{x:.0f}" y="{y:.0f}" width="{NODE_W}" height="{NODE_H}" rx="8"/>'
            f'<title>{xml_escape(t["blurb"])}</title>'
            f'<text x="{x+10:.0f}" y="{y+22:.0f}" class="nodetext">{text_lines}</text>'
            f"</g></a>"
        )

    svg = (
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'style="min-width:{width}px" xmlns="http://www.w3.org/2000/svg">'
        f'<g class="edges">{"".join(edge_svg)}</g>'
        f'<g class="headers">{"".join(header_svg)}</g>'
        f'<g class="nodes">{"".join(node_svg)}</g>'
        f"</svg>"
    )
    return svg, width


@app.route("/map")
def knowledge_map():
    svg, width = build_map_svg()
    body = (
        f"{render_nav()}"
        "<h1>Knowledge Map</h1>"
        '<p class="lede">How the 23 topics connect across tiers. Hover a box for its one-line summary, '
        "click to open it. Curves are the same cross-links shown as pills on the "
        '<a href="/">Mastery Hub</a> page.</p>'
        "<style>"
        ".wrap{max-width:min(96vw,88rem)}"
        ".edge{fill:none;stroke:var(--line);stroke-width:1.4;stroke-opacity:.65}"
        ".node rect{fill:var(--panel);stroke:var(--accent);stroke-width:1.2}"
        ".node:hover rect{fill:var(--accent-soft)}"
        ".nodetext{font-family:var(--mono);font-size:10.5px;fill:var(--ink)}"
        ".tierhead{font-family:var(--display);font-weight:700;font-size:13px;fill:var(--accent-hi)}"
        ".tierhead2{font-family:var(--mono);font-size:9px;fill:var(--muted)}"
        "a{text-decoration:none}"
        "</style>"
        f'<div class="fig" style="overflow-x:auto">{svg}</div>'
    )
    return render_doc_page("Knowledge Map", body, "mastery-hub")


# ------------------------------------------------------------ progress ---

@app.route("/api/progress", methods=["GET", "POST"])
def progress():
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        topic_id = (data.get("topic_id") or "").strip()
        status = (data.get("status") or "").strip()
        if topic_id not in TOPICS_BY_ID or status not in ("new", "learning", "mastered"):
            return jsonify({"error": "invalid topic_id or status"}), 400
        conn = get_db()
        conn.execute(
            "INSERT INTO progress (topic_id, status, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(topic_id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at",
            (topic_id, status, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()

    conn = get_db()
    rows = conn.execute("SELECT topic_id, status FROM progress").fetchall()
    conn.close()
    return jsonify({"progress": {r["topic_id"]: r["status"] for r in rows}})


# ----------------------------------------------------------------- ask ---

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True, silent=True) or {}
    topic = (data.get("topic") or "").strip()
    context = (data.get("context") or "").strip()
    question = (data.get("question") or "").strip()
    history = data.get("history") or []

    if not question:
        return jsonify({"error": "empty question"}), 400

    system_prompt = (
        "You are a study assistant embedded in a personal Mastery Hub that unifies NVIDIA "
        "NCA-GENL exam prep, a BNSF Sr/Staff Data Scientist interview loop, and DS/ML fundamentals, "
        "arranged from foundations to advanced. Answer at the depth needed to actually master the "
        "material, connecting back to easier prerequisite topics when it helps.\n"
        f'The user clicked on: "{topic}".\n'
        + (f"Relevant excerpt (for grounding, not necessarily to repeat):\n{context}\n\n" if context else "")
        + "Answer clearly and concretely. Use examples or numbers where they help. Keep it "
        "focused: a few short paragraphs or a tight list, not an essay. "
        "Never use LaTeX (no \\[, \\(, $$, or similar delimiters) — this page does not render it, "
        "so it would show up as raw backslashes. Write all math as plain text arithmetic instead, "
        "e.g. `SE = s / sqrt(n) = 10 / 5 = 2`."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-6:]:
        role, content = turn.get("role"), turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})

    url = (
        f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions"
        f"?api-version={API_VERSION}"
    )
    try:
        resp = requests.post(
            url,
            headers={"api-key": AZURE_KEY, "Content-Type": "application/json"},
            json={"messages": messages, "temperature": 0.3, "max_tokens": 700},
            timeout=60,
        )
    except requests.RequestException as exc:
        return jsonify({"error": f"could not reach Azure OpenAI: {exc}"}), 502

    if resp.status_code != 200:
        return jsonify({"error": f"Azure OpenAI error {resp.status_code}: {resp.text[:500]}"}), 502

    body = resp.json()
    try:
        answer = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return jsonify({"error": f"unexpected response shape: {body}"}), 502

    conn = get_db()
    conn.execute(
        "INSERT INTO qa_history (source, topic, question, answer, created_at) VALUES (?, ?, ?, ?, ?)",
        ("mastery-hub", topic, question, answer, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()

    return jsonify({"answer": answer})


@app.route("/api/history")
def history():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, source, topic, question, answer, created_at FROM qa_history ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return jsonify({"items": [dict(r) for r in rows]})


if __name__ == "__main__":
    print("Serving Mastery Hub on http://localhost:5001")
    app.run(port=5001, debug=False)
