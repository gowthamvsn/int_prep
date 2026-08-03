"""
Mastery Hub — Streamlit edition.

Same content, same mastery_curriculum.py topic graph as the Flask apps
(server.py port 5000, mastery_server.py port 5001) — this is a third,
independent view meant for Streamlit Community Cloud deployment (phone-friendly,
free hosting). Reads the same .md/.html files on disk; nothing is duplicated.

Scope (deliberate, "core parity"): tier/topic navigation, full markdown+code
rendering, keyword search, session/file-based progress tracking, and the
knowledge map. The Flask apps' interactive click-to-answer JS quiz engine is
NOT ported here — quiz content still renders as plain readable Q&A text.

Run locally:  streamlit run streamlit_app.py
"""
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import streamlit as st
import streamlit.components.v1 as components

from mastery_curriculum import TOPICS, TOPICS_BY_ID, TIERS

ROOT = Path(__file__).resolve().parent
PROGRESS_FILE = ROOT / "streamlit_progress.json"

HTML_FILES = {
    "nca-genl": "NCA-GENL-study-guide.html",
    "bnsf-visual": "bnsf-technical-visual.html",
    "ds-fundamentals": "ds-fundamentals-visual.html",
}

st.set_page_config(page_title="Mastery Hub", page_icon="🎓", layout="wide")

# ---------------------------------------------------------------- auth ----
# Mirrors the Flask apps' pattern: set APP_PASSWORD in Streamlit Cloud's
# "Secrets" panel to gate access; leave it unset for local dev (no login).
try:
    APP_PASSWORD = st.secrets["APP_PASSWORD"]
except Exception:
    APP_PASSWORD = None

if APP_PASSWORD:
    if "authed" not in st.session_state:
        st.session_state.authed = False
    if not st.session_state.authed:
        st.title("🎓 Mastery Hub")
        pw = st.text_input("Password", type="password")
        if st.button("Enter"):
            if pw == APP_PASSWORD:
                st.session_state.authed = True
                st.rerun()
            else:
                st.error("Wrong password.")
        st.stop()

# ---------------------------------------------------------------- progress ----
def load_progress():
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_progress(progress):
    try:
        PROGRESS_FILE.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    except Exception:
        pass  # read-only filesystem on some hosts -- progress just won't persist across restarts there


if "progress" not in st.session_state:
    st.session_state.progress = load_progress()


def set_progress(topic_id: str, status: str):
    if status == "not_started":
        st.session_state.progress.pop(topic_id, None)
    else:
        st.session_state.progress[topic_id] = status
    save_progress(st.session_state.progress)


# ---------------------------------------------------------------- link rewriting ----
# The .md files link to each other as Flask-style /topic/<id>#<anchor> paths
# (mastery_server.py's routing). Streamlit has no path-based routing, only
# query params -- rewrite those links to `?topic=<id>` so cross-references
# between theory docs and Code Drills still work by landing on the right
# PAGE (the in-page anchor scroll itself is dropped, not essential here).
_TOPIC_LINK_RE = re.compile(r"/topic/([a-z0-9-]+)(#[a-z0-9-]+)?")


def rewrite_links(markdown_text: str) -> str:
    return _TOPIC_LINK_RE.sub(lambda m: f"?topic={m.group(1)}", markdown_text)


# ---------------------------------------------------------------- knowledge map ----
TIER_X, TIER_GAP, NODE_W, NODE_H, ROW_GAP, TOP_PAD = 90, 230, 190, 58, 108, 70


def build_map_svg():
    pos = {}
    for col, tier in enumerate(TIERS):
        topics = [t for t in TOPICS if t["tier"] == tier["id"]]
        x = TIER_X + col * TIER_GAP
        for i, t in enumerate(topics):
            pos[t["id"]] = (x, TOP_PAD + i * ROW_GAP)

    max_rows = max(sum(1 for t in TOPICS if t["tier"] == tier["id"]) for tier in TIERS)
    width = TIER_X + len(TIERS) * TIER_GAP + NODE_W
    height = TOP_PAD + max_rows * ROW_GAP + 40

    edges_seen, edge_svg = set(), []
    for t in TOPICS:
        x1, y1 = pos[t["id"]]
        for r in t["related"]:
            if r not in pos:
                continue
            key = tuple(sorted((t["id"], r)))
            if key in edges_seen:
                continue
            edges_seen.add(key)
            x2, y2 = pos[r]
            sx1, sy1 = x1 + NODE_W / 2, y1 + NODE_H / 2
            sx2, sy2 = x2 + NODE_W / 2, y2 + NODE_H / 2
            mx = (sx1 + sx2) / 2
            edge_svg.append(
                f'<path d="M{sx1:.0f},{sy1:.0f} C{mx:.0f},{sy1:.0f} {mx:.0f},{sy2:.0f} {sx2:.0f},{sy2:.0f}" class="edge"/>'
            )

    header_svg = []
    for col, tier in enumerate(TIERS):
        hx = TIER_X + col * TIER_GAP
        is_ref = tier.get("is_reference", False)
        is_bonus = tier.get("is_bonus", False)
        top_label = "Reference" if is_ref else ("Bonus" if is_bonus else f'Tier {tier["id"]}')
        bottom_label = "open anytime" if is_ref else tier["name"]
        header_svg.append(
            f'<text x="{hx + NODE_W/2:.0f}" y="30" text-anchor="middle" class="tierhead">{xml_escape(top_label)}</text>'
            f'<text x="{hx + NODE_W/2:.0f}" y="46" text-anchor="middle" class="tierhead2">{xml_escape(bottom_label)}</text>'
        )

    node_svg = []
    for t in TOPICS:
        x, y = pos[t["id"]]
        title = xml_escape(t["title"])
        words, lines, cur = title.split(" "), [], ""
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
            f'<tspan x="{x+10:.0f}" dy="{"0" if i==0 else "13"}">{ln}</tspan>' for i, ln in enumerate(lines)
        )
        # target="_top": breaks out of the components.html iframe so the click
        # navigates (and reruns) the actual top-level Streamlit app, not the iframe
        node_svg.append(
            f'<a href="?topic={t["id"]}" target="_top"><g class="node">'
            f'<rect x="{x:.0f}" y="{y:.0f}" width="{NODE_W}" height="{NODE_H}" rx="8"/>'
            f'<title>{xml_escape(t["blurb"])}</title>'
            f'<text x="{x+10:.0f}" y="{y+22:.0f}" class="nodetext">{text_lines}</text>'
            f"</g></a>"
        )

    svg = (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="min-width:{width}px" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<style>'
        f'.edge{{fill:none;stroke:#8888;stroke-width:1.4}}'
        f'.node rect{{fill:#1118;stroke:#7dd3fc;stroke-width:1.2}}'
        f'.node:hover rect{{fill:#7dd3fc33}}'
        f'.nodetext{{font-family:monospace;font-size:10.5px;fill:currentColor}}'
        f'.tierhead{{font-family:sans-serif;font-weight:700;font-size:13px;fill:#7dd3fc}}'
        f'.tierhead2{{font-family:monospace;font-size:9px;fill:#888}}'
        f'a text, a tspan{{fill:currentColor}}'
        f'</style>'
        f'<g class="edges">{"".join(edge_svg)}</g>'
        f'<g class="headers">{"".join(header_svg)}</g>'
        f'<g class="nodes">{"".join(node_svg)}</g>'
        f"</svg>"
    )
    return svg, width, height


# ---------------------------------------------------------------- deep-linking ----
# Clicking a map node or a rewritten in-doc link sets ?topic=<id> and reloads
# the top-level app -- read it once at startup so the sidebar selection follows.
_query_topic = st.query_params.get("topic")
if _query_topic and _query_topic in TOPICS_BY_ID:
    st.session_state.selected_topic = _query_topic
if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = TOPICS[0]["id"]
if "view" not in st.session_state:
    st.session_state.view = "Browse"


# ---------------------------------------------------------------- sidebar ----
def status_mark(topic_id: str) -> str:
    return {"done": "✅ ", "in_progress": "🟡 "}.get(st.session_state.progress.get(topic_id), "")


with st.sidebar:
    st.title("🎓 Mastery Hub")
    total = len(TOPICS)
    done = sum(1 for t in TOPICS if st.session_state.progress.get(t["id"]) == "done")
    st.caption(f"{done} / {total} topics marked done")
    st.progress(done / total if total else 0)

    st.session_state.view = st.radio("View", ["Browse", "Search", "Knowledge Map"], horizontal=True)

    if st.session_state.view == "Browse":
        for tier in TIERS:
            topics = [t for t in TOPICS if t["tier"] == tier["id"]]
            if not topics:
                continue
            is_ref = tier.get("is_reference", False)
            is_bonus = tier.get("is_bonus", False)
            label = tier["name"] if (is_ref or is_bonus) else f'Tier {tier["id"]} · {tier["name"]}'
            prefix = "🔖 " if is_ref else ("🎁 " if is_bonus else "")
            with st.expander(f"{prefix}{label}", expanded=(tier["id"] == TOPICS_BY_ID[st.session_state.selected_topic]["tier"])):
                for t in topics:
                    if st.button(f"{status_mark(t['id'])}{t['title']}", key=f"nav-{t['id']}", use_container_width=True):
                        st.session_state.selected_topic = t["id"]
                        st.rerun()

# ---------------------------------------------------------------- main area ----
if st.session_state.view == "Knowledge Map":
    st.header("Knowledge Map")
    st.caption("How every topic connects across tiers. Click a box to open it.")
    svg, width, height = build_map_svg()
    components.html(svg, height=height + 40, scrolling=True)

elif st.session_state.view == "Search":
    st.header("Search")
    query = st.text_input("Search across every topic", placeholder="e.g. gradient clipping, LoraConfig, chunk_overlap")
    if query:
        q = query.lower()
        hits = []
        for t in TOPICS:
            if t["kind"] != "md":
                continue
            text = (ROOT / t["file"]).read_text(encoding="utf-8")
            if q in text.lower():
                idx = text.lower().find(q)
                start = max(0, idx - 60)
                snippet = text[start:idx + len(query) + 60].replace("\n", " ")
                hits.append((t, snippet))
        st.caption(f"{len(hits)} topic(s) match \"{query}\"")
        for t, snippet in hits:
            with st.container(border=True):
                st.markdown(f"**{t['title']}**")
                st.caption(f"…{snippet}…")
                if st.button("Open", key=f"search-open-{t['id']}"):
                    st.session_state.selected_topic = t["id"]
                    st.session_state.view = "Browse"
                    st.rerun()

else:  # Browse
    t = TOPICS_BY_ID[st.session_state.selected_topic]
    role_badge = " `AI Engineer`" if t.get("role") == "ai_engineer" else ""
    st.header(t["title"] + role_badge)
    st.caption(t["blurb"])

    col_a, col_b, col_c, _ = st.columns([1, 1, 1, 3])
    with col_a:
        if st.button("⬜ Not started", use_container_width=True):
            set_progress(t["id"], "not_started")
    with col_b:
        if st.button("🟡 In progress", use_container_width=True):
            set_progress(t["id"], "in_progress")
    with col_c:
        if st.button("✅ Done", use_container_width=True):
            set_progress(t["id"], "done")

    st.divider()

    if t["kind"] == "html":
        fname = HTML_FILES.get(t["id"])
        if fname:
            html_text = (ROOT / fname).read_text(encoding="utf-8")
            components.html(html_text, height=1200, scrolling=True)
        else:
            st.error(f"No HTML file mapped for topic '{t['id']}'.")
    else:
        text = (ROOT / t["file"]).read_text(encoding="utf-8")
        st.markdown(rewrite_links(text))

    if t["related"]:
        st.divider()
        st.caption("Related topics")
        cols = st.columns(min(4, len(t["related"])))
        for i, r in enumerate(t["related"]):
            related_t = TOPICS_BY_ID.get(r)
            if not related_t:
                continue
            with cols[i % len(cols)]:
                if st.button(related_t["title"], key=f"related-{t['id']}-{r}", use_container_width=True):
                    st.session_state.selected_topic = r
                    st.rerun()
