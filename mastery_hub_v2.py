"""
Mastery Hub v2 — A dynamic, personalized learning environment.

This application serves an enhanced, interactive version of the study materials,
focusing on hands-on practice, Socratic dialogue, and verifiable skill acquisition.
It runs on a separate port to keep it distinct from the existing hubs.

Run: python mastery_hub_v2.py -> http://localhost:5002
"""
import json
import os
import secrets
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, request, send_from_directory, session, url_for
from markdown import markdown
from xml.sax.saxutils import escape as xml_escape
from mastery_curriculum import TOPICS, TOPICS_BY_ID, TIERS

load_dotenv()

ROOT = Path(__file__).resolve().parent
DB_FILE = ROOT / "mastery_v2.db"  # Separate DB for this hub
TEMPLATE_FILE = ROOT / "mastery_hub_v2_template.html"

os.environ["ANTHROPIC_API_KEY"]
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5")
claude_client = anthropic.Anthropic()

APP_PASSWORD = os.environ.get("APP_PASSWORD")

app = Flask(__name__, static_folder=None)
app.secret_key = os.environ.get("FLASK_SECRET_KEY_V2") or secrets.token_hex(32)
app.permanent_session_lifetime = timedelta(days=30)
app.config["SESSION_COOKIE_SECURE"] = bool(APP_PASSWORD)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS progress_v2 ("
        "topic_id TEXT PRIMARY KEY, status TEXT NOT NULL, updated_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS qa_history_v2 ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "topic TEXT, question TEXT, answer TEXT, created_at TEXT)"
    )
    conn.commit()
    conn.close()


init_db()

# --- Authentication (copied from mastery_server.py for standalone use) ---
LOGIN_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mastery Hub v2 &mdash; sign in</title>
<style>body{font-family:system-ui,sans-serif;background:#0e0e12;color:#eee;display:flex;min-height:100vh;align-items:center;justify-content:center;margin:0}form{background:#1a1a20;padding:2rem 1.8rem;border-radius:.6rem;width:min(90vw,20rem)}h1{font-size:1.1rem;margin:0 0 1rem}input{width:100%;box-sizing:border-box;font-size:1rem;padding:.6rem .7rem;border-radius:.4rem;border:1px solid #333;background:#0e0e12;color:#eee;margin-bottom:.8rem}button{width:100%;padding:.6rem;border-radius:.4rem;border:none;background:#5b8cff;color:#fff;font-size:1rem;cursor:pointer}.err{color:#ff6b6b;font-size:.85rem;margin:-.4rem 0 .8rem}</style></head><body>
<form method="post"><h1>Mastery Hub v2</h1>{{ERROR}}<input type="password" name="password" placeholder="Password" autofocus required><button type="submit">Enter</button></form></body></html>"""

@app.before_request
def require_login():
    if not APP_PASSWORD: return None
    if request.endpoint in ("login", "static"): return None
    if session.get("authed_v2"): return None
    return redirect(url_for("login", next=request.path))

@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        if secrets.compare_digest(request.form.get("password", ""), APP_PASSWORD or ""):
            session.permanent = True
            session["authed_v2"] = True
            return redirect(request.args.get("next") or url_for("index"))
        error = '<p class="err">Wrong password.</p>'
    return LOGIN_PAGE.replace("{{ERROR}}", error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))
# --- End Authentication ---


def render_page(title: str, body_html: str, source: str = "mastery-hub-v2") -> str:
    template = TEMPLATE_FILE.read_text(encoding="utf-8")
    return template.replace("{{TITLE}}", title).replace("{{BODY}}", body_html).replace("{{SOURCE}}", source)


@app.route("/")
def index():
    """Main dashboard for the v2 hub, listing all topics."""
    sections = ""
    for tier in TIERS:
        topics_in_tier = [t for t in TOPICS if t["tier"] == tier["id"]]
        if not topics_in_tier:
            continue

        # Build the HTML for the topic cards in this tier
        cards = ""
        for t in topics_in_tier:
            href = f"/topic/{t['id']}"
            role_badge = '<span class="role-badge">AI Engineer</span>' if t.get("role") == "ai_engineer" else ""
            cards += (
                f'<div class="topic-card">'
                f'<h3><a href="{href}">{xml_escape(t["title"])}</a>{role_badge}</h3>'
                f'<p class="blurb">{xml_escape(t["blurb"])}</p>'
                f"</div>"
            )

        is_ref = tier.get("is_reference", False)
        is_bonus = tier.get("is_bonus", False)
        if is_ref:
            heading = tier["name"]
        elif is_bonus:
            heading = f'Bonus: {tier["name"]}'
        else:
            heading = f'Tier {tier["id"]} &middot; {tier["name"]}'
        
        icon = "&#128278; " if is_ref else ("&#127873; " if is_bonus else f"{tier['id']}.")
        block_class = " reference-block" if is_ref else (" bonus-block" if is_bonus else "")

        sections += (
            f'<div class="tier-block{block_class}">'
            f'<h2>{icon}{heading}</h2>'
            f'<p class="tier-sub">{xml_escape(tier["subtitle"])}</p>'
            f"{cards}</div>"
        )

    body = (
        "<h1>&#127942; Mastery Hub v2 (The Gym)</h1>"
        "<p class='lede'>Welcome to the interactive learning gym. Every code block on these pages is runnable. Select a topic to begin.</p>"
        f"{sections}"
    )
    return render_page("Mastery Hub v2", body, "mastery-hub-v2")

@app.route("/topic/<topic_id>")
def topic_page(topic_id):
    t = TOPICS_BY_ID.get(topic_id)
    if not t:
        return "Not found", 404

    # Read and render markdown, but inject new interactive elements
    text = (ROOT / t["file"]).read_text(encoding="utf-8")
    content_html = markdown(text, extensions=["fenced_code", "tables"])

    # The JS in the template will handle finding code blocks and adding "Run" buttons.
    return render_page(t["title"], content_html, f"topic-{topic_id}")

@app.route("/challenges")
def challenges_page():
    # Serve the "Prove It" tier live coding challenges
    # TODO: Implementation for live coding environment and test runner
    return render_page("Challenges", "<h1>Live Coding Challenges</h1><p>Coming soon.</p>")

@app.route("/graph")
def interactive_graph():
    # Serve the interactive D3.js knowledge graph
    # TODO: Implementation for the interactive graph view
    return render_page("Knowledge Graph", "<h1>Interactive Knowledge Graph</h1><p>Coming soon.</p>")

@app.route("/api/execute_code", methods=["POST"])
def execute_code():
    """API endpoint for the 'Run' button."""
    code = request.json.get("code", "")
    if not code:
        return jsonify({"error": "No code provided"}), 400

    # Using a venv python if available to keep dependencies isolated.
    # This path needs to be configured to your environment.
    # We can check for a few common venv names.
    venv_paths = [
        ROOT / ".venv-llm-rag" / "Scripts" / "python.exe",
        ROOT / ".venv-langchain" / "Scripts" / "python.exe",
        ROOT / ".venv" / "Scripts" / "python.exe",
    ]
    python_executable = "python" # Default to global python
    for p in venv_paths:
        if p.exists():
            python_executable = str(p)
            break

    try:
        # Use a temporary file to execute the code
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            filepath = f.name

        result = subprocess.run(
            [python_executable, filepath],
            capture_output=True, text=True, timeout=15, check=False
        )
        output = result.stdout
        error = result.stderr
    except Exception as e:
        output = ""
        error = str(e)
    finally:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)

    return jsonify({"output": output, "error": error})

@app.route("/api/socratic_explain", methods=["POST"])
def socratic_explain():
    """API endpoint for the 'Explain This Line' feature."""
    line_content = request.json.get("line", "")
    full_context = request.json.get("context", "")
    history = request.json.get("history", [])

    system_prompt = (
        "You are a Socratic mentor. A user has clicked on a specific line of code or math. "
        "Your goal is to help them understand it from first principles. Start by asking a "
        "question to probe their current understanding before you explain. Be concise. "
        f"The line is: `{line_content}`. The surrounding context is:\n{full_context}"
    )
    # ... Claude API call logic will go here ...
    return jsonify({"answer": "Let's start with a question: What do you think `*args` does here?"})


if __name__ == "__main__":
    print("Serving Mastery Hub v2 on http://localhost:5002")
    app.run(port=5002, debug=True)