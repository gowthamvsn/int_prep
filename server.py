import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import markdown as md
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

load_dotenv()

ROOT = Path(__file__).resolve().parent
GUIDE_FILE = "NCA-GENL-study-guide.html"
BNSF_VISUAL_FILE = "bnsf-technical-visual.html"
DS_FUNDAMENTALS_FILE = "ds-fundamentals-visual.html"
DOC_TEMPLATE = (ROOT / "doc_template.html").read_text(encoding="utf-8")
DB_FILE = ROOT / "qa_history.db"

AZURE_KEY = os.environ["AZURE_OPENAI_KEY"]
AZURE_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
AZURE_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")

app = Flask(__name__, static_folder=None)


def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS qa_history ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "source TEXT, topic TEXT, question TEXT, answer TEXT, created_at TEXT)"
    )
    conn.commit()
    conn.close()


init_db()

# Registry of every BNSF interview-prep doc, grouped the same way the interview loop is structured.
DOCS = [
    {"slug": "problem-formulation", "title": "Problem Formulation Framework",
     "file": "problem-formulation-framework.md", "group": "Problem Formulation"},
    {"slug": "live-coding", "title": "Live Coding Prep",
     "file": "live-coding-prep.md", "group": "Live Coding"},
    {"slug": "system-design", "title": "System Design Prep",
     "file": "system-design-prep.md", "group": "System Design"},
    {"slug": "service-impact", "title": "Service Impact & Causal Inference",
     "file": "service-impact-and-causal-inference.md", "group": "Service Impact"},
    {"slug": "behavioral", "title": "Behavioral / Partnership STAR Stories",
     "file": "behavioral-partnership-star-stories.md", "group": "Behavioral / Partnership"},
    {"slug": "core-technical", "title": "Core Technical Depth",
     "file": "core-technical-depth.md", "group": "Core Technical Depth"},
    {"slug": "domain-context", "title": "Freight Rail AI Domain Context",
     "file": "freight-rail-ai-domain-context.md", "group": "Domain Context"},
    {"slug": "my-projects", "title": "My Projects — Architecture, Code & Module Breakdown (FinSight, NaviDoc, and more)",
     "file": "my-projects-portfolio.md", "group": "My Projects Portfolio"},
    {"slug": "practice-numpy", "title": "NumPy Practice",
     "file": "numpy-practice.md", "group": "Python Data Science Practice"},
    {"slug": "practice-pandas", "title": "Pandas Practice",
     "file": "pandas-practice.md", "group": "Python Data Science Practice"},
    {"slug": "practice-sklearn", "title": "scikit-learn Practice",
     "file": "sklearn-practice.md", "group": "Python Data Science Practice"},
    {"slug": "practice-ml-models", "title": "Classical ML Models Practice",
     "file": "ml-models-practice.md", "group": "Python Data Science Practice"},
    {"slug": "practice-deep-learning", "title": "Deep Learning Practice (RNN/LSTM/CNN, PyTorch)",
     "file": "deep-learning-practice.md", "group": "Python Data Science Practice"},
    {"slug": "practice-pytorch-deep", "title": "PyTorch Deep Dive (hooks, mixed precision, DDP, GANs, ONNX)",
     "file": "pytorch-deep-dive.md", "group": "Python Data Science Practice"},
    {"slug": "practice-tf-keras-deep", "title": "TensorFlow/Keras Deep Dive (custom layers, GradientTape, transfer learning)",
     "file": "tensorflow-keras-deep-dive.md", "group": "Python Data Science Practice"},
    {"slug": "practice-visualization", "title": "Data Visualization Practice",
     "file": "visualization-practice.md", "group": "Python Data Science Practice"},
    {"slug": "practice-stats", "title": "Statistics (SciPy/statsmodels) Practice",
     "file": "stats-scipy-practice.md", "group": "Python Data Science Practice"},
    {"slug": "practice-utilities", "title": "Python Utilities Practice (datetime/regex/I-O/performance)",
     "file": "python-utilities-practice.md", "group": "Python Data Science Practice"},
    {"slug": "code-drills-basics", "title": "Code Drills 1 — Basics (variables, strings, control flow, functions)",
     "file": "code-drills-basics.md", "group": "Code Drills (Bonus)"},
    {"slug": "code-drills-data-structures", "title": "Code Drills 2 — Data Structures, JSON, Files, Exceptions",
     "file": "code-drills-data-structures.md", "group": "Code Drills (Bonus)"},
    {"slug": "code-drills-oop", "title": "Code Drills 3 — OOP, Decorators, Generators, Context Managers",
     "file": "code-drills-oop-intermediate.md", "group": "Code Drills (Bonus)"},
    {"slug": "code-drills-numpy-pandas", "title": "Code Drills 4 — NumPy & Pandas",
     "file": "code-drills-numpy-pandas.md", "group": "Code Drills (Bonus)"},
    {"slug": "code-drills-classical-ml", "title": "Code Drills 5 — Classical ML (train/eval RandomForest & friends)",
     "file": "code-drills-classical-ml.md", "group": "Code Drills (Bonus)"},
    {"slug": "code-drills-deep-learning", "title": "Code Drills 6 — PyTorch: Tensors, Training Loops, CNNs, LSTM Tuning",
     "file": "code-drills-deep-learning.md", "group": "Code Drills (Bonus)"},
    {"slug": "code-drills-llm-huggingface", "title": "Code Drills 7 — LLMs: Tokenization, HuggingFace, Decoding, Embeddings",
     "file": "code-drills-llm-huggingface.md", "group": "Code Drills (Bonus)"},
    {"slug": "code-drills-rag-langchain", "title": "Code Drills 8 — RAG & LangChain: Chunking to a Full Grounded Pipeline",
     "file": "code-drills-rag-langchain.md", "group": "Code Drills (Bonus)"},
    {"slug": "code-drills-finetuning-peft", "title": "Code Drills 9 — Fine-Tuning: LoRA, QLoRA, PEFT, the Trainer API",
     "file": "code-drills-finetuning-peft.md", "group": "Code Drills (Bonus)"},
    {"slug": "code-drills-langgraph-agents", "title": "Code Drills 10 — LangGraph: StateGraph, Tool-Calling Agents, Memory",
     "file": "code-drills-langgraph-agents.md", "group": "Code Drills (Bonus)"},
    {"slug": "practice-langchain", "title": "LangChain Practice (LCEL, RAG, agents, pitfalls)",
     "file": "langchain-practice.md", "group": "LLM App Frameworks Practice"},
    {"slug": "practice-langgraph", "title": "LangGraph Practice (StateGraph, agents, checkpointing, pitfalls)",
     "file": "langgraph-practice.md", "group": "LLM App Frameworks Practice"},
    {"slug": "llm-landscape", "title": "LLM Landscape (25 open/closed-source models + purposes)",
     "file": "llm-landscape.md", "group": "LLM Landscape Reference"},
    {"slug": "module-cheatsheet", "title": "Module & One-Line Command Cheat Sheet (94 entries: ML/NN/CNN/RNN/LSTM/LLM/Optimization/Training/Inferencing)",
     "file": "module-cheatsheet.md", "group": "Quick Reference"},
    {"slug": "common-issues", "title": "Common Issues & Failure Modes (what breaks, and why — ML/DL/LLM/production)",
     "file": "common-issues-failure-modes.md", "group": "Quick Reference"},
    {"slug": "git-scenarios", "title": "Git Commands for Real Scenarios (conflicts, rebase, bisect, LFS, undo)",
     "file": "git-scenarios-cheatsheet.md", "group": "Quick Reference"},
    {"slug": "sql-practice", "title": "SQL Practice (joins, window functions, CTEs, indexing)",
     "file": "sql-practice.md", "group": "Data & Math Foundations"},
    {"slug": "math-foundations", "title": "Math Foundations Refresher (linear algebra & probability)",
     "file": "math-foundations-refresher.md", "group": "Data & Math Foundations"},
    {"slug": "rag-deeper", "title": "RAG, Deeper (advanced retrieval, evaluation, knowledge graphs / GraphRAG)",
     "file": "rag-deeper.md", "group": "Advanced LLM Techniques"},
    {"slug": "prompt-engineering-deeper", "title": "Prompt Engineering, Deeper (ToT, Reflexion, DSPy, structured output, injection defense)",
     "file": "prompt-engineering-deeper.md", "group": "Advanced LLM Techniques"},
    {"slug": "mlops-practice", "title": "MLOps Practice (experiment tracking, model registries, DVC, CI/CD for ML)",
     "file": "mlops-practice.md", "group": "MLOps & Production"},
    {"slug": "production-ml", "title": "Production ML Practice (serving, monitoring, drift, rollback)",
     "file": "production-ml-practice.md", "group": "MLOps & Production"},
    {"slug": "leetcode-sql", "title": "LeetCode SQL (40 real problems: joins, window functions, self-joins, subqueries)",
     "file": "leetcode-sql.md", "group": "LeetCode Practice"},
    {"slug": "leetcode-pandas", "title": "LeetCode Pandas (35 real problems: filtering, groupby, merge, pivot/melt, strings/dates)",
     "file": "leetcode-pandas.md", "group": "LeetCode Practice"},
    {"slug": "leetcode-arrays-strings", "title": "LeetCode Arrays & Strings (60 problems: two pointers, sliding window, hashmap, stack, binary search, intervals)",
     "file": "leetcode-arrays-strings.md", "group": "LeetCode Practice"},
    {"slug": "leetcode-dp-trees-graphs", "title": "LeetCode DP, Trees & Graphs (43 problems: DP, backtracking, trees, BFS/DFS)",
     "file": "leetcode-dp-trees-graphs.md", "group": "LeetCode Practice"},
    {"slug": "leetcode-stats-probability", "title": "LeetCode Stats/Probability & \"From Scratch\" (25 DS-specific problems: sampling, implement ML from scratch)",
     "file": "leetcode-stats-probability.md", "group": "LeetCode Practice"},
    {"slug": "leetcode-system-design-structures", "title": "LeetCode Design-a-Data-Structure (12 problems: LRU Cache, Trie, Twitter/TinyURL/Rate Limiter, Alien Dictionary, Course Schedule II, Median of Two Sorted Arrays, Word Break II)",
     "file": "leetcode-system-design-structures.md", "group": "LeetCode Practice"},
    {"slug": "time-series", "title": "Time Series Analysis (trend/seasonality decomposition, stationarity, ACF/PACF, ARIMA, forecast evaluation)",
     "file": "time-series-analysis.md", "group": "Data & Math Foundations"},
    {"slug": "neural-net-numerical-practice", "title": "Neural Net Numericals (forward/backward pass, loss, weight updates, epochs/iterations, shapes — 20 worked MCQs)",
     "file": "neural-net-numerical-practice.md", "group": "Data & Math Foundations"},
]
DOCS_BY_SLUG = {d["slug"]: d for d in DOCS}


def render_doc_page(title: str, body_html: str, source: str) -> str:
    return DOC_TEMPLATE.replace("{{TITLE}}", title).replace("{{BODY}}", body_html).replace("{{SOURCE}}", source)


@app.route("/")
def index():
    groups = {}
    for d in DOCS:
        groups.setdefault(d["group"], []).append(d)

    sections = ""
    for group, docs in groups.items():
        items = "".join(f'<li><a href="/doc/{d["slug"]}">{d["title"]}</a></li>' for d in docs)
        sections += f"<h2>{group}</h2><ul class=\"doclist\">{items}</ul>"

    body = (
        "<h1>Study Hub</h1>"
        '<p class="lede">Everything in one place: BNSF Sr/Staff Data Scientist interview prep '
        "and the NVIDIA NCA-GENL exam guide, both with click-to-ask enabled on every heading.</p>"
        "<h2>NVIDIA NCA-GENL Exam</h2>"
        f'<ul class="doclist"><li><a href="/nca-genl">NCA-GENL Study Guide — 7 Days to Certified</a></li></ul>'
        "<h2>BNSF Interview Prep</h2>"
        '<ul class="doclist"><li><a href="/bnsf-visual">Technical Deep-Dive — Pictorial '
        "(optimization, geospatial, vector DBs, RAG/CoT/eval, GPU) — same interactive style "
        "as the transformer diagrams</a></li></ul>"
        "<h2>Data Science Fundamentals</h2>"
        '<ul class="doclist"><li><a href="/ds-fundamentals">Data Science Fundamentals — Pictorial '
        "(gradient descent, loss functions, bias-variance/overfitting, L1/L2 regularization) — "
        "same interactive diagram style</a></li></ul>"
        f"{sections}"
    )
    return render_doc_page("Study Hub", body, "study-hub-index")


@app.route("/nca-genl")
def nca_genl():
    return send_from_directory(ROOT, GUIDE_FILE)


@app.route("/bnsf-visual")
def bnsf_visual():
    return send_from_directory(ROOT, BNSF_VISUAL_FILE)


@app.route("/ds-fundamentals")
def ds_fundamentals():
    return send_from_directory(ROOT, DS_FUNDAMENTALS_FILE)


@app.route("/doc/<slug>")
def doc(slug):
    entry = DOCS_BY_SLUG.get(slug)
    if not entry:
        return "Not found", 404
    text = (ROOT / entry["file"]).read_text(encoding="utf-8")
    content_html = md.markdown(text, extensions=["fenced_code", "tables", "sane_lists", "toc"])
    body = f'<a class="backlink" href="/">&larr; All study materials</a>{content_html}'
    return render_doc_page(entry["title"], body, "bnsf-interview-prep")


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True, silent=True) or {}
    topic = (data.get("topic") or "").strip()
    context = (data.get("context") or "").strip()
    question = (data.get("question") or "").strip()
    history = data.get("history") or []
    source = (data.get("source") or "nca-genl").strip()  # older nca-genl page predates this field

    if not question:
        return jsonify({"error": "empty question"}), 400

    persona = {
        "nca-genl": (
            "You are a study assistant embedded in a personal exam-prep guide for the "
            "NVIDIA-Certified Associate: Generative AI LLMs (NCA-GENL) exam. Answer at the "
            "depth needed to actually understand and pass the exam."
        ),
        "bnsf-interview-prep": (
            "You are a study assistant embedded in a personal prep guide for a Sr/Staff Data "
            "Scientist interview loop at BNSF Railway (problem formulation, live coding, system "
            "design, service impact, and behavioral/partnership rounds). Answer at the depth "
            "expected in that interview loop."
        ),
        "ds-fundamentals": (
            "You are a study assistant embedded in a personal data-science-fundamentals guide "
            "(gradient descent, loss functions, bias-variance tradeoff, regularization, and "
            "related basics). Answer with the same rigor as the page: real formulas, real "
            "worked numbers, not hand-waving."
        ),
        "study-hub-index": "You are a study assistant helping navigate a personal interview/exam prep hub.",
    }.get(source, "You are a helpful study assistant.")

    system_prompt = (
        f"{persona}\n"
        f'The user clicked on the topic: "{topic}".\n'
        + (f"Relevant excerpt from the guide (for grounding, not necessarily to repeat):\n{context}\n\n" if context else "")
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
        (source, topic, question, answer, datetime.now(timezone.utc).isoformat()),
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
    print(f"Serving {GUIDE_FILE} on http://localhost:5000")
    app.run(port=5000, debug=False)
