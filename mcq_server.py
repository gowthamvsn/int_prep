import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

load_dotenv()

ROOT = Path(__file__).resolve().parent
FILE = "NCA-GENL-mcq-200.html"

os.environ["ANTHROPIC_API_KEY"]  # fail fast at startup if unset
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5")
claude_client = anthropic.Anthropic()

app = Flask(__name__, static_folder=None)

PERSONA = (
    "You are a study assistant embedded in a personal 200-question practice-MCQ bank for the "
    "NVIDIA-Certified Associate: Generative AI LLMs (NCA-GENL) exam. The user is asking about one "
    "specific practice question -- explain the concept clearly enough that they'd get it right next "
    "time, at the depth needed to actually pass the exam. Use examples or numbers where they help."
)


@app.route("/")
def index():
    return send_from_directory(ROOT, FILE)


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
        f"{PERSONA}\n"
        f'The user clicked on the question: "{topic}".\n'
        + (f"The question, its options, correct answer, and explanation:\n{context}\n\n" if context else "")
        + "Answer clearly and concretely. Keep it focused: a few short paragraphs or a tight list, not an essay. "
        "Never use LaTeX (no \\[, \\(, $$, or similar delimiters) — this page does not render it, "
        "so it would show up as raw backslashes. Write all math as plain text arithmetic instead, "
        "e.g. `SE = s / sqrt(n) = 10 / 5 = 2`."
    )

    messages = []
    for turn in history[-6:]:
        role, content = turn.get("role"), turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})

    try:
        resp = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            output_config={"effort": "medium"},
            system=system_prompt,
            messages=messages,
        )
    except anthropic.APIConnectionError as exc:
        return jsonify({"error": f"could not reach the Claude API: {exc}"}), 502
    except anthropic.APIStatusError as exc:
        return jsonify({"error": f"Claude API error {exc.status_code}: {exc.message}"}), 502

    if resp.stop_reason == "refusal":
        return jsonify({"error": "Claude declined to answer this one — try rephrasing the question."}), 502

    answer = next((b.text for b in resp.content if b.type == "text"), "")
    if resp.stop_reason == "max_tokens":
        answer += "\n\n*(cut off — hit the response length limit; ask a narrower follow-up to continue)*"

    return jsonify({"answer": answer})


if __name__ == "__main__":
    print(f"Serving {FILE} on http://localhost:5002")
    app.run(port=5002, debug=False)
