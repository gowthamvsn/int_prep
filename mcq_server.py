import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

load_dotenv()

ROOT = Path(__file__).resolve().parent
FILE = "NCA-GENL-mcq-200.html"

AZURE_KEY = os.environ["AZURE_OPENAI_KEY"]
AZURE_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
AZURE_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")

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

    return jsonify({"answer": answer})


if __name__ == "__main__":
    print(f"Serving {FILE} on http://localhost:5002")
    app.run(port=5002, debug=False)
