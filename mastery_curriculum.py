"""
Curriculum graph for the Mastery Hub (mastery_server.py, port 5001).

Every topic here maps to a file that already exists on disk (served today by
server.py on port 5000). This module adds no new content — it adds ordering
(tiers, easiest first), cross-links (related topics across the old silos:
NCA-GENL / BNSF interview prep / DS fundamentals / practice docs), and a
place to hang per-topic mastery progress.

kind:
  "md"    -> rendered through doc_template.html via markdown, same as server.py's /doc/<slug>
  "html"  -> an existing self-contained interactive guide, served as-is
"""

TIERS = [
    {
        "id": -1,
        "name": "Reference",
        "subtitle": "Not a rung on the ladder — the module/one-line-command lookup you keep open in a tab while working through every tier below",
        "is_reference": True,
    },
    {
        "id": 0,
        "name": "Foundations",
        "subtitle": "Data & math substrate — everything above depends on this",
    },
    {
        "id": 1,
        "name": "Core ML & Problem Framing",
        "subtitle": "Classical models, how to frame a problem, the LLM map",
    },
    {
        "id": 2,
        "name": "Deep Learning & Transformers",
        "subtitle": "Neural nets, PyTorch/TF internals, the full transformer + NCA-GENL exam",
    },
    {
        "id": 3,
        "name": "LLM Systems & Applied Engineering",
        "subtitle": "RAG, agents, fine-tuning/serving at scale, applied to the freight-rail domain",
    },
    {
        "id": 4,
        "name": "MLOps & Production",
        "subtitle": "Shipping and operating what you built — tracking, versioning, serving, monitoring",
    },
    {
        "id": 5,
        "name": "Interview Performance & Synthesis",
        "subtitle": "Putting it all together under time pressure and in a room with people",
    },
]

TOPICS = [
    # ---------------- Tier -1: Reference (outside the ladder) ----------------
    {"id": "module-cheatsheet", "title": "Module & One-Line Command Cheat Sheet", "tier": -1, "kind": "md",
     "file": "module-cheatsheet.md",
     "blurb": "94 entries: task → exact module path → one-line call, for ML/NN/CNN/RNN/LSTM/LLM/optimization/training/inferencing. A lookup tool, not a tier to finish.",
     "related": ["practice-sklearn", "practice-pytorch-deep", "practice-deep-learning", "nca-genl",
                 "core-technical", "practice-langchain", "live-coding"]},
    {"id": "common-issues", "title": "Common Issues & Failure Modes", "tier": -1, "kind": "md",
     "file": "common-issues-failure-modes.md",
     "blurb": "What breaks and why, across classical ML, deep learning, LLM/RAG, and production — the reference you check when something's wrong, not before.",
     "related": ["ds-fundamentals", "practice-pytorch-deep", "rag-deeper", "production-ml", "practice-sklearn",
                 "practice-stats", "time-series"]},
    {"id": "git-scenarios", "title": "Git Commands for Real Scenarios", "tier": -1, "kind": "md",
     "file": "git-scenarios-cheatsheet.md",
     "blurb": "Lookup by situation — conflicts, rebase, bisect, undoing history, LFS for model files — not a git tutorial.",
     "related": ["mlops-practice", "live-coding"]},

    # ---------------- Tier 0: Foundations ----------------
    {"id": "practice-numpy", "title": "NumPy Practice", "tier": 0, "kind": "md",
     "file": "numpy-practice.md",
     "blurb": "Array mechanics, broadcasting, vectorization — the substrate everything else in Python DS sits on.",
     "related": ["practice-pandas", "practice-ml-models", "live-coding", "module-cheatsheet"]},
    {"id": "practice-pandas", "title": "Pandas Practice", "tier": 0, "kind": "md",
     "file": "pandas-practice.md",
     "blurb": "DataFrame wrangling, groupby, joins — the day-to-day tool for shaping data before modeling.",
     "related": ["practice-numpy", "practice-visualization", "practice-stats", "live-coding", "leetcode-pandas"]},
    {"id": "practice-stats", "title": "Statistics (SciPy/statsmodels) Practice", "tier": 0, "kind": "md",
     "file": "stats-scipy-practice.md",
     "blurb": "Distributions, hypothesis tests, p-values — the rigor behind 'is this difference real.'",
     "related": ["practice-pandas", "service-impact", "ds-fundamentals", "practice-ml-models",
                 "common-issues", "math-foundations", "leetcode-stats-probability", "time-series"]},
    {"id": "practice-utilities", "title": "Python Utilities Practice", "tier": 0, "kind": "md",
     "file": "python-utilities-practice.md",
     "blurb": "datetime/regex/I-O/performance idioms — the small stuff that live coding rounds quietly test.",
     "related": ["live-coding", "practice-pandas"]},
    {"id": "ds-fundamentals", "title": "Data Science Fundamentals — Pictorial", "tier": 0, "kind": "html",
     "route": "/m/ds-fundamentals",
     "blurb": "Gradient descent, loss functions, bias-variance, L1/L2 regularization — worked by hand, with diagrams.",
     "related": ["nca-genl", "practice-deep-learning", "practice-ml-models", "math-foundations"]},
    {"id": "practice-visualization", "title": "Data Visualization Practice", "tier": 0, "kind": "md",
     "file": "visualization-practice.md",
     "blurb": "Choosing and building the right chart — matplotlib/seaborn mechanics for telling the data's story.",
     "related": ["practice-pandas", "service-impact"]},
    {"id": "sql-practice", "title": "SQL Practice", "tier": 0, "kind": "md",
     "file": "sql-practice.md",
     "blurb": "Joins, GROUP BY/HAVING, window functions, CTEs, indexing — how data actually gets to you before pandas touches it.",
     "related": ["practice-pandas", "practice-stats", "live-coding", "leetcode-sql"]},
    {"id": "math-foundations", "title": "Math Foundations Refresher", "tier": 0, "kind": "md",
     "file": "math-foundations-refresher.md",
     "blurb": "Linear algebra and probability, worked with small real numbers — the math every other doc here quietly assumes.",
     "related": ["ds-fundamentals", "practice-numpy", "practice-stats", "nca-genl", "nn-numericals"]},
    {"id": "nn-numericals", "title": "Neural Net Numericals (Forward/Backward Pass, Loss, Weight Updates)", "tier": 0, "kind": "md",
     "file": "neural-net-numerical-practice.md",
     "blurb": "A toy network worked entirely in real numbers — forward pass, loss, backprop by hand, weight update, epochs/iterations, shapes — then a 20-MCQ drill.",
     "related": ["math-foundations", "practice-deep-learning", "practice-pytorch-deep", "nca-genl"]},

    # ---------------- Tier 1: Core ML & Problem Framing ----------------
    {"id": "practice-sklearn", "title": "scikit-learn Practice", "tier": 1, "kind": "md",
     "file": "sklearn-practice.md",
     "blurb": "Pipelines, cross-validation, the standard fit/predict API most classical models share.",
     "related": ["practice-ml-models", "practice-numpy", "practice-stats", "module-cheatsheet"]},
    {"id": "practice-ml-models", "title": "Classical ML Models Practice", "tier": 1, "kind": "md",
     "file": "ml-models-practice.md",
     "blurb": "One question at a time on trees, boosting, linear/logistic regression, clustering — how and when each works.",
     "related": ["practice-sklearn", "practice-stats", "ds-fundamentals", "problem-formulation"]},
    {"id": "problem-formulation", "title": "Problem Formulation Framework", "tier": 1, "kind": "md",
     "file": "problem-formulation-framework.md",
     "blurb": "Turning a vague business ask into a well-posed ML problem — the meta-skill every other round assumes.",
     "related": ["system-design", "service-impact", "domain-context"]},
    {"id": "llm-landscape", "title": "LLM Landscape", "tier": 1, "kind": "md", "role": "ai_engineer",
     "file": "llm-landscape.md",
     "blurb": "25 open/closed-source models and what each is actually for — the map you need before any LLM-systems topic.",
     "related": ["nca-genl", "core-technical", "practice-langchain"]},
    {"id": "time-series", "title": "Time Series Analysis", "tier": 1, "kind": "md",
     "file": "time-series-analysis.md",
     "blurb": "Trend/seasonality decomposition, stationarity, ACF/PACF, ARIMA, and why forecasts must be evaluated chronologically, not randomly.",
     "related": ["practice-stats", "practice-pandas", "practice-ml-models", "common-issues"]},

    # ---------------- Tier 2: Deep Learning & Transformers ----------------
    {"id": "practice-deep-learning", "title": "Deep Learning Practice (RNN/LSTM/CNN, PyTorch)", "tier": 2, "kind": "md",
     "file": "deep-learning-practice.md",
     "blurb": "Neural net basics in PyTorch — the bridge from classical ML to sequence/vision architectures.",
     "related": ["ds-fundamentals", "practice-pytorch-deep", "nca-genl", "module-cheatsheet"]},
    {"id": "practice-pytorch-deep", "title": "PyTorch Deep Dive", "tier": 2, "kind": "md",
     "file": "pytorch-deep-dive.md",
     "blurb": "Hooks, mixed precision, DDP, GANs, ONNX export — the framework internals behind production training.",
     "related": ["practice-deep-learning", "core-technical", "nca-genl", "module-cheatsheet"]},
    {"id": "practice-tf-keras-deep", "title": "TensorFlow/Keras Deep Dive", "tier": 2, "kind": "md",
     "file": "tensorflow-keras-deep-dive.md",
     "blurb": "Custom layers, GradientTape, transfer learning — the same concepts as PyTorch, different API surface.",
     "related": ["practice-deep-learning", "practice-pytorch-deep"]},
    {"id": "nca-genl", "title": "NCA-GENL Study Guide — 7 Days to Certified", "tier": 2, "kind": "html", "role": "ai_engineer",
     "route": "/m/nca-genl",
     "blurb": "The full transformer teardown with real numbers, plus prompting/RAG, evaluation metrics, and trustworthy AI — internally ordered easy→hard across its own 7-day plan.",
     "related": ["ds-fundamentals", "core-technical", "bnsf-visual", "llm-landscape", "module-cheatsheet"]},

    # ---------------- Tier 3: LLM Systems & Applied Engineering ----------------
    {"id": "core-technical", "title": "Core Technical Depth", "tier": 3, "kind": "md", "role": "ai_engineer",
     "file": "core-technical-depth.md",
     "blurb": "LoRA/QLoRA, RAG internals, DDP, quantization, VRP/MILP optimization — implementation-level, not exam-level.",
     "related": ["nca-genl", "practice-pytorch-deep", "practice-langchain", "domain-context", "module-cheatsheet"]},
    {"id": "practice-langchain", "title": "LangChain Practice", "tier": 3, "kind": "md", "role": "ai_engineer",
     "file": "langchain-practice.md",
     "blurb": "LCEL, RAG pipelines, tool-calling agents, and where each piece breaks in practice.",
     "related": ["practice-langgraph", "nca-genl", "llm-landscape", "module-cheatsheet"]},
    {"id": "practice-langgraph", "title": "LangGraph Practice", "tier": 3, "kind": "md", "role": "ai_engineer",
     "file": "langgraph-practice.md",
     "blurb": "StateGraph, multi-step agents, checkpointing — orchestration once a single chain isn't enough.",
     "related": ["practice-langchain", "system-design"]},
    {"id": "bnsf-visual", "title": "BNSF Technical Deep-Dive — Pictorial", "tier": 3, "kind": "html",
     "route": "/m/bnsf-visual",
     "blurb": "Optimization, geospatial, vector DBs, RAG/CoT/eval, GPU — same interactive diagram style as the transformer guide.",
     "related": ["nca-genl", "core-technical", "domain-context"]},
    {"id": "domain-context", "title": "Freight Rail AI Domain Context", "tier": 3, "kind": "md",
     "file": "freight-rail-ai-domain-context.md",
     "blurb": "The real use-case categories (hot-bearing detection etc.) — grounding for a domain you haven't worked in yet.",
     "related": ["problem-formulation", "service-impact", "core-technical"]},
    {"id": "service-impact", "title": "Service Impact & Causal Inference", "tier": 3, "kind": "md",
     "file": "service-impact-and-causal-inference.md",
     "blurb": "Proving an ML system actually moved a business metric — the causal-inference layer above plain statistics.",
     "related": ["problem-formulation", "practice-stats", "domain-context"]},
    {"id": "rag-deeper", "title": "RAG, Deeper (Advanced Retrieval & Knowledge Graphs)", "tier": 3, "kind": "md", "role": "ai_engineer",
     "file": "rag-deeper.md",
     "blurb": "Hybrid search, re-ranking, HyDE, multi-hop, RAGAS evaluation, GraphRAG/knowledge graphs — what makes basic RAG actually reliable.",
     "related": ["core-technical", "practice-langchain", "nca-genl", "common-issues"]},
    {"id": "prompt-engineering-deeper", "title": "Prompt Engineering, Deeper", "tier": 3, "kind": "md", "role": "ai_engineer",
     "file": "prompt-engineering-deeper.md",
     "blurb": "Tree-of-Thought, Reflexion, DSPy-style optimization, structured output, prompt-injection defense — past zero-shot/CoT basics.",
     "related": ["nca-genl", "practice-langgraph", "rag-deeper"]},

    # ---------------- Tier 4: MLOps & Production ----------------
    {"id": "mlops-practice", "title": "MLOps Practice", "tier": 4, "kind": "md",
     "file": "mlops-practice.md",
     "blurb": "Experiment tracking (MLflow/W&B), model registries, data/model versioning (DVC), CI/CD for ML.",
     "related": ["core-technical", "production-ml", "git-scenarios"]},
    {"id": "production-ml", "title": "Production ML Practice", "tier": 4, "kind": "md",
     "file": "production-ml-practice.md",
     "blurb": "Batch vs. real-time serving, monitoring, model/data drift, canary deploys, rollback — what happens after training ends.",
     "related": ["mlops-practice", "common-issues", "system-design"]},

    # ---------------- Tier 5: Interview Performance & Synthesis ----------------
    {"id": "live-coding", "title": "Live Coding Prep", "tier": 5, "kind": "md",
     "file": "live-coding-prep.md",
     "blurb": "Timed problem-solving with the libraries from Tier 0 — the pressure-test of the foundations.",
     "related": ["practice-numpy", "practice-pandas", "practice-utilities", "module-cheatsheet",
                 "leetcode-arrays-strings", "leetcode-dp-trees-graphs"]},
    {"id": "leetcode-sql", "title": "LeetCode SQL", "tier": 5, "kind": "md",
     "file": "leetcode-sql.md",
     "blurb": "40 real problems — joins, self-joins, window functions, subqueries. Memorize-format: problem, solution, one-line why.",
     "related": ["sql-practice", "live-coding"]},
    {"id": "leetcode-pandas", "title": "LeetCode Pandas", "tier": 5, "kind": "md",
     "file": "leetcode-pandas.md",
     "blurb": "35 real problems — filtering, groupby, merge, pivot/melt, string/datetime ops, method chaining.",
     "related": ["practice-pandas", "practice-numpy", "live-coding"]},
    {"id": "leetcode-arrays-strings", "title": "LeetCode Arrays & Strings", "tier": 5, "kind": "md",
     "file": "leetcode-arrays-strings.md",
     "blurb": "60 canonical problems grouped by pattern — two pointers, sliding window, hashmap, stack, binary search, intervals.",
     "related": ["live-coding", "module-cheatsheet"]},
    {"id": "leetcode-dp-trees-graphs", "title": "LeetCode DP, Trees & Graphs", "tier": 5, "kind": "md",
     "file": "leetcode-dp-trees-graphs.md",
     "blurb": "43 problems — dynamic programming, backtracking, tree/graph traversal (BFS/DFS). Recursion is the throughline.",
     "related": ["live-coding", "practice-langgraph"]},
    {"id": "leetcode-stats-probability", "title": "LeetCode Stats/Probability & \"From Scratch\"", "tier": 5, "kind": "md",
     "file": "leetcode-stats-probability.md",
     "blurb": "25 DS-specific problems — reservoir sampling, weighted random pick, implementing train_test_split/k-means/linear regression from scratch.",
     "related": ["practice-stats", "math-foundations", "practice-numpy"]},
    {"id": "leetcode-system-design-structures", "title": "LeetCode Design-a-Data-Structure", "tier": 5, "kind": "md", "role": "ai_engineer",
     "file": "leetcode-system-design-structures.md",
     "blurb": "LRU Cache, Trie, Design Twitter/TinyURL/Rate Limiter, Alien Dictionary, Course Schedule II, Median of Two Sorted Arrays, Word Break II — the 'design a class' coding-round cluster reported most often in AI Engineer loops specifically.",
     "related": ["leetcode-arrays-strings", "leetcode-dp-trees-graphs", "live-coding"]},
    {"id": "system-design", "title": "System Design Prep", "tier": 5, "kind": "md",
     "file": "system-design-prep.md",
     "blurb": "Architecting a full ML system end to end — synthesizes problem framing, LLM systems, and scale tradeoffs.",
     "related": ["problem-formulation", "core-technical", "practice-langgraph", "production-ml", "my-projects"]},
    {"id": "behavioral", "title": "Behavioral / Partnership STAR Stories", "tier": 5, "kind": "md",
     "file": "behavioral-partnership-star-stories.md",
     "blurb": "STAR stories drawn from real project work — pairs naturally with the technical stories above.",
     "related": ["problem-formulation", "service-impact", "domain-context", "my-projects"]},
    {"id": "my-projects", "title": "My Projects — Architecture, Code & Module Breakdown", "tier": 5, "kind": "md",
     "file": "my-projects-portfolio.md",
     "blurb": "FinSight, NaviDoc, and every other real project on this hub — 3-level zoom (architecture → code → modules), visual, grounded only in facts stated elsewhere here.",
     "related": ["system-design", "behavioral", "core-technical", "domain-context"]},
]

TOPICS_BY_ID = {t["id"]: t for t in TOPICS}
TIERS_BY_ID = {t["id"]: t for t in TIERS}

for _t in TOPICS:
    _t["tier_name"] = TIERS_BY_ID[_t["tier"]]["name"]

# sanity: every related id must point at a real topic
for _t in TOPICS:
    for _r in _t["related"]:
        assert _r in TOPICS_BY_ID, f"{_t['id']} links to unknown topic {_r}"
