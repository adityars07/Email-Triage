# 📬 AI Email Triage & Response Environment: Project Overview

This project is a training and evaluation environment for AI agents, designed to simulate real-world email triage. It uses the **OpenEnv** framework to provide a standardized interface for agents to process incoming emails, classify them, assign priority, and generate professional responses.

---

## 🏗️ Repository Structure

The project follows a standard layout for OpenEnv environments:

```text
Email-Triage/
├── Dockerfile          # Container configuration for deployment (e.g., HF Spaces)
├── pyproject.toml      # Project metadata and dependency management
├── requirements.txt    # Python dependencies list
├── uv.lock             # Deterministic lock file for dependencies
├── openenv.yaml        # Main configuration for tasks, rewards, and evaluation
├── environment.py      # Core logic: handles reset(), step(), and state() transitions
├── tasks.py            # Dataset: 50 synthetic emails with ground truth labels
├── grader.py           # Evaluation logic: computes rewards for agent performance
├── inference.py        # Agent implementation (Rule-based & LLM) and eval harness
├── app.py              # Gradio web interface for interactive triage
└── server/             # Server entry points for automated validation
    ├── __init__.py     # Package initializer
    └── app.py          # FastAPI server for OpenEnv validation
```

---

## 🧩 Core Components & Logic

### 1. `environment.py` (The Engine)
This is the heart of the environment. It translates high-level configurations into a functional state machine.
- **`reset()`**: Loads a new email from the task pool.
- **`step(action)`**: Processes agent actions (`classify`, `priority`, `reply`), computes rewards via the grader, and checks if the episode is finished.
- **`state()`**: Returns the current observation (email body, sender, etc.) to the agent.

### 2. `tasks.py` (The Data)
Contains a diverse set of 50 synthetic emails categorized into:
- **Spam**: 15 emails (phishing, lotteries, crypto scams).
- **Critical**: 5 emails (server outages, security breaches).
- **High**: 10 emails (client escalations, legal deadlines).
- **Medium**: 10 emails (sprint reviews, meeting requests).
- **Low**: 10 emails (newsletters, social events).

### 3. `grader.py` (The Judge)
Uses a multi-dimensional reward system (Max reward: +1.0 per email):
- **Classification (+0.4)**: Correctly identifying `spam` vs `ham`.
- **Priority (+0.3)**: Correctly assigning level (`low` to `critical`). Includes partial credit for "off-by-one" errors.
- **Reply quality (+0.3)**: Uses TF-IDF cosine similarity to ground truth, length checks, and professionalism heuristics (greetings, sign-offs).

### 4. `inference.py` (The Agent)
Provides two reference implementations:
- **`RuleBasedAgent`**: Uses keyword matching and regex patterns (heuristic baseline).
- **`LLMAgent`**: Uses an LLM (e.g., GPT-3.5/4) via the OpenAI API to reason about the email content.

---

## 🛠️ Technology Stack

- **Python 3.11+**: Core programming language.
- **OpenEnv Core**: Framework for standardized environment benchmarking.
- **scikit-learn**: Used in `grader.py` for TF-IDF vectorization and text similarity.
- **Gradio / FastAPI**: For the interactive web UI and API endpoints.
- **Docker**: For consistent deployment across environments.
- **PyYAML**: For parsing the environment configuration.

---

## 🚀 How to Use

### Local Evaluation
Run the evaluation harness to test the agent on all 50 tasks:
```bash
python inference.py --episodes 50 --agent rule_based
```

### Interactive UI
Launch the Gradio interface to manually triage emails or watch the agent work:
```bash
python app.py
```

### Validation
Ensure the repository meets structure requirements for `openenv`:
```bash
pip install openenv-core
openenv validate
```

---

## 📈 Key Metrics Tracked
- **Total Reward**: Cumulative score across all episodes.
- **Classification Accuracy**: % of emails correctly identified as spam/ham.
- **Priority Accuracy**: % of correct priority levels assigned.
- **Avg Reply Score**: Mid-level quality metric for generated text.
