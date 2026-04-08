---
title: AI Email Triage
emoji: 📬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: AI email classification, priority & reply env
---

# 📬 AI Email Triage & Response Environment

An **OpenEnv** training environment where an AI agent processes incoming emails, classifies them, assigns priority, and generates appropriate responses.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│               openenv.yaml (config)             │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│            environment.py (Core Env)            │
│         reset() · step() · state()              │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐ │
│  │ tasks.py │  │ grader.py│  │ inference.py  │ │
│  │ 50 emails│  │ rewards  │  │ rule agent    │ │
│  └──────────┘  └──────────┘  └───────────────┘ │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │    app.py (Gradio)    │
         │   Interactive Demo    │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │  Dockerfile (Deploy)  │
         │  HF Spaces · Docker   │
         └───────────────────────┘
```

## 🎯 Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| Spam Detection | Easy | Classify emails as `spam` or `ham` |
| Priority Assignment | Medium | Assign `low`, `medium`, `high`, or `critical` |
| Reply Generation | Hard | Generate an appropriate text reply |

## 🏆 Reward System

| Component | Correct | Penalty |
|-----------|---------|---------|
| Classification | +0.4 | -0.2 (spam→ham), -0.3 (ham→spam) |
| Priority | +0.3 (exact), +0.15 (off-by-one) | -0.15 (wrong), -0.25 (critical miss) |
| Reply Quality | Up to +0.3 | -0.15 (empty), -0.1 (nonsensical) |

**Maximum total reward per episode: +1.0**

## 🚀 Quick Start

### Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the evaluation
python inference.py --episodes 50

# Launch the Gradio UI
python app.py
```

### Docker

```bash
docker build -t email-triage .
docker run -p 7860:7860 email-triage
```

### Hugging Face Spaces

Upload all files to a new HF Space with **Docker** SDK.

## 💻 API Reference

```python
from environment import EmailTriageEnv

# Initialize
env = EmailTriageEnv()

# Start a new episode
obs = env.reset()
# obs = {"email_id", "sender", "subject", "body", "timestamp", "metadata"}

# View current state
state = env.state()

# Take actions (all at once or step-by-step)
action = {
    "classify": "ham",
    "priority": "high",
    "reply": "Thank you for the update. I'll review this promptly."
}
obs, reward, done, info = env.step(action)

# Get running metrics
metrics = env.get_metrics()

# Pretty-print current email
env.render()
```

## 📁 Project Structure

```
├── openenv.yaml       # Environment configuration
├── environment.py     # Core OpenEnv environment
├── tasks.py           # 50 synthetic emails with ground truth
├── grader.py          # Multi-dimensional reward system
├── inference.py       # Rule-based agent & evaluation harness
├── app.py             # Gradio web interface
├── Dockerfile         # Container for deployment
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## 📊 Dataset

50 synthetic emails across 5 categories:

- **Spam** (15): Lottery, phishing, pills, SEO, crypto scams
- **Critical** (5): Server outages, security breaches, CEO urgent
- **High** (10): Client escalations, deadlines, vulnerabilities
- **Medium** (10): Meetings, reviews, reports, PRs
- **Low** (10): Newsletters, social events, FYI messages

## 🛠️ Tech Stack

- **Python 3.11+**
- **scikit-learn** — TF-IDF vectorization & cosine similarity
- **Gradio** — Interactive web UI
- **PyYAML** — Configuration parsing
- **Docker** — Containerization

## 📜 License

MIT License
