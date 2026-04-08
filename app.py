"""
app.py — FastAPI + Gradio server for the AI Email Triage Environment.

Exposes OpenEnv REST API endpoints:
  POST /reset  → reset environment, returns observation
  POST /step   → take action, returns (obs, reward, done, info)
  GET  /state  → get current observation

Also serves Gradio UI at the root path for interactive demo.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, Any

import gradio as gr
from environment import EmailTriageEnv
from inference import RuleBasedAgent, run_evaluation

# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Email Triage - OpenEnv",
    description="OpenEnv environment for email classification, prioritization & reply generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared Environment Instance ───────────────────────────────────────────────

api_env = EmailTriageEnv()
agent = RuleBasedAgent()


# ── Pydantic Models ───────────────────────────────────────────────────────────

class StepAction(BaseModel):
    classify: Optional[str] = None
    priority: Optional[str] = None
    reply: Optional[str] = None


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetResponse(BaseModel):
    observation: Dict[str, Any]


class StateResponse(BaseModel):
    state: Dict[str, Any]


# ── OpenEnv REST API Endpoints ────────────────────────────────────────────────

@app.post("/reset", response_model=ResetResponse)
def reset_env():
    """Reset the environment and return initial observation."""
    obs = api_env.reset()
    return ResetResponse(observation=obs)


@app.post("/step", response_model=StepResponse)
def step_env(action: StepAction):
    """Take an action and return (observation, reward, done, info)."""
    action_dict = {}
    if action.classify:
        action_dict["classify"] = action.classify
    if action.priority:
        action_dict["priority"] = action.priority
    if action.reply:
        action_dict["reply"] = action.reply

    obs, reward, done, info = api_env.step(action_dict)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=StateResponse)
def get_state():
    """Return the current environment state."""
    return StateResponse(state=api_env.state())


@app.get("/metrics")
def get_metrics():
    """Return running evaluation metrics."""
    return api_env.get_metrics()


@app.get("/action_space")
def get_action_space():
    """Return the action space definition."""
    return api_env.action_space


@app.get("/observation_space")
def get_observation_space():
    """Return the observation space definition."""
    return api_env.observation_space


from fastapi.responses import RedirectResponse

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "environment": "email_triage", "version": "1.0.0"}

@app.get("/")
def root():
    """Redirect to the UI."""
    return RedirectResponse(url="/ui")


# ── Gradio UI ─────────────────────────────────────────────────────────────────

ui_env = EmailTriageEnv()
current_obs = None


def format_email_display(obs: dict) -> str:
    if not obs:
        return "No email loaded. Click **New Email** to start."
    meta = obs.get("metadata", {})
    return (
        f"### {obs['subject']}\n\n"
        f"**From:** `{obs['sender']}`\n\n"
        f"**Date:** {obs['timestamp']}\n\n"
        f"**Attachments:** {'Yes' if meta.get('has_attachments') else 'No'} | "
        f"**Reply:** {'Yes' if meta.get('is_reply') else 'No'} | "
        f"**Thread:** {meta.get('thread_length', 1)} messages\n\n"
        f"---\n\n{obs['body']}"
    )


def format_reward_breakdown(info: dict) -> str:
    if not info:
        return "No actions taken yet."

    lines = ["## Reward Breakdown\n"]

    cls_info = info.get("classification", {})
    if cls_info:
        icon = "CORRECT" if cls_info.get("correct") else "WRONG"
        lines.append(
            f"**Classification** [{icon}]\n"
            f"- Predicted: `{cls_info.get('predicted', 'N/A')}` "
            f"| Actual: `{cls_info.get('actual', 'N/A')}`\n"
            f"- Reward: **{cls_info.get('reward', 0):+.2f}**\n"
        )
        if cls_info.get("reason"):
            lines.append(f"- Note: {cls_info['reason']}\n")

    pri_info = info.get("priority", {})
    if pri_info:
        icon = "CORRECT" if pri_info.get("correct") else "WRONG"
        lines.append(
            f"\n**Priority** [{icon}]\n"
            f"- Predicted: `{pri_info.get('predicted', 'N/A')}` "
            f"| Actual: `{pri_info.get('actual', 'N/A')}`\n"
            f"- Reward: **{pri_info.get('reward', 0):+.2f}**\n"
        )
        if pri_info.get("reason"):
            lines.append(f"- Note: {pri_info['reason']}\n")

    rep_info = info.get("reply", {})
    if rep_info:
        lines.append(
            f"\n**Reply Quality**\n"
            f"- Similarity: {rep_info.get('similarity', 0):.2f} "
            f"| Length: {rep_info.get('length_score', 0):.2f} "
            f"| Professionalism: {rep_info.get('professionalism_score', 0):.2f}\n"
            f"- Reward: **{rep_info.get('reward', 0):+.3f}**\n"
        )

    total = info.get("episode_reward", info.get("step_reward", 0))
    lines.append(f"\n---\n### Total Reward: **{total:+.4f}**")

    return "\n".join(lines)


def load_new_email():
    global current_obs
    current_obs = ui_env.reset()
    email_display = format_email_display(current_obs)
    state_info = ui_env.state()
    remaining = state_info.get("episode_info", {}).get("actions_remaining", [])
    return (
        email_display,
        f"Email `{current_obs['email_id']}` loaded. Actions remaining: {remaining}",
        "No actions taken yet.",
    )


def submit_actions(classification, priority, reply_text):
    global current_obs
    if current_obs is None:
        return (
            "No email loaded. Click **New Email** first.",
            "Load an email first.",
            "No actions taken yet.",
        )

    action = {}
    if classification and classification != "-- Select --":
        action["classify"] = classification.lower()
    if priority and priority != "-- Select --":
        action["priority"] = priority.lower()
    if reply_text and reply_text.strip():
        action["reply"] = reply_text.strip()

    if not action:
        return (
            format_email_display(current_obs),
            "Please select at least one action.",
            "No actions submitted.",
        )

    obs, reward, done, info = ui_env.step(action)
    reward_display = format_reward_breakdown(info)

    status = f"Step reward: **{reward:+.4f}**"
    if done:
        status += " | Episode complete!"

    return format_email_display(current_obs), status, reward_display


def run_auto_agent():
    global current_obs
    if current_obs is None:
        return (
            "No email loaded. Click **New Email** first.",
            "Load an email first.",
            "No actions taken yet.",
        )

    action = agent.act(current_obs)
    obs, reward, done, info = ui_env.step(action)
    reward_display = format_reward_breakdown(info)

    action_str = (
        f"Classify: `{action['classify']}` | "
        f"Priority: `{action['priority']}` | "
        f"Reply: _{action['reply'][:60]}..._"
    )
    status = f"Agent action: {action_str}\n\nReward: **{reward:+.4f}**"
    if done:
        status += " | Episode complete!"

    return format_email_display(current_obs), status, reward_display


def run_batch_evaluation(num_episodes):
    batch_env = EmailTriageEnv()
    batch_agent = RuleBasedAgent()
    results = run_evaluation(batch_agent, batch_env, int(num_episodes), verbose=False)
    m = results["metrics"]

    summary = (
        "## Batch Evaluation Results\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Episodes | {m['total_episodes']} |\n"
        f"| Total Reward | {m['total_reward']:+.4f} |\n"
        f"| Avg Reward | {m['avg_reward']:+.4f} |\n"
        f"| Classification Accuracy | {m['classification_accuracy']:.2%} |\n"
        f"| Priority Accuracy | {m['priority_accuracy']:.2%} |\n"
        f"| Avg Reply Score | {m['avg_reply_score']:+.4f} |\n"
    )
    return summary


# Build Gradio interface
CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; }
.email-display { font-family: 'Segoe UI', Tahoma, Geneva, sans-serif; }
"""

with gr.Blocks(
    title="AI Email Triage - OpenEnv",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
) as demo:
    gr.Markdown(
        "# AI Email Triage & Response Environment\n"
        "*An OpenEnv training environment for email classification, "
        "prioritization, and response generation.*"
    )

    with gr.Tabs():
        with gr.Tab("Interactive Mode"):
            with gr.Row():
                with gr.Column(scale=3):
                    email_display = gr.Markdown(
                        "Click **New Email** to start.",
                        elem_classes=["email-display"],
                    )
                with gr.Column(scale=2):
                    status_display = gr.Markdown("Ready.")
                    reward_display = gr.Markdown("No actions taken yet.")

            gr.Markdown("---")

            with gr.Row():
                new_email_btn = gr.Button("New Email", variant="primary", size="lg")
                auto_agent_btn = gr.Button("Auto Agent", variant="secondary", size="lg")

            with gr.Row():
                classify_dropdown = gr.Dropdown(
                    choices=["-- Select --", "spam", "ham"],
                    value="-- Select --",
                    label="Classification",
                )
                priority_dropdown = gr.Dropdown(
                    choices=["-- Select --", "low", "medium", "high", "critical"],
                    value="-- Select --",
                    label="Priority",
                )

            reply_input = gr.Textbox(
                label="Reply",
                placeholder="Type your reply to the email...",
                lines=3,
            )

            submit_btn = gr.Button("Submit Actions", variant="primary", size="lg")

            new_email_btn.click(
                load_new_email,
                outputs=[email_display, status_display, reward_display],
            )
            submit_btn.click(
                submit_actions,
                inputs=[classify_dropdown, priority_dropdown, reply_input],
                outputs=[email_display, status_display, reward_display],
            )
            auto_agent_btn.click(
                run_auto_agent,
                outputs=[email_display, status_display, reward_display],
            )

        with gr.Tab("Batch Evaluation"):
            gr.Markdown(
                "Run the **Rule-Based Agent** on multiple emails and "
                "see aggregate performance metrics."
            )
            with gr.Row():
                num_episodes_slider = gr.Slider(
                    minimum=10, maximum=50, value=50, step=10,
                    label="Number of Episodes",
                )
                run_eval_btn = gr.Button(
                    "Run Evaluation", variant="primary", size="lg"
                )
            eval_results = gr.Markdown("Click **Run Evaluation** to start.")

            run_eval_btn.click(
                run_batch_evaluation,
                inputs=[num_episodes_slider],
                outputs=[eval_results],
            )

        with gr.Tab("About"):
            gr.Markdown(
                "## About This Environment\n\n"
                "This is an **OpenEnv** training environment for AI email triage.\n\n"
                "### Tasks\n"
                "1. **Spam Detection** (Easy) - Classify emails as spam or ham\n"
                "2. **Priority Assignment** (Medium) - Assign low/medium/high/critical\n"
                "3. **Reply Generation** (Hard) - Write an appropriate response\n\n"
                "### Reward System\n"
                "| Component | Max Reward | Penalty |\n"
                "|-----------|-----------|----------|\n"
                "| Classification | +0.4 | -0.2 to -0.3 |\n"
                "| Priority | +0.3 | -0.15 to -0.25 |\n"
                "| Reply Quality | +0.3 | -0.1 to -0.15 |\n\n"
                "### REST API Endpoints\n"
                "| Method | Endpoint | Description |\n"
                "|--------|----------|-------------|\n"
                "| POST | `/reset` | Reset environment |\n"
                "| POST | `/step` | Take action |\n"
                "| GET | `/state` | Get current state |\n"
                "| GET | `/metrics` | Get evaluation metrics |\n"
                "| GET | `/health` | Health check |\n\n"
                "### Python API\n"
                "```python\n"
                "from environment import EmailTriageEnv\n\n"
                "env = EmailTriageEnv()\n"
                "obs = env.reset()\n"
                "action = {'classify': 'ham', 'priority': 'high', "
                "'reply': 'Thank you...'}\n"
                "obs, reward, done, info = env.step(action)\n"
                "```\n\n"
                "---\n"
                "*Built for the OpenEnv AI Training Framework*"
            )

# Mount Gradio on FastAPI
app = gr.mount_gradio_app(app, demo, path="/ui")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
