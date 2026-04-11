"""
inference.py — Reference Agent & Evaluation Harness.

Includes:
  - RuleBasedAgent: heuristic baseline for email triage
  - run_evaluation: full evaluation loop with metrics
  - CLI interface for running evaluations
"""

import argparse
import json
import re
import os
from typing import Dict, List
from openai import OpenAI

from environment import EmailTriageEnv


# ── LLM Agent ─────────────────────────────────────────────────────────────────

class LLMAgent:
    def __init__(self):
        self.name = "LLMAgent"
        import os
        from openai import OpenAI
        self.client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"]
        )

    def act(self, observation: Dict) -> Dict[str, str]:
        prompt = f'''You are an AI Email triage agent.
Read the following email and output a JSON object with three keys:
1. "classify": either "spam" or "ham"
2. "priority": if ham, assign "low", "medium", "high", or "critical". if spam, assign "none".
3. "reply": if ham, generate a short professional reply. if spam, return "".

Email content:
Sender: {observation.get('sender')}
Subject: {observation.get('subject')}
Body: {observation.get('body')}

Output ONLY valid JSON.'''

        import json
        import re

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            content = response.choices[0].message.content
            
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(content)
        except Exception as e:
            print(f"[LLM ERROR] Exception during chat.completions.create: {e}", flush=True)
            print("[LLM FALLBACK] Returning default response", flush=True)
            return {"classify": "ham", "priority": "low", "reply": "Thank you for your email. We will process it shortly."}


# ── Rule-Based Agent ──────────────────────────────────────────────────────────

SPAM_KEYWORDS = [
    "winner", "lottery", "prize", "congratulations", "claim",
    "click here", "verify your", "act now", "limited time",
    "free", "guaranteed", "no credit check", "unsubscribe",
    "buy now", "order now", "discount", "cheap", "pills",
    "weight loss", "earn money", "work from home", "bitcoin",
    "crypto", "moonshot", "1000x", "nigerian", "prince",
    "account suspended", "account locked", "verify identity",
    "bank details", "credit card", "prescription", "pharmacy",
    "singles", "dating", "hot", "bulk email", "million emails",
    "rolex", "luxury watches", "gift card", "survey",
    "pre-approved", "loan", "0% interest",
]

SPAM_SENDER_PATTERNS = [
    r"\.xyz$", r"\.biz$", r"\.click$", r"\.ru$", r"\.net$",
    r"verify", r"alert", r"security.*\.(com|net|xyz)",
    r"cheap", r"discount", r"pharmacy", r"pills",
    r"lottery", r"prize", r"dating", r"singles",
]

URGENCY_KEYWORDS = {
    "critical": [
        "critical", "down", "outage", "breach", "failure",
        "unreachable", "500 error", "security breach", "data exposed",
        "payment.*fail", "emergency", "immediately", "ASAP",
        "revenue impact", "regulatory deadline",
    ],
    "high": [
        "urgent", "escalation", "cancel contract", "deadline",
        "threatening", "resignation", "launch", "approval needed",
        "vulnerability", "CVE", "critical bug", "investment",
        "term sheet", "press release", "downtime window",
    ],
    "medium": [
        "review", "feedback", "meeting", "sprint", "update",
        "workshop", "report", "performance review", "pull request",
        "documentation", "quarterly", "mentor", "KPI",
    ],
}

REPLY_TEMPLATES = {
    "critical": (
        "Acknowledged — this is top priority. I'm taking immediate action "
        "and will provide a status update within 15 minutes. "
        "I'll coordinate with the relevant teams right away."
    ),
    "high": (
        "Thank you for flagging this. I'll prioritize this and address it "
        "promptly. I'll review the details and provide my response by "
        "end of day. Let's schedule a follow-up if needed."
    ),
    "medium": (
        "Thanks for the update. I'll review this and follow up with my "
        "input within the next few days. Please let me know if anything "
        "changes in the meantime."
    ),
    "low": (
        "Thanks for sharing! Noted — I'll take a look when I have a chance. "
        "Appreciate the heads up."
    ),
}


class RuleBasedAgent:
    """Heuristic-based agent for email triage tasks."""

    def __init__(self):
        self.name = "RuleBasedAgent"

    def act(self, observation: Dict) -> Dict[str, str]:
        """
        Given an email observation, return all three actions at once.

        Args:
            observation: dict with sender, subject, body, metadata

        Returns:
            {"classify": ..., "priority": ..., "reply": ...}
        """
        classification = self._classify(observation)
        priority = self._prioritize(observation) if classification == "ham" else "none"
        reply = self._generate_reply(observation, priority) if classification == "ham" else ""

        return {
            "classify": classification,
            "priority": priority,
            "reply": reply,
        }

    def _classify(self, obs: Dict) -> str:
        """Spam detection via keyword + sender pattern matching."""
        text = f"{obs.get('subject', '')} {obs.get('body', '')}".lower()
        sender = obs.get("sender", "").lower()

        # Count spam keyword hits
        spam_score = sum(1 for kw in SPAM_KEYWORDS if kw in text)

        # Check sender patterns
        for pattern in SPAM_SENDER_PATTERNS:
            if re.search(pattern, sender):
                spam_score += 2

        # Suspicious indicators
        if "!" in obs.get("subject", "") and "$" in text:
            spam_score += 2
        if any(emoji in obs.get("subject", "") for emoji in ["🎉", "🚀", "💰"]):
            spam_score += 1
        if "http://" in text and "https://" not in text:
            spam_score += 1

        return "spam" if spam_score >= 3 else "ham"

    def _prioritize(self, obs: Dict) -> str:
        """Priority assignment via urgency keyword analysis."""
        text = f"{obs.get('subject', '')} {obs.get('body', '')}".lower()

        for priority in ["critical", "high", "medium"]:
            keywords = URGENCY_KEYWORDS[priority]
            hits = sum(1 for kw in keywords if re.search(kw, text))
            if priority == "critical" and hits >= 2:
                return "critical"
            elif priority == "high" and hits >= 2:
                return "high"
            elif priority == "medium" and hits >= 1:
                return "medium"

        return "low"

    def _generate_reply(self, obs: Dict, priority: str) -> str:
        """Template-based reply generation."""
        template = REPLY_TEMPLATES.get(priority, REPLY_TEMPLATES["medium"])
        return template


# ── Evaluation Harness ────────────────────────────────────────────────────────

def clamp_score(score: float) -> float:
    """Ensure score is strictly between 0 and 1 (exclusive)."""
    epsilon = 1e-6
    return max(epsilon, min(1.0 - epsilon, float(score)))

def run_evaluation(
    agent,
    env: EmailTriageEnv,
    num_episodes: int = 50,
    verbose: bool = True,
) -> Dict:
    """
    Run a full evaluation of the agent on the environment.

    Args:
        agent: object with an act(observation) → action method
        env: EmailTriageEnv instance
        num_episodes: number of emails to evaluate
        verbose: print per-episode results

    Returns:
        Summary dict with metrics and per-episode results
    """
    results = []

    if verbose:
        print("=" * 70)
        print(f"  EVALUATION: {agent.name} | {num_episodes} episodes")
        print("=" * 70)

    for ep in range(num_episodes):
        obs = env.reset()
        task_name = str(obs.get('email_id', f'task_{ep+1}'))
        print(f"[START] task={task_name}", flush=True)

        action = agent.act(obs)
        obs_next, reward, done, info = env.step(action)

        clamped = clamp_score(reward)
        print(f"[STEP] step=1 reward={clamped}", flush=True)
        print(f"[END] task={task_name} score={clamped} steps=1", flush=True)

        result = {
            "episode": ep + 1,
            "email_id": obs["email_id"],
            "reward": reward,
            "info": info,
        }
        results.append(result)

        if verbose:
            cls_icon = "Y" if info.get("classification", {}).get("correct") else "N"
            pri_icon = "Y" if info.get("priority", {}).get("correct") else "N"
            rep_score = info.get("reply", {}).get("reward", 0.0)

            print(
                f"  [{ep+1:3d}/{num_episodes}] {obs['email_id']:<12} | "
                f"Class: {cls_icon}  Pri: {pri_icon}  "
                f"Reply: {rep_score:+.3f}  | "
                f"Total: {reward:+.4f}",
                flush=True
            )

    metrics = env.get_metrics()

    if verbose:
        print("=" * 70)
        print("  RESULTS SUMMARY")
        print("-" * 70)
        print(f"  Total Episodes:          {metrics['total_episodes']}")
        print(f"  Total Reward:            {metrics['total_reward']:+.4f}")
        print(f"  Avg Reward per Episode:  {metrics['avg_reward']:+.4f}")
        print(f"  Classification Accuracy: {metrics['classification_accuracy']:.2%}")
        print(f"  Priority Accuracy:       {metrics['priority_accuracy']:.2%}")
        print(f"  Avg Reply Score:         {metrics['avg_reply_score']:+.4f}")
        print("=" * 70)

    return {
        "metrics": metrics,
        "per_episode": results,
    }


# ── CLI Interface ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Email Triage — Inference & Evaluation"
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of episodes to evaluate (default: 50)"
    )
    parser.add_argument(
        "--agent", type=str, default="llm",
        choices=["rule_based", "llm"],
        help="Agent to use (default: llm)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save results JSON"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-episode output"
    )
    args = parser.parse_args()

    env = EmailTriageEnv()

    if args.agent == "llm":
        agent = LLMAgent()
    else:
        agent = RuleBasedAgent()

    results = run_evaluation(
        agent=agent,
        env=env,
        num_episodes=args.episodes,
        verbose=not args.quiet,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
