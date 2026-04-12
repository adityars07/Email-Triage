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
        
        # Best practice: Load from environment variables (Secrets in Hugging Face)
        base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("API_KEY") 

        self.models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        self.model_index = 0

        if not base_url or not api_key:
            print("[LLM WARNING] API_BASE_URL or API_KEY not found in environment.")
            self.client = None
        else:
            self.client = OpenAI(base_url=base_url, api_key=api_key)

    def act(self, observation: Dict, retry: bool = True) -> Dict[str, str]:
        if self.client is None:
            return {"classify": "ham", "priority": "low", "reply": "Thank you for your email. [Agent unconfigured]"}

        prompt = f'''You are a Senior AI Email Triage Specialist. You are graded on accuracy and technical helpfulness.

### GRADING RUBRIC:
1. **Classification**: Correct "spam" vs "ham".
2. **Priority**: Assign based on business impact: "low", "medium", "high", "critical".
3. **Reply Quality (TF-IDF Similarity)**:
   - Use a formal greeting (e.g., "Dear [Sender],") and sign-off.
   - **ENTITY MATCHING**: To get a high score, your reply MUST explicitly mention specific technical names, server IDs (e.g., db-prod-01), project names, and key terms found in the email body.
   - Be professional and helpful.

### CURRENT EMAIL TO PROCESS:
- **Sender**: {observation.get('sender')}
- **Subject**: {observation.get('subject')}
- **Body**: {observation.get('body')}

### OUTPUT FORMAT:
You MUST output a JSON object with exactly three keys: "classify", "priority", "reply".

JSON:'''

        try:
            model = self.models[self.model_index]
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise triage assistant. Always output valid JSON. Use specific technical terms, names, and IDs from the email in your replies."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            err_msg = str(e).lower()
            # If the model is not found or access is denied, try the next one in the list
            if ("model_not_found" in err_msg or "not exist" in err_msg or "access" in err_msg) and self.model_index < len(self.models) - 1:
                print(f"[LLM REDIRECT] Model {self.models[self.model_index]} failed, trying {self.models[self.model_index+1]}...", flush=True)
                self.model_index += 1
                return self.act(observation, retry=retry)
                
            print(f"[LLM ERROR] Try failed: {e}", flush=True)
            if retry:
                return self.act(observation, retry=False)
            
            return {"classify": "ham", "priority": "low", "reply": "Thank you for your email. We have received your message and will follow up with the relevant team members shortly. Best regards, AI Triage Team."}


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
        "critical", "down", "outage", "breach", "failure", "emergency",
        "unreachable", "500 error", "security breach", "data exposed",
        "payment.*fail", "billing failure", "immediately", "ASAP",
        "revenue impact", "regulatory deadline", "hacked", "stolen",
        "incident", "production", "live", "db down",
    ],
    "high": [
        "urgent", "escalation", "cancel contract", "deadline", "important",
        "threatening", "resignation", "launch", "approval needed",
        "vulnerability", "CVE", "critical bug", "investment", "legal",
        "term sheet", "press release", "downtime window", "overdue",
    ],
    "medium": [
        "review", "feedback", "meeting", "sprint", "update", "task",
        "workshop", "report", "performance review", "pull request",
        "documentation", "quarterly", "mentor", "KPI", "question",
    ],
}

REPLY_TEMPLATES = {
    "critical": (
        "Dear Team,\n\nI have received the critical alert and am prioritizing this immediately. "
        "I am initiating an emergency investigation into the root cause and will coordinate "
        "with all relevant stakeholders for a rapid manual failover if necessary. "
        "I will provide status updates every 15 minutes until resolution.\n\n"
        "Best regards,\nAI Triage Assistant"
    ),
    "high": (
        "Hello,\n\nThank you for flagging this high-priority issue. I have reviewed the "
        "details and am assigning the necessary senior resources to address this urgently. "
        "I will ensure we meet the current project deadlines and provide a detailed "
        "status update on our remediation progress by end of day.\n\n"
        "Sincerely,\nAI Triage Assistant"
    ),
    "medium": (
        "Hi,\n\nThanks for reaching out with this update. I have received the information "
        "and will review it in detail during my next scheduled task block to ensure alignment "
        "with our goals. I'll follow up with any specific questions or feedback within the "
        "next 48 hours.\n\n"
        "Best regards,\nAI Triage Assistant"
    ),
    "low": (
        "Hi there,\n\nThank you for sharing this information with me! I have noted the "
        "details for future reference and will review the materials when my current "
        "high-priority milestones are completed. I appreciate the heads-up and your proactive "
        "communication.\n\n"
        "Best regards,\nAI Triage Assistant"
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

        try:
            action = agent.act(obs)
            obs_next, reward, done, info = env.step(action)
            clamped = clamp_score(reward)
            
            print(f"[STEP] step=1 reward={clamped}", flush=True)
            print(f"[END] task={task_name} score={clamped} steps=1", flush=True)

            result = {
                "episode": ep + 1,
                "email_id": obs.get("email_id", "unknown"),
                "reward": reward,
                "info": info,
            }
            results.append(result)

            if verbose:
                cls_icon = "Y" if info.get("classification", {}).get("correct") else "N"
                pri_icon = "Y" if info.get("priority", {}).get("correct") else "N"
                rep_score = info.get("reply", {}).get("reward", 0.0)

                print(
                    f"  [{ep+1:3d}/{num_episodes}] {obs.get('email_id', 'unknown'):<12} | "
                    f"Class: {cls_icon}  Pri: {pri_icon}  "
                    f"Reply: {rep_score:+.3f}  | "
                    f"Total: {reward:+.4f}",
                    flush=True
                )
        except Exception as e:
            fallback_score = clamp_score(0.01)
            print(f"[LLM ERROR] Task {task_name} failed: {e}", flush=True)
            print(f"[STEP] step=1 reward={fallback_score}", flush=True)
            print(f"[END] task={task_name} score={fallback_score} steps=1", flush=True)
            
            results.append({
                "episode": ep + 1,
                "email_id": obs.get("email_id", "unknown"),
                "reward": 0.01,
                "info": {"error": str(e)},
            })

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
