"""
environment.py — Core OpenEnv Environment for AI Email Triage.

Provides a gymnasium-like interface with:
  - reset()  → start a new episode (new email)
  - state()  → observe current email
  - step()   → take an action and receive a reward
"""

import os
import copy
from typing import Dict, Tuple, Optional, Any

import yaml

from tasks import TaskPool, ALL_EMAILS
from grader import EmailTriageGrader


class EmailTriageEnv:
    """
    AI Email Triage & Response Environment.

    Each episode presents one email. The agent can take up to 3 actions:
      1. classify  → "spam" or "ham"
      2. priority  → "low", "medium", "high", "critical"
      3. reply     → free-text response

    Actions can be submitted individually or all at once.
    The episode ends when all 3 actions are taken or max steps reached.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the environment.

        Args:
            config_path: Path to openenv.yaml. Defaults to ./openenv.yaml.
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "openenv.yaml")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        episode_cfg = self.config.get("episode", {})
        self.max_steps = episode_cfg.get("max_steps_per_email", 3)
        shuffle = episode_cfg.get("shuffle_emails", True)

        self.task_pool = TaskPool(shuffle=shuffle)
        self.grader = EmailTriageGrader(self.config)

        # Episode state
        self._current_email: Optional[Dict] = None
        self._actions_taken: Dict[str, str] = {}
        self._step_count: int = 0
        self._done: bool = True
        self._episode_reward: float = 0.0

        # Running metrics
        self._total_episodes: int = 0
        self._total_reward: float = 0.0
        self._classification_correct: int = 0
        self._classification_total: int = 0
        self._priority_correct: int = 0
        self._priority_total: int = 0
        self._reply_scores: list = []

    # ── Core API ──────────────────────────────────────────────────────────

    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment and start a new episode.

        Returns:
            Initial observation dict (email without ground truth).
        """
        self._current_email = self.task_pool.sample()
        self._actions_taken = {}
        self._step_count = 0
        self._done = False
        self._episode_reward = 0.0
        return self._get_observation()

    def state(self) -> Dict[str, Any]:
        """
        Return the current state / observation.

        Returns:
            Current observation dict with episode metadata.
        """
        if self._current_email is None:
            return {"error": "No active episode. Call reset() first."}

        obs = self._get_observation()
        obs["episode_info"] = {
            "step": self._step_count,
            "max_steps": self.max_steps,
            "actions_taken": list(self._actions_taken.keys()),
            "actions_remaining": [
                a for a in ["classify", "priority", "reply"]
                if a not in self._actions_taken
            ],
            "done": self._done,
            "current_reward": self._episode_reward,
        }
        return obs

    def step(self, action: Dict[str, str]) -> Tuple[Dict, float, bool, Dict]:
        """
        Take an action in the environment.

        Args:
            action: Dict with one or more of:
                - "classify": "spam" or "ham"
                - "priority": "low", "medium", "high", "critical"
                - "reply": free-text string

        Returns:
            (observation, reward, done, info)
        """
        if self._done:
            return (
                self._get_observation(),
                0.0,
                True,
                {"error": "Episode is done. Call reset() to start a new one."},
            )

        if self._current_email is None:
            return ({}, 0.0, True, {"error": "No active episode. Call reset() first."})

        # Validate and record actions
        step_reward = 0.0
        step_info = {}
        gt = self._current_email["ground_truth"]

        for action_type in ["classify", "priority", "reply"]:
            if action_type in action and action_type not in self._actions_taken:
                self._actions_taken[action_type] = action[action_type]

                if action_type == "classify":
                    r, detail = self.grader.grade_classification(
                        action[action_type], gt["classification"]
                    )
                    step_reward += r
                    step_info["classification"] = detail
                    # Track metrics
                    self._classification_total += 1
                    if detail.get("correct"):
                        self._classification_correct += 1

                elif action_type == "priority":
                    r, detail = self.grader.grade_priority(
                        action[action_type], gt["priority"]
                    )
                    step_reward += r
                    step_info["priority"] = detail
                    # Track metrics
                    if gt["priority"] != "none":
                        self._priority_total += 1
                        if detail.get("correct"):
                            self._priority_correct += 1

                elif action_type == "reply":
                    r, detail = self.grader.grade_reply(
                        action[action_type],
                        gt.get("reference_reply", ""),
                        self._current_email,
                    )
                    step_reward += r
                    step_info["reply"] = detail
                    if gt["classification"] != "spam":
                        self._reply_scores.append(r)

        self._step_count += 1
        self._episode_reward += step_reward

        # Check if episode is done
        all_actions_taken = all(
            a in self._actions_taken for a in ["classify", "priority", "reply"]
        )
        if all_actions_taken or self._step_count >= self.max_steps:
            self._done = True
            self._total_episodes += 1
            self._total_reward += self._episode_reward

        step_info["step_reward"] = round(step_reward, 4)
        step_info["episode_reward"] = round(self._episode_reward, 4)

        return self._get_observation(), round(step_reward, 4), self._done, step_info

    # ── Metrics ───────────────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """Return running evaluation metrics."""
        return {
            "total_episodes": self._total_episodes,
            "total_reward": round(self._total_reward, 4),
            "avg_reward": round(
                self._total_reward / max(self._total_episodes, 1), 4
            ),
            "classification_accuracy": round(
                self._classification_correct / max(self._classification_total, 1), 4
            ),
            "priority_accuracy": round(
                self._priority_correct / max(self._priority_total, 1), 4
            ),
            "avg_reply_score": round(
                sum(self._reply_scores) / max(len(self._reply_scores), 1), 4
            ),
            "num_emails_in_pool": len(self.task_pool),
        }

    def render(self) -> str:
        """Pretty-print the current email state."""
        if self._current_email is None:
            return "No active episode. Call reset() first."

        e = self._current_email
        meta = e.get("metadata", {})
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            f"║  EMAIL ID: {e['id']:<50}║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║  From:    {e['sender']:<51}║",
            f"║  Subject: {e['subject'][:51]:<51}║",
            f"║  Time:    {e['timestamp']:<51}║",
            f"║  Attach:  {'Yes' if meta.get('has_attachments') else 'No':<51}║",
            f"║  Reply:   {'Yes' if meta.get('is_reply') else 'No':<51}║",
            f"║  Thread:  {str(meta.get('thread_length', 1)):<51}║",
            "╠══════════════════════════════════════════════════════════════╣",
        ]
        # Wrap body text
        body = e["body"]
        for i in range(0, len(body), 58):
            chunk = body[i : i + 58]
            lines.append(f"║  {chunk:<59}║")
        lines.append("╠══════════════════════════════════════════════════════════════╣")
        lines.append(f"║  Step: {self._step_count}/{self.max_steps}  |  "
                      f"Actions: {list(self._actions_taken.keys())!s:<34}║")
        lines.append(f"║  Episode Reward: {self._episode_reward:<44}║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")

        output = "\n".join(lines)
        print(output)
        return output

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_observation(self) -> Dict[str, Any]:
        """Build the observation dict (email without ground truth)."""
        if self._current_email is None:
            return {}
        e = self._current_email
        return {
            "email_id": e["id"],
            "sender": e["sender"],
            "subject": e["subject"],
            "body": e["body"],
            "timestamp": e["timestamp"],
            "metadata": copy.deepcopy(e.get("metadata", {})),
        }

    @property
    def action_space(self) -> Dict:
        """Describe the available actions."""
        return {
            "classify": {"type": "categorical", "values": ["spam", "ham"]},
            "priority": {
                "type": "categorical",
                "values": ["low", "medium", "high", "critical"],
            },
            "reply": {"type": "text", "max_length": 1024},
        }

    @property
    def observation_space(self) -> Dict:
        """Describe the observation structure."""
        return {
            "email_id": "string",
            "sender": "string",
            "subject": "string",
            "body": "string",
            "timestamp": "string",
            "metadata": {
                "has_attachments": "boolean",
                "is_reply": "boolean",
                "thread_length": "integer",
            },
        }
