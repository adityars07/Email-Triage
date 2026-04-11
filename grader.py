"""
grader.py — Multi-dimensional Reward System for the Email Triage Environment.

Evaluates agent actions across three dimensions:
  1. Classification (spam/ham)   → up to +0.4
  2. Priority assignment         → up to +0.3
  3. Reply quality               → up to +0.3
"""

import re
import math
from typing import Dict, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Priority ordering (for off-by-one detection) ─────────────────────────────

PRIORITY_LEVELS = ["low", "medium", "high", "critical"]
PRIORITY_INDEX = {p: i for i, p in enumerate(PRIORITY_LEVELS)}


class EmailTriageGrader:
    """Computes rewards for each dimension of the email triage task."""

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        rewards_cfg = cfg.get("rewards", {})

        # Classification rewards
        cls_cfg = rewards_cfg.get("classification", {})
        self.cls_correct = cls_cfg.get("correct", 0.4)
        self.cls_incorrect = cls_cfg.get("incorrect", -0.2)
        self.ham_as_spam_penalty = cls_cfg.get("ham_as_spam_penalty", -0.3)
        self.spam_as_ham_penalty = cls_cfg.get("spam_as_ham_penalty", -0.2)

        # Priority rewards
        pri_cfg = rewards_cfg.get("priority", {})
        self.pri_exact = pri_cfg.get("exact_match", 0.3)
        self.pri_off_by_one = pri_cfg.get("off_by_one", 0.15)
        self.pri_incorrect = pri_cfg.get("incorrect", -0.15)
        self.pri_critical_miss = pri_cfg.get("critical_miss_penalty", -0.25)

        # Reply rewards
        rep_cfg = rewards_cfg.get("reply", {})
        self.rep_max = rep_cfg.get("max_score", 0.3)
        self.rep_sim_weight = rep_cfg.get("similarity_weight", 0.5)
        self.rep_len_weight = rep_cfg.get("length_weight", 0.2)
        self.rep_prof_weight = rep_cfg.get("professionalism_weight", 0.3)
        self.rep_empty_penalty = rep_cfg.get("empty_penalty", -0.15)
        self.rep_nonsensical_penalty = rep_cfg.get("nonsensical_penalty", -0.1)

        # TF-IDF vectorizer (lazy-initialized)
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

    # ── Classification Grading ────────────────────────────────────────────

    def grade_classification(
        self, predicted: str, actual: str
    ) -> Tuple[float, Dict]:
        """
        Grade the spam/ham classification.

        Returns:
            (reward, details_dict)
        """
        predicted = predicted.strip().lower()
        actual = actual.strip().lower()

        if predicted == actual:
            return self.cls_correct, {
                "correct": True,
                "predicted": predicted,
                "actual": actual,
                "reward": self.cls_correct,
            }

        # Asymmetric penalties
        if actual == "ham" and predicted == "spam":
            penalty = self.ham_as_spam_penalty
            reason = "Misclassified legitimate email as spam (blocks important email)"
        elif actual == "spam" and predicted == "ham":
            penalty = self.spam_as_ham_penalty
            reason = "Misclassified spam as legitimate (lets spam through)"
        else:
            penalty = self.cls_incorrect
            reason = "Incorrect classification"

        return penalty, {
            "correct": False,
            "predicted": predicted,
            "actual": actual,
            "reward": penalty,
            "reason": reason,
        }

    # ── Priority Grading ──────────────────────────────────────────────────

    def grade_priority(
        self, predicted: str, actual: str
    ) -> Tuple[float, Dict]:
        """
        Grade the priority assignment.

        Returns:
            (reward, details_dict)
        """
        predicted = predicted.strip().lower()
        actual = actual.strip().lower()

        # Spam emails have "none" priority — skip grading
        if actual == "none":
            if predicted == "none" or predicted == "":
                return 0.0, {
                    "correct": True,
                    "predicted": predicted,
                    "actual": actual,
                    "reward": 0.0,
                    "note": "Spam email — priority not applicable",
                }
            return 0.0, {
                "correct": False,
                "predicted": predicted,
                "actual": actual,
                "reward": 0.0,
                "note": "Spam email — priority not applicable, no penalty",
            }

        if predicted not in PRIORITY_INDEX:
            return self.pri_incorrect, {
                "correct": False,
                "predicted": predicted,
                "actual": actual,
                "reward": self.pri_incorrect,
                "reason": f"Invalid priority value: {predicted}",
            }

        if predicted == actual:
            return self.pri_exact, {
                "correct": True,
                "predicted": predicted,
                "actual": actual,
                "reward": self.pri_exact,
            }

        # Check off-by-one
        diff = abs(PRIORITY_INDEX[predicted] - PRIORITY_INDEX[actual])
        if diff == 1:
            return self.pri_off_by_one, {
                "correct": False,
                "predicted": predicted,
                "actual": actual,
                "reward": self.pri_off_by_one,
                "reason": "Off by one priority level (partial credit)",
            }

        # Critical miss penalty
        if actual == "critical" and predicted in ("low", "medium"):
            return self.pri_critical_miss, {
                "correct": False,
                "predicted": predicted,
                "actual": actual,
                "reward": self.pri_critical_miss,
                "reason": "Missed critical priority — dangerous underestimation",
            }

        return self.pri_incorrect, {
            "correct": False,
            "predicted": predicted,
            "actual": actual,
            "reward": self.pri_incorrect,
            "reason": f"Priority mismatch (off by {diff} levels)",
        }

    # ── Reply Grading ─────────────────────────────────────────────────────

    def grade_reply(
        self, generated: str, reference: str, email: Dict
    ) -> Tuple[float, Dict]:
        """
        Grade the generated reply against the reference.

        Components:
          - Semantic similarity (TF-IDF cosine)
          - Length appropriateness
          - Professionalism heuristics

        Returns:
            (reward, details_dict)
        """
        # Spam emails should NOT get replies
        if email.get("ground_truth", {}).get("classification") == "spam":
            if not generated or generated.strip() == "":
                return 0.0, {
                    "reward": 0.0,
                    "note": "Spam email — no reply expected (correct)",
                }
            return self.rep_nonsensical_penalty, {
                "reward": self.rep_nonsensical_penalty,
                "reason": "Replied to a spam email — should not respond",
            }

        # Empty reply check
        if not generated or generated.strip() == "":
            return self.rep_empty_penalty, {
                "reward": self.rep_empty_penalty,
                "similarity": 0.0,
                "length_score": 0.0,
                "professionalism_score": 0.0,
                "reason": "Empty reply",
            }

        generated = generated.strip()

        # 1. Semantic Similarity (TF-IDF cosine)
        sim_score = self._compute_similarity(generated, reference)

        # 2. Length Appropriateness
        len_score = self._compute_length_score(generated, reference)

        # 3. Professionalism
        prof_score = self._compute_professionalism(generated)

        # Weighted combination
        raw_score = (
            sim_score * self.rep_sim_weight
            + len_score * self.rep_len_weight
            + prof_score * self.rep_prof_weight
        )
        reward = round(raw_score * self.rep_max, 4)

        return reward, {
            "reward": reward,
            "similarity": round(sim_score, 4),
            "length_score": round(len_score, 4),
            "professionalism_score": round(prof_score, 4),
            "generated_preview": generated[:100] + ("..." if len(generated) > 100 else ""),
        }

    def _compute_similarity(self, generated: str, reference: str) -> float:
        """TF-IDF cosine similarity between generated and reference reply."""
        try:
            tfidf = self._vectorizer.fit_transform([reference, generated])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            return float(max(0.0, min(1.0, sim)))
        except Exception:
            return 0.0

    def _compute_length_score(self, generated: str, reference: str) -> float:
        """Score based on how close the generated length is to the reference."""
        gen_len = len(generated.split())
        ref_len = max(len(reference.split()), 1)
        ratio = gen_len / ref_len

        if 0.5 <= ratio <= 2.0:
            # Gaussian-like score centered at ratio=1.0
            return math.exp(-0.5 * ((ratio - 1.0) / 0.5) ** 2)
        elif 0.2 <= ratio < 0.5 or 2.0 < ratio <= 3.0:
            return 0.3
        else:
            return 0.1

    def _compute_professionalism(self, text: str) -> float:
        """Heuristic professionalism scoring."""
        score = 0.0
        text_lower = text.lower()

        # Check for greeting (0.3)
        greetings = [
            "hi", "hello", "dear", "good morning", "good afternoon",
            "thanks", "thank you", "greetings",
        ]
        if any(g in text_lower[:50] for g in greetings):
            score += 0.3

        # Check for sign-off (0.2)
        signoffs = [
            "regards", "best", "sincerely", "thanks", "thank you",
            "cheers", "warm regards", "kind regards",
        ]
        if any(s in text_lower[-80:] for s in signoffs):
            score += 0.2

        # Sentence structure (0.2) — at least 2 sentences
        sentences = re.split(r'[.!?]+', text.strip())
        sentences = [s for s in sentences if len(s.strip()) > 5]
        if len(sentences) >= 2:
            score += 0.2

        # No excessive caps / spam-like language (0.15)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio < 0.3:
            score += 0.15

        # Reasonable word count (0.15)
        word_count = len(text.split())
        if 10 <= word_count <= 200:
            score += 0.15

        return min(score, 1.0)

    # ── Aggregate Scoring ─────────────────────────────────────────────────

    def compute_total_reward(
        self, actions: Dict, ground_truth: Dict, email: Dict
    ) -> Tuple[float, Dict]:
        """
        Compute total reward across all three dimensions.

        Args:
            actions: {"classify": str, "priority": str, "reply": str}
            ground_truth: {"classification": str, "priority": str, "reference_reply": str}
            email: full email dict

        Returns:
            (total_reward, breakdown_dict)
        """
        breakdown = {}
        total = 0.0

        # Classification
        if "classify" in actions:
            cls_reward, cls_detail = self.grade_classification(
                actions["classify"], ground_truth["classification"]
            )
            total += cls_reward
            breakdown["classification"] = cls_detail
        else:
            breakdown["classification"] = {"reward": 0.0, "note": "Not attempted"}

        # Priority
        if "priority" in actions:
            pri_reward, pri_detail = self.grade_priority(
                actions["priority"], ground_truth["priority"]
            )
            total += pri_reward
            breakdown["priority"] = pri_detail
        else:
            breakdown["priority"] = {"reward": 0.0, "note": "Not attempted"}

        # Reply
        if "reply" in actions:
            rep_reward, rep_detail = self.grade_reply(
                actions["reply"], ground_truth.get("reference_reply", ""), email
            )
            total += rep_reward
            breakdown["reply"] = rep_detail
        else:
            breakdown["reply"] = {"reward": 0.0, "note": "Not attempted"}

        epsilon = 1e-6
        total = max(epsilon, min(1.0 - epsilon, round(total, 4)))
        breakdown["total_reward"] = total
        return total, breakdown
