from __future__ import annotations
from typing import List

from models import Observation


def grade_episode(
    observations: List[Observation],
    task: str,
) -> float:
    """
    Deterministic grader. Returns score in [0.0, 1.0].

    Scoring components:
    - final_health    (weighted by task)
    - avg_health      (rewards sustained performance)
    - min_health      (resilience: penalises dangerous dips)
    - misinfo_bonus   (low final misinformation)
    - stability_bonus (long consecutive healthy streaks)
    - collapse_penalty (if health ever < 0.05)
    """
    if not observations:
        return 0.0

    final_obs = observations[-1]
    final_health = final_obs.system_health
    min_health = min(o.system_health for o in observations)
    avg_health = sum(o.system_health for o in observations) / len(observations)
    max_consecutive = max(o.consecutive_healthy_steps for o in observations)

    weights = {
        "easy":   {"final": 0.55, "avg": 0.25, "min": 0.20},
        "medium": {"final": 0.50, "avg": 0.30, "min": 0.20},
        "hard":   {"final": 0.45, "avg": 0.30, "min": 0.25},
    }.get(task, {"final": 0.55, "avg": 0.25, "min": 0.20})

    raw_score = (
        weights["final"] * final_health
        + weights["avg"]  * avg_health
        + weights["min"]  * min_health
    )

    # Stability bonus: up to 0.05 for long healthy streaks
    max_steps = max(len(observations) - 1, 1)  # exclude initial reset observation
    stability_bonus = min(0.05, 0.05 * (max_consecutive / max(max_steps, 1)))
    raw_score += stability_bonus

    # Low final misinfo bonus
    misinfo_bonus = max(0.0, 0.05 * (1.0 - final_obs.misinformation_ratio))
    raw_score = min(1.0, raw_score + misinfo_bonus)

    # Collapse penalty (health ever below 0.05)
    collapsed = any(o.system_health < 0.05 for o in observations)
    if collapsed:
        raw_score *= 0.30

    return round(float(raw_score), 4)


def grade_easy(observations: List[Observation]) -> float:
    return grade_episode(observations, "easy")


def grade_medium(observations: List[Observation]) -> float:
    return grade_episode(observations, "medium")


def grade_hard(observations: List[Observation]) -> float:
    return grade_episode(observations, "hard")
