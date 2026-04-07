from __future__ import annotations
import math
from models import Observation, Reward


COLLAPSE_THRESHOLD = 0.15       # system_health below this → collapse penalty
STABILITY_HEALTH_FLOOR = 0.65   # must be above this for stability bonus
DELAY_PENALTY_BASE = 0.25       # base penalty per step of delayed detection
DELAY_PENALTY_EXPONENT = 1.4    # exponential growth factor for delay penalty


def compute_reward(
    prev_obs: Observation,
    curr_obs: Observation,
    early_detections: int = 0,
) -> Reward:
    """
    Compute the step reward.

    Components:
    - health_delta × 10          (penalises decline, rewards improvement)
    - misinfo_spike × -8         (penalises misinformation surges)
    - collapse_penalty = -5.0    (if health < 0.15)
    - early_detection_bonus      (+1.5 per detected attack)
    - stability_bonus            (+0.3 per consecutive healthy step, capped at +2.0)
    - delay_penalty              (exponential penalty for late response to known attacks)
    """
    health_delta = curr_obs.system_health - prev_obs.system_health
    health_reward = health_delta * 10.0

    # Misinformation surge penalty
    misinfo_spike = max(0.0, curr_obs.misinformation_ratio - prev_obs.misinformation_ratio)
    misinfo_penalty = -misinfo_spike * 8.0

    # Deepfake surge penalty (additional)
    deepfake_spike = max(0.0, curr_obs.deepfake_ratio - prev_obs.deepfake_ratio)
    deepfake_penalty = -deepfake_spike * 5.0

    # Collapse penalty
    collapse_penalty = 0.0
    if curr_obs.system_health < COLLAPSE_THRESHOLD:
        collapse_penalty = -5.0

    # Early detection bonus
    early_detection_bonus = early_detections * 1.5

    # Stability bonus: reward sustained healthy periods
    # Bonus grows with consecutive healthy steps but is capped
    stability_bonus = 0.0
    if curr_obs.system_health >= STABILITY_HEALTH_FLOOR:
        # 0.15 per consecutive step, capped at 2.0
        stability_bonus = min(curr_obs.consecutive_healthy_steps * 0.15, 2.0)
    else:
        # Partial stability: smaller bonus if health is between 0.5 and threshold
        if curr_obs.system_health >= 0.5:
            stability_bonus = min(curr_obs.consecutive_healthy_steps * 0.05, 0.5)

    # Delay penalty: exponentially increasing penalty for ignoring known attacks
    delay_penalty = 0.0
    undetected_attacks = [a for a in curr_obs.attack_signals if not a.detected]
    if undetected_attacks and curr_obs.steps_since_last_detection > 3:
        delay_steps = curr_obs.steps_since_last_detection - 3
        max_intensity = max(a.intensity for a in undetected_attacks)
        delay_penalty = -(
            DELAY_PENALTY_BASE
            * max_intensity
            * math.pow(delay_steps, DELAY_PENALTY_EXPONENT)
        )
        delay_penalty = max(delay_penalty, -3.0)  # cap at -3.0 per step

    total = (
        health_reward
        + misinfo_penalty
        + deepfake_penalty
        + collapse_penalty
        + early_detection_bonus
        + stability_bonus
        + delay_penalty
    )

    return Reward(
        total=round(total, 4),
        health_delta=round(health_delta, 4),
        misinfo_penalty=round(misinfo_penalty + deepfake_penalty, 4),
        collapse_penalty=round(collapse_penalty, 4),
        early_detection_bonus=round(early_detection_bonus, 4),
        stability_bonus=round(stability_bonus, 4),
        delay_penalty=round(delay_penalty, 4),
        breakdown={
            "health_reward":        round(health_reward, 4),
            "misinfo_penalty":      round(misinfo_penalty, 4),
            "deepfake_penalty":     round(deepfake_penalty, 4),
            "collapse_penalty":     round(collapse_penalty, 4),
            "early_detection_bonus": round(early_detection_bonus, 4),
            "stability_bonus":      round(stability_bonus, 4),
            "delay_penalty":        round(delay_penalty, 4),
        },
    )
