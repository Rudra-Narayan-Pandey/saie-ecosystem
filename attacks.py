from __future__ import annotations
import random
from typing import Dict, List

from models import AttackSignal, AttackHistoryEntry


# ──────────────────────────────────────────────
# Base spawn probabilities
# ──────────────────────────────────────────────
BASE_PROBS: Dict[str, Dict[str, float]] = {
    "easy":   {"misinfo_burst": 0.08, "deepfake_wave": 0.03, "bot_cluster": 0.02, "trust_erosion": 0.02},
    "medium": {"misinfo_burst": 0.15, "deepfake_wave": 0.12, "bot_cluster": 0.10, "trust_erosion": 0.08},
    "hard":   {"misinfo_burst": 0.25, "deepfake_wave": 0.22, "bot_cluster": 0.18, "trust_erosion": 0.20},
}


def _compute_adaptive_probs(
    difficulty: str,
    history: List[AttackHistoryEntry],
    recent_window: int = 8,
) -> Dict[str, float]:
    """
    Adapt attack probabilities based on AI's recent responses.

    Rules:
    - AI uses fact-check / suppress frequently → attackers pivot to deepfakes
    - AI bans agents frequently → attackers spawn new bot clusters
    - AI uses deepfake detection frequently → attackers increase stealth (lower observable prob)
    - AI uses retrain_algorithm → attackers increase trust_erosion
    """
    probs = {k: v for k, v in BASE_PROBS[difficulty].items()}

    if not history:
        return probs

    recent = history[-recent_window:]
    action_counts: Dict[str, int] = {}
    for entry in recent:
        if entry.ai_response:
            action_counts[entry.ai_response] = action_counts.get(entry.ai_response, 0) + 1

    total_recent = len(recent)
    if total_recent == 0:
        return probs

    def _freq(action: str) -> float:
        return action_counts.get(action, 0) / total_recent

    # Counter-adaptations
    suppress_freq = _freq("deploy_fact_check") + _freq("suppress_content") + _freq("attach_warning_label")
    ban_freq = _freq("ban_agent") + _freq("quarantine_cluster")
    detect_freq = _freq("run_deepfake_detection")
    retrain_freq = _freq("retrain_algorithm")

    # Attackers pivot to deepfakes when misinfo is being suppressed
    if suppress_freq > 0.3:
        probs["deepfake_wave"] = min(probs["deepfake_wave"] * (1.0 + suppress_freq), 0.9)
        probs["misinfo_burst"] *= max(1.0 - suppress_freq * 0.5, 0.5)

    # Attackers spawn new bots when agents are being banned
    if ban_freq > 0.2:
        probs["bot_cluster"] = min(probs["bot_cluster"] * (1.0 + ban_freq * 2.0), 0.9)

    # Attackers increase stealth when detection is high (modeled via intensity reduction + stealth flag)
    # This is communicated back through spawn parameters

    # Retrain → trust erosion escalation
    if retrain_freq > 0.25:
        probs["trust_erosion"] = min(probs["trust_erosion"] * (1.0 + retrain_freq), 0.9)

    return probs


def maybe_spawn_attacks(
    step: int,
    rng: random.Random,
    difficulty: str,
    existing_attacks: List[AttackSignal],
    history: List[AttackHistoryEntry] | None = None,
) -> List[AttackSignal]:
    """
    Probabilistically spawn new attack signals.
    Applies adaptive logic based on AI history.
    Returns the full (updated) list of attack signals.
    """
    history = history or []
    probs = _compute_adaptive_probs(difficulty, history)

    # Determine stealth boost from history (detect_freq)
    recent = history[-8:] if len(history) >= 8 else history
    detect_freq = sum(
        1 for e in recent if e.ai_response == "run_deepfake_detection"
    ) / max(len(recent), 1)
    stealth_boost = min(detect_freq * 0.6, 0.5)

    new_attacks: List[AttackSignal] = []
    for attack_type, prob in probs.items():
        if rng.random() < prob:
            # Stealth attacks have lower intensity but are harder to detect
            stealth = min(stealth_boost + rng.uniform(0.0, 0.2), 0.8)
            intensity = rng.uniform(0.3, 1.0) * (1.0 - stealth * 0.3)
            new_attacks.append(
                AttackSignal(
                    attack_type=attack_type,
                    intensity=max(intensity, 0.15),
                    detected=False,
                    step_occurred=step,
                    stealth=stealth,
                )
            )

    # Keep undetected attacks and recently detected ones
    surviving = [
        a for a in existing_attacks
        if not a.detected or (step - a.step_occurred) < 3
    ]
    return surviving + new_attacks


def apply_attack_effects(
    attack: AttackSignal,
    rng: random.Random,
) -> dict:
    """
    Returns effect multipliers. Stealth attacks have reduced observable effects
    but build up silently.
    """
    i = attack.intensity
    # Stealth attacks are harder to detect and their effects are partially hidden
    visible_fraction = 1.0 - attack.stealth * 0.4

    if attack.attack_type == "misinfo_burst":
        return {
            "misinfo_delta": 0.05 * i * visible_fraction,
            "trust_delta": -0.02 * i,
            "hidden_misinfo_delta": 0.05 * i * (1.0 - visible_fraction),
        }
    elif attack.attack_type == "deepfake_wave":
        return {
            "deepfake_delta": 0.06 * i * visible_fraction,
            "trust_delta": -0.03 * i,
            "virality_boost": 0.1 * i,
            "hidden_deepfake_delta": 0.06 * i * (1.0 - visible_fraction),
        }
    elif attack.attack_type == "bot_cluster":
        return {
            "misinfo_delta": 0.04 * i,
            "deepfake_delta": 0.02 * i,
            "spam_content_count": int(3 * i),
        }
    elif attack.attack_type == "trust_erosion":
        return {
            "trust_delta": -0.05 * i,
            "health_delta": -0.03 * i,
        }
    return {}
