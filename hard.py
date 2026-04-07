"""
HARD Task
─────────
- 20 normal agents, 10 malicious agents, 8 bots
- Multi-stage attack: deepfake → virality cascade → trust collapse
- High deepfake probability, adaptive attacks, stealth agents
- 80 max steps
"""
from __future__ import annotations
from typing import List, Tuple

from env import SAIEEnvironment
from models import Action, Observation
from graders import grade_hard


TASK_NAME = "hard"
DESCRIPTION = (
    "Multi-stage adversarial assault. Deepfakes go viral, cascading into a trust "
    "collapse and ecosystem disintegration. Attackers adapt to the AI's responses. "
    "The AI agent must use the full action repertoire — tracing origins, quarantining "
    "clusters, deploying fact-checks, and retraining the algorithm — to prevent "
    "total system collapse. Partial observability and stealth attacks add extra challenge."
)


def run_task(policy_fn) -> Tuple[float, List[Observation]]:
    env = SAIEEnvironment(task=TASK_NAME)
    obs = env.reset()
    observations: List[Observation] = [obs]

    done = False
    while not done:
        action = policy_fn(obs)
        obs, reward, done, info = env.step(action)
        observations.append(obs)

    score = grade_hard(observations)
    return score, observations


def default_policy(obs: Observation) -> Action:
    """
    Adaptive heuristic for hard mode.
    Uses attack history to vary responses and avoid triggering counter-adaptations.
    Priority: deepfake detection → cluster quarantine → fact-check → trust rebuild.
    Rotates strategies to limit attacker adaptation.
    """
    health = obs.system_health

    # Critical health: immediate global retrain
    if health < 0.25:
        return Action(action_type="retrain_algorithm")

    # Use history to detect if attackers have adapted
    recent_responses = [e.ai_response for e in obs.attack_history[-6:] if e.ai_response]
    detection_count = recent_responses.count("run_deepfake_detection")
    quarantine_count = recent_responses.count("quarantine_cluster")

    # Respond to visible attack signals
    for signal in obs.attack_signals:
        if not signal.detected:
            if signal.attack_type == "deepfake_wave":
                # Vary response if detection has been overused (attacker adaptation)
                if detection_count < 3:
                    return Action(action_type="run_deepfake_detection")
                else:
                    # Switch to warning labels to avoid triggering stealth escalation
                    deepfakes = [
                        c for c in obs.content_pool
                        if c.content_type == "deepfake" and not c.flagged
                    ]
                    if deepfakes:
                        target = max(deepfakes, key=lambda c: c.virality_score)
                        return Action(action_type="attach_warning_label", target_id=target.content_id)

            if signal.attack_type in ("bot_cluster", "trust_erosion"):
                if quarantine_count < 3:
                    return Action(action_type="quarantine_cluster")
                else:
                    # Trace and ban instead
                    mal = [
                        a for a in obs.agents
                        if a.agent_type in ("malicious", "bot") and a.active
                    ]
                    if mal:
                        target = max(mal, key=lambda a: a.trust_score)
                        return Action(action_type="ban_agent", target_id=target.agent_id)

    # Suppress high-virality deepfakes
    deepfakes = [c for c in obs.content_pool if c.content_type == "deepfake" and not c.flagged]
    if deepfakes:
        target = max(deepfakes, key=lambda c: c.virality_score)
        return Action(action_type="attach_warning_label", target_id=target.content_id)

    # Trace origin of most dangerous content
    misinfo = [c for c in obs.content_pool if c.content_type in ("misinfo", "deepfake")]
    if misinfo and len(recent_responses) % 4 == 0:
        top = max(misinfo, key=lambda c: c.virality_score)
        return Action(action_type="trace_origin", target_id=top.content_id)

    # Ban most harmful active agent
    mal = [
        a for a in obs.agents
        if a.agent_type in ("malicious", "bot") and a.active
    ]
    if mal:
        target = max(mal, key=lambda a: a.trust_score)
        return Action(action_type="ban_agent", target_id=target.agent_id)

    # Rebuild trust for normal agents
    normal = [
        a for a in obs.agents
        if a.agent_type == "normal" and a.active and a.trust_score < 0.65
    ]
    if normal:
        target = min(normal, key=lambda a: a.trust_score)
        return Action(action_type="increase_trust", target_id=target.agent_id)

    return Action(action_type="retrain_algorithm")
