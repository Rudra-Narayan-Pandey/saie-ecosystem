"""
MEDIUM Task
───────────
- 20 normal agents, 6 malicious agents, 4 bots
- Coordinated bot cluster attack
- Mixed misinformation + deepfakes
- 50 max steps
"""
from __future__ import annotations
from typing import List, Tuple

from env import SAIEEnvironment
from models import Action, Observation
from graders import grade_medium


TASK_NAME = "medium"
DESCRIPTION = (
    "Coordinated adversarial campaign. Bot clusters amplify deepfake content while "
    "malicious agents erode community trust. The AI agent must quarantine clusters, "
    "run deepfake detection, and stabilise trust to keep system health above 0.5."
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

    score = grade_medium(observations)
    return score, observations


def default_policy(obs: Observation) -> Action:
    """
    Heuristic for medium mode.
    Priority: deepfake detection → cluster quarantine → suppress misinfo → trust rebuild.
    Uses history to avoid repetitive detection when no new deepfakes exist.
    """
    health = obs.system_health

    # Critical health: global retrain
    if health < 0.35:
        return Action(action_type="retrain_algorithm")

    # Respond to attack signals early (delay penalty avoidance)
    for signal in obs.attack_signals:
        if not signal.detected:
            if signal.attack_type == "deepfake_wave":
                return Action(action_type="run_deepfake_detection")
            if signal.attack_type == "bot_cluster":
                return Action(action_type="quarantine_cluster")
            if signal.attack_type == "trust_erosion":
                normal = [
                    a for a in obs.agents
                    if a.agent_type == "normal" and a.active and a.trust_score < 0.7
                ]
                if normal:
                    target = min(normal, key=lambda a: a.trust_score)
                    return Action(action_type="increase_trust", target_id=target.agent_id)

    # Detect undetected deepfakes in pool
    deepfakes = [c for c in obs.content_pool if c.content_type == "deepfake" and not c.flagged]
    if deepfakes:
        return Action(action_type="run_deepfake_detection")

    # Suppress high-virality misinfo
    misinfo = [c for c in obs.content_pool if c.content_type == "misinfo" and not c.flagged]
    if misinfo:
        target = max(misinfo, key=lambda c: c.virality_score)
        return Action(action_type="suppress_content", target_id=target.content_id)

    # Boost low-trust normal agents
    normal = [
        a for a in obs.agents
        if a.agent_type == "normal" and a.active and a.trust_score < 0.6
    ]
    if normal:
        target = min(normal, key=lambda a: a.trust_score)
        return Action(action_type="increase_trust", target_id=target.agent_id)

    return Action(action_type="retrain_algorithm")
