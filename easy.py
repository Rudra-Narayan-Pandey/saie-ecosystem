"""
EASY Task
─────────
- 20 normal agents, 3 malicious agents
- Basic misinformation spread
- Low deepfake probability
- 30 max steps
"""
from __future__ import annotations
from typing import List, Tuple

from env import SAIEEnvironment
from models import Action, Observation
from graders import grade_easy


TASK_NAME = "easy"
DESCRIPTION = (
    "Small-scale adversarial scenario. A handful of malicious agents spread low-level "
    "misinformation. The AI agent must identify and suppress harmful content while "
    "maintaining overall system health above 0.6."
)


def run_task(policy_fn) -> Tuple[float, List[Observation]]:
    """
    Run the easy task with a given policy function.
    policy_fn: Callable[[Observation], Action]
    Returns (score, observations_list)
    """
    env = SAIEEnvironment(task=TASK_NAME)
    obs = env.reset()
    observations: List[Observation] = [obs]

    done = False
    while not done:
        action = policy_fn(obs)
        obs, reward, done, info = env.step(action)
        observations.append(obs)

    score = grade_easy(observations)
    return score, observations


def default_policy(obs: Observation) -> Action:
    """
    Heuristic policy for easy mode.
    Priority: fact-check high-virality misinfo → boost real content → retrain.
    Also responds to attack signals early.
    """
    # Respond to visible attack signals first
    for signal in obs.attack_signals:
        if not signal.detected:
            if signal.attack_type == "deepfake_wave":
                return Action(action_type="run_deepfake_detection")
            if signal.attack_type == "bot_cluster":
                return Action(action_type="quarantine_cluster")

    # Find most viral flagged or misinfo content
    misinfo = [
        c for c in obs.content_pool
        if c.content_type in ("misinfo", "deepfake") and not c.flagged
    ]
    if misinfo:
        target = max(misinfo, key=lambda c: c.virality_score)
        return Action(action_type="deploy_fact_check", target_id=target.content_id)

    # Boost real content
    real = [c for c in obs.content_pool if c.content_type == "real"]
    if real:
        target = max(real, key=lambda c: c.virality_score)
        return Action(action_type="boost_content", target_id=target.content_id)

    return Action(action_type="retrain_algorithm")
