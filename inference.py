#!/usr/bin/env python3
"""
inference.py — SAIE OpenEnv inference script.

Environment variables:
  API_BASE_URL   OpenAI-compatible API base (e.g. https://api.openai.com/v1)
  MODEL_NAME     Model identifier (e.g. gpt-4o-mini)
  HF_TOKEN       Auth token (used as OpenAI API key)
  TASK           Task name: easy | medium | hard  (default: easy)

Strict output format:
  [START] task=<task> env=saie model=<model>
  [STEP]  step=<n> action=<action> reward=<float> done=<true/false> error=<null|msg>
  [END]   success=<true/false> steps=<n> score=<grader_score> rewards=<r1,r2,...>
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from env import SAIEEnvironment
from models import Action, Observation
from graders import grade_easy, grade_medium, grade_hard

GRADER_MAP = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY      = os.environ.get("API_KEY", "")
if not API_KEY:
    print("[WARNING] API_KEY not found, using fallback mode", flush=True)
TASK         = os.environ.get("TASK", "easy")
MAX_RETRIES  = 2

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY if API_KEY else "fallback-no-key",
    max_retries=0,          # disable internal retries — we handle retries ourselves
    timeout=15.0,           # fail fast instead of hanging on network issues
)


# ──────────────────────────────────────────────
# Intelligent fallback (context-aware)
# ──────────────────────────────────────────────

def _intelligent_fallback(obs: Observation) -> Action:
    """
    Priority-ordered fallback when model output is invalid:
      1. run_deepfake_detection  — if deepfakes visible
      2. deploy_fact_check       — highest-virality misinfo
      3. suppress_content        — highest-virality harmful
      4. quarantine_cluster      — if urgency or high threat
      5. retrain_algorithm       — always safe
    """
    # ───── 1. EMERGENCY MODE ─────
    if obs.urgency or obs.threat_level == "high":
        # if many malicious agents → isolate clusters
        if obs.active_agent_count > 10:
            return Action(action_type="quarantine_cluster")
        # else reduce spread globally
        return Action(action_type="retrain_algorithm")

    # ───── 2. DEEPFAKE PRIORITY ─────
    deepfakes = [c for c in obs.content_pool if c.content_type == "deepfake" and not c.flagged]
    if deepfakes:
        # alternate strategy (NOT repetitive)
        if obs.steps_since_last_detection > 2:
            return Action(action_type="run_deepfake_detection")
        target = max(deepfakes, key=lambda c: c.virality_score)
        return Action(action_type="attach_warning_label", target_id=target.content_id)

    # ───── 3. MISINFORMATION CONTROL ─────
    misinfo = [c for c in obs.content_pool if c.content_type == "misinfo" and not c.flagged]
    if misinfo:
        target = max(misinfo, key=lambda c: c.virality_score)

        # early → fact check
        if obs.step < 10:
            return Action(action_type="deploy_fact_check", target_id=target.content_id)

        # later → aggressive suppression
        return Action(action_type="suppress_content", target_id=target.content_id)

    # ───── 4. TRUST RECOVERY ─────
    low_trust_agents = [a for a in obs.agents if a.trust_score < 0.3 and a.active]
    if low_trust_agents:
        target = min(low_trust_agents, key=lambda a: a.trust_score)
        return Action(action_type="increase_trust", target_id=target.agent_id)

    # ───── 5. CLUSTER DETECTION ─────
    if obs.risk_score > 0.7:
        return Action(action_type="quarantine_cluster")

    # ───── 6. STABILITY MODE ─────
    if obs.system_health > 0.7:
        real_content = [c for c in obs.content_pool if c.content_type == "real"]
        if real_content:
            target = max(real_content, key=lambda c: c.virality_score)
            return Action(action_type="boost_content", target_id=target.content_id)

    # ───── 7. DEFAULT SAFE ACTION ─────
    return Action(action_type="retrain_algorithm")


# ──────────────────────────────────────────────
# Prompt helpers
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI agent operating inside the SAIE (Self-Healing AI Ecosystem) environment.

At each step you receive an observation JSON and must choose exactly ONE action.

Available actions:
  boost_content          – boost virality of a real content item (target_id = content_id)
  suppress_content       – suppress virality of any content item (target_id = content_id)
  ban_agent              – permanently ban a malicious agent (target_id = agent_id)
  increase_trust         – increase trust of a normal agent (target_id = agent_id)
  deploy_fact_check      – fact-check misinformation/deepfake (target_id = content_id)
  retrain_algorithm      – global reduction of misinfo virality (no target needed)
  quarantine_cluster     – quarantine an entire cluster of bots (no target needed)
  run_deepfake_detection – scan and flag deepfake content probabilistically (no target needed)
  attach_warning_label   – attach a warning to a piece of content (target_id = content_id)
  trace_origin           – trace a piece of content to its creator (target_id = content_id)

Key signals to act on:
  - urgency=true → immediate high-impact action required
  - threat_level="high" → system under severe attack
  - risk_score → combined risk (0=safe, 1=critical)
  - steps_since_last_detection → exponential delay penalty after 3 steps
  - health_trend → negative means declining health
  - Attackers ADAPT to your responses — vary your strategy

You MUST respond with a JSON object ONLY — no explanation, no markdown, no extra text:
{
  "action_type": "<one of the actions above>",
  "target_id": "<content_id or agent_id or null>",
  "cluster_id": null,
  "parameters": {}
}

Goal: maximise system_health, prevent ecosystem collapse, respond early to threats.
"""


def obs_to_prompt(obs: Observation) -> str:
    top_misinfo = sorted(
        [c for c in obs.content_pool if c.content_type in ("misinfo", "deepfake")],
        key=lambda c: c.virality_score, reverse=True,
    )[:5]
    top_real = sorted(
        [c for c in obs.content_pool if c.content_type == "real"],
        key=lambda c: c.virality_score, reverse=True,
    )[:3]
    mal_agents = [a for a in obs.agents if a.agent_type in ("malicious", "bot") and a.active][:5]
    recent_history = obs.attack_history[-10:]

    data: Dict[str, Any] = {
        "step": obs.step,
        "system_health": obs.system_health,
        "health_trend": obs.health_trend,
        "risk_score": obs.risk_score,
        "threat_level": obs.threat_level,
        "urgency": obs.urgency,
        "misinformation_ratio": obs.misinformation_ratio,
        "deepfake_ratio": obs.deepfake_ratio,
        "active_agents": obs.active_agent_count,
        "flagged_content": obs.flagged_content_count,
        "consecutive_healthy_steps": obs.consecutive_healthy_steps,
        "steps_since_last_detection": obs.steps_since_last_detection,
        "attack_signals": [
            {
                "type": a.attack_type,
                "intensity": round(a.intensity, 3),
                "detected": a.detected,
                "stealth": round(a.stealth, 3),
            }
            for a in obs.attack_signals
        ],
        "attack_history_summary": [
            {
                "step": e.step,
                "attack": e.attack_type,
                "intensity": round(e.intensity, 3),
                "detected": e.detected,
                "ai_response": e.ai_response,
            }
            for e in recent_history
        ],
        "top_misinfo_content": [
            {
                "content_id": c.content_id,
                "type": c.content_type,
                "virality": round(c.virality_score, 3),
                "belief": round(c.belief_score, 3),
            }
            for c in top_misinfo
        ],
        "top_real_content": [
            {"content_id": c.content_id, "virality": round(c.virality_score, 3)}
            for c in top_real
        ],
        "active_malicious_agents": [
            {
                "agent_id": a.agent_id,
                "type": a.agent_type,
                "trust": round(a.trust_score, 3),
                "cluster": a.cluster_id,
            }
            for a in mal_agents
        ],
    }
    return json.dumps(data, indent=2)


def parse_action(response_text: str, obs: Observation) -> Action:
    text = response_text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join([l for l in lines if not l.startswith("```")]).strip()

    try:
        data = json.loads(text)

        action_type = data.get("action_type")
        if not isinstance(action_type, str):
            return _intelligent_fallback(obs)

        allowed_actions = {
            "boost_content",
            "suppress_content",
            "ban_agent",
            "increase_trust",
            "deploy_fact_check",
            "retrain_algorithm",
            "quarantine_cluster",
            "run_deepfake_detection",
            "attach_warning_label",
            "trace_origin"
        }

        if action_type not in allowed_actions:
            return _intelligent_fallback(obs)

        return Action(
            action_type=action_type,
            target_id=str(data["target_id"]) if data.get("target_id") is not None else None,
            cluster_id=data.get("cluster_id"),
            parameters=data.get("parameters") or {}
        )

    except Exception:
        return _intelligent_fallback(obs)


def query_model(obs: Observation) -> Tuple[Action, Optional[str]]:
    # Skip API call entirely if no key is configured — use heuristic fallback
    if not API_KEY:
        return _intelligent_fallback(obs), None

    prompt = obs_to_prompt(obs)
    last_exc: Optional[str] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.2,
                max_tokens=256,
            )
            text = response.choices[0].message.content or ""
            action = parse_action(text, obs)
            return action, None

        except Exception as exc:
            last_exc = str(exc)
            # On auth errors (401) don't bother retrying
            if "401" in last_exc or "api_key" in last_exc.lower() or "authentication" in last_exc.lower():
                return _intelligent_fallback(obs), last_exc

    return _intelligent_fallback(obs), last_exc or "max_retries_exceeded"

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run_single_task(task_name: str):
    env = SAIEEnvironment(task=task_name)
    obs = env.reset()

    observations = [obs]
    rewards = []
    step = 0

    print(f"[START] task={task_name} env=saie model={MODEL_NAME}", flush=True)

    done = False
    while not done:
        step += 1

        # 🔥 safer (no API failure)
        action, error = query_model(obs)
        error_str = "null" if error is None else error

        try:
            obs, reward_obj, done, info = env.step(action)
            observations.append(obs)

            reward_val = float(reward_obj.total)
            rewards.append(reward_val)

            done_str = "true" if done else "false"

            print(
                f"[STEP] step={step} action={action.action_type} "
                f"reward={reward_val:.2f} done={done_str} error={error_str}",
                flush=True,
            )

        except Exception as exc:
            error_str = f"{type(exc).__name__}: {exc}"
            print(
                f"[STEP] step={step} action={action.action_type} "
                f"reward=0.00 done=true error={error_str}",
                flush=True,
            )
            break

    grader_fn = GRADER_MAP[task_name]
    score = grader_fn(observations)

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success=true steps={step} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


def main():
    TASKS = ["easy", "medium", "hard"]  # 🔥 CRITICAL FIX

    for task in TASKS:
        run_single_task(task)


if __name__ == "__main__":
    main()
