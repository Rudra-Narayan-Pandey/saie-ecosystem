from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Tuple

from models import Action, Observation, Reward
from agents import create_agents
from attacks import maybe_spawn_attacks
from ecosystem import Ecosystem
from reward import compute_reward


# Safe fallback action — never requires a target_id
_SAFE_FALLBACK = Action(action_type="retrain_algorithm")


class SAIEEnvironment:
    """
    Self-Healing AI Ecosystem with Synthetic Reality Defense (SAIE)
    OpenEnv-compliant environment.

    Step order (corrected):
      1. Spawn new attacks  (adaptive, history-aware)
      2. Validate + apply AI action  ← AI acts FIRST
      3. ecosystem.tick()            ← World evolves AFTER
      4. Compute reward
    """

    TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
        "easy": {
            "num_normal": 20,
            "num_malicious": 3,
            "num_bots": 0,
            "max_steps": 30,
            "difficulty": "easy",
            "seed": 101,
        },
        "medium": {
            "num_normal": 20,
            "num_malicious": 6,
            "num_bots": 4,
            "max_steps": 50,
            "difficulty": "medium",
            "seed": 202,
        },
        "hard": {
            "num_normal": 20,
            "num_malicious": 10,
            "num_bots": 8,
            "max_steps": 80,
            "difficulty": "hard",
            "seed": 303,
        },
    }

    def __init__(self, task: str = "easy") -> None:
        if task not in self.TASK_CONFIGS:
            raise ValueError(f"Unknown task '{task}'. Choose from {list(self.TASK_CONFIGS)}")
        self.task = task
        cfg = self.TASK_CONFIGS[task]
        self._cfg = cfg
        self._eco: Optional[Ecosystem] = None
        self._prev_obs: Optional[Observation] = None
        self._step_count: int = 0
        self._done: bool = False
        self._rng = random.Random(cfg["seed"])

    # ──────────────────────────────────────────
    # OpenEnv interface
    # ──────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset environment and return initial observation."""
        cfg = self._cfg
        self._rng = random.Random(cfg["seed"])
        agents = create_agents(
            num_normal=cfg["num_normal"],
            num_malicious=cfg["num_malicious"],
            num_bots=cfg["num_bots"],
            seed=cfg["seed"],
        )
        self._eco = Ecosystem(agents, difficulty=cfg["difficulty"], seed=cfg["seed"])
        self._step_count = 0
        self._done = False
        obs = self._eco.get_observation()
        self._prev_obs = obs
        return obs

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Returns (observation, reward, done, info).
        """
        if self._eco is None or self._done:
            raise RuntimeError("Call reset() before step().")

        self._step_count += 1
        eco = self._eco

        # 1. Spawn new attacks (adaptive)
        new_attacks = maybe_spawn_attacks(
            step=self._step_count,
            rng=self._rng,
            difficulty=self._cfg["difficulty"],
            existing_attacks=eco.attack_signals,
            history=eco.attack_history,
        )
        eco.attack_signals = new_attacks
        prev_attacks_undetected = sum(1 for a in eco.attack_signals if not a.detected)

        # 2. Validate action; fall back to safe default if target is invalid
        validated_action = self._validate_action(action)
        action_success, action_info = self._apply_action(validated_action)

        # Record action in history for adaptive attack system
        eco.record_action_in_history(validated_action.action_type)

        # 3. Ecosystem evolves AFTER action
        eco.tick(eco.attack_signals)

        # 4. Post-tick observation and reward
        curr_obs = eco.get_observation()

        curr_attacks_undetected = sum(1 for a in eco.attack_signals if not a.detected)
        newly_detected = max(0, prev_attacks_undetected - curr_attacks_undetected)
        if newly_detected > 0:
            eco.steps_since_last_detection = 0

        reward = compute_reward(
            self._prev_obs,
            curr_obs,
            early_detections=newly_detected,
        )

        max_steps = self._cfg["max_steps"]
        collapsed = curr_obs.system_health < 0.05
        self._done = (self._step_count >= max_steps) or collapsed

        info: Dict[str, Any] = {
            "step": self._step_count,
            "action_success": action_success,
            "action_info": action_info,
            "action_validated": validated_action.action_type != action.action_type,
            "newly_detected_attacks": newly_detected,
            "collapsed": collapsed,
        }

        self._prev_obs = curr_obs
        return curr_obs, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return raw current state dict (OpenEnv required)."""
        if self._eco is None:
            return {}
        return self._eco.get_observation().model_dump()

    # ──────────────────────────────────────────
    # Action validation layer
    # ──────────────────────────────────────────

    def _validate_action(self, action: Action) -> Action:
        """
        Validate that targeted actions have resolvable targets.
        Falls back to an intelligent safe action when target is missing/invalid.

        Targeted actions:  boost_content, suppress_content, ban_agent,
                           increase_trust, deploy_fact_check,
                           attach_warning_label, trace_origin
        Untargeted actions: retrain_algorithm, quarantine_cluster,
                            run_deepfake_detection
        """
        eco = self._eco
        a = action.action_type
        tid = action.target_id or ""

        content_targeted = a in (
            "boost_content", "suppress_content", "deploy_fact_check",
            "attach_warning_label", "trace_origin",
        )
        agent_targeted = a in ("ban_agent", "increase_trust")

        if content_targeted and tid:
            # Check target exists in visible pool
            ids = {c.content_id for c in eco.content_pool if c.observable}
            if tid not in ids:
                return self._intelligent_fallback(eco)

        if agent_targeted and tid:
            ids = {ag.agent_id for ag in eco.agents if ag.active}
            if tid not in ids:
                return self._intelligent_fallback(eco)

        # No target provided for targeted action — use intelligent fallback
        if content_targeted and not tid:
            return self._intelligent_fallback(eco)
        if agent_targeted and not tid:
            return self._intelligent_fallback(eco)

        return action

    def _intelligent_fallback(self, eco: Ecosystem) -> Action:
        """
        Priority-ordered fallback:
          1. run_deepfake_detection  (if deepfakes present)
          2. deploy_fact_check       (highest-virality misinfo)
          3. suppress_content        (highest-virality any harmful)
          4. quarantine_cluster      (if active cluster exists)
          5. retrain_algorithm       (always safe)
        """
        # 1. Deepfakes in pool
        deepfakes = [c for c in eco.content_pool if c.content_type == "deepfake" and not c.flagged and c.observable]
        if deepfakes:
            return Action(action_type="run_deepfake_detection")

        # 2. Misinfo to fact-check
        misinfo = [c for c in eco.content_pool if c.content_type in ("misinfo", "deepfake") and not c.flagged and c.observable]
        if misinfo:
            target = max(misinfo, key=lambda c: c.virality_score)
            return Action(action_type="deploy_fact_check", target_id=target.content_id)

        # 3. Any harmful content to suppress
        harmful = [c for c in eco.content_pool if c.content_type != "real" and not c.flagged and c.observable]
        if harmful:
            target = max(harmful, key=lambda c: c.virality_score)
            return Action(action_type="suppress_content", target_id=target.content_id)

        # 4. Active cluster
        clusters: Dict[str, int] = {}
        for ag in eco.agents:
            if ag.cluster_id and ag.active:
                clusters[ag.cluster_id] = clusters.get(ag.cluster_id, 0) + 1
        if clusters:
            return Action(action_type="quarantine_cluster")

        return _SAFE_FALLBACK

    # ──────────────────────────────────────────
    # Action dispatcher
    # ──────────────────────────────────────────

    def _apply_action(self, action: Action) -> Tuple[bool, str]:
        eco = self._eco
        a = action.action_type
        tid = action.target_id or ""
        cid = action.cluster_id or ""

        if a == "boost_content":
            ok = eco.boost_content(tid) if tid else False
            return ok, f"boost_content -> {tid}"

        elif a == "suppress_content":
            ok = eco.suppress_content(tid) if tid else False
            return ok, f"suppress_content -> {tid}" if ok else f"suppress_content FAILED -> {tid}"

        elif a == "ban_agent":
            ok = eco.ban_agent(tid) if tid else False
            return ok, f"ban_agent -> {tid}"

        elif a == "increase_trust":
            ok = eco.increase_trust(tid) if tid else False
            return ok, f"increase_trust -> {tid}"

        elif a == "deploy_fact_check":
            ok = eco.deploy_fact_check(tid) if tid else False
            return ok, f"deploy_fact_check -> {tid}"

        elif a == "retrain_algorithm":
            ok = eco.retrain_algorithm()
            return ok, "retrain_algorithm"

        elif a == "quarantine_cluster":
            params = action.parameters or {}
            target_cluster = cid or params.get("cluster_id", "")
            if not target_cluster:
                clusters: Dict[str, int] = {}
                for ag in eco.agents:
                    if ag.cluster_id and ag.active:
                        clusters[ag.cluster_id] = clusters.get(ag.cluster_id, 0) + 1
                target_cluster = max(clusters, key=clusters.get) if clusters else ""
            ok = eco.quarantine_cluster(target_cluster) if target_cluster else False
            return ok, f"quarantine_cluster -> {target_cluster}"

        elif a == "run_deepfake_detection":
            count = eco.run_deepfake_detection()
            return count > 0, f"run_deepfake_detection -> detected {count}"

        elif a == "attach_warning_label":
            ok = eco.attach_warning_label(tid) if tid else False
            return ok, f"attach_warning_label -> {tid}"

        elif a == "trace_origin":
            origin = eco.trace_origin(tid) if tid else None
            return origin is not None, f"trace_origin -> {origin}"

        return False, f"unknown action: {a}"

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def get_most_viral_misinfo(self) -> Optional[str]:
        if not self._eco:
            return None
        candidates = [
            c for c in self._eco.content_pool
            if c.content_type in ("misinfo", "deepfake") and c.observable
        ]
        return max(candidates, key=lambda c: c.virality_score).content_id if candidates else None

    def get_most_harmful_agent(self) -> Optional[str]:
        if not self._eco:
            return None
        mal = [a for a in self._eco.agents if a.agent_type in ("malicious", "bot") and a.active]
        return max(mal, key=lambda a: a.trust_score).agent_id if mal else None
