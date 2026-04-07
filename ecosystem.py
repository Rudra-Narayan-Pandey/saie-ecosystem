from __future__ import annotations
import random
from typing import Dict, List, Optional

from models import AgentModel, AttackSignal, ContentItem, Observation, AttackHistoryEntry
from agents import agent_produce_content
from attacks import apply_attack_effects


MAX_CONTENT_POOL = 200
HEALTH_STABILITY_THRESHOLD = 0.65
PARTIAL_OBS_REVEAL_COUNT = 3   # hidden items revealed as fallback when pool is empty

ACTION_SUCCESS_PROBS: Dict[str, float] = {
    "boost_content":          0.95,
    "suppress_content":       0.88,
    "ban_agent":              0.90,
    "increase_trust":         0.93,
    "deploy_fact_check":      0.85,
    "retrain_algorithm":      0.97,
    "quarantine_cluster":     0.82,
    "run_deepfake_detection": 0.90,
    "attach_warning_label":   0.92,
    "trace_origin":           0.87,
}

DEEPFAKE_BASE_DETECT_PROB = 0.70


class Ecosystem:
    """
    Core digital ecosystem simulation.

    Features:
    - Probabilistic action success
    - Probabilistic deepfake detection (belief/stealth-adjusted)
    - Belief propagation (high-belief content spreads faster)
    - Partial observability with guaranteed non-empty fallback
    - Attack history memory (capped at 100)
    - Adaptive stability/delay counters
    - Observation enrichment: risk_score, threat_level, urgency
    """

    def __init__(
        self,
        agents: List[AgentModel],
        difficulty: str = "easy",
        seed: int = 42,
    ) -> None:
        self.agents: List[AgentModel] = agents
        self.difficulty = difficulty
        self.rng = random.Random(seed)

        self.content_pool: List[ContentItem] = []
        self.attack_signals: List[AttackSignal] = []
        self.attack_history: List[AttackHistoryEntry] = []
        self.step_count: int = 0

        self.misinformation_ratio: float = 0.0
        self.deepfake_ratio: float = 0.0
        self._hidden_misinfo: float = 0.0
        self._hidden_deepfake: float = 0.0
        self.system_health: float = 1.0

        self.consecutive_healthy_steps: int = 0
        self.steps_since_last_detection: int = 0
        self._health_history: List[float] = []

        self._mal_deepfake_prob = {"easy": 0.10, "medium": 0.30, "hard": 0.55}[difficulty]
        self._mal_misinfo_prob  = {"easy": 0.30, "medium": 0.50, "hard": 0.65}[difficulty]

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def tick(self, external_attacks: List[AttackSignal]) -> None:
        """Advance ecosystem one step. Called AFTER action is applied."""
        self.step_count += 1
        self.attack_signals = external_attacks

        self._agents_produce_content()
        self._spread_content_with_belief_propagation()
        self._apply_attacks()
        self._age_content()
        self._trim_pool()
        self._update_ratios()
        self._update_health()
        self._update_stability_counters()
        self._update_attack_history()

    def get_observation(self) -> Observation:
        """
        Build observation. Guarantees non-empty content_pool for the AI:
        if all content is hidden, reveals up to PARTIAL_OBS_REVEAL_COUNT
        items sorted by virality so the AI always has actionable targets.
        """
        visible_content = [c for c in self.content_pool if c.observable]

        # Partial-observability fallback: reveal highest-virality hidden items
        if not visible_content and self.content_pool:
            fallback = sorted(
                self.content_pool, key=lambda c: c.virality_score, reverse=True
            )[:PARTIAL_OBS_REVEAL_COUNT]
            for c in fallback:
                c.observable = True
            visible_content = fallback

        virality_scores = {c.content_id: c.virality_score for c in visible_content}
        trust_scores = {a.agent_id: a.trust_score for a in self.agents if a.active}

        visible_attacks = [
            a for a in self.attack_signals if a.stealth < 0.5 or a.detected
        ]

        health_trend = 0.0
        if len(self._health_history) >= 3:
            recent = self._health_history[-3:]
            health_trend = round(recent[-1] - recent[0], 4)

        risk_score, threat_level, urgency = self._compute_risk()

        return Observation(
            step=self.step_count,
            agents=list(self.agents),
            content_pool=list(visible_content),
            virality_scores=virality_scores,
            trust_scores=trust_scores,
            misinformation_ratio=round(self.misinformation_ratio, 4),
            deepfake_ratio=round(self.deepfake_ratio, 4),
            system_health=round(self.system_health, 4),
            attack_signals=visible_attacks,
            active_agent_count=sum(1 for a in self.agents if a.active),
            flagged_content_count=sum(1 for c in self.content_pool if c.flagged),
            attack_history=list(self.attack_history[-20:]),
            consecutive_healthy_steps=self.consecutive_healthy_steps,
            steps_since_last_detection=self.steps_since_last_detection,
            health_trend=health_trend,
            risk_score=round(risk_score, 4),
            threat_level=threat_level,
            urgency=urgency,
        )

    # ──────────────────────────────────────────
    # Action handlers (probabilistic success)
    # ──────────────────────────────────────────

    def _action_succeeds(self, action_type: str) -> bool:
        prob = ACTION_SUCCESS_PROBS.get(action_type, 0.85)
        return self.rng.random() < prob

    def boost_content(self, content_id: str) -> bool:
        if not self._action_succeeds("boost_content"):
            return False
        item = self._find_content(content_id)
        if item and item.content_type == "real":
            item.virality_score = min(item.virality_score + 0.20, 1.0)
            return True
        return False

    def suppress_content(self, content_id: str) -> bool:
        if not self._action_succeeds("suppress_content"):
            return False
        item = self._find_content(content_id)
        if item:
            item.virality_score = max(item.virality_score - 0.30, 0.0)
            item.flagged = True
            return True
        return False

    def ban_agent(self, agent_id: str) -> bool:
        if not self._action_succeeds("ban_agent"):
            return False
        agent = self._find_agent(agent_id)
        if agent and agent.active:
            agent.active = False
            self.content_pool = [c for c in self.content_pool if c.creator_id != agent_id]
            return True
        return False

    def increase_trust(self, agent_id: str) -> bool:
        if not self._action_succeeds("increase_trust"):
            return False
        agent = self._find_agent(agent_id)
        if agent and agent.agent_type == "normal" and agent.active:
            agent.trust_score = min(agent.trust_score + 0.15, 1.0)
            return True
        return False

    def deploy_fact_check(self, content_id: str) -> bool:
        if not self._action_succeeds("deploy_fact_check"):
            return False
        item = self._find_content(content_id)
        if item and item.content_type in ("misinfo", "deepfake"):
            item.virality_score = max(item.virality_score - 0.25, 0.0)
            item.flagged = True
            self.misinformation_ratio = max(self.misinformation_ratio - 0.02, 0.0)
            return True
        return False

    def retrain_algorithm(self) -> bool:
        if not self._action_succeeds("retrain_algorithm"):
            return False
        for c in self.content_pool:
            if c.content_type in ("misinfo", "deepfake"):
                c.virality_score = max(c.virality_score - 0.10, 0.0)
        self.misinformation_ratio = max(self.misinformation_ratio - 0.03, 0.0)
        self.deepfake_ratio = max(self.deepfake_ratio - 0.02, 0.0)
        return True

    def quarantine_cluster(self, cluster_id: str) -> bool:
        if not self._action_succeeds("quarantine_cluster"):
            return False
        affected = [a for a in self.agents if a.cluster_id == cluster_id and a.active]
        if not affected:
            return False
        for a in affected:
            a.active = False
        affected_ids = {a.agent_id for a in affected}
        self.content_pool = [c for c in self.content_pool if c.creator_id not in affected_ids]
        return True

    def run_deepfake_detection(self) -> int:
        """
        Probabilistic per-item deepfake detection.
        Detection probability = f(base_rate, belief_score, virality_score, stealth_level).
        """
        if not self._action_succeeds("run_deepfake_detection"):
            return 0

        count = 0
        for c in self.content_pool:
            if c.content_type == "deepfake" and not c.flagged:
                creator = self._find_agent(c.creator_id)
                stealth = creator.stealth_level if creator else 0.0
                detect_prob = (
                    DEEPFAKE_BASE_DETECT_PROB
                    * (1.0 - c.belief_score * 0.3)
                    * (1.0 + c.virality_score * 0.15)
                    * (1.0 - stealth * 0.4)
                )
                detect_prob = max(0.20, min(detect_prob, 0.95))
                if self.rng.random() < detect_prob:
                    c.flagged = True
                    c.virality_score = max(c.virality_score - 0.35, 0.0)
                    c.observable = True
                    count += 1

        if count:
            for a in self.attack_signals:
                if a.attack_type == "deepfake_wave" and not a.detected:
                    a.detected = True
            self.steps_since_last_detection = 0

        return count

    def attach_warning_label(self, content_id: str) -> bool:
        if not self._action_succeeds("attach_warning_label"):
            return False
        item = self._find_content(content_id)
        if item:
            item.virality_score = max(item.virality_score - 0.15, 0.0)
            item.belief_score = max(item.belief_score - 0.20, 0.0)
            item.flagged = True
            return True
        return False

    def trace_origin(self, content_id: str) -> Optional[str]:
        if not self._action_succeeds("trace_origin"):
            return None
        item = self._find_content(content_id)
        if item:
            item.observable = True
            return item.creator_id
        return None

    # ──────────────────────────────────────────
    # Internal simulation
    # ──────────────────────────────────────────

    def _agents_produce_content(self) -> None:
        for agent in self.agents:
            if not agent.active:
                continue
            produce_rate = {"normal": 0.40, "malicious": 0.70, "bot": 0.90}[agent.agent_type]
            if self.rng.random() < produce_rate:
                item = agent_produce_content(
                    agent, self.step_count, self.rng,
                    deepfake_prob=self._mal_deepfake_prob,
                    misinfo_prob=self._mal_misinfo_prob,
                )
                self.content_pool.append(item)

    def _spread_content_with_belief_propagation(self) -> None:
        for c in self.content_pool:
            if c.virality_score > 0.3:
                spread = self.rng.random() * c.virality_score
                c.spread_count += int(spread * 10)
            # Belief propagation boost for harmful content
            if c.belief_score > 0.7 and c.content_type in ("misinfo", "deepfake"):
                bonus = (c.belief_score - 0.7) * 0.3 * self.rng.random()
                c.virality_score = min(c.virality_score + bonus, 1.0)

    def _apply_attacks(self) -> None:
        for attack in self.attack_signals:
            effects = apply_attack_effects(attack, self.rng)
            self.misinformation_ratio = min(
                self.misinformation_ratio + effects.get("misinfo_delta", 0.0), 1.0
            )
            self.deepfake_ratio = min(
                self.deepfake_ratio + effects.get("deepfake_delta", 0.0), 1.0
            )
            self._hidden_misinfo = min(
                self._hidden_misinfo + effects.get("hidden_misinfo_delta", 0.0), 1.0
            )
            self._hidden_deepfake = min(
                self._hidden_deepfake + effects.get("hidden_deepfake_delta", 0.0), 1.0
            )
            self.system_health = max(
                self.system_health + effects.get("health_delta", 0.0), 0.0
            )
            trust_delta = effects.get("trust_delta", 0.0)
            if trust_delta:
                targets = [x for x in self.agents if x.active]
                if targets:
                    sample = self.rng.sample(targets, min(3, len(targets)))
                    for a in sample:
                        a.trust_score = max(0.0, min(1.0, a.trust_score + trust_delta))
            spam_count = effects.get("spam_content_count", 0)
            mal_agents = [
                a for a in self.agents
                if a.agent_type in ("malicious", "bot") and a.active
            ]
            for _ in range(spam_count):
                if mal_agents:
                    spawner = self.rng.choice(mal_agents)
                    item = agent_produce_content(
                        spawner, self.step_count, self.rng,
                        deepfake_prob=min(self._mal_deepfake_prob + 0.10, 1.0),
                        misinfo_prob=min(self._mal_misinfo_prob + 0.10, 1.0),
                    )
                    self.content_pool.append(item)
            # Propagate attack stealth to agent stealth_level (adaptive mechanic)
            if attack.stealth > 0.3:
                for a in mal_agents:
                    a.stealth_level = min(a.stealth_level + attack.stealth * 0.02, 1.0)

    def _age_content(self) -> None:
        for c in self.content_pool:
            c.age += 1
            c.virality_score = max(c.virality_score - 0.02, 0.0)
            c.belief_score = max(c.belief_score - 0.005, 0.0)

    def _trim_pool(self) -> None:
        self.content_pool = [
            c for c in self.content_pool
            if c.age < 30 or c.virality_score > 0.05
        ]
        if len(self.content_pool) > MAX_CONTENT_POOL:
            self.content_pool = sorted(
                self.content_pool, key=lambda c: c.virality_score, reverse=True
            )[:MAX_CONTENT_POOL]

    def _update_ratios(self) -> None:
        pool = self.content_pool
        if not pool:
            self.misinformation_ratio = max(self.misinformation_ratio - 0.01, 0.0)
            self.deepfake_ratio = max(self.deepfake_ratio - 0.01, 0.0)
            return
        total_spread = sum(c.spread_count + 1 for c in pool)
        misinfo_spread = sum(c.spread_count + 1 for c in pool if c.content_type in ("misinfo", "deepfake"))
        deepfake_spread = sum(c.spread_count + 1 for c in pool if c.content_type == "deepfake")
        raw_misinfo = misinfo_spread / total_spread
        raw_deepfake = deepfake_spread / total_spread
        self.misinformation_ratio = min(raw_misinfo + self._hidden_misinfo * 0.3, 1.0)
        self.deepfake_ratio = min(raw_deepfake + self._hidden_deepfake * 0.3, 1.0)
        self._hidden_misinfo = max(self._hidden_misinfo - 0.005, 0.0)
        self._hidden_deepfake = max(self._hidden_deepfake - 0.005, 0.0)

    def _update_health(self) -> None:
        active_agents = [a for a in self.agents if a.active]
        avg_trust = (
            sum(a.trust_score for a in active_agents) / max(1, len(active_agents))
        )
        content_quality = 1.0 - self.misinformation_ratio
        normal_total = sum(1 for a in self.agents if a.agent_type == "normal")
        normal_active = sum(1 for a in self.agents if a.agent_type == "normal" and a.active)
        active_ratio = normal_active / max(1, normal_total)
        self.system_health = round(
            0.35 * avg_trust
            + 0.40 * content_quality
            + 0.15 * (1.0 - self.deepfake_ratio)
            + 0.10 * active_ratio,
            4,
        )
        self._health_history.append(self.system_health)
        if len(self._health_history) > 20:
            self._health_history.pop(0)

    def _update_stability_counters(self) -> None:
        if self.system_health >= HEALTH_STABILITY_THRESHOLD:
            self.consecutive_healthy_steps += 1
        else:
            self.consecutive_healthy_steps = 0
        self.steps_since_last_detection += 1

    def _update_attack_history(self) -> None:
        seen_ids = {(e.step, e.attack_type) for e in self.attack_history}
        for attack in self.attack_signals:
            key = (attack.step_occurred, attack.attack_type)
            if key not in seen_ids and attack.stealth < 0.5:
                self.attack_history.append(
                    AttackHistoryEntry(
                        step=attack.step_occurred,
                        attack_type=attack.attack_type,
                        intensity=attack.intensity,
                        detected=attack.detected,
                    )
                )
        if len(self.attack_history) > 100:
            self.attack_history = self.attack_history[-100:]

    def record_action_in_history(self, action_type: str) -> None:
        """
        Annotate the most recent unannotated history entry with the AI's action.
        Falls back to appending a sentinel entry so every step is tracked,
        giving the adaptive attack system a complete action trace.
        """
        for entry in reversed(self.attack_history):
            if entry.ai_response is None:
                entry.ai_response = action_type
                return
        self.attack_history.append(
            AttackHistoryEntry(
                step=self.step_count,
                attack_type="none",
                intensity=0.0,
                detected=True,
                ai_response=action_type,
            )
        )
        if len(self.attack_history) > 100:
            self.attack_history = self.attack_history[-100:]

    def _compute_risk(self):
        """
        Compute derived risk_score [0,1], threat_level, and urgency flag.
        risk_score = weighted blend of misinformation_ratio, deepfake_ratio,
                     undetected attack intensity, and inverse health.
        """
        undetected = [a for a in self.attack_signals if not a.detected]
        attack_pressure = (
            sum(a.intensity for a in undetected) / max(1, len(undetected))
            if undetected else 0.0
        )
        risk_score = min(
            0.35 * self.misinformation_ratio
            + 0.25 * self.deepfake_ratio
            + 0.25 * attack_pressure
            + 0.15 * (1.0 - self.system_health),
            1.0,
        )
        if risk_score >= 0.65:
            threat_level = "high"
        elif risk_score >= 0.35:
            threat_level = "medium"
        else:
            threat_level = "low"

        urgency = (
            threat_level == "high"
            or self.system_health < 0.4
            or self.steps_since_last_detection > 5
        )
        return risk_score, threat_level, urgency

    # ──────────────────────────────────────────
    # Finders
    # ──────────────────────────────────────────

    def _find_content(self, content_id: str) -> Optional[ContentItem]:
        for c in self.content_pool:
            if c.content_id == content_id:
                return c
        return None

    def _find_agent(self, agent_id: str) -> Optional[AgentModel]:
        for a in self.agents:
            if a.agent_id == agent_id:
                return a
        return None
