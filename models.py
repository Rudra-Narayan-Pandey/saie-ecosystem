from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Content item
# ──────────────────────────────────────────────
class ContentItem(BaseModel):
    content_id: str
    creator_id: str
    content_type: Literal["real", "misinfo", "deepfake"]
    virality_score: float = Field(ge=0.0, le=1.0)
    belief_score: float = Field(ge=0.0, le=1.0)
    virality_boost: float = Field(ge=1.0)
    spread_count: int = 0
    flagged: bool = False
    age: int = 0                        # steps since creation
    observable: bool = True             # partial observability flag


# ──────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────
class AgentModel(BaseModel):
    agent_id: str
    agent_type: Literal["normal", "malicious", "bot"]
    trust_score: float = Field(ge=0.0, le=1.0)
    active: bool = True
    content_produced: int = 0
    cluster_id: Optional[str] = None
    stealth_level: float = Field(default=0.0, ge=0.0, le=1.0)


# ──────────────────────────────────────────────
# Attack signal
# ──────────────────────────────────────────────
class AttackSignal(BaseModel):
    attack_type: Literal["misinfo_burst", "deepfake_wave", "bot_cluster", "trust_erosion"]
    intensity: float = Field(ge=0.0, le=1.0)
    detected: bool = False
    step_occurred: int = 0
    stealth: float = Field(default=0.0, ge=0.0, le=1.0)


# ──────────────────────────────────────────────
# Attack history entry (memory system)
# ──────────────────────────────────────────────
class AttackHistoryEntry(BaseModel):
    step: int
    attack_type: str          # str (not Literal) to allow sentinel "none"
    intensity: float
    detected: bool
    ai_response: Optional[str] = None


# ──────────────────────────────────────────────
# Observation
# ──────────────────────────────────────────────
class Observation(BaseModel):
    step: int
    agents: List[AgentModel]
    content_pool: List[ContentItem]
    virality_scores: Dict[str, float]
    trust_scores: Dict[str, float]
    misinformation_ratio: float = Field(ge=0.0, le=1.0)
    deepfake_ratio: float = Field(ge=0.0, le=1.0)
    system_health: float = Field(ge=0.0, le=1.0)
    attack_signals: List[AttackSignal]
    active_agent_count: int
    flagged_content_count: int
    # Memory / history
    attack_history: List[AttackHistoryEntry] = Field(default_factory=list)
    consecutive_healthy_steps: int = 0
    steps_since_last_detection: int = 0
    health_trend: float = 0.0
    # Enriched situational awareness
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    threat_level: Literal["low", "medium", "high"] = "low"
    urgency: bool = False       # True when immediate action is strongly recommended


# ──────────────────────────────────────────────
# Action
# ──────────────────────────────────────────────
ActionType = Literal[
    "boost_content",
    "suppress_content",
    "ban_agent",
    "increase_trust",
    "deploy_fact_check",
    "retrain_algorithm",
    "quarantine_cluster",
    "run_deepfake_detection",
    "attach_warning_label",
    "trace_origin",
]


class Action(BaseModel):
    action_type: ActionType
    target_id: Optional[str] = None
    cluster_id: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────
# Reward
# ──────────────────────────────────────────────
class Reward(BaseModel):
    total: float
    health_delta: float
    misinfo_penalty: float
    collapse_penalty: float
    early_detection_bonus: float
    stability_bonus: float = 0.0
    delay_penalty: float = 0.0
    breakdown: Dict[str, float] = Field(default_factory=dict)
