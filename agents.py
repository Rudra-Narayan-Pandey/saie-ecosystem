from __future__ import annotations
import random
import uuid
from typing import List

from models import AgentModel, ContentItem


def create_agents(
    num_normal: int,
    num_malicious: int,
    num_bots: int = 0,
    seed: int = 42,
) -> List[AgentModel]:
    rng = random.Random(seed)
    agents: List[AgentModel] = []

    for _ in range(num_normal):
        agents.append(
            AgentModel(
                agent_id=f"agent_{uuid.uuid4().hex[:8]}",
                agent_type="normal",
                trust_score=rng.uniform(0.6, 1.0),
                stealth_level=0.0,
            )
        )

    cluster_id = f"cluster_{uuid.uuid4().hex[:6]}"
    for _ in range(num_malicious):
        agents.append(
            AgentModel(
                agent_id=f"mal_{uuid.uuid4().hex[:8]}",
                agent_type="malicious",
                trust_score=rng.uniform(0.3, 0.6),
                cluster_id=cluster_id,
                stealth_level=rng.uniform(0.1, 0.3),
            )
        )

    for _ in range(num_bots):
        agents.append(
            AgentModel(
                agent_id=f"bot_{uuid.uuid4().hex[:8]}",
                agent_type="bot",
                trust_score=rng.uniform(0.2, 0.5),
                cluster_id=cluster_id,
                stealth_level=rng.uniform(0.2, 0.5),
            )
        )

    return agents


def agent_produce_content(
    agent: AgentModel,
    step: int,
    rng: random.Random,
    deepfake_prob: float = 0.0,
    misinfo_prob: float = 0.0,
) -> ContentItem:
    """Generate a content item based on agent type and stealth level."""
    if agent.agent_type == "normal":
        content_type = "real"
        virality = rng.uniform(0.1, 0.5)
        belief = rng.uniform(0.5, 0.9)
        boost = 1.0
        # High-stealth agents produce slightly stealthier real content
        observable = True

    elif agent.agent_type == "malicious":
        roll = rng.random()
        # Stealth raises effective deepfake/misinfo threshold slightly
        effective_deepfake = deepfake_prob * (1.0 + agent.stealth_level * 0.3)
        effective_misinfo = misinfo_prob * (1.0 + agent.stealth_level * 0.2)

        if roll < effective_deepfake:
            content_type = "deepfake"
            virality = rng.uniform(0.5, 0.9)
            belief = rng.uniform(0.6, 0.95)
            boost = rng.uniform(1.5, 2.5)
        elif roll < effective_deepfake + effective_misinfo:
            content_type = "misinfo"
            virality = rng.uniform(0.4, 0.8)
            belief = rng.uniform(0.4, 0.8)
            boost = rng.uniform(1.2, 1.8)
        else:
            content_type = "real"
            virality = rng.uniform(0.1, 0.4)
            belief = rng.uniform(0.3, 0.6)
            boost = 1.0

        # High-stealth malicious agents may hide harmful content from direct observation
        observable = rng.random() > agent.stealth_level * 0.6

    else:  # bot
        content_type = "misinfo"
        virality = rng.uniform(0.3, 0.7)
        belief = rng.uniform(0.3, 0.7)
        boost = rng.uniform(1.1, 1.6)
        observable = rng.random() > agent.stealth_level * 0.5

    agent.content_produced += 1
    return ContentItem(
        content_id=f"c_{uuid.uuid4().hex[:10]}",
        creator_id=agent.agent_id,
        content_type=content_type,
        virality_score=min(virality * boost, 1.0),
        belief_score=belief,
        virality_boost=boost,
        age=0,
        observable=observable,
    )
