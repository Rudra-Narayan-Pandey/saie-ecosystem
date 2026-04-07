---
title: SAIE Self-Healing AI Ecosystem
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 7860
---


# SAIE — Self-Healing AI Ecosystem with Synthetic Reality Defense

> A research-grade OpenEnv environment simulating adversarial digital ecosystems.  
> Version 2.0 — with adaptive attacks, partial observability, belief propagation, and memory.

---

## Problem Statement

Modern digital platforms face unprecedented threats from **misinformation**, **synthetic deepfakes**, and **coordinated bot networks**. Left unchecked, these forces erode user trust, distort public discourse, and cause platform-wide health collapse.

SAIE simulates a digital content ecosystem under attack and challenges an AI agent to act as an autonomous **platform health guardian** — suppressing harmful content, banning adversarial agents, detecting deepfakes early, and rebuilding trust before the system collapses.

Crucially, **adversaries adapt**. Overusing any single strategy causes attackers to pivot, making the environment non-trivial for fixed policies and requiring genuine strategic reasoning.

---

## Project Structure

```
saie_env/
├── env.py           # OpenEnv-compliant environment (action validation layer)
├── models.py        # Pydantic models (Observation, Action, Reward, AttackHistoryEntry)
├── ecosystem.py     # Core simulation engine
├── agents.py        # Agent creation and content generation (stealth mechanics)
├── attacks.py       # Adversarial attack logic (adaptive attack system)
├── reward.py        # Reward computation (stability bonus, delay penalty)
├── easy.py          # Easy task definition and heuristic policy
├── medium.py        # Medium task definition and heuristic policy
├── hard.py          # Hard task definition and adaptive heuristic policy
├── graders.py       # Deterministic episode graders (score 0–1)
├── inference.py     # OpenAI-compatible inference script
├── openenv.yaml     # OpenEnv metadata
├── Dockerfile       # Container definition
├── requirements.txt
└── README.md
```

---

## Environment Design

### System Health Formula

```
health = 0.35 × avg_trust
       + 0.40 × (1 − misinformation_ratio)
       + 0.15 × (1 − deepfake_ratio)
       + 0.10 × normal_agent_ratio
```

### State Components

| Field | Type | Description |
|---|---|---|
| `agents` | List[AgentModel] | All agents (normal / malicious / bot) |
| `content_pool` | List[ContentItem] | Visible active content (partial observability) |
| `virality_scores` | Dict[str, float] | Per-content virality |
| `trust_scores` | Dict[str, float] | Per-agent trust level |
| `misinformation_ratio` | float [0–1] | Weighted reach from harmful content |
| `deepfake_ratio` | float [0–1] | Weighted reach from deepfakes |
| `system_health` | float [0–1] | Overall platform health |
| `attack_signals` | List[AttackSignal] | Visible adversarial attacks |
| `attack_history` | List[AttackHistoryEntry] | Last 20 attack events + AI responses |
| `consecutive_healthy_steps` | int | Steps health ≥ 0.65 continuously |
| `steps_since_last_detection` | int | Drives exponential delay penalty |
| `health_trend` | float | 3-step rolling health delta |
| `risk_score` | float [0–1] | Composite situational risk |
| `threat_level` | low/medium/high | Categorical threat assessment |
| `urgency` | bool | True when immediate action is critical |

### Content Types

| Type | Virality Boost | Belief Score | Effect |
|---|---|---|---|
| `real` | 1.0× | High | Improves content quality |
| `misinfo` | 1.2–1.8× | Medium | Raises misinformation_ratio |
| `deepfake` | 1.5–2.5× | High | Raises deepfake_ratio + trust erosion |

---

## Action Space

| Action | Target | Effect |
|---|---|---|
| `boost_content` | content_id | +0.20 virality on real content |
| `suppress_content` | content_id | −0.30 virality, flag content |
| `ban_agent` | agent_id | Deactivate agent + remove their content |
| `increase_trust` | agent_id | +0.15 trust for a normal agent |
| `deploy_fact_check` | content_id | −0.25 virality, flag misinfo/deepfake |
| `retrain_algorithm` | — | Global −0.10 misinfo virality |
| `quarantine_cluster` | — | Ban entire bot cluster |
| `run_deepfake_detection` | — | Probabilistic detection of all deepfakes |
| `attach_warning_label` | content_id | −0.15 virality, −0.20 belief, flag |
| `trace_origin` | content_id | Reveal creator, remove stealth |

---

## Advanced Mechanics

### Adaptive Attack System
Adversaries analyze the AI's last 8 actions and counter-adapt:
- Frequent **fact-checking / suppression** → attackers pivot to deepfake waves
- Frequent **banning / quarantine** → attackers spawn new bot clusters
- Frequent **deepfake detection** → attackers increase stealth level
- Frequent **retrain_algorithm** → attackers escalate trust erosion

### Partial Observability
- High-stealth agents produce content marked `observable=False`
- AI only sees observable content in `content_pool`
- `trace_origin` and `run_deepfake_detection` reveal hidden items
- Guaranteed fallback: if all content is hidden, the 3 highest-virality items are auto-revealed

### Belief Propagation
- Content with `belief_score > 0.7` receives an additional per-step virality push
- High-belief deepfakes are harder to detect and spread more aggressively

### Probabilistic Actions
All actions have a base success rate (82–97%). Failed actions have no effect, requiring the AI to account for outcome uncertainty.

### Memory System
`attack_history` in `Observation` stores the last 20 attack events with the AI's responses annotated. The adaptive attack system reads a sliding window of the last 8 entries.

### Observation Enrichment
- `risk_score`: composite of misinformation, deepfake, attack pressure, and inverse health
- `threat_level`: `low / medium / high` categorical assessment
- `urgency`: boolean flag indicating when immediate high-impact action is critical

---

## Reward Function

```
reward = health_delta × 10
       − misinfo_spike × 8
       − deepfake_spike × 5
       − 5.0  (collapse penalty if health < 0.15)
       + 1.5  (per newly detected attack)
       + stability_bonus  (0.15 × consecutive_healthy_steps, capped +2.0)
       − delay_penalty    (0.25 × intensity × delay_steps^1.4, capped −3.0)
```

Range: approximately `[−13, +12]` per step.

---

## Grader

Each task uses a deterministic grader returning a score in `[0.0, 1.0]`:

```
score = w_final × final_health
      + w_avg   × avg_health
      + w_min   × min_health
      + stability_bonus  (up to 0.05)
      + misinfo_bonus    (up to 0.05)
```

Collapse (health < 0.05 at any point) reduces score by 70%.

---

## Tasks

### Easy
- **Agents**: 20 normal, 3 malicious  
- **Max steps**: 30  
- **Threat**: Basic misinformation, low deepfake probability  
- **Goal**: Maintain `system_health > 0.6`

### Medium
- **Agents**: 20 normal, 6 malicious, 4 bots  
- **Max steps**: 50  
- **Threat**: Coordinated bot clusters + deepfake waves  
- **Goal**: Prevent trust collapse, keep `system_health > 0.5`

### Hard
- **Agents**: 20 normal, 10 malicious, 8 bots  
- **Max steps**: 80  
- **Threat**: Multi-stage adaptive assault — deepfake → virality cascade → trust erosion  
- **Goal**: Prevent ecosystem collapse, keep `system_health > 0.4`

---

## Setup

### Local

```bash
cd saie_env
pip install -r requirements.txt
```

### Docker

```bash
docker build -t saie-env .
docker run \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="<your_token>" \
  -e TASK="hard" \
  saie-env
```

---

## Running Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="<your_token>"
export TASK="medium"

python inference.py
```

### Example Output

```
[START] task=medium env=saie model=gpt-4o-mini
[STEP] step=1 action=run_deepfake_detection reward=0.42 done=false error=null
[STEP] step=2 action=quarantine_cluster reward=1.15 done=false error=null
[STEP] step=3 action=deploy_fact_check reward=0.78 done=false error=null
...
[STEP] step=50 action=retrain_algorithm reward=0.31 done=true error=null
[END] success=true steps=50 rewards=0.42,1.15,0.78,...,0.31
```

---

## Running with Heuristic Policy (No API required)

```python
import sys
sys.path.insert(0, ".")
from medium import run_task, default_policy

score, observations = run_task(default_policy)
print(f"Score: {score:.4f}")
print(f"Final health: {observations[-1].system_health:.4f}")
```

---

## License

MIT
