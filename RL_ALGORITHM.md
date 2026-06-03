# SIRLO: Reinforcement Learning Algorithm — Deep Dive & Improvement Guide

> **Algorithm:** REINFORCE (Monte Carlo Policy Gradient)  
> **Paradigm:** Multi-Agent Reinforcement Learning (MARL)  
> **Implementation:** `src/training/trainer.py` → `MARLTrainer`

---

## Table of Contents

1. [What Algorithm is Used?](#1-what-algorithm-is-used)
2. [How REINFORCE Works in SIRLO](#2-how-reinforce-works-in-sirlo)
3. [The MARL Setup](#3-the-marl-setup)
4. [Weaknesses of the Current Approach](#4-weaknesses-of-the-current-approach)
5. [Improvement Roadmap](#5-improvement-roadmap)
6. [Code-Level Changes](#6-code-level-changes)

---

## 1. What Algorithm is Used?

SIRLO uses **REINFORCE**, also known as the **Monte Carlo Policy Gradient** method. It is one of the foundational algorithms in deep RL, introduced by Ronald Williams (1992).

The core idea is simple:

> *If an action led to a good outcome, make it more likely. If it led to a bad outcome, make it less likely.*

This is formalized as:

```
∇J(θ) = E [ ∇ log π(a|s; θ) × G_t ]
```

Where:
- `π(a|s; θ)` — the policy network (the `ViTAgent`) that outputs action probabilities given a state
- `G_t` — the **discounted return** from timestep `t` onward
- `θ` — the neural network weights being optimized

---

## 2. How REINFORCE Works in SIRLO

The training loop runs one full **episode** at a time (up to 100 steps = 100 simulated minutes), then performs a single weight update at the end.

### Step-by-Step

#### Step 1 — Reset the Environment
```
env.reset() → initial world_state (2, 64, 64) + satellite_states (N, 2)
```
Each satellite starts with 100% battery, 0 MB storage, and the current GEE world state (or mocked noise).

---

#### Step 2 — Action Selection (per timestep)

The shared `ViTAgent` runs a single batched forward pass for all N satellites simultaneously:

```
world_state (tiled N times) + satellite_states (N, 2)
         ↓
    ViTAgent forward pass
         ↓
   logits (N, 3)  →  softmax  →  probabilities (N, 3)
         ↓
   Categorical distribution → sample action per satellite
         ↓
   Store log_prob(action) for later
```

Actions available per satellite:

| Code | Action | Energy Cost |
|---|---|---|
| `0` | Idle | −0.1% battery |
| `1` | Image Target | −5.0% battery |
| `2` | Downlink Data | −2.0% battery, frees 100 MB |

---

#### Step 3 — Environment Step

`env.step(actions)` advances the simulation clock by 60 seconds and computes:
- Physics (battery drain/charge, storage fill)
- Visibility check (can the satellite actually see the target?)
- Reward signals (positive for downlink, negative for battery death)

---

#### Step 4 — Discounted Return Calculation

After the episode ends, rewards are converted to **discounted returns** per satellite:

```
G_T     = r_T
G_{T-1} = r_{T-1} + γ · G_T
G_{T-2} = r_{T-2} + γ · G_{T-1}
...
G_0     = r_0 + γ · r_1 + γ² · r_2 + ...
```

With `γ = 0.99`, a reward received 100 steps in the future is still worth `0.99^100 ≈ 0.37` today.

---

#### Step 5 — Return Normalization

Returns are normalized to reduce gradient variance (standard practice, not strictly part of REINFORCE):

```python
sat_returns = (sat_returns - sat_returns.mean()) / (sat_returns.std() + 1e-9)
```

---

#### Step 6 — Policy Loss & Weight Update

```
Loss = -Σ [ log_prob(action_t) × G_t ]
```

The negative sign turns maximization into minimization (standard in PyTorch). The Adam optimizer then updates all weights of the `ViTAgent` via backpropagation.

```python
policy_loss.backward()
self.optimizer.step()
```

---

## 3. The MARL Setup

SIRLO uses the **Centralized Training, Decentralized Execution (CTDE)** paradigm:

| Property | Value |
|---|---|
| Number of agents | N satellites (default 5) |
| Shared policy | ✅ Yes — one `ViTAgent` for all satellites |
| Communication between agents | ❌ No — satellites act independently |
| Observation type | Local (each satellite sees its own state + the shared world map) |
| Reward type | Individual per satellite |

All satellites share weights, so when one satellite learns "don't image clouds", all satellites benefit. This is parameter-efficient and works well for homogeneous fleets.

---

## 4. Weaknesses of the Current Approach

| Problem | Impact | Root Cause |
|---|---|---|
| **High gradient variance** | Slow, unstable learning | REINFORCE uses full Monte Carlo returns with no baseline |
| **No value function** | Cannot separate "good state" from "good action" | No critic head on the ViTAgent |
| **No inter-satellite communication** | Satellites may redundantly target the same fire | No shared message passing |
| **Sparse reward** | Agent gets very little signal per step | Only rewarded on data downlink |
| **Short episode horizon** | 100 steps ≈ 1 LEO orbit — not enough to learn long-term strategy | Hard-coded `range(100)` in `train_episode` |
| **No trust-region constraint** | A single bad gradient step can destroy learned policy | REINFORCE has no clipping |
| **Mocked world state** | Policy trained on noise, not real cloud/fire patterns | GEE calls are mocked for speed |

---

## 5. Improvement Roadmap

Listed in order of implementation priority and expected impact.

---

### 🔴 Priority 1 — Add a Baseline (Actor-Critic / A2C)

**What:** Add a value head to `ViTAgent` that predicts `V(s)` — the expected future reward from a state. Use the **advantage** `A_t = G_t - V(s_t)` instead of raw returns.

**Why:** The biggest single source of variance in REINFORCE is that `G_t` can be high even when an action was mediocre — the state itself was just good. Subtracting a learned baseline eliminates this bias.

**New loss:**
```
Actor loss:  -Σ log_prob(action_t) × A_t
Critic loss:  Σ (G_t - V(s_t))²
Total loss: actor_loss + 0.5 × critic_loss
```

**Expected gain:** 2–5× faster convergence, significantly more stable training curves.

---

### 🔴 Priority 2 — Upgrade to PPO

**What:** Replace the REINFORCE update with **Proximal Policy Optimization (PPO)**. PPO adds a clipped surrogate loss that prevents the policy from changing too drastically in one update step.

**PPO clipped objective:**
```
L_CLIP = E[ min( r_t × A_t,  clip(r_t, 1-ε, 1+ε) × A_t ) ]
```
Where `r_t = π_new(a|s) / π_old(a|s)` is the probability ratio and `ε = 0.2`.

**Why:** PPO is the current industry standard for on-policy RL. It's significantly more sample-efficient than REINFORCE and rarely diverges.

---

### 🟡 Priority 3 — Add Entropy Regularization

**What:** Add an entropy bonus to the loss to prevent the policy from becoming prematurely deterministic.

```python
entropy_bonus = dist.entropy().mean()
total_loss = policy_loss - 0.01 * entropy_bonus
```

**Why:** Without this, the agent often converges to always choosing the same action (e.g., always "Idle") within a few hundred episodes.

---

### 🟡 Priority 4 — Reward Shaping

**What:** Add dense intermediate rewards so the agent gets useful signal every step, not just on downlink.

| Condition | Shaped Reward |
|---|---|
| Imaged a fire pixel with no cloud cover | `+0.5` |
| Downlinked data over a ground station | `+1.0 × MB / 100` (already exists) |
| Battery dropped below 10% | `−2.0` warning penalty |
| Imaged a cloud-covered area | `−0.3` (wasted energy) |
| Idle while a fire is clearly visible and battery > 50% | `−0.2` opportunity cost |

**Why:** Sparse rewards make credit assignment nearly impossible. With 100 steps and only a terminal reward on downlink, the agent often cannot figure out *which* of its actions mattered.

---

### 🟡 Priority 5 — Inter-Satellite Communication

**What:** Add a communication Transformer between satellites. Before each action selection, satellites broadcast a learned embedding and attend to each other's messages.

```
Satellite embeddings (N, D)
        ↓
Multi-Head Self-Attention across satellites
        ↓
Contextualized embeddings (N, D) → action heads
```

**Why:** Currently, two satellites may both choose to image the same fire pixel simultaneously — wasting one satellite's energy. With communication, they can coordinate.

---

### 🟢 Priority 6 — Gradient Clipping

**What:** A single line added after `loss.backward()`:
```python
torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
```

**Why:** Large gradient steps can destabilize the ViT (especially its attention layers). Clipping is cheap insurance against training blowups.

---

### 🟢 Priority 7 — Extend Episode Horizon

**What:** Change `range(100)` to `range(1440)` for a full 24-hour episode (1440 minutes). Introduce curriculum learning — start short, increase horizon as training progresses.

```python
# Curriculum: ramp from 100 → 1440 steps over 500 episodes
max_steps = min(100 + episode_num * 2, 1440)
```

---

### 🟢 Priority 8 — Use Real GEE Data During Training

**What:** Replace the mocked `world_state` (random noise) with real calls to `WorldStateGenerator` at a lower frequency (e.g., every 10 steps).

**Why:** The policy currently learns to act in pure noise — it cannot develop meaningful cloud-avoidance or fire-targeting behaviors. Real data is the only path to a policy that transfers to deployment.

---

## 6. Code-Level Changes

### A2C upgrade to `vit_agent.py`

```python
# Add a value head alongside the existing action_head
self.value_head = nn.Sequential(
    nn.Linear(embed_dim + sat_embed_dim, embed_dim),
    nn.GELU(),
    nn.Linear(embed_dim, 1)  # outputs scalar V(s)
)

def forward(self, world_state, satellite_state):
    # ... existing code ...
    fused = torch.cat([cls_output, sat_embed], dim=-1)
    action_logits = self.action_head(fused)
    state_value = self.value_head(fused)   # NEW
    return action_logits, state_value      # return both
```

### A2C update to `trainer.py` — `update_policy`

```python
def update_policy(self, log_probs, rewards, values):
    actor_losses = []
    critic_losses = []

    for s in range(num_sats):
        # ... compute sat_returns as before ...
        advantages = sat_returns - values_tensor[s]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        for t in range(num_steps):
            actor_losses.append(-log_probs[t][s] * advantages[t].detach())
            critic_losses.append(F.mse_loss(values[t][s], sat_returns[t]))

    total_loss = torch.stack(actor_losses).sum() + 0.5 * torch.stack(critic_losses).sum()
    self.optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
    self.optimizer.step()
```

### Entropy bonus

```python
entropy = dist.entropy().mean()
policy_loss = reinforce_loss - 0.01 * entropy
```

---

## Upgrade Path Summary

```
Current: REINFORCE (vanilla, no baseline, no clipping)
    ↓
Step 1: A2C  (add value head → advantage estimation)
    ↓
Step 2: PPO  (add clipped surrogate loss → stable large-batch updates)
    ↓
Step 3: MAPPO  (PPO with inter-satellite attention → coordinated fleet)
    ↓
Step 4: MAPPO + Real GEE data  (policy generalizes to real satellite conditions)
```

Each step is incrementally buildable on top of the previous — no need to rewrite the environment or the ViTAgent architecture from scratch.

---

*Generated by Antigravity — companion document to `DEEP_DIVE.md`.*
