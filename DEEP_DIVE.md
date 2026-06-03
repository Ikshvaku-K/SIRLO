# SIRLO: A Deep-Dive into Every Process

> **S**atellite **I**ntelligence via **R**einforcement **L**earning for **O**bservation

This document explains every process in the SIRLO framework in detail — from how real-world satellite data enters the system, to how an AI agent is trained to make autonomous decisions in space.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Process 1 — Authentication & GEE Connection](#2-process-1--authentication--gee-connection)
3. [Process 2 — World State Generation](#3-process-2--world-state-generation)
4. [Process 3 — Orbital Propagation](#4-process-3--orbital-propagation)
5. [Process 4 — The Satellite Subsystem Model](#5-process-4--the-satellite-subsystem-model)
6. [Process 5 — The ViT Agent (The Brain)](#6-process-5--the-vit-agent-the-brain)
7. [Process 6 — The MARL Environment (The Game World)](#7-process-6--the-marl-environment-the-game-world)
8. [Process 7 — Training with REINFORCE](#8-process-7--training-with-reinforce)
9. [Process 8 — Data Generation & Visualization Pipeline](#9-process-8--data-generation--visualization-pipeline)
10. [End-to-End Data Flow Diagram](#10-end-to-end-data-flow-diagram)

---

## 1. The Big Picture

SIRLO is a **Digital Twin** of a satellite constellation over the Amazon rainforest. The core problem it solves is:

> *Traditional satellites follow pre-programmed schedules sent from the ground. But what if clouds roll in? Or a wildfire starts unexpectedly? SIRLO trains satellites to make their own smart decisions in real-time using AI.*

The framework does this in three broad stages:

| Stage | What Happens |
|---|---|
| **World Simulation** | Real satellite data (clouds, fires) is downloaded and converted into a spatial "world map" the AI can read. |
| **Agent Policy** | A Vision Transformer AI model observes the world map and the satellite's own battery/storage status, and decides what to do next. |
| **RL Training** | The AI is rewarded for smart actions (imaging a fire) and penalized for dumb ones (wasting power on clouds), and learns over thousands of episodes. |

---

## 2. Process 1 — Authentication & GEE Connection

**File:** `src/utils/gee_auth.py`

Before any real data can flow into the system, the code must establish a trusted connection with Google Earth Engine (GEE) — the massive cloud platform that holds petabytes of real satellite imagery.

### How it works

GEE authentication is **not** a hardcoded password in the code. Instead, it uses a secure, layered approach:

1. **One-Time Setup:** A developer runs `earthengine authenticate` in their terminal once. This opens a Google login page in the browser.
2. **Token Saved Locally:** After a successful login, GEE saves a persistent OAuth2 token to the local filesystem at `~/.config/earthengine/credentials`.
3. **Runtime Discovery:** When `initialize_gee()` is called in the code, the `ee` Python library automatically finds and uses this token file. No secrets are ever written into source code.
4. **Project Resolution:** The function resolves which GEE *Project* to bill API calls to. It checks, in order:
    - An explicit `project` argument passed to the function.
    - The `DEFAULT_GEE_PROJECT` environment variable.
    - The `EARTHENGINE_PROJECT` environment variable (from a `.env` file).
5. **Service Account Option:** For server deployments (where there's no browser), you can set `GOOGLE_APPLICATION_CREDENTIALS` to a path pointing to a JSON key file for a GEE Service Account.

```python
# The single line that connects to all of Google's satellite data
ee.Initialize(project=resolved)
```

---

## 3. Process 2 — World State Generation

**File:** `src/env/world_state.py`  
**Class:** `WorldStateGenerator`

This is the "eye" of the system. It downloads real data from space and synthesizes it into a structured tensor that the AI can consume.

### What data is pulled

| Data Source | GEE Collection | What it tells us |
|---|---|---|
| **GOES-16** (Geostationary weather sat) | `NOAA/GOES/16/MCMIPF` | Cloud cover and brightness temperature. Band `CMI_C13` (Clean Infrared Window) works day and night. |
| **FIRMS** (Fire watch system) | `FIRMS` | Thermal anomaly brightness temperature. Band `T21` identifies active fire hotspots. |

### The pipeline

1. **Time Window Filtering:** For cloud data, a ±15 minute window around the target time is sampled. For fire data (which updates daily), a ±12 hour window is used to ensure data coverage.
2. **Spatial Filtering (`.filterBounds`):** Only data within the Area of Interest (e.g., the Amazon bounding box) is downloaded — not the entire globe.
3. **Band Selection & Combination:** The relevant bands from each dataset are selected and stacked into a single multi-channel `ee.Image`:
    ```python
    combined = ee.Image.cat([cloud_img, fire_img]).rename(['cloud', 'fire'])
    ```
4. **Rasterization to NumPy:** The combined image is sampled at a fixed pixel resolution over the region using `.sampleRectangle()`, converting the GEE cloud object into a local NumPy array of shape `(2, H, W)` — two channels (Cloud, Fire) over a height × width grid.
5. **Fallback:** If the GEE download fails for any reason, a `(2, 100, 100)` array of zeros is returned so training can continue without crashing.

### Output

```
WorldState Tensor: shape (2, 64, 64)
  Channel 0 → Cloud Brightness Temperature (float, normalized)
  Channel 1 → Fire Hotspot Intensity (float, normalized)
```

---

## 4. Process 3 — Orbital Propagation

**File:** `src/env/propagator.py`  
**Class:** `OrbitalPropagator`

This is the physics engine. It tells the simulation exactly *where* each satellite is in its orbit at any moment in time, with real orbital mechanics.

### What is a TLE?

Every satellite in orbit is described by a **Two-Line Element (TLE)** set — a standardized format of two 69-character lines that encode all the orbital parameters: inclination, eccentricity, altitude, current position, and more. TLEs are publicly available for thousands of satellites (e.g., from [CelesTrak](https://celestrak.org)).

### SGP4 Propagation via Skyfield

The `OrbitalPropagator` class wraps the [Skyfield](https://rhodesmill.org/skyfield/) library, which implements the **SGP4** (Simplified General Perturbations Model 4) algorithm. This is the same algorithm used to track real satellites and predict their positions.

```python
self.satellite = EarthSatellite(tle_line1, tle_line2, name, self.ts)
```

### Key capabilities

| Method | What it does |
|---|---|
| `get_position(time)` | Returns `(latitude°, longitude°, altitude_km)` of the satellite at a given UTC time. |
| `check_visibility(target_lat, target_lon, time)` | Checks if a ground target is above the satellite's sensor horizon (>10° elevation angle). Returns `True`/`False`. |
| `propagate_orbit(start_time, duration_min)` | Generates a time series of positions over an entire orbit arc, useful for plotting ground tracks. |

### Elevation angle check

Visibility is not just about the satellite being "above" the target. The satellite must be high enough above the local horizon for the camera to see it clearly, which is controlled by `min_elevation=10.0` degrees (a commonly used real-world threshold).

---

## 5. Process 4 — The Satellite Subsystem Model

**File:** `src/env/satellite.py`  
**Class:** `Satellite`

Each satellite is not just a position in space — it is a physical system with limited resources that must be managed. This class models the **power and data subsystems** of a satellite.

### Resource budget

| Resource | Initial Value | Unit |
|---|---|---|
| Battery | 100 | % |
| Storage | 0 | MB (max 1000 MB) |

### Energy costs per action

| Action Code | Action | Energy Cost (per step) |
|---|---|---|
| `0` | Idle | −0.1% |
| `1` | Image Target | −5.0% (taking a photo is expensive!) |
| `2` | Downlink Data | −2.0% + frees up to 100 MB of storage |

### Charging

If the satellite is in sunlight (`is_sunlit=True`), it charges at `+1.5%` per step from its solar panels.

### Reward & Penalty signals

- `+reward` for each MB of data successfully downlinked to a ground station.
- `−10.0 penalty` if the battery drops to 0% (the satellite goes "dead").

### State vector output

The satellite's state is compressed into a minimal 2-element normalized vector for the AI to read:

```
[Battery%, StorageFill%]  →  e.g., [0.87, 0.43]
```

---

## 6. Process 5 — The ViT Agent (The Brain)

**File:** `src/agents/vit_agent.py`  
**Class:** `ViTAgent`

This is the AI model — the actual neural network that makes decisions. It is a **Vision Transformer (ViT)**, a state-of-the-art architecture originally designed for image classification that has been repurposed here as a decision-making policy network.

### Why a Vision Transformer?

The world state is a **spatial image** (2 channels × 64 × 64 pixels). CNNs are the classic choice for image data, but ViTs use a **self-attention mechanism** that captures long-range spatial dependencies — for example, understanding that a large cloud formation 100km away might drift over the fire zone on the next timestep.

### Architecture breakdown

```
Input World State: (Batch, 2 channels, 64×64 pixels)
         │
         ▼
┌─────────────────────────────┐
│ 1. Patch Embedding (Conv2d) │  → Splits the 64×64 image into 64 patches
│    Kernel: 8×8, Stride 8    │    of size 8×8. Each patch → 128-dim vector.
└─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│ 2. CLS Token + Positional Embed  │  → A special [CLS] token is prepended.
│                                  │    Positional embeddings tell the model
│                                  │    where each patch is in the image.
└──────────────────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│ 3. Transformer Encoder (×4)    │  → 4 stacked attention layers with 4
│    Self-Attention + FFN        │    attention heads each. This is the core
│                                │    "understanding" step.
└────────────────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│ 4. CLS Token Output          │  → The [CLS] token output acts as a global
│                              │    summary of what the satellite "sees".
└──────────────────────────────┘
         │  (concat)
         ├──────────────────────────────────────────────────┐
         │                                                  │
         │                         ┌──────────────────────────────────────┐
         │                         │  Satellite State Embedding (Linear)  │
         │                         │  [Battery%, Storage%] → 128-dim      │
         │                         └──────────────────────────────────────┘
         │                                                  │
         └────────────────────┬─────────────────────────────┘
                              ▼
         ┌──────────────────────────────────┐
         │ 5. Action Head (Linear)          │  → Two-layer MLP outputting 3 logits
         │                                  │    (one per possible action).
         └──────────────────────────────────┘
                              │
                              ▼
         Output Logits: [Idle_score, Image_score, Downlink_score]
         → Softmax → Action Probabilities → Sample Action
```

### State Fusion

The critical innovation in this architecture is **State Fusion** (step 4 above). The CLS token output (what the satellite *sees*) is concatenated with the embedded satellite internal state (battery, storage). This means the agent doesn't just look at the world — it looks at both the world *and* its own physical condition before deciding what to do. A satellite with 5% battery should act *very* differently than one at 100%, even if they see the same fire below.

---

## 7. Process 6 — The MARL Environment (The Game World)

**File:** `src/training/marl_env.py`  
**Class:** `MARLEnvironment`

This class is the "game engine" that ties everything together. It follows the standard **OpenAI Gym** interface (`reset()` / `step()`) so it's compatible with any standard RL training loop.

### Initialization

When the environment is created, it spawns:
- **N `Satellite` objects** — each with their own battery, storage, and power model.
- **N `OrbitalPropagator` objects** — each initialized with slightly different TLEs to simulate a real distributed constellation (satellites at different orbital plane longitudes).

### `reset()` — Starting a new episode

1. Sets the simulation clock back to "now".
2. Fetches (or mocks) the initial world state tensor.
3. Queries all satellites for their current state vectors.
4. Returns a dictionary with `world_state` and `satellite_states`.

### `step(actions)` — One tick of time

Each call to `step()` advances the simulation clock by **60 seconds** and performs:

1. **Clock Advance:** `current_time += 60 seconds`
2. **World State Update:** The world state tensor is refreshed (in production via GEE; in mocked mode, via random noise to simulate changing conditions).
3. **Satellite Loop (per satellite):**
   - **Solar Check:** If the simulation hour is between 6:00 and 18:00, the satellite is in sunlight and can charge.
   - **Visibility Check:** Whether the imaging target is actually visible from the satellite's current orbital position.
   - **Action Override:** If the agent chose to "Image" but the target isn't visible, the action is downgraded to "Idle" (you can't take a photo of something you can't see).
   - **Physics Update:** The `Satellite.step()` method is called, consuming energy, filling storage, and computing the reward.
4. **Return:** The new state dictionary, a list of rewards (one per satellite), a list of `done` flags (True if a satellite battery hit 0), and debug info.

---

## 8. Process 7 — Training with REINFORCE

**File:** `src/training/trainer.py`  
**Class:** `MARLTrainer`

This is where machine learning happens. The trainer implements the **REINFORCE** algorithm (also known as the Monte Carlo Policy Gradient method).

### Setup

- A single **shared `ViTAgent`** model is used for *all* satellites. This is the "centralized training, decentralized execution" (CTDE) paradigm — all satellites share the same brain, but each acts independently based on its own local observation.
- The optimizer is **Adam** (`lr=1e-3`), a standard adaptive learning rate optimizer.
- `gamma=0.99` is the discount factor, meaning the agent values near-future rewards almost as much as immediate ones.

### Training loop (`train_episode`)

1. **Reset:** Get the initial world state and satellite states.
2. **Action Selection (per step):**
   - The shared world state is tiled/repeated so every satellite gets its own copy.
   - All satellite state vectors are batched together.
   - A single forward pass through the `ViTAgent` produces action logits for all N satellites simultaneously.
   - Actions are **sampled** from the resulting probability distribution (`Categorical` distribution). This sampling is what makes exploration possible — the agent doesn't always pick the "most likely" action.
   - The `log_prob` of the chosen action is saved (needed for the gradient calculation later).
3. **Environment Step:** The sampled actions are sent to `MARLEnvironment.step()`.
4. **Repeat** for 100 steps or until all satellites are "dead".

### Policy Update (`update_policy` — the REINFORCE algorithm)

This runs *after* a full episode completes. It's the "learning" step.

1. **Discounted Returns:** For each satellite's trajectory, the received rewards are converted into **discounted returns** — the total future reward from each timestep, discounted by `gamma` for each step into the future:
   ```
   G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
   ```
2. **Normalization:** Returns are normalized (zero mean, unit variance) to reduce variance and stabilize training. This is not strictly part of REINFORCE but is standard practice.
3. **Loss Calculation:** The REINFORCE loss is:
   ```
   Loss = -Σ [ log_prob(action_t) × G_t ]
   ```
   Intuitively: if an action led to a high return, increase its probability. If it led to a low/negative return, decrease its probability.
4. **Backpropagation:** `.backward()` computes the gradient of this loss through the entire `ViTAgent` neural network.
5. **Optimizer Step:** Adam updates the model weights to reduce the loss.

This entire `train_episode` → `update_policy` cycle is what causes the satellites to gradually learn better behaviors over thousands of episodes.

---

## 9. Process 8 — Data Generation & Visualization Pipeline

**Files:** `src/data/generate_dataset.py`, `src/data/run_and_visualise.py`

This pipeline is the "proof of work" layer — it generates fully synthetic but physically-correlated world state data and then renders it into human-readable images and animations.

### `generate_dataset.py` — Synthetic Data Generation

Because calling GEE for every training step is too slow and expensive, this script pre-generates a **synthetic dataset** that mimics real-world correlations:

| Physical Correlation | How it's modelled |
|---|---|
| Fire ↑ → LST ↑ | Land surface temperature increases near fire pixels |
| Fire ↑ → NDVI ↓ | Vegetation health drops in burned areas |
| Cloud ↑ → Precip ↑ | Precipitation is correlated with cloud cover |
| Cloud ↑ → LST ↓ | Clouds shade the ground, lowering temperature |
| Cloud over fire → Imaging penalty | Cloud mask boolean suppresses imaging reward |

The generator produces an **8-channel tensor** per timestep:
1. GOES-16 Cloud BT
2. FIRMS Fire BT
3. MODIS NDVI (Vegetation)
4. MODIS LST (Land Surface Temperature)
5. SMAP Soil Moisture
6. GPM Precipitation
7. Cloud Mask (binary, threshold > 0.55)
8. Fire Mask (binary, threshold > 0.45)

Each channel is spatially coherent (not pure noise) — clouds drift, fires flicker, and physical correlations hold across all channels.

### `run_and_visualise.py` — Visualization

After datasets are generated, this script renders five types of output (saved to `data/viz/`):

| Output File | Description |
|---|---|
| `ep{N}_world_state.png` | All 8 data channels shown as color-mapped panels at a single timestep. Satellite positions overlaid as dots. |
| `ep{N}_timeline.png` | Time-series plots of battery, storage fill, action heatmap, and cumulative reward across the full 24-hour episode. |
| `ep{N}_ground_tracks.png` | A world map with orbital paths drawn. Action markers show what each satellite did at each position. Orange stars show successful fire captures. |
| `fleet_summary.png` | Cross-episode bar charts summarizing total fleet reward, fire captures, cloud penalties, and battery depletions. |
| `world_state_evolution.gif` | An animation showing the world state evolving over time across all 8 channels, with satellite dots moving in real time. |

---

## 10. End-to-End Data Flow Diagram

```
                       ┌─────────────────────────┐
                       │   Google Earth Engine    │
                       │  (GOES-16, FIRMS, etc.)  │
                       └────────────┬────────────┘
                                    │ Real satellite data
                                    ▼
                       ┌─────────────────────────┐
                       │   WorldStateGenerator   │
                       │   (world_state.py)       │
                       │                          │
                       │ GEE → NumPy tensor       │
                       │ shape: (2, 64, 64)       │
                       └────────────┬────────────┘
                                    │
              ┌────────────────────┼─────────────────────┐
              │                    │                     │
              ▼                    ▼                     ▼
   ┌────────────────┐   ┌────────────────────┐  ┌────────────────┐
   │ OrbitalProp.   │   │  Satellite (×N)    │  │  ViTAgent      │
   │ (propagator.py)│   │  (satellite.py)    │  │  (vit_agent.py)│
   │                │   │                    │  │                │
   │ TLE → lat/lon  │   │ Battery, Storage,  │  │ ViT Encoder    │
   │ visibility?    │   │ Energy model       │  │ + State Fusion │
   └───────┬────────┘   └─────────┬──────────┘  └───────┬────────┘
           │                      │                      │
           │ position & visibility│ satellite_state[]    │ action logits
           └──────────────────────┴──────────────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │    MARLEnvironment       │
                       │    (marl_env.py)         │
                       │                          │
                       │  reset() / step()        │
                       │  → rewards, next state   │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │      MARLTrainer         │
                       │      (trainer.py)        │
                       │                          │
                       │  REINFORCE algorithm     │
                       │  → update ViTAgent       │
                       │    weights via Adam      │
                       └────────────┬────────────┘
                                    │ after training
                                    ▼
                       ┌─────────────────────────┐
                       │  Visualization Pipeline  │
                       │ generate_dataset.py      │
                       │ run_and_visualise.py     │
                       │                          │
                       │ → .png plots, .gif       │
                       └─────────────────────────┘
```

---

*Generated by Antigravity — based on full source code analysis of the SIRLO project.*
