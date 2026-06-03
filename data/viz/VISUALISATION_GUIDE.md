# SIRLO Data Visualisation Guide

All outputs are saved to `data/viz/` when you run:
```bash
python src/data/run_and_visualise.py
```

---

## 1. `ep{N}_world_state.png` — World-State Tensor Dashboard
> **3 files, one per episode/region**

Shows a **64×64 synthetic satellite snapshot** at the **mid-episode timestep (step 24 = T+12h)**. Each of the 8 channels is a separate panel:

| Panel | Meaning | Colourmap |
|---|---|---|
| **GOES-16 Cloud BT** | Cloud brightness temperature — blue = cold/thick cloud, red = clear sky | `RdYlBu_r` |
| **FIRMS Fire BT** | Fire hotspot intensity | `hot` |
| **MODIS NDVI** | Vegetation health — green = healthy, yellow = burned/stressed | `YlGn` |
| **MODIS LST** | Land surface temperature — bright = hot (fire/bare soil) | `inferno` |
| **SMAP Soil Moisture** | Moisture — blue = wet, inversely correlated with LST | `Blues` |
| **GPM Precip** | Precipitation — correlated with cloud cover | `PuBu` |
| **Cloud Mask (binary)** | Hard threshold (>0.55) — white = cloudy pixel | `binary` |
| **Fire Mask (binary)** | Hard threshold (>0.45) — red = confirmed fire pixel | `Reds` |
| **Composite (R=Fire, G=NDVI, B=Cloud)** | False-colour RGB: red patches = fire, green = vegetation, blue = clouds | RGB |
| **Channel Statistics** | Per-channel Mean / Max / ">0.5 cover" bar chart across the full grid | Bar chart |

> White/coloured dots overlaid on every panel = **6 satellites' current positions** in normalised lat/lon space.

---

## 2. `ep{N}_timeline.png` — Constellation Episode Timeline
> **3 files, one per episode**

4-panel time-series over the full **48 steps × 30 min = 24 hours** for all 6 satellites:

| Panel | Meaning |
|---|---|
| **Battery SoC** | Charge per satellite over time. Red dashed = 0.15 low-battery threshold (triggers IDLE). |
| **Storage Fill** | On-board buffer fill. Orange dashed = 0.90 overflow threshold (triggers DOWNLINK). |
| **Action Heatmap** | Grid (satellites × time), colour-coded: dark grey = IDLE, green = IMAGE, blue = STEER, orange = DOWNLINK |
| **Cumulative Reward** | Running reward score per satellite. Fleet total `Σ` annotated. Blue band = best/worst satellite spread. |

---

## 3. `ep{N}_ground_tracks.png` — Orbital Ground Tracks & Coverage
> **3 files, one per episode**

| Panel | Meaning |
|---|---|
| **Left — Ground Tracks** | Orbital path of each satellite (6 colours). Path markers = action taken: `*`=IMAGE, `^`/`v`=STEER, `■`=DOWNLINK, `.`=IDLE |
| **Right — Coverage Density** | 2D heatmap of where IMAGE actions occurred. Orange ★ = high-reward **fire captures** (reward ≥ +10) |

---

## 4. `fleet_summary.png` — Multi-Episode Fleet Statistics
> **1 file across all 3 episodes**

2×3 bar charts + action distribution pie comparing episodes:

| Panel | Meaning |
|---|---|
| **Total Fleet Reward** | Sum of all shaped rewards across 6 satellites |
| **Fire Captures ★** | Successful fire-pixel imaging events |
| **Cloud Penalties** | Imaging wasted on cloudy pixels (reward = −2) |
| **Successful Downlinks** | Data downlink events during ground station contact |
| **Battery Depletions** | Times a satellite critically drained (<0.02) |
| **Action Distribution (pie)** | Aggregate % of IDLE / IMAGE / STEER / DOWNLINK across all episodes |

---

## 5. `world_state_evolution.gif` — Animated World State
> **1 file, Episode 1 only**

Animates the 8-channel world state through **Episode 1** (every 3rd step, ~16 frames at 5 fps).

- Cloud patterns **drift** over the 24-hour period
- Fire intensity **fluctuates** with small spatial jitter
- **NDVI, LST, soil moisture, precipitation** respond dynamically
- **Satellite dots** move across all 8 panels in real time
- Bottom counter shows current mission time: `T+Xh | Step Y/48`

---

## Physical Correlations in the Synthetic Data

All channels are physically correlated by design to produce realistic training data:

| Cause | Effect |
|---|---|
| Fire ↑ | LST ↑, NDVI ↓, Soil Moisture ↓ |
| Cloud ↑ | Precipitation ↑, LST ↓ (shading), imaging reward −2 |
| IMAGE action | Battery drains faster, storage fill increases |
| DOWNLINK action | Requires ground station contact window (`gs_flag=1`, ~600 s per orbit) |
