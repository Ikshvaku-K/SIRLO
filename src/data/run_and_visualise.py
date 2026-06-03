"""
SIRLO — Synthetic Dataset Runner + Live Visualiser
====================================================
Generates a synthetic SIRLO dataset (no GEE required) and produces a
rich interactive visualisation dashboard saved to data/viz/.

Simulates:
  • 8-channel world-state tensor (cloud, fire, NDVI, LST, soil, rain, masks)
  • N-satellite orbital constellation dynamics
  • Autonomous action & shaped reward computation
  • Episode progression over 24 hours

Run:
    python src/data/run_and_visualise.py
"""

from __future__ import annotations

import os
import sys
import datetime
import random
import math
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

# ── aesthetic ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.titleweight": "bold",
    "axes.titlecolor":  "#58a6ff",
})

CHANNEL_NAMES  = ["GOES-16\nCloud BT", "FIRMS\nFire BT", "MODIS\nNDVI",
                  "MODIS\nLST",        "SMAP\nSoil Moist", "GPM\nPrecip",
                  "Cloud\nMask (bin)", "Fire\nMask (bin)"]
ACTION_NAMES   = ["IDLE", "IMAGE", "STEER↑", "STEER↓", "DOWNLINK"]
ACTION_COLORS  = ["#30363d", "#238636", "#1f6feb", "#1f6feb", "#f0883e"]
REGION_COLORS  = {"amazon_wildfire": "#f85149", "cerrado": "#ffa657", "pantanal": "#3fb950"}

N_SATS         = 6
N_ACTIONS      = 5
N_CHANNELS     = 8
H, W           = 64, 64
STEPS_PER_EP   = 48   # 48 × 30 min = 24 h
N_EPISODES     = 3
OUTPUT_DIR     = Path("data/viz")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Synthetic data generators
# ══════════════════════════════════════════════════════════════════════════════

def _simplex_noise(H: int, W: int, scale: float = 0.08) -> np.ndarray:
    """Fast pseudo-perlin noise on an H×W grid."""
    xs = np.linspace(0, scale * W, W)
    ys = np.linspace(0, scale * H, H)
    xg, yg = np.meshgrid(xs, ys)
    noise = (
        np.sin(xg * 2.1) * np.cos(yg * 1.7) * 0.40
        + np.sin(xg * 4.3 + 0.5) * np.cos(yg * 3.8 + 0.7) * 0.25
        + np.sin(xg * 8.7 + 1.2) * np.cos(yg * 6.5 + 2.1) * 0.15
        + np.random.randn(H, W) * 0.05
    )
    return (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)


def gen_world_state(rng: np.random.Generator, t: float, fire_locs: list) -> np.ndarray:
    """Generate a [8, H, W] world-state tensor for a fractional time t ∈ [0,1]."""
    channels = []

    # Ch0: Cloud BT — organic cloud patterns
    cloud_base = _simplex_noise(H, W, 0.07 + t * 0.01)
    cloud_base = np.clip(cloud_base + 0.15 * math.sin(t * math.pi * 4), 0, 1)
    channels.append(cloud_base)

    # Ch1: Fire BT — sparse hotspot blobs
    fire_ch = np.zeros((H, W), np.float32)
    for (fr, fc) in fire_locs:
        row_off = int(fr * H + rng.integers(-3, 4))
        col_off = int(fc * W + rng.integers(-3, 4))
        rr, cc = np.ogrid[:H, :W]
        sigma = rng.uniform(2, 5)
        blob = np.exp(-((rr - row_off)**2 + (cc - col_off)**2) / (2 * sigma**2))
        fire_ch += blob * rng.uniform(0.6, 1.0)
    fire_ch = np.clip(fire_ch, 0, 1).astype(np.float32)
    channels.append(fire_ch)

    # Ch2: NDVI — vegetation health (inversely correlated with fire)
    ndvi = np.clip(1.0 - fire_ch * 0.8 + _simplex_noise(H, W, 0.05) * 0.3, 0, 1)
    channels.append(ndvi.astype(np.float32))

    # Ch3: LST — land surface temperature (correlated with fire + cloud inverse)
    lst = np.clip(fire_ch * 0.6 + (1 - cloud_base) * 0.3 + _simplex_noise(H, W, 0.06) * 0.2, 0, 1)
    channels.append(lst.astype(np.float32))

    # Ch4: Soil moisture — anti-correlated with LST
    ssm = np.clip(1.0 - lst * 0.7 + _simplex_noise(H, W, 0.09) * 0.2, 0, 1)
    channels.append(ssm.astype(np.float32))

    # Ch5: Precipitation — correlated with cloud cover
    precip = np.clip(cloud_base * 0.8 + rng.random((H, W)).astype(np.float32) * 0.1, 0, 1)
    channels.append(precip.astype(np.float32))

    # Ch6: Cloud mask (binary, threshold 0.55)
    cloud_mask = (cloud_base > 0.55).astype(np.float32)
    channels.append(cloud_mask)

    # Ch7: Fire mask (binary, threshold 0.45)
    fire_mask = (fire_ch > 0.45).astype(np.float32)
    channels.append(fire_mask)

    return np.stack(channels, axis=0)  # [8, 64, 64]


def gen_satellite_positions(step: int, n_sats: int = N_SATS) -> np.ndarray:
    """Return [N, 2] normalised (lat_n, lon_n) positions within ROI."""
    positions = []
    for i in range(n_sats):
        period = 5400 + i * 900   # seconds (~90-105 min)
        t_sec  = step * 1800 + i * (period / n_sats)
        phase  = (2 * math.pi * t_sec / period)
        lat_n  = (math.sin(phase) + 1) / 2
        lon_n  = ((math.cos(phase * 0.7) + 1) / 2 + i / n_sats) % 1.0
        positions.append([lat_n, lon_n])
    return np.array(positions, dtype=np.float32)  # [N, 2]


def gen_sat_state(step: int, positions: np.ndarray, cloud_mask: np.ndarray,
                  battery: list, storage: list, rng: np.random.Generator) -> np.ndarray:
    states = []
    H_, W_ = cloud_mask.shape
    for i in range(N_SATS):
        lat_n, lon_n = positions[i]
        r = int(np.clip(lat_n * H_, 0, H_ - 1))
        c = int(np.clip(lon_n * W_, 0, W_ - 1))
        half = 4
        patch = cloud_mask[max(0, r-half):min(H_, r+half), max(0, c-half):min(W_, c+half)]
        cloud_frac = float(patch.mean()) if patch.size > 0 else 0.0

        period_s  = 5400 + i * 900
        t_sec     = step * 1800 + i * 300
        phase     = t_sec % period_s
        gs_flag   = 1.0 if phase < 600 else 0.0
        ttc       = min((period_s - phase) / 60.0, 120.0) / 120.0

        states.append([battery[i], storage[i], lat_n, lon_n,
                       (500.0 + i * 10 - 450) / 150.0, cloud_frac, gs_flag, ttc])
    return np.array(states, dtype=np.float32)


def compute_actions_rewards(sat_states: np.ndarray, world_state: np.ndarray,
                             rng: np.random.Generator) -> tuple:
    """Epsilon-greedy-ish policy for realistic action distribution."""
    cloud_mask = world_state[6]
    fire_mask  = world_state[7]
    H_, W_     = cloud_mask.shape
    actions, rewards = [], []

    for i in range(N_SATS):
        lat_n, lon_n = sat_states[i, 2], sat_states[i, 3]
        r = int(np.clip(lat_n * H_, 0, H_ - 1))
        c = int(np.clip(lon_n * W_, 0, W_ - 1))
        batt    = sat_states[i, 0]
        stor    = sat_states[i, 1]
        gs_flag = sat_states[i, 6]
        c_frac  = sat_states[i, 5]

        # Greedy-ish policy
        if batt < 0.15:
            a = 0   # IDLE to recharge
        elif gs_flag > 0.5 and stor > 0.5:
            a = 4   # DOWNLINK
        elif c_frac > 0.7:
            a = 0   # IDLE, skip cloud
        elif fire_mask[r, c] > 0.5:
            a = 1   # IMAGE fire target
        elif stor > 0.9:
            a = 4   # DOWNLINK before overflow
        else:
            a = int(rng.choice([0, 1, 2, 3, 4], p=[0.25, 0.45, 0.1, 0.1, 0.1]))

        # Reward
        rew = 0.0
        if a == 1:
            if cloud_mask[r, c] > 0.5:  rew = -2.0
            elif fire_mask[r, c] > 0.5: rew = +10.0
            else:                        rew = +1.0
        elif a == 4 and gs_flag > 0.5:  rew = +2.0
        if batt < 0.02: rew -= 5.0
        if stor > 0.98: rew -= 3.0

        actions.append(a)
        rewards.append(rew)

    return np.array(actions, dtype=np.int32), np.array(rewards, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — World State Tensor Dashboard
# ══════════════════════════════════════════════════════════════════════════════

CMAPS = [
    "RdYlBu_r",  # cloud BT (blue=warm clear, red=cold cloud)
    "hot",        # fire BT
    "YlGn",       # NDVI
    "inferno",    # LST
    "Blues",      # soil moisture
    "PuBu",       # precipitation
    "binary",     # cloud mask
    "Reds",       # fire mask
]


def plot_world_state_dashboard(world_state: np.ndarray, sat_positions: np.ndarray,
                                step: int, region: str, out_path: Path):
    fig = plt.figure(figsize=(20, 11), facecolor="#0d1117")
    fig.suptitle(
        f"SIRLO World-State Tensor  |  Region: {region.replace('_', ' ').title()}  |  Step {step:03d}",
        fontsize=14, fontweight="bold", color="#58a6ff", y=0.98
    )

    # 8 channel panels + 1 composite satellite overlay
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.04, right=0.97, top=0.93, bottom=0.04)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]

    for idx, ax in enumerate(axes):
        ch   = world_state[idx]
        cmap = CMAPS[idx]
        im   = ax.imshow(ch, cmap=cmap, vmin=0, vmax=1, origin="upper", interpolation="bilinear")
        ax.set_title(CHANNEL_NAMES[idx], pad=4)
        ax.axis("off")

        # Satellite footprint dots
        for si in range(N_SATS):
            lat_n, lon_n = sat_positions[si]
            px = lon_n * W
            py = lat_n * H
            ax.plot(px, py, "o", color="#f0f0f0", markersize=4, zorder=5)
            ax.plot(px, py, "o", color=f"C{si}", markersize=2.5, zorder=6)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="#8b949e", labelsize=7)

    # Bottom row — composite + statistics
    ax_comp = fig.add_subplot(gs[2, 0:2])  # composite RGB-ish view
    # Pseudo-RGB: R=fire, G=ndvi, B=cloud
    rgb = np.stack([
        world_state[1],           # R: fire
        world_state[2] * 0.8,     # G: NDVI
        world_state[0] * 0.6,     # B: cloud
    ], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    ax_comp.imshow(rgb, origin="upper", interpolation="bilinear")
    ax_comp.set_title("Composite (R=Fire | G=NDVI | B=Cloud)", pad=4)
    ax_comp.axis("off")

    # Satellite positions overlay on composite
    for si in range(N_SATS):
        lat_n, lon_n = sat_positions[si]
        px = lon_n * W
        py = lat_n * H
        circle = plt.Circle((px, py), 3.5, color=f"C{si}", fill=False, linewidth=1.5)
        ax_comp.add_patch(circle)
        ax_comp.plot(px, py, "x", color="white", markersize=4, markeredgewidth=1.2)
        ax_comp.text(px + 2, py - 2, f"S{si+1}", fontsize=6, color="white")

    # Stats panel
    ax_stat = fig.add_subplot(gs[2, 2:4])
    ax_stat.set_facecolor("#161b22")
    ax_stat.set_title("Channel Statistics", pad=4)
    stat_labels = [cn.replace("\n", " ") for cn in CHANNEL_NAMES]
    means  = world_state.mean(axis=(1, 2))
    maxs   = world_state.max(axis=(1, 2))
    covers = (world_state > 0.5).mean(axis=(1, 2))

    x = np.arange(N_CHANNELS)
    w = 0.28
    ax_stat.bar(x - w,   means,  w, label="Mean",       color="#238636", alpha=0.85)
    ax_stat.bar(x,       maxs,   w, label="Max",        color="#f0883e", alpha=0.85)
    ax_stat.bar(x + w,   covers, w, label=">0.5 Cover", color="#58a6ff", alpha=0.85)
    ax_stat.set_xticks(x)
    ax_stat.set_xticklabels([c.replace("\n", "\n") for c in
                              ["Cloud BT", "Fire BT", "NDVI", "LST",
                               "Soil", "Precip", "CloudMask", "FireMask"]],
                             fontsize=7, rotation=30, ha="right")
    ax_stat.set_ylim(0, 1.15)
    ax_stat.legend(fontsize=7, framealpha=0.3, loc="upper right")
    ax_stat.grid(axis="y", alpha=0.3)
    ax_stat.set_ylabel("Value [0-1]")

    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ World state dashboard → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Constellation Episode Timeline
# ══════════════════════════════════════════════════════════════════════════════

def plot_episode_timeline(history: list, episode: int, out_path: Path):
    """history = list of (world_state[8,H,W], sat_states[N,8], actions[N], rewards[N]) per step."""
    T = len(history)

    batteries = np.array([[h[1][i, 0] for i in range(N_SATS)] for h in history])  # [T, N]
    storages  = np.array([[h[1][i, 1] for i in range(N_SATS)] for h in history])
    rewards   = np.array([[h[3][i]     for i in range(N_SATS)] for h in history])
    actions   = np.array([[h[2][i]     for i in range(N_SATS)] for h in history])
    cum_rew   = rewards.cumsum(axis=0)

    fig, axes = plt.subplots(4, 1, figsize=(18, 12), facecolor="#0d1117",
                              gridspec_kw={"hspace": 0.52})
    fig.suptitle(
        f"SIRLO Constellation Timeline  |  Episode {episode+1}  |  {T} Steps × 30 min",
        fontsize=13, fontweight="bold", color="#58a6ff"
    )

    steps = np.arange(T)
    hours = steps * 0.5   # 30 min per step

    # ── 1. Battery SOC ────────────────────────────────────────────────────────
    ax = axes[0]
    for i in range(N_SATS):
        ax.plot(hours, batteries[:, i], lw=1.8, label=f"SAT-{i+1}", color=f"C{i}", alpha=0.9)
    ax.axhline(0.15, ls="--", lw=0.9, color="#f85149", alpha=0.7, label="Low-batt threshold")
    ax.fill_between(hours, 0, 0.15, alpha=0.08, color="#f85149")
    ax.set_ylabel("Battery SoC")
    ax.set_ylim(0, 1.05)
    ax.set_title("Battery State-of-Charge per Satellite")
    ax.legend(fontsize=7, ncol=N_SATS + 1, framealpha=0.2, loc="upper right")
    ax.grid(alpha=0.25)

    # ── 2. Data Storage Fill ──────────────────────────────────────────────────
    ax = axes[1]
    for i in range(N_SATS):
        ax.plot(hours, storages[:, i], lw=1.8, color=f"C{i}", alpha=0.9)
    ax.axhline(0.90, ls="--", lw=0.9, color="#f0883e", alpha=0.7, label="Overflow threshold")
    ax.fill_between(hours, 0.9, 1.0, alpha=0.08, color="#f0883e")
    ax.set_ylabel("Storage Fill [0-1]")
    ax.set_ylim(0, 1.05)
    ax.set_title("On-Board Storage Buffer Fill Fraction")
    ax.legend(fontsize=7, framealpha=0.2)
    ax.grid(alpha=0.25)

    # ── 3. Action Heatmap ─────────────────────────────────────────────────────
    ax = axes[2]
    cmap_act = mcolors.ListedColormap(ACTION_COLORS)
    im = ax.imshow(actions.T, aspect="auto", cmap=cmap_act, vmin=0, vmax=4,
                   extent=[0, hours[-1], N_SATS - 0.5, -0.5], interpolation="nearest")
    ax.set_yticks(range(N_SATS))
    ax.set_yticklabels([f"SAT-{i+1}" for i in range(N_SATS)], fontsize=8)
    ax.set_title("Action Space Heatmap  (colour = action type)")
    ax.set_ylabel("Satellite")
    patches = [mpatches.Patch(color=ACTION_COLORS[a], label=ACTION_NAMES[a]) for a in range(5)]
    ax.legend(handles=patches, fontsize=7, ncol=5, loc="lower right", framealpha=0.2)
    ax.grid(False)

    # ── 4. Cumulative Reward ──────────────────────────────────────────────────
    ax = axes[3]
    for i in range(N_SATS):
        ax.plot(hours, cum_rew[:, i], lw=1.8, color=f"C{i}", label=f"SAT-{i+1}", alpha=0.9)
    ax.fill_between(hours, cum_rew.min(axis=1), cum_rew.max(axis=1), alpha=0.08, color="#58a6ff")
    ax.axhline(0, lw=0.7, color="#8b949e", ls="--", alpha=0.5)
    total = cum_rew[-1].sum()
    ax.text(hours[-1] * 0.98, cum_rew[-1].max(), f"Fleet Σ = {total:.1f}",
             ha="right", va="bottom", fontsize=8, color="#3fb950")
    ax.set_title("Cumulative Shaped Reward per Satellite")
    ax.set_xlabel("Mission Time (hours)")
    ax.set_ylabel("Cumulative Reward")
    ax.legend(fontsize=7, ncol=N_SATS, framealpha=0.2, loc="upper left")
    ax.grid(alpha=0.25)

    for ax in axes:
        ax.set_xlim(0, hours[-1])

    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Episode timeline → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Orbital Ground Track + Coverage Map
# ══════════════════════════════════════════════════════════════════════════════

def plot_ground_tracks(all_positions: np.ndarray, all_actions: np.ndarray,
                        all_rewards: np.ndarray, out_path: Path):
    """
    all_positions: [T, N, 2]  (lat_n, lon_n) in ROI frame
    all_actions:   [T, N]
    all_rewards:   [T, N]
    """
    T, N, _ = all_positions.shape
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="#0d1117")
    fig.suptitle("Satellite Ground Tracks & Coverage Density",
                 fontsize=13, fontweight="bold", color="#58a6ff")

    ax_track = axes[0]
    ax_cover = axes[1]

    ax_track.set_facecolor("#0d1117")
    ax_track.set_xlim(0, 1)
    ax_track.set_ylim(1, 0)   # lat 0=top, 1=bottom (image convention)
    ax_track.set_xlabel("Longitude (normalised)")
    ax_track.set_ylabel("Latitude (normalised)")
    ax_track.set_title("Orbital Ground Tracks  (colour=satellite, marker=action)")
    ax_track.grid(alpha=0.15)

    markers = {0: ".", 1: "*", 2: "^", 3: "v", 4: "s"}
    for i in range(N):
        lons = all_positions[:, i, 1]
        lats = all_positions[:, i, 0]
        # trajectory line
        points  = np.array([lons, lats]).T.reshape(-1, 1, 2)
        segs    = np.concatenate([points[:-1], points[1:]], axis=1)
        lc      = LineCollection(segs, color=f"C{i}", linewidth=0.8, alpha=0.4)
        ax_track.add_collection(lc)
        # action markers
        for t in range(0, T, 4):
            a  = all_actions[t, i]
            ax_track.scatter(lons[t], lats[t], c=f"C{i}", marker=markers[a],
                             s=20 if a == 1 else 8, zorder=5, alpha=0.8)

    # Legend for action markers
    action_handles = [
        plt.scatter([], [], marker=markers[a], color="white", s=20 if a==1 else 10,
                    label=ACTION_NAMES[a]) for a in range(5)
    ]
    ax_track.legend(handles=action_handles, fontsize=7, framealpha=0.3,
                    title="Actions", title_fontsize=7, loc="lower right")

    # Satellite legend
    sat_handles = [mpatches.Patch(color=f"C{i}", label=f"SAT-{i+1}") for i in range(N)]
    ax_track.legend(handles=sat_handles + action_handles, fontsize=7, framealpha=0.3,
                    ncol=2, loc="upper left")

    # ── Coverage Density Map ──────────────────────────────────────────────────
    ax_cover.set_title("Imaging Coverage Density  (fire captures highlighted)")
    ax_cover.set_facecolor("#0d1117")

    # 2D histogram of IMAGE actions
    image_mask = (all_actions == 1)  # [T, N]
    if image_mask.any():
        img_lats = all_positions[:, :, 0][image_mask]
        img_lons = all_positions[:, :, 1][image_mask]
        h, xe, ye = np.histogram2d(img_lats, img_lons, bins=32, range=[[0,1],[0,1]])
        ax_cover.imshow(h, origin="upper", cmap="plasma", extent=[0,1,1,0],
                        interpolation="gaussian", aspect="auto")
        plt.colorbar(ax_cover.images[0], ax=ax_cover, label="Image count", fraction=0.046)

    # Highlight high-reward captures
    high_rew = all_rewards >= 8.0   # [T, N]  fire captures
    if high_rew.any():
        fr_lats = all_positions[:, :, 0][high_rew]
        fr_lons = all_positions[:, :, 1][high_rew]
        ax_cover.scatter(fr_lons, fr_lats, c="#f0883e", s=60, marker="*",
                         zorder=5, label="Fire capture (+10)", alpha=0.9)
        ax_cover.legend(fontsize=8, framealpha=0.3)

    ax_cover.set_xlabel("Longitude (normalised)")
    ax_cover.set_ylabel("Latitude (normalised)")
    ax_cover.set_xlim(0, 1)
    ax_cover.set_ylim(1, 0)
    ax_cover.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Ground tracks → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Fleet Statistics Summary
# ══════════════════════════════════════════════════════════════════════════════

def plot_fleet_summary(all_episode_stats: list, out_path: Path):
    """all_episode_stats = list of dicts per episode."""
    n_ep = len(all_episode_stats)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#0d1117")
    fig.suptitle("SIRLO Fleet Performance Summary — Multi-Episode Statistics",
                 fontsize=13, fontweight="bold", color="#58a6ff")

    episodes = np.arange(1, n_ep + 1)

    def _bar(ax, key, title, color, ylabel=""):
        vals = [s[key] for s in all_episode_stats]
        ax.bar(episodes, vals, color=color, alpha=0.85, width=0.6)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel or key)
        ax.set_xticks(episodes)
        ax.grid(axis="y", alpha=0.3)
        for ep, v in zip(episodes, vals):
            ax.text(ep, v + max(vals) * 0.01, f"{v:.1f}", ha="center", fontsize=8, color="#c9d1d9")

    _bar(axes[0, 0], "total_fleet_reward",  "Total Fleet Reward",       "#3fb950", "Σ Reward")
    _bar(axes[0, 1], "fire_captures",       "Fire Captures (★)",        "#f0883e", "Count")
    _bar(axes[0, 2], "cloud_penalties",     "Cloud-Obscured Wastes",    "#f85149", "Count")
    _bar(axes[1, 0], "downlinks",           "Successful Downlinks",     "#58a6ff", "Count")
    _bar(axes[1, 1], "battery_depletions",  "Battery Depletions",       "#ffa657", "Count")
    _bar(axes[1, 2], "avg_storage_fill",    "Avg Storage Fill at End",  "#bc8cff", "Fill [0-1]")

    # Action distribution pie (aggregated)
    action_counts = np.zeros(N_ACTIONS)
    for s in all_episode_stats:
        for a in range(N_ACTIONS):
            action_counts[a] += s["action_counts"][a]

    # Replace last subplot with pie
    axes[1, 2].remove()
    ax_pie = fig.add_subplot(2, 3, 6, facecolor="#161b22")
    ax_pie.pie(action_counts, labels=ACTION_NAMES, colors=ACTION_COLORS,
               autopct="%1.1f%%", textprops={"fontsize": 8, "color": "#c9d1d9"},
               wedgeprops={"linewidth": 0.5, "edgecolor": "#0d1117"})
    ax_pie.set_title("Fleet-Wide Action Distribution")
    ax_pie.set_facecolor("#161b22")

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Fleet summary → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — GIF: World State Animation
# ══════════════════════════════════════════════════════════════════════════════

def make_animation(frames_ws: list, frames_pos: list, out_path: Path, fps: int = 4):
    """frames_ws: list of [8,H,W], frames_pos: list of [N,2]"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor="#0d1117")
    fig.suptitle("World-State Evolution (30-min cadence, 24-h episode)",
                 fontsize=11, color="#58a6ff")
    plt.subplots_adjust(hspace=0.4, wspace=0.25, left=0.03, right=0.97,
                        top=0.92, bottom=0.03)

    imgs, dots = [], []
    for idx, ax in enumerate(axes.flat):
        im = ax.imshow(frames_ws[0][idx], cmap=CMAPS[idx], vmin=0, vmax=1,
                       origin="upper", interpolation="bilinear", animated=True)
        ax.set_title(CHANNEL_NAMES[idx], pad=3, fontsize=8)
        ax.axis("off")
        imgs.append(im)
        sat_dots = [ax.plot([], [], "o", color=f"C{i}", markersize=3, zorder=5)[0]
                    for i in range(N_SATS)]
        dots.append(sat_dots)

    time_text = fig.text(0.5, 0.005, "", ha="center", fontsize=9, color="#8b949e")

    def update(frame_idx):
        ws  = frames_ws[frame_idx]
        pos = frames_pos[frame_idx]
        for idx, im in enumerate(imgs):
            im.set_data(ws[idx])
        for idx in range(N_CHANNELS):
            for si in range(N_SATS):
                dots[idx][si].set_data([pos[si, 1] * W], [pos[si, 0] * H])
        t_h = frame_idx * 0.5
        time_text.set_text(f"T+{t_h:.1f}h  |  Step {frame_idx + 1}/{len(frames_ws)}")
        return imgs + [d for row in dots for d in row] + [time_text]

    anim = FuncAnimation(fig, update, frames=len(frames_ws), interval=200,
                         blit=False, repeat=True)
    writer = PillowWriter(fps=fps, metadata={"title": "SIRLO World State"})
    anim.save(str(out_path), writer=writer, dpi=90)
    plt.close(fig)
    print(f"  ✓ Animation GIF → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    rng_seed = 42
    rng      = np.random.default_rng(rng_seed)
    py_rng   = random.Random(rng_seed)

    regions = ["amazon_wildfire", "cerrado", "pantanal"]
    all_ep_stats = []

    print("\n" + "═"*60)
    print("  SIRLO — Synthetic Dataset Generator + Visualiser")
    print("  Generating & visualising without GEE — synthetic mode")
    print("═"*60 + "\n")

    animation_frames_ws  = []
    animation_frames_pos = []
    anim_episode         = 0   # use episode 0 for animation

    for ep in range(N_EPISODES):
        region = regions[ep % len(regions)]
        print(f"[Episode {ep+1}/{N_EPISODES}] Region: {region}")

        # Randomly place 3-5 persistent fire clusters
        n_fires  = py_rng.randint(3, 6)
        fire_locs = [(py_rng.random(), py_rng.random()) for _ in range(n_fires)]

        # Per-satellite mutable state
        battery = [py_rng.uniform(0.4, 0.95) for _ in range(N_SATS)]
        storage = [py_rng.uniform(0.0, 0.25) for _ in range(N_SATS)]

        history    = []
        all_pos    = []
        all_act    = []
        all_rew    = []

        stats = {"total_fleet_reward": 0, "fire_captures": 0, "cloud_penalties": 0,
                 "downlinks": 0, "battery_depletions": 0, "avg_storage_fill": 0,
                 "action_counts": [0] * N_ACTIONS}

        for step in range(STEPS_PER_EP):
            t_frac = step / STEPS_PER_EP
            ws     = gen_world_state(rng, t_frac, fire_locs)
            pos    = gen_satellite_positions(step)
            ss     = gen_sat_state(step, pos, ws[6], battery, storage, rng)
            acs, rews = compute_actions_rewards(ss, ws, rng)

            # Update persistent satellite state from actions
            for i in range(N_SATS):
                a = acs[i]
                discharge = {0: 0.002, 1: 0.015, 2: 0.005, 3: 0.005, 4: 0.020}[a]
                battery[i] = float(np.clip(battery[i] - discharge + py_rng.uniform(0.005, 0.012), 0, 1))
                if a == 1:
                    storage[i] = float(np.clip(storage[i] + py_rng.uniform(0.02, 0.07), 0, 1))
                elif a == 4 and ss[i, 6] > 0.5:
                    storage[i] = float(np.clip(storage[i] - py_rng.uniform(0.3, 0.6), 0, 1))

            # Accumulate stats
            stats["total_fleet_reward"] += float(rews.sum())
            for i in range(N_SATS):
                if acs[i] == 1 and ws[7, int(np.clip(pos[i,0]*H, 0, H-1)), int(np.clip(pos[i,1]*W, 0, W-1))] > 0.5:
                    stats["fire_captures"] += 1
                if rews[i] == -2.0:
                    stats["cloud_penalties"] += 1
                if acs[i] == 4 and rews[i] > 0:
                    stats["downlinks"] += 1
                if battery[i] < 0.02:
                    stats["battery_depletions"] += 1
                stats["action_counts"][acs[i]] += 1

            history.append((ws.copy(), ss.copy(), acs.copy(), rews.copy()))
            all_pos.append(pos.copy())
            all_act.append(acs.copy())
            all_rew.append(rews.copy())

            # Save animation frames for episode 0
            if ep == anim_episode:
                animation_frames_ws.append(ws.copy())
                animation_frames_pos.append(pos.copy())

            if step % 10 == 0:
                print(f"  step {step+1:3d}/{STEPS_PER_EP}  "
                      f"reward={rews.sum():+.1f}  "
                      f"fires_so_far={stats['fire_captures']}")

        stats["avg_storage_fill"] = float(np.mean(storage))
        all_ep_stats.append(stats)

        # ── PLOTS ─────────────────────────────────────────────────────────────
        print(f"  → Plotting episode {ep+1}…")
        all_pos_arr = np.array(all_pos)   # [T, N, 2]
        all_act_arr = np.stack(all_act, axis=0)   # [T, N]
        all_rew_arr = np.stack(all_rew, axis=0)   # [T, N]

        # World state snapshot (mid-episode)
        mid = STEPS_PER_EP // 2
        plot_world_state_dashboard(
            world_state=history[mid][0],   # ws saved as 1st element of history tuple
            sat_positions=all_pos[mid],
            step=mid,
            region=region,
            out_path=OUTPUT_DIR / f"ep{ep+1}_world_state.png"
        )

        # Episode timeline
        plot_episode_timeline(
            history=history,
            episode=ep,
            out_path=OUTPUT_DIR / f"ep{ep+1}_timeline.png"
        )

        # Ground tracks
        plot_ground_tracks(
            all_positions=all_pos_arr,
            all_actions=all_act_arr,
            all_rewards=all_rew_arr,
            out_path=OUTPUT_DIR / f"ep{ep+1}_ground_tracks.png"
        )

    # ── Fleet summary (all episodes) ──────────────────────────────────────────
    print("\n  → Plotting fleet summary…")
    plot_fleet_summary(all_ep_stats, out_path=OUTPUT_DIR / "fleet_summary.png")

    # ── Animation (episode 0) ─────────────────────────────────────────────────
    print("  → Generating world-state animation (this takes ~15s)…")
    # Use every 3rd frame for speed
    stride = 3
    make_animation(
        frames_ws=animation_frames_ws[::stride],
        frames_pos=animation_frames_pos[::stride],
        out_path=OUTPUT_DIR / "world_state_evolution.gif",
        fps=5,
    )

    # ── Save dataset stats JSON ───────────────────────────────────────────────
    with open(OUTPUT_DIR / "run_stats.json", "w") as fh:
        json.dump(all_ep_stats, fh, indent=2)

    print("\n" + "═"*60)
    print(f"  ✅ All outputs saved to: {OUTPUT_DIR.resolve()}")
    print("═"*60)
    for f in sorted(OUTPUT_DIR.iterdir()):
        sz = f.stat().st_size / 1024
        print(f"    {f.name:<40} {sz:>7.1f} KB")
    print()


if __name__ == "__main__":
    main()
