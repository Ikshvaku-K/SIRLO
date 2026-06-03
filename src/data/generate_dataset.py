"""
SIRLO — GEE Dataset Generator
==============================
Generates a richly annotated, multi-channel geospatial dataset from Google Earth Engine
for training Autonomous Multi-Agent Reinforcement Learning (MARL) agents to optimise
high-cadence environmental monitoring missions (wildfire detection, deforestation, etc.).

Dataset structure per timestep (episode step):
─────────────────────────────────────────────────────────────────────────────────
world_state  : np.ndarray  [C, H, W]  — multi-channel GEE image tensor
               C0: GOES-16 IR brightness temperature (cloud proxy)
               C1: FIRMS fire brightness temperature
               C2: MODIS NDVI (vegetation health)
               C3: MODIS LST (land surface temperature)
               C4: SMAP surface soil moisture
               C5: GPM precipitation rate
               C6: Cloud mask (derived binary)
               C7: Fire mask (derived binary)

sat_states   : np.ndarray  [N, F]     — per-satellite internal state (N agents, F features)
               F0: battery SOC [0,1]
               F1: storage fill fraction [0,1]
               F2: sub-sat lat (normalised to ROI)
               F3: sub-sat lon (normalised to ROI)
               F4: altitude (km, normalised)
               F5: footprint cloud fraction
               F6: ground-station contact flag
               F7: minutes to next GS contact (normalised)

actions      : np.ndarray  [N]        — discrete actions taken (0–4)
rewards      : np.ndarray  [N]        — per-agent reward signal
done         : bool                   — episode termination flag

metadata     : dict — timestamp, region, split, episode_id, step_id
─────────────────────────────────────────────────────────────────────────────────

Usage:
    python src/data/generate_dataset.py --config config/dataset_config.yaml
    python src/data/generate_dataset.py --region amazon_wildfire --samples 500
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ee
import numpy as np
import yaml

# ── optional heavy deps (fail gracefully) ──────────────────────────────────────
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.gee_auth import initialize_gee          # noqa: E402
from src.env.propagator import OrbitalPropagator       # noqa: E402
from src.env.satellite import Satellite                 # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("sirlo.dataset")


# ══════════════════════════════════════════════════════════════════════════════
# Configuration helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> Dict[str, Any]:
    """Load YAML dataset config."""
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def parse_date(s: str) -> datetime.datetime:
    return datetime.datetime.strptime(s, "%Y-%m-%d").replace(
        tzinfo=datetime.timezone.utc
    )


def sample_timestamps(cfg: Dict) -> List[datetime.datetime]:
    """
    Generate a deterministic list of UTC timestamps within the configured
    date range at the given cadence.
    """
    rng = random.Random(cfg.get("seed", 42))
    start = parse_date(cfg["temporal"]["start_date"])
    end   = parse_date(cfg["temporal"]["end_date"])
    step  = datetime.timedelta(minutes=cfg["temporal"]["cadence_minutes"])
    max_n = cfg["temporal"].get("max_samples", 5000)

    all_ts: List[datetime.datetime] = []
    t = start
    while t <= end:
        all_ts.append(t)
        t += step

    rng.shuffle(all_ts)
    return all_ts[:max_n]


# ══════════════════════════════════════════════════════════════════════════════
# GEE image fetchers  (one per data source)
# ══════════════════════════════════════════════════════════════════════════════

class GEEFetcher:
    """
    Wraps all Google Earth Engine data fetching with retries,
    normalisation, and graceful fallback on API failures.
    """

    MIN_RETRY_WAIT_S = 3
    MAX_RETRIES      = 4

    def __init__(self, cfg: Dict):
        self.cfg = cfg["data_sources"]
        self.spatial = cfg["spatial"]

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _roi_from_bbox(bbox: List[float]) -> ee.Geometry:
        """bbox = [lon_min, lat_min, lon_max, lat_max]"""
        return ee.Geometry.Rectangle(bbox)

    @staticmethod
    def _time_window(ts: datetime.datetime, minutes: int) -> Tuple[str, str]:
        half = datetime.timedelta(minutes=minutes)
        fmt  = "%Y-%m-%dT%H:%M:%S"
        return (ts - half).strftime(fmt), (ts + half).strftime(fmt)

    @staticmethod
    def _day_window(ts: datetime.datetime, days: int) -> Tuple[str, str]:
        half = datetime.timedelta(days=days)
        fmt  = "%Y-%m-%d"
        return (ts - half).strftime(fmt), (ts + half).strftime(fmt)

    def _safe_sample(
        self,
        image: ee.Image,
        roi: ee.Geometry,
        band_name: str,
        grid_size: Tuple[int, int],
        resolution_m: int,
    ) -> np.ndarray:
        """
        Export a single-band image to a numpy array with retries.
        Falls back to zeros on repeated failure.
        """
        H, W = grid_size
        for attempt in range(self.MAX_RETRIES):
            try:
                url = image.getDownloadURL(
                    {
                        "region": roi,
                        "dimensions": f"{W}x{H}",
                        "format": "NPY",
                        "bands": [band_name],
                    }
                )
                import urllib.request, io
                with urllib.request.urlopen(url) as resp:
                    data = np.load(io.BytesIO(resp.read()))
                # getDownloadURL NPY returns structured array; extract first field
                if data.dtype.names:
                    data = data[data.dtype.names[0]]
                return data.astype(np.float32).reshape(H, W)
            except Exception as exc:
                wait = self.MIN_RETRY_WAIT_S * (2 ** attempt) + random.random()
                log.warning(
                    "GEE fetch attempt %d/%d failed for %s: %s — retrying in %.1fs",
                    attempt + 1,
                    self.MAX_RETRIES,
                    band_name,
                    exc,
                    wait,
                )
                time.sleep(wait)
        log.error("All retries exhausted for %s — using zeros.", band_name)
        return np.zeros((H, W), dtype=np.float32)

    # ── normalisation ────────────────────────────────────────────────────────

    @staticmethod
    def _minmax(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
        denom = hi - lo
        return np.clip((arr - lo) / denom if denom != 0 else arr * 0, 0.0, 1.0)

    @staticmethod
    def _percentile(arr: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
        lo, hi = np.percentile(arr, [p_lo, p_hi])
        denom = hi - lo
        return np.clip((arr - lo) / denom if denom != 0 else arr * 0, 0.0, 1.0)

    # ── per-source fetchers ──────────────────────────────────────────────────

    def fetch_goes16(
        self,
        ts: datetime.datetime,
        roi: ee.Geometry,
        grid_size: Tuple[int, int],
        resolution_m: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (cloud_channel [H,W], cloud_mask [H,W]).
        cloud_channel is normalised brightness temperature.
        cloud_mask is binary (1 = cloud-obscured, 0 = clear).
        """
        src = self.cfg["goes16_cloud"]
        t0, t1 = self._time_window(ts, src["temporal_window_min"])
        collection = (
            ee.ImageCollection(src["collection"])
            .filterDate(t0, t1)
            .filterBounds(roi)
            .select(src["band"])
        )
        image = collection.mean()  # temporal average within window
        raw = self._safe_sample(image, roi, src["band"], grid_size, resolution_m)

        norm_cfg = src["normalise"]
        channel = self._minmax(raw, norm_cfg["clip_min"], norm_cfg["clip_max"])

        # Cloud mask: T < 273 K  ↔  normalised value < threshold
        threshold = (273.0 - norm_cfg["clip_min"]) / (
            norm_cfg["clip_max"] - norm_cfg["clip_min"]
        )
        cloud_mask = (channel < threshold).astype(np.float32)
        return channel, cloud_mask

    def fetch_firms(
        self,
        ts: datetime.datetime,
        roi: ee.Geometry,
        grid_size: Tuple[int, int],
        resolution_m: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (fire_channel [H,W], fire_mask [H,W]).
        """
        src = self.cfg["firms"]
        t0, t1 = self._time_window(ts, src["temporal_window_min"])
        collection = (
            ee.ImageCollection(src["collection"])
            .filterDate(t0, t1)
            .filterBounds(roi)
        )
        image = collection.mosaic().select(src["band"])
        raw = self._safe_sample(image, roi, src["band"], grid_size, resolution_m)

        norm_cfg = src["normalise"]
        channel = self._minmax(raw, norm_cfg["clip_min"], norm_cfg["clip_max"])

        # Fire mask: anomalously high brightness temperature
        threshold = (380.0 - norm_cfg["clip_min"]) / (
            norm_cfg["clip_max"] - norm_cfg["clip_min"]
        )
        fire_mask = (channel > threshold).astype(np.float32)
        return channel, fire_mask

    def fetch_modis_ndvi(
        self,
        ts: datetime.datetime,
        roi: ee.Geometry,
        grid_size: Tuple[int, int],
        resolution_m: int,
    ) -> np.ndarray:
        src = self.cfg["modis_ndvi"]
        t0, t1 = self._day_window(ts, src["temporal_window_days"])
        collection = (
            ee.ImageCollection(src["collection"])
            .filterDate(t0, t1)
            .filterBounds(roi)
            .select(src["band"])
        )
        image = collection.mean()
        raw = self._safe_sample(image, roi, src["band"], grid_size, resolution_m)
        raw = raw * src.get("scale_factor", 1.0)
        return self._minmax(raw, 0.0, 1.0).astype(np.float32)

    def fetch_modis_lst(
        self,
        ts: datetime.datetime,
        roi: ee.Geometry,
        grid_size: Tuple[int, int],
        resolution_m: int,
    ) -> np.ndarray:
        src = self.cfg["modis_lst"]
        t0, t1 = self._day_window(ts, src["temporal_window_days"])
        collection = (
            ee.ImageCollection(src["collection"])
            .filterDate(t0, t1)
            .filterBounds(roi)
            .select(src["band"])
        )
        image = collection.mean()
        raw = self._safe_sample(image, roi, src["band"], grid_size, resolution_m)
        raw = raw * src.get("scale_factor", 0.02)          # DN → Kelvin
        return self._minmax(raw, 273.0, 350.0).astype(np.float32)

    def fetch_smap(
        self,
        ts: datetime.datetime,
        roi: ee.Geometry,
        grid_size: Tuple[int, int],
        resolution_m: int,
    ) -> np.ndarray:
        src = self.cfg["smap_soil_moisture"]
        t0, t1 = self._day_window(ts, src["temporal_window_days"])
        collection = (
            ee.ImageCollection(src["collection"])
            .filterDate(t0, t1)
            .filterBounds(roi)
            .select(src["band"])
        )
        image = collection.mean()
        raw = self._safe_sample(image, roi, src["band"], grid_size, resolution_m)
        norm_cfg = src["normalise"]
        return self._minmax(raw, norm_cfg["clip_min"], norm_cfg["clip_max"]).astype(np.float32)

    def fetch_gpm(
        self,
        ts: datetime.datetime,
        roi: ee.Geometry,
        grid_size: Tuple[int, int],
        resolution_m: int,
    ) -> np.ndarray:
        src = self.cfg["gpm_precipitation"]
        t0, t1 = self._time_window(ts, src["temporal_window_min"])
        collection = (
            ee.ImageCollection(src["collection"])
            .filterDate(t0, t1)
            .filterBounds(roi)
            .select(src["band"])
        )
        image = collection.mean()
        raw = self._safe_sample(image, roi, src["band"], grid_size, resolution_m)
        norm_cfg = src["normalise"]
        return self._minmax(raw, norm_cfg["clip_min"], norm_cfg["clip_max"]).astype(np.float32)

    # ── full world state ─────────────────────────────────────────────────────

    def build_world_state(
        self,
        ts: datetime.datetime,
        region_cfg: Dict,
    ) -> np.ndarray:
        """
        Assemble the full [C, H, W] world-state tensor for a given timestamp.

        Channels:
          0  GOES-16 IR brightness temp (cloud proxy)
          1  FIRMS fire brightness temp
          2  MODIS NDVI
          3  MODIS LST
          4  SMAP soil moisture
          5  GPM precipitation
          6  Cloud mask (binary)
          7  Fire mask  (binary)
        """
        roi         = self._roi_from_bbox(region_cfg["bbox"])
        grid_size   = tuple(region_cfg["grid_size"])
        resolution  = region_cfg["resolution_m"]

        log.debug("Fetching GOES-16 …")
        cloud_ch, cloud_mask = self.fetch_goes16(ts, roi, grid_size, resolution)

        log.debug("Fetching FIRMS …")
        fire_ch,  fire_mask  = self.fetch_firms(ts, roi, grid_size, resolution)

        log.debug("Fetching MODIS NDVI …")
        ndvi = self.fetch_modis_ndvi(ts, roi, grid_size, resolution)

        log.debug("Fetching MODIS LST …")
        lst  = self.fetch_modis_lst(ts, roi, grid_size, resolution)

        log.debug("Fetching SMAP …")
        ssm  = self.fetch_smap(ts, roi, grid_size, resolution)

        log.debug("Fetching GPM …")
        precip = self.fetch_gpm(ts, roi, grid_size, resolution)

        state = np.stack(
            [cloud_ch, fire_ch, ndvi, lst, ssm, precip, cloud_mask, fire_mask],
            axis=0,
        ).astype(np.float32)  # [8, H, W]
        return state


# ══════════════════════════════════════════════════════════════════════════════
# Satellite State Simulator
# ══════════════════════════════════════════════════════════════════════════════

class SatelliteStateSimulator:
    """
    Generates synthetic but physically plausible per-satellite internal states,
    integrated with the GEE world-state for footprint cloud coverage.
    """

    def __init__(self, cfg: Dict, region_cfg: Dict, rng: random.Random):
        self.n_sats    = cfg["satellite_state"]["n_satellites"]
        self.region    = region_cfg["bbox"]           # [lon_min, lat_min, lon_max, lat_max]
        self.rng       = rng

        # Initialise satellite instances (wraps OrbitalPropagator if available)
        try:
            self.propagator = OrbitalPropagator()
            self.use_propagator = True
        except Exception:
            self.use_propagator = False
            log.warning("OrbitalPropagator unavailable — using stochastic satellite positions.")

        # Per-satellite persistent state
        self._battery   = [self.rng.uniform(0.3, 1.0) for _ in range(self.n_sats)]
        self._storage   = [self.rng.uniform(0.0, 0.3) for _ in range(self.n_sats)]
        self._altitude  = [self.rng.uniform(450.0, 600.0) for _ in range(self.n_sats)]

    def _step_battery(self, i: int, action: int) -> float:
        """Simple discharge model: imaging/downlink costs more energy."""
        discharge_map = {0: 0.001, 1: 0.015, 2: 0.005, 3: 0.005, 4: 0.020}
        self._battery[i] = max(0.0, self._battery[i] - discharge_map.get(action, 0.005))
        # Solar panel recharge (stochastic clear-sky model)
        self._battery[i] = min(1.0, self._battery[i] + self.rng.uniform(0.005, 0.015))
        return self._battery[i]

    def _step_storage(self, i: int, action: int) -> float:
        if action == 1:   # IMAGE → adds data
            self._storage[i] = min(1.0, self._storage[i] + self.rng.uniform(0.02, 0.08))
        elif action == 4: # DOWNLINK → empties buffer
            self._storage[i] = max(0.0, self._storage[i] - self.rng.uniform(0.3, 0.7))
        return self._storage[i]

    def _sat_position(self, sat_idx: int, ts: datetime.datetime) -> Tuple[float, float]:
        """
        Returns (lat_norm, lon_norm) ∈ [0,1] within the ROI bounding box.
        Uses propagator if available, else synthetic sinusoidal orbit.
        """
        lon_min, lat_min, lon_max, lat_max = self.region
        if self.use_propagator:
            try:
                lat, lon = self.propagator.get_subpoint(sat_idx, ts)
                lat_n = np.clip((lat - lat_min) / (lat_max - lat_min), 0, 1)
                lon_n = np.clip((lon - lon_min) / (lon_max - lon_min), 0, 1)
                return float(lat_n), float(lon_n)
            except Exception:
                pass
        # Fallback: synthetic circular orbit
        epoch = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
        t_sec = (ts - epoch).total_seconds()
        period_s = 5400.0 + sat_idx * 900.0          # ~90–105 min orbits
        phase    = (2 * np.pi * t_sec / period_s) + (sat_idx * 2 * np.pi / self.n_sats)
        lat_n = (np.sin(phase) + 1) / 2
        lon_n = ((np.cos(phase) * 0.7 + sat_idx / self.n_sats) % 1.0)
        return float(lat_n), float(lon_n)

    def get_footprint_cloud_fraction(
        self,
        lat_n: float,
        lon_n: float,
        cloud_mask: np.ndarray,   # [H, W]  binary
        patch_size_frac: float = 0.15,
    ) -> float:
        """
        Sample the cloud mask within the satellite's footprint and return
        the fraction of cloudy pixels.
        """
        H, W   = cloud_mask.shape
        row    = int(np.clip(lat_n * H, 0, H - 1))
        col    = int(np.clip(lon_n * W, 0, W - 1))
        half_p = max(1, int(patch_size_frac * min(H, W) / 2))

        r0, r1 = max(0, row - half_p), min(H, row + half_p)
        c0, c1 = max(0, col - half_p), min(W, col + half_p)
        patch  = cloud_mask[r0:r1, c0:c1]
        return float(patch.mean()) if patch.size > 0 else 0.0

    def get_gs_contact(self, ts: datetime.datetime, sat_idx: int) -> Tuple[int, float]:
        """
        Estimate ground-station contact.
        Ground stations assumed at: Kourou (FG), Inuvik (CA), Svalbard (NO).
        Returns (contact_flag, minutes_to_next_contact).
        """
        # Simplified: GS contact every ~100 min orbit, window ~10 min
        period_s    = 5400.0 + sat_idx * 900.0
        epoch       = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
        t_sec       = (ts - epoch).total_seconds() + sat_idx * 300.0
        phase       = t_sec % period_s
        window_s    = 600.0   # 10-min contact window per pass
        contact_flag = 1 if phase < window_s else 0
        next_contact = (period_s - phase) / 60.0 if not contact_flag else 0.0
        return contact_flag, min(next_contact, 120.0)

    def step(
        self,
        ts: datetime.datetime,
        cloud_mask: np.ndarray,
        actions: List[int],
    ) -> np.ndarray:
        """
        Step all satellite states forward. Returns [N, 8] state array.
        """
        states = []
        for i, action in enumerate(actions):
            batt  = self._step_battery(i, action)
            stor  = self._step_storage(i, action)
            lat_n, lon_n = self._sat_position(i, ts)
            cloud_frac   = self.get_footprint_cloud_fraction(lat_n, lon_n, cloud_mask)
            gs_flag, ttc = self.get_gs_contact(ts, i)
            alt_norm     = (self._altitude[i] - 450.0) / 150.0

            states.append([
                batt,
                stor,
                lat_n,
                lon_n,
                alt_norm,
                cloud_frac,
                float(gs_flag),
                ttc / 120.0,          # normalise to [0,1]
            ])
        return np.array(states, dtype=np.float32)  # [N, 8]


# ══════════════════════════════════════════════════════════════════════════════
# Reward Calculator
# ══════════════════════════════════════════════════════════════════════════════

class RewardCalculator:
    """
    Computes the reward vector for all agents based on their actions and the
    current world state.  Mirrors the reward shaping defined in config.
    """

    def __init__(self, cfg: Dict):
        r = cfg["reward"]
        self.fire_capture_bonus  = r["fire_capture_bonus"]
        self.clear_image_bonus   = r["clear_image_bonus"]
        self.cloud_waste_penalty = r["cloud_waste_penalty"]
        self.batt_depleted_pen   = r["battery_depleted_penalty"]
        self.buf_overflow_pen    = r["buffer_overflow_penalty"]
        self.downlink_bonus      = r["downlink_success_bonus"]

    def compute(
        self,
        actions: List[int],
        sat_states_before: np.ndarray,   # [N, 8] state BEFORE step
        sat_states_after:  np.ndarray,   # [N, 8] state AFTER step
        world_state: np.ndarray,          # [C, H, W]
    ) -> np.ndarray:
        """Returns reward vector [N]."""
        N       = len(actions)
        rewards = np.zeros(N, dtype=np.float32)
        H, W    = world_state.shape[1], world_state.shape[2]

        cloud_mask = world_state[6]   # channel 6
        fire_mask  = world_state[7]   # channel 7

        for i, action in enumerate(actions):
            lat_n   = sat_states_after[i, 2]
            lon_n   = sat_states_after[i, 3]
            row     = int(np.clip(lat_n * H, 0, H - 1))
            col     = int(np.clip(lon_n * W, 0, W - 1))

            if action == 1:  # IMAGE_TARGET
                is_cloudy = cloud_mask[row, col] > 0.5
                is_fire   = fire_mask[row, col]  > 0.5
                if is_cloudy:
                    rewards[i] += self.cloud_waste_penalty
                elif is_fire:
                    rewards[i] += self.fire_capture_bonus
                else:
                    rewards[i] += self.clear_image_bonus

            elif action == 4:  # DOWNLINK
                gs_flag = sat_states_after[i, 6]
                if gs_flag > 0.5:
                    rewards[i] += self.downlink_bonus

            # Penalty checks
            if sat_states_after[i, 0] <= 0.01:   # battery depleted
                rewards[i] += self.batt_depleted_pen
            if sat_states_after[i, 1] >= 0.99:   # buffer overflow
                rewards[i] += self.buf_overflow_pen

        return rewards


# ══════════════════════════════════════════════════════════════════════════════
# Dataset Writer
# ══════════════════════════════════════════════════════════════════════════════

class DatasetWriter:
    """
    Persists episode steps to disk in NPZ and/or HDF5 format.
    Also accumulates a flat CSV of scalar features.
    """

    CHANNEL_NAMES = [
        "goes16_bt",     # GOES-16 brightness temp (cloud proxy)
        "firms_bt",      # FIRMS brightness temp (fire)
        "modis_ndvi",    # NDVI
        "modis_lst",     # Land surface temperature
        "smap_ssm",      # Soil moisture
        "gpm_precip",    # Precipitation
        "cloud_mask",    # Derived binary cloud mask
        "fire_mask",     # Derived binary fire mask
    ]

    SAT_FEATURE_NAMES = [
        "battery_soc",
        "storage_fill",
        "lat_norm",
        "lon_norm",
        "altitude_norm",
        "footprint_cloud_fraction",
        "gs_contact_flag",
        "time_to_next_contact_norm",
    ]

    def __init__(self, cfg: Dict, output_dir: Path):
        out_cfg        = cfg["output"]
        self.out_dir   = output_dir
        self.npz_dir   = output_dir / out_cfg["episode_dir"]
        self.npz_dir.mkdir(parents=True, exist_ok=True)
        self.compress  = out_cfg.get("compress", True)
        self.hdf5_cfg  = out_cfg.get("hdf5", {})
        self.csv_cfg   = out_cfg.get("csv_export", {})

        self._records: List[Dict] = []   # flat scalar records for CSV

        # Optional HDF5 writer
        self._h5: Optional[Any] = None
        self._h5_step: int = 0
        if HDF5_AVAILABLE and self.hdf5_cfg.get("enabled", False):
            h5_path = output_dir / self.hdf5_cfg["filename"]
            self._h5 = h5py.File(h5_path, "w")
            log.info("HDF5 writer opened: %s", h5_path)

    def write_step(
        self,
        episode_id: str,
        step_id: int,
        ts: datetime.datetime,
        region_name: str,
        split: str,
        world_state: np.ndarray,    # [C, H, W]
        sat_states: np.ndarray,     # [N, F]
        actions: List[int],
        rewards: np.ndarray,        # [N]
        done: bool,
    ):
        meta = {
            "episode_id":  episode_id,
            "step_id":     step_id,
            "timestamp":   ts.isoformat(),
            "region":      region_name,
            "split":       split,
            "done":        int(done),
        }
        actions_arr = np.array(actions, dtype=np.int32)

        # ── NPZ ──────────────────────────────────────────────────────────────
        npz_path = self.npz_dir / f"{episode_id}_step{step_id:05d}.npz"
        save_fn  = np.savez_compressed if self.compress else np.savez
        save_fn(
            str(npz_path),
            world_state=world_state,
            sat_states=sat_states,
            actions=actions_arr,
            rewards=rewards,
            **{f"meta_{k}": np.array(str(v)) for k, v in meta.items()},
        )

        # ── HDF5 ─────────────────────────────────────────────────────────────
        if self._h5 is not None:
            grp = self._h5.require_group(f"episode_{episode_id}")
            sgrp = grp.require_group(f"step_{step_id:05d}")
            sgrp.create_dataset("world_state", data=world_state, compression="gzip")
            sgrp.create_dataset("sat_states",  data=sat_states,  compression="gzip")
            sgrp.create_dataset("actions",     data=actions_arr)
            sgrp.create_dataset("rewards",     data=rewards)
            for k, v in meta.items():
                sgrp.attrs[k] = str(v)

        # ── CSV record ───────────────────────────────────────────────────────
        record: Dict = {**meta}
        # aggregate world-state stats per channel
        for c_idx, c_name in enumerate(self.CHANNEL_NAMES):
            ch = world_state[c_idx]
            record[f"{c_name}_mean"]  = float(ch.mean())
            record[f"{c_name}_max"]   = float(ch.max())
            record[f"{c_name}_cover"] = float((ch > 0.5).mean())   # fraction above threshold
        # per-sat scalar features
        N = sat_states.shape[0]
        for sat_i in range(N):
            for f_idx, f_name in enumerate(self.SAT_FEATURE_NAMES):
                record[f"sat{sat_i}_{f_name}"] = float(sat_states[sat_i, f_idx])
            record[f"sat{sat_i}_action"] = int(actions[sat_i])
            record[f"sat{sat_i}_reward"] = float(rewards[sat_i])
        self._records.append(record)

    def write_metadata(self, meta: Dict, cfg: Dict):
        path = self.out_dir / "metadata.json"
        with open(path, "w") as fh:
            json.dump(
                {"generation_config": cfg, "run_metadata": meta},
                fh,
                indent=2,
                default=str,
            )
        log.info("Metadata written → %s", path)

    def flush(self):
        """Write out the CSV and close HDF5."""
        if PANDAS_AVAILABLE and self.csv_cfg.get("enabled", False) and self._records:
            df = pd.DataFrame(self._records)
            csv_path = self.out_dir / self.csv_cfg["filename"]
            df.to_csv(csv_path, index=False)
            log.info("CSV written: %s  (%d rows)", csv_path, len(df))

        if self._h5 is not None:
            self._h5.flush()
            self._h5.close()
            log.info("HDF5 closed.")


# ══════════════════════════════════════════════════════════════════════════════
# Episode / Policy Sampler
# ══════════════════════════════════════════════════════════════════════════════

class RandomPolicy:
    """Uniform random action sampler — captures the full action space distribution."""

    def __init__(self, n_actions: int, n_sats: int, seed: int = 42):
        self.n_actions = n_actions
        self.n_sats    = n_sats
        self.rng       = random.Random(seed)

    def act(self, _world_state: np.ndarray, _sat_states: np.ndarray) -> List[int]:
        return [self.rng.randint(0, self.n_actions - 1) for _ in range(self.n_sats)]


# ══════════════════════════════════════════════════════════════════════════════
# Main Dataset Generator
# ══════════════════════════════════════════════════════════════════════════════

class DatasetGenerator:

    STEPS_PER_EPISODE = 48          # each episode = 48 × 30-min steps = 24 h

    def __init__(self, cfg: Dict, output_dir: Path):
        self.cfg        = cfg
        self.output_dir = output_dir
        self.rng        = random.Random(cfg.get("seed", 42))
        self.fetcher    = GEEFetcher(cfg)
        self.writer     = DatasetWriter(cfg, output_dir)
        self.policy     = RandomPolicy(
            n_actions=cfg["actions"]["n_actions"],
            n_sats=cfg["satellite_state"]["n_satellites"],
            seed=cfg.get("seed", 42),
        )
        self.reward_calc = RewardCalculator(cfg)

        # Dataset splits
        split_cfg = cfg["splits"]
        self._split_thresholds = {
            "train":      split_cfg["train"],
            "validation": split_cfg["train"] + split_cfg["validation"],
        }

    def _assign_split(self, idx: int, total: int) -> str:
        frac = idx / max(total, 1)
        if frac < self._split_thresholds["train"]:
            return "train"
        elif frac < self._split_thresholds["validation"]:
            return "validation"
        return "test"

    def run(self):
        """Main generation loop."""
        timestamps = sample_timestamps(self.cfg)
        n_total    = len(timestamps)
        log.info("Generating dataset: %d timestep samples across all regions.", n_total)

        regions    = self.cfg["spatial"]["regions"]
        region_names = list(regions.keys())

        global_step  = 0
        episode_step = 0
        episode_id   = str(uuid.uuid4())[:8]

        curr_region_name = self._pick_region(region_names)
        region_cfg       = regions[curr_region_name]
        sat_sim          = self._new_sat_sim(region_cfg)

        for idx, ts in enumerate(timestamps):
            split = self._assign_split(idx, n_total)

            # ── Fetch world state ────────────────────────────────────────────
            log.info(
                "[%d/%d] %s  region=%s  episode=%s  step=%d",
                idx + 1, n_total, ts.strftime("%Y-%m-%d %H:%M"), curr_region_name,
                episode_id, episode_step,
            )
            try:
                world_state = self.fetcher.build_world_state(ts, region_cfg)
            except Exception as exc:
                log.error("World state generation failed at step %d: %s — skipping.", idx, exc)
                global_step += 1
                continue

            cloud_mask = world_state[6]  # binary cloud mask channel

            # ── Satellite state & actions ────────────────────────────────────
            dummy_sat_state = sat_sim.step(ts, cloud_mask, [0] * self.cfg["satellite_state"]["n_satellites"])
            actions         = self.policy.act(world_state, dummy_sat_state)
            sat_states_after = sat_sim.step(ts, cloud_mask, actions)
            rewards          = self.reward_calc.compute(
                actions, dummy_sat_state, sat_states_after, world_state
            )

            # ── Episode boundary ─────────────────────────────────────────────
            done = (episode_step >= self.STEPS_PER_EPISODE - 1)
            self.writer.write_step(
                episode_id=episode_id,
                step_id=episode_step,
                ts=ts,
                region_name=curr_region_name,
                split=split,
                world_state=world_state,
                sat_states=sat_states_after,
                actions=actions,
                rewards=rewards,
                done=done,
            )

            global_step  += 1
            episode_step += 1

            if done:
                log.info("Episode %s complete (%d steps).", episode_id, episode_step)
                episode_id   = str(uuid.uuid4())[:8]
                episode_step = 0
                curr_region_name = self._pick_region(region_names)
                region_cfg       = regions[curr_region_name]
                sat_sim          = self._new_sat_sim(region_cfg)

        # ── Finalise ─────────────────────────────────────────────────────────
        run_meta = {
            "total_steps":    global_step,
            "regions":        region_names,
            "generated_at":   datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "seed":           self.cfg.get("seed", 42),
            "world_state_channels": DatasetWriter.CHANNEL_NAMES,
            "sat_feature_names":    DatasetWriter.SAT_FEATURE_NAMES,
            "actions_map":          self.cfg["actions"]["action_map"],
        }
        self.writer.write_metadata(run_meta, self.cfg)
        self.writer.flush()
        log.info("Dataset generation complete. Output: %s", self.output_dir)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _pick_region(self, region_names: List[str]) -> str:
        return self.rng.choice(region_names)

    def _new_sat_sim(self, region_cfg: Dict) -> SatelliteStateSimulator:
        return SatelliteStateSimulator(self.cfg, region_cfg, random.Random(self.rng.random()))


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SIRLO GEE Dataset Generator for MARL training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/dataset_config.yaml",
        help="Path to dataset YAML config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output directory (default: from config).",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Restrict to a single named region (e.g. amazon_wildfire).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override max_samples from config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and GEE auth without running generation.",
    )
    parser.add_argument(
        "--gee-project",
        type=str,
        default=None,
        help="GEE Cloud Project ID (overrides DEFAULT_GEE_PROJECT env var).",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        log.error("Config file not found: %s", cfg_path)
        sys.exit(1)
    cfg = load_config(str(cfg_path))

    if args.samples:
        cfg["temporal"]["max_samples"] = args.samples

    if args.region:
        available = list(cfg["spatial"]["regions"].keys())
        if args.region not in available:
            log.error("Unknown region '%s'. Available: %s", args.region, available)
            sys.exit(1)
        cfg["spatial"]["regions"] = {args.region: cfg["spatial"]["regions"][args.region]}

    output_dir = Path(args.output) if args.output else Path(cfg["output"]["base_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Authenticate GEE ─────────────────────────────────────────────────────
    project_id = (
        args.gee_project
        or os.environ.get("DEFAULT_GEE_PROJECT")
        or os.environ.get("EARTHENGINE_PROJECT")
    )
    try:
        initialize_gee(project=project_id)
        log.info("GEE initialised (project=%s).", project_id or "<default>")
    except Exception as exc:
        log.error("GEE authentication failed: %s", exc)
        log.error("Run: earthengine authenticate")
        sys.exit(1)

    if args.dry_run:
        log.info("Dry-run complete — config and auth look good.")
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    generator = DatasetGenerator(cfg, output_dir)
    generator.run()


if __name__ == "__main__":
    main()
