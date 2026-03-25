#!/usr/bin/env python3
from __future__ import annotations

import time, uuid, json, threading, queue
from pathlib import Path
from datetime import datetime, timezone
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, Dict, Any, Tuple
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # headless backend for server
import matplotlib.pyplot as plt
import contextlib
import dash
from dash import html, dcc, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
# ---- Large-file download registry / logging ----
from flask import Response, request, stream_with_context, send_file
import csv
import secrets
import  os
import base64, io
import pandas as pd
from dash import dash_table
from astropy.io import fits
import urllib.request
import urllib.error


try:
    from dash import ctx
except Exception:
    # older Dash
    from dash import callback_context as ctx

# =========================
# Config
# =========================
BASE_SIZE = 1024
TAU_PERCENTILE = 1e-5
SOFTPLUS_BETA = 2.0

EPS = 1e-2

REPO_ROOT = Path(__file__).resolve().parent.parent
base_dir = REPO_ROOT

print("base_dir:",base_dir)

meta_dir = Path(base_dir) / "data/processed_libraries"
model_dir = Path(base_dir) / "models"
OUTPUT_DIR = Path(base_dir) / "lookup_out"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Model ↔ files mapping ---
MODEL_CONFIG = {
    "PHOENIX-husser": {
        "label": "PHOENIX–Husser",
        "npz": "phoenix_husser_all.npz",
        "model": "lookup_model_best_husser.pt",
        "meta": "phoenix_husser_all.meta.npz",
    },
    "PHOENIX-allard": {
        "label": "PHOENIX–Allard (non-optimal)",
        "npz": "phoenix_allard_all.npz",
        "model": "lookup_model_best_allard.pt",
        "meta": "phoenix_allard_all.meta.npz",
    },
    "C3K-conroy": {
        "label": "C3K-Conroy",
        "npz": "C3K_all.npz",
        "model": "lookup_model_best_C3K.pt",
        "meta": "C3K_all.meta.npz",
    },
    "TMAP-Werner": {
        "label": "TMAP-Werner",
        "npz": "TMAP_all.npz",
        "model": "lookup_model_best_TMAP.pt",
        "meta": "TMAP_all.meta.npz",
    },
    "OB-PoWR": {
        "label": "OB-PoWR",
        "npz": "OB_PoWR_all.npz",
        "model": "lookup_model_best_OB_PoWR.pt",
        "meta": "OB_PoWR_all.meta.npz",
    },
    "TMAP-C3K-husser-combined": {
        "label": "Husser + C3K + TMAP (combined grid)",
        "npz": "phoenix_husser_c3k_tmap_all.npz",
        "model": "lookup_model_TMAP_C3K_husser_combined.pt",
        "meta": "phoenix_husser_c3k_tmap_all.meta.npz",
    },
    "TMAP-C3K-husser-OB-combined": {
        "label": "Husser + C3K + TMAP + OB (combined grid)",
        "npz": "phoenix_husser_c3k_tmap_OB_all.npz",
        "model": "lookup_model_TMAP_C3K_husser_OB_combined.pt",
        "meta": "phoenix_husser_c3k_tmap_OB_all.meta.npz",
    },
}

DEFAULT_MODEL_KEY = "TMAP-C3K-husser-OB-combined"


# =========================
# Weights download (direct .pt)
# =========================

ZENODO_RECORD_ID = "18883385"
ZENODO_DOI = "10.5281/zenodo.18883385"

def zenodo_file_url(filename: str) -> str:
    return f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{filename}?download=1"

# All weight files you uploaded (filenames on Zenodo)
# Map keys in MODEL_CONFIG -> filename
MODEL_WEIGHTS_FILES: dict[str, str] = {
    "PHOENIX-husser": "lookup_model_best_husser.pt",
    "PHOENIX-allard": "lookup_model_best_allard.pt",
    "C3K-conroy": "lookup_model_best_C3K.pt",
    "TMAP-Werner": "lookup_model_best_TMAP.pt",
    "OB-PoWR": "lookup_model_best_OB_PoWR.pt",
    "TMAP-C3K-husser-combined": "lookup_model_TMAP_C3K_husser_combined.pt",
    "TMAP-C3K-husser-OB-combined": "lookup_model_TMAP_C3K_husser_OB_combined.pt",  # default
    # Note: you also listed lookup_model_C3K_husser_combined.pt, but it's not referenced by MODEL_CONFIG currently.
}

def weights_url_for(model_key: str) -> str | None:
    fname = MODEL_WEIGHTS_FILES.get(model_key)
    return zenodo_file_url(fname) if fname else None

def _expected_weight_path(model_key: str) -> Path:
    cfg = MODEL_CONFIG.get(model_key, MODEL_CONFIG[DEFAULT_MODEL_KEY])
    return model_dir / cfg["model"]

def _weights_present(model_key: str) -> bool:
    return _expected_weight_path(model_key).is_file()

class WeightsDownloadManager:
    """
    Downloads a single model weight file (.pt) with progress reporting.
    Writes to a temp file then atomically renames to the final path.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.active = False
        self.model_key: str | None = None
        self.status = "idle"     # idle|downloading|done|failed
        self.msg = ""
        self.progress = 0.0      # 0..100
        self.bytes_done = 0
        self.bytes_total = 0
        self.last_error = ""
        self.thread: threading.Thread | None = None

    def start(self, model_key: str, url: str) -> bool:
        with self.lock:
            if self.active:
                return False
            self.active = True
            self.model_key = model_key
            self.status = "downloading"
            label = MODEL_CONFIG.get(model_key, {}).get("label", model_key)
            self.msg = f"Downloading weights for {label}…"
            self.progress = 0.0
            self.bytes_done = 0
            self.bytes_total = 0
            self.last_error = ""

        def _run():
            tmp_path = None
            try:
                model_dir.mkdir(parents=True, exist_ok=True)
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                dest = _expected_weight_path(model_key)
                tmp_path = dest.with_suffix(dest.suffix + f".part_{uuid.uuid4().hex[:8]}")

                print(f"[WEIGHTS] Downloading '{model_key}' → {dest}", flush=True)
                print(f"[WEIGHTS] URL: {url}", flush=True)

                req = urllib.request.Request(url, headers={"User-Agent": "SM-Net/1.0"})
                with urllib.request.urlopen(req) as resp:
                    total = resp.headers.get("Content-Length")
                    total = int(total) if total and total.isdigit() else 0
                    with self.lock:
                        self.bytes_total = total

                    chunk = 1024 * 1024  # 1 MiB
                    done = 0
                    t_last = time.time()

                    with open(tmp_path, "wb") as f:
                        while True:
                            b = resp.read(chunk)
                            if not b:
                                break
                            f.write(b)
                            done += len(b)

                            now = time.time()
                            if now - t_last >= 0.2:
                                with self.lock:
                                    self.bytes_done = done
                                    if self.bytes_total > 0:
                                        self.progress = 100.0 * done / self.bytes_total
                                t_last = now

                    with self.lock:
                        self.bytes_done = done
                        if self.bytes_total > 0:
                            self.progress = min(100.0, 100.0 * done / self.bytes_total)

                # Atomic replace into final path
                tmp_path.replace(dest)

                if not dest.exists() or dest.stat().st_size == 0:
                    raise IOError(f"Downloaded file is missing/empty: {dest}")

                print(f"[WEIGHTS] Ready: {dest} ({dest.stat().st_size/1e9:.2f} GB)", flush=True)
                with self.lock:
                    self.status = "done"
                    self.msg = f"Weights ready for {MODEL_CONFIG.get(model_key, {}).get('label', model_key)}."
                    self.progress = 100.0

            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                print(f"[WEIGHTS][ERROR] {err}", flush=True)
                with self.lock:
                    self.status = "failed"
                    self.last_error = err
                    self.msg = f"Failed downloading weights for {model_key}: {err}"

                # cleanup temp file on failure
                if tmp_path is not None:
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except Exception:
                        pass

            finally:
                with self.lock:
                    self.active = False

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()
        return True

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "active": self.active,
                "model_key": self.model_key,
                "status": self.status,
                "msg": self.msg,
                "progress": float(self.progress),
                "bytes_done": int(self.bytes_done),
                "bytes_total": int(self.bytes_total),
                "last_error": self.last_error,
            }

WEIGHTS = WeightsDownloadManager()

# Keep these for argparse defaults – they will be overridden at runtime
npz_file = MODEL_CONFIG[DEFAULT_MODEL_KEY]["npz"]
saved_model = MODEL_CONFIG[DEFAULT_MODEL_KEY]["model"]




DOWNLOAD_REGISTRY = {}
DOWNLOAD_LOCK = threading.Lock()
DOWNLOAD_DELETE_AFTER_S = int(os.environ.get("DELETE_AFTER_SECONDS", "600"))  # 10 min
DELETE_ONCE_DOWNLOADED = bool(int(os.environ.get("DELETE_ONCE_DOWNLOADED", "1")))  # 1 to delete once done
DOWNLOAD_LOG_CSV = OUTPUT_DIR / "download_log.csv"

DELETE_EXPIRED = int(os.environ.get("DELETE_EXPIRED", "1"))  # 1 = delete on expiry
DELETE_FILE_PATTERNS = [s.strip() for s in os.environ.get(
    "DELETE_FILE_PATTERNS", ".fits,.fits.tar.gz"
).split(",") if s.strip()]

HARD = {"Teff": (2000.0, 190000.0), "logG": (-1.0, 9.0), "logZ": (-4.0, 1.0)}
MAX_SAMPLES = {"Teff": 50, "logG": 50, "logZ": 50}

SUN_EXACT = {
    "Teff": 5772.0,
    "logG": 4.44,
    "logZ": 0.0,
}
# CSV / explicit-triplets mode
MAX_CSV_ROWS = int(os.environ.get("MAX_CSV_ROWS", "1000000"))  # safety cap for uploaded rows
CSV_REQUIRED_COLS = {"teff", "logg", "logz"}


# =========================
# Small utils
# =========================

def hard_cuda_teardown(*objs_to_drop):
    """
    Attempt to aggressively free CUDA memory held by this process.
    Returns a dict with before/after memory stats (bytes), when available.
    """
    stats = {"before_reserved": 0, "after_reserved": 0, "freed_bytes": 0}
    try:
        import gc
        try:
            import torch
            if torch.cuda.is_available():
                dev = torch.cuda.current_device()
                torch.cuda.synchronize(dev)
                stats["before_reserved"] = torch.cuda.memory_reserved(dev)
        except Exception:
            pass

        # Drop references explicitly (if any were passed in)
        for o in objs_to_drop:
            try:
                del o
            except Exception:
                pass

        # Close any matplotlib figures if used during plotting
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception:
            pass

        # GC + CUDA cache clear
        try:
            gc.collect()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # Optional: clear cuBLASLt autotuning cache (if available)
                try:
                    torch._C._cuda_clearCublasLtCache()
                except Exception:
                    pass
                dev = torch.cuda.current_device()
                stats["after_reserved"] = torch.cuda.memory_reserved(dev)
                stats["freed_bytes"] = max(0, stats["before_reserved"] - stats["after_reserved"])
        except Exception:
            pass
    except Exception:
        pass
    return stats



def _sanitize_filename(name: str) -> str:
    """
    Keep only A–Z, a–z, 0–9, dot, underscore, and hyphen.
    Collapse anything else into a single hyphen. Trim leading/trailing hyphens.
    """
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
    return name.strip("-")


def _parse_csv_bytes(b: bytes) -> pd.DataFrame:
    """
    Parse CSV/TSV/whitespace text into a pandas DataFrame with columns [Teff, logG, logZ].
    - Uses header names (teff, logg, logz) if present (case-insensitive).
    - Otherwise scans numeric tokens per line and picks the first (t,g,z) triplet that
      lies within HARD ranges.
    - Drops duplicate rows.
    """
    text = b.decode("utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame(columns=["Teff", "logG", "logZ"])


    sep = "," if ("," in lines[0]) else None  # None => split on whitespace
    header_tokens = (lines[0].split(",") if sep == "," else re.split(r"\s+", lines[0]))
    header = [h.strip().lower() for h in header_tokens]

    def _is_number(s: str) -> bool:
        s = s.replace(".", "", 1).replace("-", "", 1)
        return s.isdigit()

    HAVE_HEADER = any(not _is_number(h) for h in header)
    name_to_idx = {}
    if HAVE_HEADER:
        for i, name in enumerate(header):
            if name in ("teff", "t_eff", "t"):
                name_to_idx["teff"] = i
            elif name in ("logg", "log_g", "g"):
                name_to_idx["logg"] = i
            elif name in ("logz", "log_z", "z"):
                name_to_idx["logz"] = i

    rows = []
    def _inside(val, lo_hi): return (lo_hi[0] <= val <= lo_hi[1])

    data_iter = lines[1:] if HAVE_HEADER else lines
    for ln in data_iter:
        parts = (ln.split(",") if (sep == ",") else re.split(r"\s+", ln))

        # Header-directed parse
        if HAVE_HEADER and all(k in name_to_idx for k in ("teff", "logg", "logz")):
            try:
                t = float(parts[name_to_idx["teff"]])
                g = float(parts[name_to_idx["logg"]])
                z = float(parts[name_to_idx["logz"]])
                rows.append((t, g, z))
                continue
            except Exception:
                pass  # fall through

        # Heuristic scan for (t,g,z) in-range triplet
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except Exception:
                continue

        found = False
        for i in range(len(nums) - 2):
            t, g, z = nums[i], nums[i + 1], nums[i + 2]
            if _inside(t, HARD["Teff"]) and _inside(g, HARD["logG"]) and _inside(z, HARD["logZ"]):
                rows.append((t, g, z))
                found = True
                break

        if not found and len(nums) == 3:
            rows.append(tuple(nums))

    if not rows:
        return pd.DataFrame(columns=["Teff", "logG", "logZ"])

    df = pd.DataFrame(rows, columns=["Teff", "logG", "logZ"])
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def _inrange_mask(arr: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask for rows whose (Teff, logG, logZ) lie within HARD bounds.
    arr: (M,3) -> columns [Teff, logG, logZ]
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return np.zeros((0,), dtype=bool)
    t_ok = (HARD["Teff"][0] <= arr[:, 0]) & (arr[:, 0] <= HARD["Teff"][1])
    g_ok = (HARD["logG"][0] <= arr[:, 1]) & (arr[:, 1] <= HARD["logG"][1])
    z_ok = (HARD["logZ"][0] <= arr[:, 2]) & (arr[:, 2] <= HARD["logZ"][1])
    return (t_ok & g_ok & z_ok)

def _validate_csv_grid(grid):
    """
    Accepts either:
      - numpy.ndarray shape (M,3) [Teff, logG, logZ], or
      - pandas.DataFrame with columns ['Teff','logG','logZ'] (any order ok).
    Returns a list of error strings (empty if OK).
    """
    try:
        import pandas as pd  # optional
    except Exception:
        pd = None

    # Coerce to ndarray (M,3)
    if pd is not None and isinstance(grid, pd.DataFrame):
        # Reorder if necessary, then to numpy
        cols = [c for c in grid.columns]
        # Try best-effort mapping
        name_map = {c.lower(): c for c in cols}
        want = ["teff", "logg", "logz"]
        if all(w in name_map for w in want):
            grid = grid[[name_map["teff"], name_map["logg"], name_map["logz"]]].to_numpy(dtype=float)
        else:
            # assume current order is [Teff, logG, logZ]
            grid = grid.to_numpy(dtype=float)
    else:
        grid = np.asarray(grid, dtype=float)

    errs = []
    if grid.size == 0:
        errs.append("CSV has no valid rows.")
        return errs

    # Inclusive bounds check
    t_ok = (HARD["Teff"][0] <= grid[:, 0]).all() and (grid[:, 0] <= HARD["Teff"][1]).all()
    g_ok = (HARD["logG"][0] <= grid[:, 1]).all() and (grid[:, 1] <= HARD["logG"][1]).all()
    z_ok = (HARD["logZ"][0] <= grid[:, 2]).all() and (grid[:, 2] <= HARD["logZ"][1]).all()

    if not t_ok:
        errs.append(f"At least one Teff is outside {HARD['Teff']}.")
    if not g_ok:
        errs.append(f"At least one logG is outside {HARD['logG']}.")
    if not z_ok:
        errs.append(f"At least one logZ is outside {HARD['logZ']}.")

    return errs


def _make_grid3d_figure(
        grid_np: np.ndarray,
        title: str = "3D Grid",
        model_key: str | None = None,
):
    """
    grid_np: (M,3) array of [Teff, logG, logZ].
    Collapses duplicates to speed up plotting, scales marker size by counts.

    For the TMAP+C3K+Husser combined model, points are coloured by which
    underlying library span they fall into, using the individual
    meta files (phoenix_husser_all.meta.npz, C3K_all.meta.npz, TMAP_all.meta.npz).
    """
    if grid_np.size == 0:
        grid_np = np.zeros((0, 3), dtype=float)

    # collapse duplicates for fast scatter
    df = pd.DataFrame(grid_np, columns=["Teff", "logG", "logZ"])
    agg = df.value_counts().reset_index(name="count")  # columns: Teff, logG, logZ, count


    teff_plot = np.log10(agg["Teff"].to_numpy(dtype=float))


    counts = agg["count"].to_numpy()

    # --- Bigger marker sizes (also enlarges legend circles) ---
    # previously: sizes = np.clip(3 + np.log1p(counts), 3, 11)
    sizes = np.clip(6 + 1.5 * np.log1p(counts), 6, 20)

    #print("model_key: ",model_key)

    if model_key == "TMAP-C3K-husser-OB-combined":
        # Component libraries and short names for legend
        component_keys = [
            ("PHOENIX-husser", "Husser"),
            ("C3K-conroy",     "C3K"),
            ("TMAP-Werner",    "TMAP"),
            ("OB-PoWR",        "OB"),
        ]

        # Build simple axis-aligned boxes from each library's meta cache
        boxes: dict[str, dict[str, float]] = {}
        for mkey, short in component_keys:
            cfg = MODEL_CONFIG.get(mkey)
            if not cfg:
                continue
            meta_path = meta_dir / cfg["meta"]
            meta = _load_meta_cache(meta_path)
            if meta is None:
                # If the meta file is missing, skip this component gracefully
                #print("meta cache missing for", short)
                continue
            boxes[short] = {
                "Teff_min": float(meta["Teff_min"]),
                "Teff_max": float(meta["Teff_max"]),
                "logG_min": float(meta["logG_min"]),
                "logG_max": float(meta["logG_max"]),
                "logZ_min": float(meta["logZ_min"]),
                "logZ_max": float(meta["logZ_max"]),
            }
        #print("boxes:",boxes)

        coords = agg[["Teff", "logG", "logZ"]].to_numpy(dtype=float)

        # Boolean masks per library; if there is no box for a lib,
        # its mask is all False.
        def _mask_for(short: str) -> np.ndarray:
            box = boxes.get(short)
            if box is None:
                return np.zeros(coords.shape[0], dtype=bool)

            t = coords[:, 0]
            g = coords[:, 1]
            z = coords[:, 2]

            return (
                    (box["Teff_min"] - EPS <= t) & (t <= box["Teff_max"] + EPS) &
                    (box["logG_min"] - EPS <= g) & (g <= box["logG_max"] + EPS) &
                    (box["logZ_min"] - EPS <= z) & (z <= box["logZ_max"] + EPS)
            )

        mask_h = _mask_for("Husser")
        mask_c = _mask_for("C3K")
        mask_t = _mask_for("TMAP")
        mask_o = _mask_for("OB")

        # Encode membership as bits: Husser=1, C3K=2, TMAP=4, OB=8
        codes = (
                mask_h.astype(int) * 1 +
                mask_c.astype(int) * 2 +
                mask_t.astype(int) * 4 +
                mask_o.astype(int) * 8
        )
        #print("codes:", codes)

        # Define categories: name + colour
        # (Only categories that are actually present will be plotted.)
        patterns = {
            1:  ("Husser only",              "#1f77b4"),
            2:  ("C3K only",                 "#ff7f0e"),
            4:  ("TMAP only",                "#2ca02c"),
            8:  ("OB only",                  "#17becf"),

            3:  ("C3K + Husser overlap",     "#d62728"),
            5:  ("Husser + TMAP overlap",    "#9467bd"),
            9:  ("Husser + OB overlap",      "#bcbd22"),
            6:  ("C3K + TMAP overlap",       "#8c564b"),
            10: ("C3K + OB overlap",         "#e377c2"),
            12: ("TMAP + OB overlap",        "#7f7f7f"),

            7:  ("Husser + C3K + TMAP",      "#aec7e8"),
            11: ("Husser + C3K + OB",        "#ffbb78"),
            13: ("Husser + TMAP + OB",       "#98df8a"),
            14: ("C3K + TMAP + OB",          "#ff9896"),

            15: ("All four overlap",         "#c5b0d5"),
            0:  ("Extrapolated by model",    "#bbbbbb"),
        }

        # Nice-ish legend order: singles → pairs → triples → all four → extrapolated
        pattern_order = [
            1, 2, 4, 8,
            3, 5, 6, 9, 10, 12,
            7, 11, 13, 14,
            15,
            0,
        ]

        fig = go.Figure()
        for code in pattern_order:
            if code not in patterns:
                continue
            name, color = patterns[code]
            mask = (codes == code)
            if not np.any(mask):
                continue

            # text array so we can use %{text} in hovertemplate
            region_text = np.full(mask.sum(), name)

            fig.add_trace(
                go.Scatter3d(
                    x=teff_plot[mask],
                    y=agg["logG"][mask],
                    z=agg["logZ"][mask],
                    mode="markers",
                    marker=dict(
                        size=sizes[mask],
                        opacity=1,
                        color=color,
                    ),
                    # --- Library / overlap info in hover ---
                    text=region_text,
                    hovertemplate=(
                        "Teff=%{customdata[0]:.0f} K<br>"
                        "logG=%{y:.2f}<br>"
                        "logZ=%{z:.2f}<br>"
                        "Region: %{text}"
                        "<extra></extra>"
                    ),
                    customdata=np.stack([agg["Teff"].to_numpy(dtype=float), counts], axis=1),
                    name=name,
                    showlegend=True,
                )
            )

    elif model_key == "TMAP-C3K-husser-combined":
        # Component libraries and short names for legend
        component_keys = [
            ("PHOENIX-husser", "Husser"),
            ("C3K-conroy",     "C3K"),
            ("TMAP-Werner",    "TMAP"),
        ]

        # Build simple axis-aligned boxes from each library's meta cache
        boxes: dict[str, dict[str, float]] = {}
        for mkey, short in component_keys:
            cfg = MODEL_CONFIG.get(mkey)
            if not cfg:
                continue
            meta_path = meta_dir / cfg["meta"]
            meta = _load_meta_cache(meta_path)
            if meta is None:
                # If the meta file is missing, skip this component gracefully
                continue
            boxes[short] = {
                "Teff_min": float(meta["Teff_min"]),
                "Teff_max": float(meta["Teff_max"]),
                "logG_min": float(meta["logG_min"]),
                "logG_max": float(meta["logG_max"]),
                "logZ_min": float(meta["logZ_min"]),
                "logZ_max": float(meta["logZ_max"]),
            }

        coords = agg[["Teff", "logG", "logZ"]].to_numpy(dtype=float)

        # Boolean masks per library; if we don't have a box for a lib,
        # its mask is all False.
        def _mask_for(short: str) -> np.ndarray:
            box = boxes.get(short)
            if box is None:
                return np.zeros(coords.shape[0], dtype=bool)

            t = coords[:, 0]
            g = coords[:, 1]
            z = coords[:, 2]

            return (
                    (box["Teff_min"] - EPS <= t) & (t <= box["Teff_max"] + EPS) &
                    (box["logG_min"] - EPS <= g) & (g <= box["logG_max"] + EPS) &
                    (box["logZ_min"] - EPS <= z) & (z <= box["logZ_max"] + EPS)
            )

        mask_h = _mask_for("Husser")
        mask_c = _mask_for("C3K")
        mask_t = _mask_for("TMAP")

        # Encode membership as bits: Husser=1, C3K=2, TMAP=4
        codes = (
                mask_h.astype(int) * 1 +
                mask_c.astype(int) * 2 +
                mask_t.astype(int) * 4
        )

        # Define categories: name + colour
        # (Only categories that are actually present will be plotted.)
        patterns = {
            1:  ("Husser only",              "#1f77b4"),
            2:  ("C3K only",                 "#ff7f0e"),
            4:  ("TMAP only",                "#2ca02c"),
            8:  ("OB only",                  "#17becf"),

            3:  ("C3K + Husser overlap",     "#d62728"),
            5:  ("Husser + TMAP overlap",    "#9467bd"),
            6:  ("C3K + TMAP overlap",       "#8c564b"),

            0:  ("Extrapolated by model",    "#bbbbbb"),
        }

        pattern_order = [
            1, 2, 4,
            3, 5, 6,
            7,
            0,
        ]

        fig = go.Figure()
        for code in pattern_order:
            if code not in patterns:
                continue
            name, color = patterns[code]
            mask = (codes == code)
            if not np.any(mask):
                continue

            # text array so we can use %{text} in hovertemplate
            region_text = np.full(mask.sum(), name)

            fig.add_trace(
                go.Scatter3d(
                    x=teff_plot[mask],
                    y=agg["logG"][mask],
                    z=agg["logZ"][mask],
                    mode="markers",
                    marker=dict(
                        size=sizes[mask],
                        opacity=1,
                        color=color,
                    ),
                    # --- Library / overlap info in hover ---
                    text=region_text,
                    hovertemplate=(
                        "Teff=%{customdata[0]:.0f} K<br>"
                        "logG=%{y:.2f}<br>"
                        "logZ=%{z:.2f}<br>"
                        "Region: %{text}"
                        "<extra></extra>"
                    ),
                    customdata=np.stack([agg["Teff"].to_numpy(dtype=float), counts], axis=1),
                    name=name,
                    showlegend=True,
                )
            )
    else:
        # ------------------------------------------------------------------
        # Default: single-colour scatter for non-combined models
        # ------------------------------------------------------------------
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=teff_plot,
                    y=agg["logG"],
                    z=agg["logZ"],
                    mode="markers",
                    marker=dict(size=sizes, opacity=1),
                    hovertemplate=(
                        "Teff=%{customdata[0]:.0f} K<br>"
                        "logG=%{y:.2f}<br>"
                        "logZ=%{z:.2f}<br>"
                        "count=%{customdata[1]:d}"
                        "<extra></extra>"
                    ),
                    customdata=np.stack([counts], axis=1),
                    name="points",
                    showlegend=False,
                )
            ]
        )

    fig.update_layout(
        legend=dict(
            x=0.80,
            y=0.05,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.85)",  # optional: cleaner visibility
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
            font=dict(size=16),
        ),
        template="plotly_white",
        height=1000,
        margin=dict(l=10, r=10, t=40, b=10),
        title=title,
    )

    fig.update_scenes(
        xaxis=dict(
            title="log Teff(K)",
            title_font=dict(size=20),   # <- axis title size
            tickfont=dict(size=16),     # <- tick label size
        ),
        yaxis=dict(
            title="log g (cgs)",
            title_font=dict(size=20),
            tickfont=dict(size=16),
        ),
        zaxis=dict(
            title="log Z",
            title_font=dict(size=20),
            tickfont=dict(size=16),
        ),
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
    )
    return fig, int(agg["count"].sum()), int(len(agg))



def _preview_candidates_for(p: Path) -> list[Path]:
    """Return [first_png, last_png] that match the given artifact path p."""
    base = p.name
    for suf in (".fits.tar.gz", ".fits"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    first = p.with_name(base + "_first.png")
    last  = p.with_name(base + "_last.png")
    return [first, last]

def _delete_artifacts_and_previews(p: Path) -> None:
    """Delete the main artifact p and its preview PNGs (if present)."""
    try:
        os.remove(p)
    except Exception:
        pass
    # Do NOT log image deletions (per your request)
    for q in _preview_candidates_for(p):
        try:
            if q.exists():
                os.remove(q)
        except Exception:
            pass


def purge_gpu_memory(device: int | None = None):
    """
    Aggressively release GPU memory from this process.
    Returns a dict with before/after stats for logging.
    """
    try:
        import gc, torch
        if not torch.cuda.is_available():
            return {"freed_bytes": 0, "note": "cuda_not_available"}

        if device is None:
            device = torch.cuda.current_device()

        # Record before
        torch.cuda.synchronize(device)
        before_alloc = torch.cuda.memory_allocated(device)
        before_reserved = torch.cuda.memory_reserved(device)

        # Best-effort: drop any big objects your code might hold
        # (adjust attribute names to your codebase as needed)
        for name in ("_model", "_optimizer", "_tensors", "_cache", "_module"):
            if hasattr(globals().get("RUN"), name):
                setattr(globals()["RUN"], name, None)

        # Run GC a couple of times to break cycles
        gc.collect()
        gc.collect()

        # Ask the caching allocator to release unused cached blocks back to driver
        torch.cuda.empty_cache()
        # Collect any CUDA IPC handles that can pin memory
        torch.cuda.ipc_collect()
        torch.cuda.synchronize(device)

        after_alloc = torch.cuda.memory_allocated(device)
        after_reserved = torch.cuda.memory_reserved(device)
        freed = max(0, before_reserved - after_reserved)

        # Optional: reset peaks (purely cosmetic for stats)
        with contextlib.suppress(Exception):
            torch.cuda.reset_peak_memory_stats(device)

        return {
            "freed_bytes": int(freed),
            "before_alloc": int(before_alloc),
            "before_reserved": int(before_reserved),
            "after_alloc": int(after_alloc),
            "after_reserved": int(after_reserved),
        }
    except Exception as e:
        print(f"[GPU] purge_gpu_memory failed: {e}", flush=True)
        return {"freed_bytes": 0, "error": str(e)}



def _append_download_log(row: dict):
    # Ensure header once
    new_file = not DOWNLOAD_LOG_CSV.exists()
    with DOWNLOAD_LOG_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ts_start_iso", "ts_end_iso", "duration_s",
            "client_ip", "filename", "size_bytes", "kind", "bytes_sent", "status", "token"
        ])
        if new_file:
            w.writeheader()
        w.writerow(row)

def _client_ip():
    # Honor X-Forwarded-For if behind a proxy; fallback to remote_addr
    xff = request.headers.get("X-Forwarded-For", "")
    return xff.split(",")[0].strip() if xff else (request.remote_addr or "unknown")

def register_download(path: Path, kind: str) -> str:
    token = secrets.token_urlsafe(16)
    with DOWNLOAD_LOCK:
        DOWNLOAD_REGISTRY[token] = {
            "path": Path(path),
            "kind": kind,
            "created": time.time(),
            "delete_after": DOWNLOAD_DELETE_AFTER_S,
            "downloaded": False,   # NEW: mark on successful completion
        }
    return token


def _purge_expired_downloads():
    now = time.time()
    to_delete = []
    with DOWNLOAD_LOCK:
        expired = []
        for t, meta in list(DOWNLOAD_REGISTRY.items()):
            if (now - meta["created"]) > meta["delete_after"]:
                expired.append((t, meta))
        for t, meta in expired:
            DOWNLOAD_REGISTRY.pop(t, None)
            # Optionally delete the file if we timed out before/after download
            if DELETE_EXPIRED:
                p = meta.get("path")
                if p and isinstance(p, Path) and p.exists():
                    try:
                        size_bytes = os.path.getsize(p)
                        os.remove(p)
                        _append_download_log({
                            "ts_start_iso": datetime.now(timezone.utc).isoformat(),
                            "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                            "duration_s": 0.0,
                            "client_ip": "janitor",
                            "filename": str(p),
                            "size_bytes": size_bytes,
                            "kind": meta.get("kind", "file"),
                            "bytes_sent": 0,
                            "status": "expired_deleted",
                            "token": t,
                        })
                    except Exception:
                        pass


def minmax_norm(x, lo=None, hi=None):
    lo = np.min(x) if lo is None else lo
    hi = np.max(x) if hi is None else hi
    rng = hi - lo if (hi - lo) != 0 else 1.0
    return (x - lo) / rng, (float(lo), float(hi))

def log1p01_flux(x: np.ndarray, tau: float | None = None,
                 tau_percentile: float = TAU_PERCENTILE,
                 ensure_nonneg: bool = True) -> Tuple[np.ndarray, dict]:
    x = np.asarray(x, dtype=np.float64)
    x_clipped = np.maximum(x, 0.0) if ensure_nonneg else x
    if tau is None:
        pos = x_clipped[x_clipped > 0]
        if pos.size == 0:
            return np.zeros_like(x_clipped, dtype=np.float32), {"x_max": 0.0, "tau": 1.0}
        tau = np.percentile(pos, tau_percentile)
        tau = float(max(tau, np.finfo(np.float64).tiny))
    xmax = float(np.nanmax(x_clipped)) if np.isfinite(x_clipped).any() else 0.0
    if xmax <= 0:
        return np.zeros_like(x_clipped, dtype=np.float32), {"x_max": xmax, "tau": tau}
    num = np.log1p(x_clipped / tau)
    den = np.log1p(xmax / tau)
    y = (num / den).astype(np.float32)
    return y, {"x_max": xmax, "tau": tau}

def inv_log1p01_flux(y: np.ndarray, tau: float, x_max: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    den = np.log1p(x_max / tau)
    return tau * (np.exp(y * den) - 1.0)

def _load_meta_cache(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.is_file():
            z = np.load(path)
            return {
                "wave_ang": z["wave"].astype(np.float64),
                "tau": float(z["tau"]),
                "x_max": float(z["x_max"]),
                "Teff_min": float(z["Teff_min"]),
                "Teff_max": float(z["Teff_max"]),
                "logG_min": float(z["logG_min"]),
                "logG_max": float(z["logG_max"]),
                "logZ_min": float(z["logZ_min"]),
                "logZ_max": float(z["logZ_max"]),
            }
    except Exception:
        pass
    return None

def _save_meta_cache(path: Path, wave_ang: np.ndarray,
                     tau: float, x_max: float,
                     Teff_min: float, Teff_max: float,
                     logG_min: float, logG_max: float,
                     logZ_min: float, logZ_max: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        wave=wave_ang.astype(np.float32, copy=False),
        tau=np.array(tau, dtype=np.float64),
        x_max=np.array(x_max, dtype=np.float64),
        Teff_min=np.array(Teff_min), Teff_max=np.array(Teff_max),
        logG_min=np.array(logG_min), logG_max=np.array(logG_max),
        logZ_min=np.array(logZ_min), logZ_max=np.array(logZ_max),
    )

def ensure_limits_for_model(model_key: str) -> dict:
    cfg = MODEL_CONFIG.get(model_key, MODEL_CONFIG[DEFAULT_MODEL_KEY])
    meta_path = meta_dir / cfg["meta"]

    meta = _load_meta_cache(meta_path)
    if meta is None:
        raise FileNotFoundError(
            f"Required meta file not found: {meta_path}. "
            f"Please download the model resources for '{model_key}'."
        )

    HARD["Teff"] = (float(meta["Teff_min"]), float(meta["Teff_max"]))
    HARD["logG"] = (float(meta["logG_min"]), float(meta["logG_max"]))
    HARD["logZ"] = (float(meta["logZ_min"]), float(meta["logZ_max"]))
    return meta


def _auto_fits_name(args, M: int, L: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    tok = uuid.uuid4().hex[:6]
    flux_tag = "lin" if args.linear_flux else "scaled"

    def _rng(lo, hi, n):  # no brackets
        return f"{lo:g}-{hi:g}x{n}"

    raw = (
        f"phx_T{_rng(args.teff_min, args.teff_max, args.teff_n)}"
        f"_g{_rng(args.logg_min, args.logg_max, args.logg_n)}"
        f"_Z{_rng(args.logz_min, args.logz_max, args.logz_n)}"
        f"_L{L}_{flux_tag}_{stamp}_{tok}.fits"
    )
    return _sanitize_filename(raw)


def build_grid(teff_min, teff_max, teff_n, logg_min, logg_max, logg_n, logz_min, logz_max, logz_n):
    Teff = np.linspace(teff_min, teff_max, teff_n)
    logG = np.linspace(logg_min, logg_max, logg_n)
    logZ = np.linspace(logz_min, logz_max, logz_n)
    T, G, Z = np.meshgrid(Teff, logG, logZ, indexing="xy")
    return np.stack([T.ravel(), G.ravel(), Z.ravel()], axis=1)

def _validate_params(teff_min, teff_max, teff_n, logg_min, logg_max, logg_n, logz_min, logz_max, logz_n):
    errors = []
    if not (HARD["Teff"][0] <= teff_min < teff_max <= HARD["Teff"][1]):
        errors.append(f"Teff must be within {HARD['Teff']} and min < max.")
    if not (HARD["logG"][0] <= logg_min < logg_max <= HARD["logG"][1]):
        errors.append(f"logG must be within {HARD['logG']} and min < max.")
    if not (HARD["logZ"][0] <= logz_min < logz_max <= HARD["logZ"][1]):
        errors.append(f"logZ must be within {HARD['logZ']} and min < max.")
    if not (2 <= teff_n <= MAX_SAMPLES["Teff"]):
        errors.append(f"Teff samples must be 2–{MAX_SAMPLES['Teff']}.")
    if not (2 <= logg_n <= MAX_SAMPLES["logG"]):
        errors.append(f"logG samples must be 2–{MAX_SAMPLES['logG']}.")
    if not (2 <= logz_n <= MAX_SAMPLES["logZ"]):
        errors.append(f"logZ samples must be 2–{MAX_SAMPLES['logZ']}.")
    return errors

def _unique_outname(params: dict) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tok = uuid.uuid4().hex[:6]
    raw = (
        "dash_phx_"
        f"T{params['teff_min']}-{params['teff_max']}x{params['teff_n']}"
        f"_g{params['logg_min']}-{params['logg_max']}x{params['logg_n']}"
        f"_Z{params['logz_min']}-{params['logz_max']}x{params['logz_n']}"
        f"_{stamp}_{tok}.fits"
    )
    return OUTPUT_DIR / _sanitize_filename(raw)


def _png_to_data_url(p: Path) -> Optional[str]:
    try:
        b = p.read_bytes()
        return "data:image/png;base64," + base64.b64encode(b).decode("ascii")
    except Exception:
        return None

# =========================
# Model
# =========================
class ConditionalRefineNet(nn.Module):
    def __init__(self, cond_dim=3, spec_len=1024, dropout_p=0.0,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.spec_len = spec_len
        self.cond_proj = nn.Linear(cond_dim, spec_len, device=device, dtype=dtype)
        self.register_buffer("pos_encoding", None, persistent=False)
        self.branch3_conv = nn.Conv1d(5, 16, kernel_size=7)
        self.branch5_conv = nn.Conv1d(5, 16, kernel_size=15)
        self.branch7_conv = nn.Conv1d(5, 16, kernel_size=23)
        self.fuse_conv1 = nn.Conv1d(48, 16, kernel_size=3)
        self.fuse_conv2 = nn.Conv1d(16, 1, kernel_size=3)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)
    def _ensure_pos(self, device, dtype):
        if self.pos_encoding is None:
            freqs = torch.linspace(0, 1, self.spec_len, device=device, dtype=dtype).unsqueeze(0)
            pe = torch.cat([freqs, torch.sin(2 * torch.pi * freqs), torch.cos(2 * torch.pi * freqs)], dim=0)
            self.register_buffer("pos_encoding", pe, persistent=False)
    def forward(self, coarse_spec, cond):
        self._ensure_pos(coarse_spec.device, coarse_spec.dtype)
        cond_proj = self.cond_proj(cond)
        x = torch.stack([coarse_spec, cond_proj], dim=1)
        pos = self.pos_encoding.unsqueeze(0).expand(x.size(0), -1, -1)
        x = torch.cat([x, pos], dim=1)
        x3 = F.pad(x, pad=(3, 3), mode='replicate')
        x5 = F.pad(x, pad=(7, 7), mode='replicate')
        x7 = F.pad(x, pad=(11, 11), mode='replicate')
        out3 = self.dropout(self.activation(self.branch3_conv(x3)))
        out5 = self.dropout(self.activation(self.branch5_conv(x5)))
        out7 = self.dropout(self.activation(self.branch7_conv(x7)))
        x_cat = torch.cat([out3, out5, out7], dim=1)
        x_cat = F.pad(x_cat, pad=(1, 1), mode='replicate')
        fuse_out = self.dropout(self.activation(self.fuse_conv1(x_cat)))
        fuse_out = F.pad(fuse_out, pad=(1, 1), mode='replicate')
        correction = self.fuse_conv2(fuse_out).squeeze(1)
        return coarse_spec + correction

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_p=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(out_features, out_features)
    def forward(self, x):
        out = self.fc1(x); residual = out
        out = self.act(out); out = self.dropout(out); out = self.fc2(out)
        return self.act(out + residual)

class LookupModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=1024, base_size=BASE_SIZE,
                 dropout_p=0.0, out_softplus_beta: float = SOFTPLUS_BETA,
                 init_zero: bool = True, device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.output_dim = output_dim
        self.out_softplus_beta = float(out_softplus_beta)
        H1 = BASE_SIZE * 4
        H2 = BASE_SIZE * 16
        self.input_proj_1 = nn.Linear(input_dim, H1, device=device, dtype=dtype)
        self.input_proj_2 = nn.Linear(input_dim, H2, device=device, dtype=dtype)
        self.input_proj_3 = nn.Linear(input_dim, output_dim, device=device, dtype=dtype)
        self.layer1 = ResidualBlock(input_dim, H1, dropout_p)
        self.layer2 = ResidualBlock(H1, H2, dropout_p)
        self.layer3 = nn.Linear(H2, output_dim, device=device, dtype=dtype)
        self.refine = ConditionalRefineNet(cond_dim=input_dim, spec_len=output_dim,
                                           dropout_p=dropout_p, device=device, dtype=dtype)
        self.baseline = nn.Parameter(torch.zeros(output_dim, device=device, dtype=dtype))
        self.softplus = nn.Softplus(beta=self.out_softplus_beta, threshold=20.0)
        if init_zero:
            self._zero_init()
    def _zero_init(self):
        nn.init.zeros_(self.layer3.weight); nn.init.zeros_(self.layer3.bias)
        nn.init.zeros_(self.input_proj_1.weight); nn.init.zeros_(self.input_proj_1.bias)
        nn.init.zeros_(self.input_proj_2.weight); nn.init.zeros_(self.input_proj_2.bias)
        nn.init.zeros_(self.input_proj_3.weight); nn.init.zeros_(self.input_proj_3.bias)
        nn.init.zeros_(self.refine.fuse_conv2.weight)
        if self.refine.fuse_conv2.bias is not None:
            nn.init.zeros_(self.refine.fuse_conv2.bias)
    def forward(self, x):
        x_clean = x
        h1 = self.layer1(x) + self.input_proj_1(x_clean)
        h2 = self.layer2(h1) + self.input_proj_2(x_clean)
        coarse_spec = self.layer3(h2) + self.input_proj_3(x_clean)
        refined = self.refine(coarse_spec, x_clean)
        return self.softplus(refined + self.baseline) + 1e-8

# =========================
# Inference (returns dict of paths)
# =========================
def run_inference(cli_args: Optional[list[str]] = None) -> Optional[dict]:
    import argparse

    class SectionTimer:
        def __init__(self, label: str):
            self.label = label
            self.t0 = None
        def __enter__(self):
            self.t0 = time.perf_counter(); print(f"[TIME] {self.label}…", flush=True)
        def __exit__(self, exc_type, exc, tb):
            dt = time.perf_counter() - self.t0
            print(f"[TIME] {self.label}: {dt:.3f}s", flush=True)

    p = argparse.ArgumentParser()
    p.add_argument("--npz-path", type=Path, default=meta_dir / npz_file)
    p.add_argument("--model-path", type=Path, default=model_dir / saved_model)
    p.add_argument("--outdir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--teff-min", type=float, default=2300.0)
    p.add_argument("--teff-max", type=float, default=12000.0)
    p.add_argument("--logg-min", type=float, default=0.0)
    p.add_argument("--logg-max", type=float, default=6.0)
    p.add_argument("--logz-min", type=float, default=-4.0)
    p.add_argument("--logz-max", type=float, default=1.0)
    p.add_argument("--teff-n", type=int, default=20)
    p.add_argument("--logg-n", type=int, default=15)
    p.add_argument("--logz-n", type=int, default=15)
    p.add_argument("--csv-path", type=Path, default=None,
                   help="Optional path to a JSON file containing [[Teff, logG, logZ], ...]. If set, grid bounds are ignored.")

    try:
        from argparse import BooleanOptionalAction
    except Exception:
        BooleanOptionalAction = None
    if BooleanOptionalAction is not None:
        p.add_argument("--linear-flux", dest="linear_flux", action=BooleanOptionalAction, default=True)
        p.add_argument("--compress-fits", dest="compress_fits", action=BooleanOptionalAction, default=False)
    else:
        p.add_argument("--linear-flux", dest="linear_flux", action="store_true", default=True)
        p.add_argument("--no-linear-flux", dest="linear_flux", action="store_false")
        p.add_argument("--compress-fits", dest="compress_fits", action="store_true", default=False)
        p.add_argument("--no-compress-fits", dest="compress_fits", action="store_false")

    p.add_argument("--save-fits", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--meta-cache", type=Path, default=meta_dir / npz_file)
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")  # stays fp32 unless you change it
    p.add_argument("--tf32", choices=["auto","on","off"], default="auto")
    args = p.parse_args(cli_args)

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.tf32 in ("on","auto") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    print("[INFO] Starting…", flush=True)
    t_start = time.perf_counter()

    if args.csv_path is None:
        if not (HARD["Teff"][0] <= args.teff_min < args.teff_max <= HARD["Teff"][1]):
            raise ValueError(f"Teff bounds must be within {HARD['Teff']}.")
        if not (HARD["logG"][0] <= args.logg_min < args.logg_max <= HARD["logG"][1]):
            raise ValueError(f"logG bounds must be within {HARD['logG']}.")
        if not (HARD["logZ"][0] <= args.logz_min < args.logz_max <= HARD["logZ"][1]):
            raise ValueError(f"logZ bounds must be within {HARD['logZ']}.")


    with SectionTimer("Load NPZ/meta"):
        meta_cache_path = args.meta_cache or args.model_path.with_suffix(".meta.npz")
        cached = _load_meta_cache(meta_cache_path)
        if cached is not None:
            wave_ang = cached["wave_ang"]; tau = cached["tau"]; x_max = cached["x_max"]
            Teff_min, Teff_max = cached["Teff_min"], cached["Teff_max"]
            logG_min, logG_max = cached["logG_min"], cached["logG_max"]
            logZ_min, logZ_max = cached["logZ_min"], cached["logZ_max"]
        else:
            npz = np.load(args.npz_path, mmap_mode="r")
            wave_ang = np.asarray(npz["wave"], dtype=np.float64)

            # --- Load and round labels to 2 decimal places ---
            Teff_all = np.round(np.asarray(npz["Teff"], dtype=float), 2)
            logG_all = np.round(np.asarray(npz["logG"], dtype=float), 2)
            logZ_all = np.round(np.asarray(npz["logZ"], dtype=float), 2)

            # Compute min/max after rounding
            _, (Teff_min, Teff_max) = minmax_norm(Teff_all)
            _, (logG_min, logG_max) = minmax_norm(logG_all)
            _, (logZ_min, logZ_max) = minmax_norm(logZ_all)

            spectra_lin_all = np.asarray(npz["spectra"], dtype=np.float64)
            _, meta = log1p01_flux(spectra_lin_all, tau=None, tau_percentile=TAU_PERCENTILE)
            tau, x_max = float(meta["tau"]), float(meta["x_max"])

            try:
                _save_meta_cache(
                    meta_cache_path,
                    wave_ang, tau, x_max,
                    Teff_min, Teff_max,
                    logG_min, logG_max,
                    logZ_min, logZ_max
                )
            except Exception as e:
                print(f"[WARN] Could not write meta cache: {e}", flush=True)


    with SectionTimer("Build grid + normalize"):
        if args.csv_path is not None:
            # Expect a JSON array of [Teff,logG,logZ]
            try:
                rows = json.loads(Path(args.csv_path).read_text())
                grid = np.asarray(rows, dtype=np.float32)
            except Exception as e:
                raise ValueError(f"Failed to read --csv-path {args.csv_path}: {e}")
            if grid.ndim != 2 or grid.shape[1] != 3:
                raise ValueError(f"--csv-path must contain [[Teff,logG,logZ], ...]; got shape {grid.shape}")

            # Validate bounds again here (defense in depth)
            csv_errs = _validate_csv_grid(grid)
            if csv_errs:
                raise ValueError(" ; ".join(csv_errs))
        else:
            grid = build_grid(args.teff_min, args.teff_max, args.teff_n,
                              args.logg_min, args.logg_max, args.logg_n,
                              args.logz_min, args.logz_max, args.logz_n)

        M = grid.shape[0]

        # vectorized normalization (dataset min/max)
        T = (grid[:,0]-Teff_min)/max(Teff_max-Teff_min,1e-12)
        G = (grid[:,1]-logG_min)/max(logG_max-logG_min,1e-12)
        Z = (grid[:,2]-logZ_min)/max(logZ_max-logZ_min,1e-12)
        grid_norm = np.clip(np.stack([T,G,Z], axis=1), 0.0, 1.0).astype(np.float32)

        L = int(np.asarray(wave_ang).shape[0])


    device = torch.device(args.device)
    tgt_dtype = torch.bfloat16 if args.precision=="bf16" else (torch.float16 if args.precision=="fp16" else torch.float32)

    try:
        from torch.nn.utils import skip_init
        use_skip = True
    except Exception:
        use_skip = False

    if use_skip:
        print("[INFO] Create model", flush=True)
        model = skip_init(LookupModel, input_dim=3, output_dim=L, base_size=BASE_SIZE,
                          dropout_p=0.0, out_softplus_beta=SOFTPLUS_BETA,
                          init_zero=False, device=torch.device("meta"), dtype=tgt_dtype)
    else:
        print("[INFO] Create model", flush=True)
        prev_default = torch.get_default_dtype()
        try:
            torch.set_default_dtype(tgt_dtype)
            model = LookupModel(input_dim=3, output_dim=L, base_size=BASE_SIZE,
                                dropout_p=0.0, out_softplus_beta=SOFTPLUS_BETA, init_zero=False)
        finally:
            torch.set_default_dtype(prev_default)

    print("[INFO] Load weights", flush=True)
    ckpt = torch.load(model_dir / saved_model, map_location="cpu") if args.model_path is None else torch.load(args.model_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if tgt_dtype is not torch.float32:
        for k, v in list(state.items()):
            if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype in (torch.float32, torch.float64):
                state[k] = v.to(dtype=tgt_dtype)
    for k in list(state.keys()):
        if k.endswith("pos_encoding"):
            state.pop(k)
    try:
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    except TypeError:
        if hasattr(model, "to_empty"):
            model = model.to_empty(device="cpu", dtype=tgt_dtype)
            missing, unexpected = model.load_state_dict(state, strict=False)
        else:
            raise
    if unexpected: print(f"[WARN] Unexpected keys ignored: {unexpected}", flush=True)
    if missing: print(f"[INFO] Missing keys (okay): {missing}", flush=True)
    model.eval()
    print("[INFO] Weights loaded", flush=True)

    print("[INFO] Move to device", flush=True)
    model = model.to(device, non_blocking=True)

    BATCH = max(1, int(args.batch_size))
    print("[INFO] Inference starting…", flush=True)
    use_amp = (device.type == "cuda") and (args.precision in ("bf16","fp16"))
    autocast_dtype = torch.bfloat16 if args.precision=="bf16" else (torch.float16 if args.precision=="fp16" else None)

    # -----------------------------------
    # INFERENCE (raw, fast) + TIMING
    # -----------------------------------
    t_inf0 = time.perf_counter()
    last_pulse = t_inf0

    # store raw network outputs (scaled domain) to avoid extra copies
    preds_scaled = np.empty((M, L), dtype=np.float32)

    with torch.inference_mode():
        cm = (torch.autocast(device_type="cuda", dtype=autocast_dtype) if use_amp else torch.no_grad())
        with cm:
            for i in range(0, M, BATCH):
                j = min(M, i + BATCH)
                cond_cpu = (torch.from_numpy(grid_norm[i:j]).pin_memory()
                            if device.type == "cuda" else torch.from_numpy(grid_norm[i:j]))
                cond = cond_cpu.to(device, non_blocking=True)

                y_scaled = model(cond).float().cpu().numpy()  # raw model output in "scaled" space
                preds_scaled[i:j] = y_scaled  # stash as-is (no postproc in loop)

                now = time.perf_counter()
                if (now - last_pulse) >= 1.0 or j == M:
                    done = j
                    pct = 100.0 * done / M if M else 100.0
                    print(f"[PROG] {done}/{M} ({pct:.1f}%)", flush=True)
                    last_pulse = now

    dt_inf = time.perf_counter() - t_inf0
    n_batches = (M + BATCH - 1) // BATCH
    thr = (M / dt_inf) if dt_inf > 0 else float("nan")
    print(f"[STAT] Inference: batches={n_batches}  batch_size={BATCH}  M={M}  L={L}  "
          f"-> {thr:.1f} spectra/s  (time={dt_inf:.3f}s)", flush=True)

    # -----------------------------------
    # POST-PROCESSING (separate timer)
    # - Invert to linear + normalize to ∫=1 for --linear-flux
    # -----------------------------------
    if args.linear_flux:
        with SectionTimer("Converting flux to linear & unit-integral normalization"):
            # 1) invert scaled -> linear (float64 for stable integration)
            flux_lin = inv_log1p01_flux(preds_scaled, tau=float(tau), x_max=float(x_max)).astype(np.float64, copy=False)

            # 2) normalize each spectrum so ∫ F(λ) dλ = 1, robust to non-uniform/descending λ
            _wave = np.asarray(wave_ang, dtype=np.float64)
            _order = np.argsort(_wave, kind="mergesort")
            _inv_order = np.empty_like(_order); _inv_order[_order] = np.arange(_order.size)
            _wave_sorted = _wave[_order]

            F_sorted = flux_lin[:, _order]
            areas_pre = np.trapezoid(F_sorted, _wave_sorted, axis=1)
            safe = np.where(np.abs(areas_pre) > 1e-30, areas_pre, 1.0)  # guard 0-area rows
            F_sorted /= safe[:, None]
            flux_lin_norm = F_sorted[:, _inv_order]

            # final output for downstream (FITS, plots)
            preds_out = flux_lin_norm.astype(np.float32, copy=False)

    else:
        # no linear normalization in scaled mode; keep raw scaled output
        preds_out = preds_scaled
        areas_pre = None



    grid_bounds = {
        "mode": "csv" if args.csv_path is not None else "grid",
    }
    if args.csv_path is None:
        grid_bounds.update({
            "Teff":[args.teff_min,args.teff_max,args.teff_n],
            "logG":[args.logg_min,args.logg_max,args.logg_n],
            "logZ":[args.logz_min,args.logz_max,args.logz_n],
        })
    else:
        grid_bounds.update({"rows": int(M)})

    manifest = {
        "npz_path": str(args.npz_path),
        "model_path": str(args.model_path),
        "tau": float(tau), "x_max": float(x_max),
        "label_minmax": {"Teff":[float(Teff_min),float(Teff_max)],
                         "logG":[float(logG_min),float(logG_max)],
                         "logZ":[float(logZ_min),float(logZ_max)]},
        "grid_bounds": grid_bounds,
        "M": int(M), "L": int(L),
        "flags":{"linear_flux": bool(args.linear_flux)},
    }

    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.save_fits is None:
        args.save_fits = args.outdir / _auto_fits_name(args, M, L)

    # ---- Write FITS (ascending wavelength, labels table, optional compression) ----
    print("[INFO] Writing FITS…", flush=True)


    # 1) Ensure wavelength is strictly increasing; if not, flip both wave & flux
    wave = np.asarray(wave_ang, dtype=np.float32)
    if wave.size != preds_out.shape[1]:
        raise RuntimeError(f"WAVE length {wave.size} != FLUX columns {preds_out.shape[1]}")
    if wave[0] > wave[-1]:
        wave = wave[::-1].copy()
        preds_out = preds_out[:, ::-1].copy()
        print("[INFO] Wavelength axis was descending; flipped WAVE and FLUX columns.", flush=True)

    M, L = preds_out.shape

    # 2) Primary header with useful metadata
    hdr = fits.Header()
    hdr["N_SPECT"] = (int(M), "Number of spectra")
    hdr["N_WAVE"]  = (int(L), "Number of wavelength points")
    hdr["FLUXTYPE"] = ("linear" if args.linear_flux else "scaled", "Flux domain")
    hdr["WUNIT"]   = ("Angstrom", "Wavelength unit")
    hdr["BUNIT"]   = ("arb", "Flux unit (arbitrary unless calibrated)")
    hdr["TAU"]     = (float(tau), "log1p scaling tau (training meta)")
    hdr["X_MAX"]   = (float(x_max), "log1p scaling x_max (training meta)")
    phdu = fits.PrimaryHDU(header=hdr)



    if args.compress_fits:
        print("[INFO] Compressing (may take some time)", flush=True)
        # flux_hdu = fits.CompImageHDU(
        #     data=np.asarray(preds_out, dtype=np.float32, order="C"),
        #     name="FLUX",
        #     compression_type="GZIP_1",
        # )
        # L = preds_out.shape[1]
        # flux_hdu = fits.CompImageHDU(
        #     data=np.asarray(preds_out, dtype=np.float32, order="C"),
        #     name="FLUX",
        #     compression_type="GZIP_2",           # faster than GZIP_1
        #     tile_shape=(1, L),                   # one tile per spectrum row
        # )
        L = preds_out.shape[1]
        flux_hdu = fits.CompImageHDU(
            data=np.asarray(preds_out, dtype=np.float32, order="C"),
            name="FLUX",
            compression_type="RICE_1",
            tile_shape=(1, L),                   # big tiles = lower overhead
            quantize_level=16,                   # ~1/16 sigma step (tweak as needed)
        )

    else:
        flux_hdu = fits.ImageHDU(
            data=np.asarray(preds_out, dtype=np.float32, order="C"),
            name="FLUX",
        )
    # Helpful per-HDU cards
    flux_hdu.header["CTYPE1"] = "WAVE"
    flux_hdu.header["CUNIT1"] = "Angstrom"
    flux_hdu.header["CTYPE2"] = "SPECTRUM"
    flux_hdu.header["CUNIT2"] = "index"

    # 4) Wavelength vector as a simple image (1D). You can also compress this if you wish.
    wave_hdu = fits.ImageHDU(data=wave.astype(np.float32, copy=False), name="WAVE_ANGSTROM")
    wave_hdu.header["WFORM"]  = ("vector", "1D wavelength grid")

    # 5) Labels table: the exact Teff/logg/logZ per spectrum row
    #    'grid' already contains the labels for each output row, shape (M, 3).
    #    If you don’t have 'grid' in scope, build it the same way you did for inference.
    cols = [
        fits.Column(name="TEFF", format="E", array=grid[:, 0].astype(np.float32, copy=False)),
        fits.Column(name="LOGG", format="E", array=grid[:, 1].astype(np.float32, copy=False)),
        fits.Column(name="LOGZ", format="E", array=grid[:, 2].astype(np.float32, copy=False)),
    ]
    labels_hdu = fits.BinTableHDU.from_columns(cols, name="LABELS")
    labels_hdu.header["NROWS"] = (int(M), "One row per spectrum in FLUX")

    # 6) (Optional) Store original grid bounds when in grid mode
    if args.csv_path is None:
        labels_hdu.header["TMIN"] = float(args.teff_min)
        labels_hdu.header["TMAX"] = float(args.teff_max)
        labels_hdu.header["TN"]   = int(args.teff_n)
        labels_hdu.header["GMIN"] = float(args.logg_min)
        labels_hdu.header["GMAX"] = float(args.logg_max)
        labels_hdu.header["GN"]   = int(args.logg_n)
        labels_hdu.header["ZMIN"] = float(args.logz_min)
        labels_hdu.header["ZMAX"] = float(args.logz_max)
        labels_hdu.header["ZN"]   = int(args.logz_n)
    else:
        labels_hdu.header["CSV_ROWS"] = int(M)

    # 7) Write file with FITS checksums
    hdul = fits.HDUList([phdu, flux_hdu, wave_hdu, labels_hdu])
    hdul.writeto(args.save_fits, overwrite=True, checksum=True)
    print(f"[OK] FITS written: {args.save_fits}", flush=True)


    # ---- Quick summary plots (first & last, or single) — log-λ (x) and log-y (flux) from ORIGINAL SCALED OUTPUT ----
    with SectionTimer("Plots"):
        def _prep_wave_for_log_plot(wave_arr: np.ndarray) -> np.ndarray:
            w = np.asarray(wave_arr, dtype=np.float64)
            min_pos = np.nanmin(w[w > 0]) if np.any(w > 0) else None
            if min_pos is None:
                raise RuntimeError("All wavelengths are non-positive; cannot plot on a log axis.")
            return np.where(w > 0, w, min_pos * 1e-6)

        def _sorted_view(wave_arr: np.ndarray, flux_row: np.ndarray):
            w = _prep_wave_for_log_plot(wave_arr)
            order = np.argsort(w, kind="mergesort")
            return w[order], np.asarray(flux_row, dtype=np.float64)[order]

        def _plot_one(wave_arr, flux_row, title, out_path):
            w_plot, f_plot = _sorted_view(wave_arr, flux_row)

            # Guard: log-y needs strictly positive values
            min_pos = np.nanmin(f_plot[f_plot > 0]) if np.any(f_plot > 0) else 1e-12
            eps = max(min_pos * 1e-6, 1e-12)
            f_plot = np.where(f_plot > 0, f_plot, eps)

            fig = plt.figure(figsize=(10, 5))
            ax = plt.gca()
            ax.plot(w_plot, f_plot, lw=0.9)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("Wavelength (Å)")
            if args.linear_flux:
                ax.set_ylabel("Flux / Jy")
            else:
                ax.set_ylabel("log10(Flux / Jy)")
            ax.grid(True, which="both", alpha=0.35)
            ax.set_title(title)
            ax.set_xlim(w_plot.min(), w_plot.max())
            fig.tight_layout()
            fig.savefig(out_path, dpi=200)
            plt.close(fig)

        figdir = Path(args.save_fits).parent

        # Always make the "first" plot
        t0, g0, z0 = grid[0]
        first_png = figdir / (Path(args.save_fits).stem + "_first.png")
        single = (M == 1)
        first_title = (
            f"{'Single triplet' if single else 'First label'} — "
            f"Teff={t0:.0f} K, logg={g0:.2f}, logZ={z0:.2f} (log-λ, log-flux)"
        )
        _plot_one(wave_ang, preds_out[0], title=first_title, out_path=first_png)

        # Only create "last" if we have more than one spectrum
        last_png = None
        if M > 1:
            t1, g1, z1 = grid[-1]
            last_png = figdir / (Path(args.save_fits).stem + "_last.png")
            _plot_one(
                wave_ang, preds_out[-1],
                title=f"Last label — Teff={t1:.0f} K, logg={g1:.2f}, logZ={z1:.2f} (log-λ, log-flux)",
                out_path=last_png
            )

    if last_png is not None:
        print(f"[OK] Wrote summary plots: {first_png.name}, {last_png.name}", flush=True)
    else:
        print(f"[OK] Wrote summary plot: {first_png.name}", flush=True)


    print(f"[TIME] Total: {time.perf_counter()-t_start:.3f}s", flush=True)
    print("[OK] Inference complete.", flush=True)

    # freed = purge_gpu_memory()
    # if freed:
    #     print(f"[GPU] Freed GPU memory.")

    return {
        "fits": Path(args.save_fits),
        "first_png": Path(first_png),
        "last_png": (Path(last_png) if last_png is not None else None),
    }


# =========================
# Real-time log streamer + run manager
# =========================
class LiveLog:
    """File-like object that writes to both a memory buffer and a Queue for streaming."""
    def __init__(self):
        self.buf = io.StringIO()
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.total_len = 0
    def write(self, s: str):
        if not s: return 0
        with self.lock:
            self.buf.write(s)
            self.total_len += len(s)
        self.q.put(s)
        return len(s)
    def flush(self): pass
    def drain_chunks(self, max_chars: int = 20000) -> str:
        out = []
        size = 0
        try:
            while True:
                s = self.q.get_nowait()
                out.append(s)
                size += len(s)
                if size >= max_chars:
                    break
        except queue.Empty:
            pass
        return "".join(out)
    def snapshot(self) -> str:
        with self.lock:
            return self.buf.getvalue()

class RunManager:
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._done: bool = False
        self._failed: bool = False
        self._paths: dict | None = None
        self._log = LiveLog()
        self._started_at: Optional[float] = None
        self._ended_at: Optional[float] = None
        self._oom = False

    def start(self, cli_args: list[str]) -> bool:
        if self._running:
            return False
        # reset state
        self._oom = False
        self._log = LiveLog()
        self._running = True; self._done = False; self._failed = False
        self._paths = None; self._started_at = time.time(); self._ended_at = None

        def _target():
            def _is_cuda_oom(err: Exception) -> bool:
                try:
                    import torch
                    if isinstance(err, getattr(torch.cuda, "OutOfMemoryError", tuple())):
                        return True
                except Exception:
                    pass
                return "cuda out of memory" in str(err).lower()

            # locals we might want to drop explicitly
            _local_model = None
            _local_outputs = None
            try:
                with redirect_stdout(self._log), redirect_stderr(self._log):
                    print(f"[CMD] run_inference {' '.join(cli_args)}", flush=True)
                    try:
                        # run_inference should encapsulate model/tensors; if it returns any large
                        # objects you keep, assign them to _local_* so teardown can drop them.
                        p = run_inference(cli_args)
                        self._paths = p
                        self._done = True
                    except Exception as e:
                        if _is_cuda_oom(e):
                            try:
                                import torch
                                dev = torch.cuda.current_device() if torch.cuda.is_available() else None
                            except Exception:
                                dev = None
                            print("[ERROR] CUDA out of memory — the GPU is currently busy, sorry.\n\nPlease try again after some time.\n", flush=True)
                            # stats = purge_gpu_memory(dev)
                            # print(
                            #     f"[GPU] Freed ~{stats.get('freed_bytes',0)/1e6:.1f} MB "
                            #     f"(reserved {stats.get('before_reserved',0)/1e6:.1f}"
                            #     f"→{stats.get('after_reserved',0)/1e6:.1f} MB).",
                            #     flush=True
                            # )
                            self._oom = True
                            self._failed = True
                        else:
                            print(f"[ERROR] {e}", flush=True)
                            self._failed = True
            finally:
                # hard CUDA teardown always, regardless of success/failure
                with redirect_stdout(self._log), redirect_stderr(self._log):
                    print("[CLEANUP] Releasing CUDA memory…", flush=True)
                    stats2 = hard_cuda_teardown(_local_model, _local_outputs)
                    if stats2.get("before_reserved", 0) or stats2.get("after_reserved", 0):
                        print(
                            f"[CLEANUP] CUDA reserved {stats2['before_reserved']/1e6:.1f}"
                            f"→{stats2['after_reserved']/1e6:.1f} MB "
                            f"(freed ~{stats2['freed_bytes']/1e6:.1f} MB).",
                            flush=True
                        )
                    else:
                        print("[CLEANUP] Teardown complete.", flush=True)

                self._running = False
                self._ended_at = time.time()

        self._thread = threading.Thread(target=_target, daemon=True)
        self._thread.start()
        return True

    def status(self) -> str:
        if self._running: return "Running"
        if self._failed:  return "Failed"
        if self._done:    return "Done"
        return "Idle"

    def pop_log_chunks(self) -> str:
        return self._log.drain_chunks()

    def full_log(self) -> str:
        return self._log.snapshot()

    def ready(self) -> bool:
        return (self._paths is not None) and self._done and not self._failed

    def fits_path(self) -> Optional[Path]:
        return None if not self._paths else self._paths.get("fits")

    #def tar_path(self) -> Optional[Path]:
    #return None if not self._paths else self._paths.get("tar")

    def plot_paths(self) -> Tuple[Optional[Path], Optional[Path]]:
        if not self._paths: return (None, None)
        return self._paths.get("first_png"), self._paths.get("last_png")

RUN = RunManager()

# =========================
# Dash UI
# =========================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LITERA])
app.title = "Inference (Dash, live logs)"

server = app.server

@server.route("/static/sample_star_track.csv")
def download_sample_star_track():
    csv_path = Path(base_dir) / "sample_star_track.csv"
    return send_file(
        csv_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name="sample_star_track.csv",
    )

num_style = {"width": "100%"}
small_label_style = {"fontSize": "0.8rem", "marginBottom": "0.15rem"}
small_input_style = {
    "fontSize": "0.8rem",
    "height": "1.6rem",
    "padding": "0.1rem 0.3rem",
    "width": "100%",
}
small_header_style = {"fontSize": "0.9rem", "padding": "0.25rem 0.5rem"}
small_body_style = {"fontSize": "0.85rem", "padding": "0.5rem"}
model_selector = dbc.Card(
    dbc.CardBody(
        dbc.Row(
            [
                dbc.Col(
                    html.Div("Model / library", className="fw-bold me-3"),
                    md="auto",
                    align="center",
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="model-select",
                        options=[
                            {"label": MODEL_CONFIG[k]["label"], "value": k}
                            for k in MODEL_CONFIG.keys()
                        ],
                        value=DEFAULT_MODEL_KEY,
                        clearable=False,
                    ),
                    md=True,
                ),
            ],
            align="center",
            justify="start",
        )
    ),
    className="mb-2",
)

weights_card = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(html.Div("Model weights", className="fw-bold"), md="auto"),
                    dbc.Col(html.Div(id="weights-status-text", className="text-muted"), md=True),
                    dbc.Col(
                        dbc.Button("Open Zenodo DOI", id="open-zenodo", color="secondary", outline=True, size="sm"),
                        md="auto",
                    ),
                ],
                align="center",
            ),
            dbc.Progress(id="weights-progress", value=0, striped=True, animated=True, className="mt-2"),
            dcc.Interval(id="weights-tick", interval=500, n_intervals=0),
            dcc.Interval(id="startup-tick", interval=800, n_intervals=0, max_intervals=1),
        ]
    ),
    className="mb-2",
)

weights_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Download model weights?")),
        dbc.ModalBody(
            [
                html.Div(id="weights-modal-body"),
                html.Div(
                    [
                        html.Div("If you prefer manual download, use Zenodo DOI:", className="mt-2"),
                        html.Code(ZENODO_DOI),
                    ],
                    className="small text-muted",
                ),
            ]
        ),
        dbc.ModalFooter(
            [
                dbc.Button("Download", id="weights-confirm", color="primary"),
                dbc.Button("Cancel", id="weights-cancel", color="secondary", outline=True),
            ]
        ),
    ],
    id="weights-modal",
    is_open=False,
    backdrop="static",
)

mode_selector = dbc.Card(
    dbc.CardBody(
        dbc.Row(
            [
                dbc.Col(html.Div("Input mode", className="fw-bold me-3"), md="auto", align="center"),
                dbc.Col(
                    dbc.RadioItems(
                        id="mode-radio",
                        options=[
                            {"label": "Grid (ranges + samples)", "value": "grid"},
                            {"label": "CSV upload (explicit list)", "value": "csv"},
                            {"label": "Single triplet (Teff, logG, logZ)", "value": "single"},
                        ],
                        value="grid",
                        inline=True,
                    ),
                    md=True,
                ),
            ],
            align="center",
            justify="start",
        )
    ),
    className="mb-2",
)


single_card = dbc.Card(
    [
        dbc.CardHeader("Single triplet"),
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col([
                        dbc.Label("Teff (K)"),
                        dbc.Input(id="teff-one", type="number",
                                  value=5772.0, min=HARD["Teff"][0], max=HARD["Teff"][1], step=1),
                    ], md=4),
                    dbc.Col([
                        dbc.Label("logG"),
                        dbc.Input(id="logg-one", type="number",
                                  value=4.44, min=HARD["logG"][0], max=HARD["logG"][1], step=0.001),
                    ], md=4),
                    dbc.Col([
                        dbc.Label("logZ"),
                        dbc.Input(id="logz-one", type="number",
                                  value=0.0, min=HARD["logZ"][0], max=HARD["logZ"][1], step=0.001),
                    ], md=4),
                ],
                className="g-2",
            )
        ),
    ]
)


logs_card = dbc.Card(
    [
        dbc.CardHeader("Logs"),
        dbc.CardBody(
            dcc.Textarea(
                id="log-box",
                value="",
                readOnly=True,
                style={
                    "height": "25vh",        # adjust as you like
                    "width": "100%",
                    "overflowY": "auto",     # scrolling container
                    "whiteSpace": "pre",
                    "fontFamily": "monospace",
                    "fontSize": "12px",
                },
            )
        ),
    ]
)


card_T = dbc.Card(
    [
        dbc.CardHeader("Teff (K)", style=small_header_style),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Min", style=small_label_style),
                                dbc.Input(
                                    id="teff-min",
                                    type="number",
                                    value=2300.0,
                                    min=HARD["Teff"][0],
                                    max=HARD["Teff"][1],
                                    step=.01,
                                    style=small_input_style,
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Max", style=small_label_style),
                                dbc.Input(
                                    id="teff-max",
                                    type="number",
                                    value=12000.0,
                                    min=HARD["Teff"][0],
                                    max=HARD["Teff"][1],
                                    step=.01,
                                    style=small_input_style,
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("# Samples (>1 & ≤50)", style=small_label_style),
                                dbc.Input(
                                    id="teff-n",
                                    type="number",
                                    value=20,
                                    min=2,
                                    max=MAX_SAMPLES["Teff"],
                                    step=1,
                                    style=small_input_style,
                                ),
                            ],
                            md=4,
                        ),
                    ]
                ),
                html.Div(id="teff-preview", className="small text-muted mt-2"),
            ],
            style=small_body_style,
        ),
    ]
)

card_G = dbc.Card(
    [
        dbc.CardHeader("logG", style=small_header_style),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Min", style=small_label_style),
                                dbc.Input(
                                    id="logg-min",
                                    type="number",
                                    value=0.0,
                                    min=HARD["logG"][0],
                                    max=HARD["logG"][1],
                                    step=0.001,
                                    style=small_input_style,
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Max", style=small_label_style),
                                dbc.Input(
                                    id="logg-max",
                                    type="number",
                                    value=6.0,
                                    min=HARD["logG"][0],
                                    max=HARD["logG"][1],
                                    step=0.001,
                                    style=small_input_style,
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("# Samples (>1 & ≤50)", style=small_label_style),
                                dbc.Input(
                                    id="logg-n",
                                    type="number",
                                    value=15,
                                    min=2,
                                    max=MAX_SAMPLES["logG"],
                                    step=1,
                                    style=small_input_style,
                                ),
                            ],
                            md=4,
                        ),
                    ]
                ),
                html.Div(id="logg-preview", className="small text-muted mt-2"),
            ],
            style=small_body_style,
        ),
    ]
)

card_Z = dbc.Card(
    [
        dbc.CardHeader("logZ", style=small_header_style),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Min", style=small_label_style),
                                dbc.Input(
                                    id="logz-min",
                                    type="number",
                                    value=-4.0,
                                    min=HARD["logZ"][0],
                                    max=HARD["logZ"][1],
                                    step=0.001,
                                    style=small_input_style,
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Max", style=small_label_style),
                                dbc.Input(
                                    id="logz-max",
                                    type="number",
                                    value=1.0,
                                    min=HARD["logZ"][0],
                                    max=HARD["logZ"][1],
                                    step=0.001,
                                    style=small_input_style,
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("# Samples (>1 & ≤50)", style=small_label_style),
                                dbc.Input(
                                    id="logz-n",
                                    type="number",
                                    value=15,
                                    min=2,
                                    max=MAX_SAMPLES["logZ"],
                                    step=1,
                                    style=small_input_style,
                                ),
                            ],
                            md=4,
                        ),
                    ]
                ),
                html.Div(id="logz-preview", className="small text-muted mt-2"),
            ],
            style=small_body_style,
        ),
    ]
)

controls1 = dbc.Card([
    dbc.CardHeader("Upload CSV"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col(dcc.Upload(
                id="csv-upload",
                children=html.Div(["Drag & drop a CSV file"]),
                multiple=False,
                style={
                    "width": "100%", "height": "56px", "lineHeight": "56px",
                    "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "6px",
                    "textAlign": "center",
                },
            ), md=6),

            # New: explicit Browse button that opens the same picker
            dbc.Col(dbc.Button("Browse…", id="browse-csv", color="primary", outline=True), md="auto"),

            dbc.Col(html.Div(id="csv-preview", className="small text-muted"), md=True),
        ], className="g-2"),

        # harmless target for clientside side-effect callback
        dcc.Store(id="csv-browse-dummy"),
    ])
])



controls2 = dbc.Card([
    dbc.CardHeader("Run"),
    dbc.CardBody([

        dbc.Row([
            dbc.Col(dbc.Checklist(
                id="linear-flux",
                options=[{"label": "Generate FITs with linear flux", "value": "linear"}],
                value=["linear"],
                switch=True,
            ), md=2),

            dbc.Col(dbc.Checklist(
                id="compress-fits",
                options=[{"label": "Compress FITS (RICE_1)", "value": "compress"}],
                value=[],  # OFF by default
                switch=True,
            ), md=3),

            dbc.Col(dbc.Button("Show Parameter Grid", id="show-grid3d", color="secondary", outline=False), md="auto"),
            dbc.Col(dbc.Button("Start Inference", id="go", color="primary"), md="auto"),
            dbc.Col(dbc.Button("Clear Logs", id="clear-logs", color="secondary", outline=True), md="auto"),

            dbc.Col(
                html.A(
                    dbc.Button("Download FITS", id="download-fits", color="success", outline=False, disabled=True),
                    id="href-fits", href="", target="_self", style={"pointerEvents": "none"},
                ),
                width="auto",
            ),

            dbc.Col(dbc.Button("Download Log", id="download-log", color="secondary", outline=True), md="auto"),
            dbc.Col(html.Span(id="run-status", className="ms-3 fw-bold"), md=True),

            dcc.Download(id="dl-log"),
        ], className="g-2"),
        dcc.Interval(id="tick", interval=1500, n_intervals=0, disabled=True),
    ])
])


plots_card = dbc.Card([
    dbc.CardHeader("Summary Plots"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col(html.Img(id="plot-first", style={"width": "100%", "display": "none"}), md=6),
            dbc.Col(html.Img(id="plot-last",  style={"width": "100%", "display": "none"}), md=6),
        ]),
        html.Div(id="plot-note", className="small text-muted mt-2"),
    ])
])

csv_store = dcc.Store(id="csv-data")

grid3d_modal = dbc.Modal(
    [
        dbc.ModalBody(
            [
                html.Div(id="grid3d-stats", className="mb-2 text-muted"),
                dcc.Graph(
                    id="grid3d-graph",
                    style={"flex": "1 1 auto", "minHeight": 0},   # key
                    config={"responsive": True},
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "column",
                "height": "100vh",
                "paddingBottom": "0.5rem",
            },
        ),
        dbc.ModalFooter(
            dbc.Button("Close", id="grid3d-close", color="primary"),
            style={"flex": "0 0 auto"},
        ),
    ],
    id="grid3d-modal",
    is_open=False,
    fullscreen=True,
    scrollable=False,   # key: disable body scrolling
    backdrop="static",
)


csv_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("CSV Preview")),
        dbc.ModalBody([
            html.Div(id="csv-stats", className="mb-2 fw-semibold"),
            dbc.Row([
                dbc.Col(dbc.Input(id="csv-search", placeholder="Search (Teff, logG, logZ)…", type="text"), md=6),
            ], className="mb-2"),
            dash_table.DataTable(
                id="csv-table",
                page_size=15,
                sort_action="native",
                filter_action="none",  # <— avoid async-highlight.js
                style_table={"maxHeight": "60vh", "overflowY": "auto"},
                style_cell={"fontSize": 12, "padding": "6px"},
            ),
        ]),
        dbc.ModalFooter(dbc.Button("OK", id="csv-ok", color="primary", n_clicks=0)),
    ],
    id="csv-modal",
    size="xl",
    is_open=False,
    backdrop="static",
)

readme_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("SM-Net — Quick Start")),
        dbc.ModalBody(
            dbc.Stack(
                [
                    html.P(
                        "This is a local Dash interface for generating synthetic spectra with SM-Net. "
                        "Use the steps below to run inference and download outputs as FITS.",
                        className="mb-2",
                    ),

                    html.H6("Model weights (required)"),
                    html.Ul([
                        html.Li([
                            "Model weights are hosted on Zenodo (DOI: ",
                            html.Code("10.5281/zenodo.18883385"),
                            ")."
                        ]),
                        html.Li([
                            "On first launch, the app will automatically download the ",
                            html.B("default model"),
                            " weights (if not already present). Download progress is shown in the UI and console logs."
                        ]),
                        html.Li([
                            "When you select a different model, the app will check if its weights exist in ",
                            html.Code("models/"),
                            ". If missing, it will prompt you to download them."
                        ]),
                        html.Li([
                            "Manual option: download the relevant ",
                            html.Code(".pt"),
                            " file(s) from Zenodo and place them in ",
                            html.Code("models/"),
                            "."
                        ]),
                    ]),

                    html.H6("Model / Library", className="mt-2"),
                    html.Ul([
                        html.Li([
                            "Choose a model/library based on your coverage requirements. "
                            "The combined grids provide wider parameter coverage."
                        ]),
                        html.Li([
                            html.B("Husser + C3K + TMAP + OB (combined grid)"),
                            " provides the broadest coverage across (Teff, logG, logZ) for most use cases."
                        ]),
                    ]),

                    html.H6("Input modes", className="mt-2"),
                    html.Ul([
                        html.Li([
                            "Default is ", html.B("Grid mode"),
                            ": choose ranges and number of samples for Teff, logG, logZ."
                        ]),
                        html.Li([
                            "For ", html.B("CSV mode"),
                            ": toggle the switch at the top to the right, then upload a CSV with columns ",
                            html.Code("teff, logg, logz"),
                            " (case-insensitive). A template CSV is available ",
                            html.A("here", href="/static/sample_star_track.csv", target="_blank"),
                            ".",
                        ]),
                    ]),

                    html.H6("Limits & validation"),
                    html.Ul([
                        html.Li(
                            "Parameter limits are taken from the selected model/library metadata. "
                            "Out-of-range values are rejected and reported under the previews."
                        ),
                        html.Li("# samples per axis: 2–50. Errors will show under the previews."),
                    ]),

                    html.H6("Running inference"),
                    html.Ul([
                        html.Li("Click “Start Inference”. The live log will stream below."),
                        html.Li("Two preview plots (first/last label) appear when ready."),
                    ]),

                    html.H6("Downloads & cleanup"),
                    html.Ul([
                        html.Li("When complete, “Download FITS” activates."),
                        html.Li("Large downloads are streamed; progress is handled by your browser."),
                        html.Li(
                            "Artifacts (FITS + preview PNGs) may be auto-deleted after a timeout; "
                            "completed downloads may be deleted immediately after download."
                        ),
                    ]),

                    html.H6("CSV tips"),
                    html.Ul([
                        html.Li("Header can be any case: teff/logg/logz; otherwise the parser scans triples per line."),
                        html.Li("Duplicates are dropped automatically."),
                    ]),

                    html.H6("Show Parameter Grid"),
                    html.Ul([
                        html.Li([
                            "Click ", html.B("Show Parameter Grid"),
                            " to open a popup with a 3D scatter of (Teff, logG, logZ)."
                        ]),
                        html.Li([
                            "In ", html.B("Grid mode"),
                            " it visualises the current ranges; in ", html.B("CSV mode"),
                            " it visualises the uploaded rows (if valid)."
                        ]),
                    ]),

                    html.H6("Project links"),
                    html.Ul([
                        html.Li([
                            "Code repository: ",
                            html.A("https://github.com/ICRAR/SM_Net", href="https://github.com/ICRAR/SM_Net", target="_blank"),
                        ]),
                        html.Li([
                            "Weights (Zenodo): ",
                            html.A("https://doi.org/10.5281/zenodo.18883385", href="https://doi.org/10.5281/zenodo.18883385", target="_blank"),
                        ]),
                    ]),

                    html.H6("Logs"),
                    html.Ul([
                        html.Li("Logs show progress and errors (including download issues)."),
                        html.Li("Use “Download Log” to save a full snapshot."),
                    ]),

                    html.Hr(),
                    html.P(
                        "If something fails, check the logs first. Missing weights, out-of-range parameters, "
                        "and oversized grids are the most common issues."
                    ),
                    html.P("Email: omar.anwar@uwa.edu.au"),
                ],
                gap=2,
            )
        ),
        dbc.ModalFooter(
            dbc.Button("Close", id="readme-close", color="primary", n_clicks=0)
        ),
    ],
    id="readme-modal",
    is_open=False,
    size="lg",
    scrollable=True,
    backdrop="static",
)

app.layout = dbc.Container([
    dbc.Row(
        [
            dbc.Col(
                html.H2("SM-Net: A Learned Continuous Spectral Manifold from Multiple Stellar Libraries"),
                md=True,
            ),
            dbc.Col(
                dbc.Button("Read me", id="readme-btn", size="sm", color="info", outline=True),
                md="auto", className="text-end", align="center",
            ),
        ],
        align="center",
    ),

    # NEW: model selector card
    dbc.Row([dbc.Col(model_selector, lg=12)], className="g-3"),

    # Mode selector card
    dbc.Row([dbc.Col(mode_selector, lg=12)], className="g-3"),

    dbc.Row([dbc.Col(weights_card, lg=12)], className="g-3"),

    weights_modal,

    dcc.Input(id="input-mode", value="grid", type="text", style={"display": "none"}),

    # Grid controls (unchanged)
    html.Div(
        id="grid-cards-row",
        children=[dbc.Row([dbc.Col(card_T, lg=4), dbc.Col(card_G, lg=4), dbc.Col(card_Z, lg=4)], className="g-3")],
    ),

    # CSV controls (unchanged)
    html.Div(
        id="csv-card-row",
        children=[dbc.Row([dbc.Col(controls1, lg=12)], className="g-3")],
        style={"display": "none"},
    ),

    # NEW: Single-triplet controls
    html.Div(
        id="single-card-row",
        children=[dbc.Row([dbc.Col(single_card, lg=12)], className="g-3")],
        style={"display": "none"},
    ),


    dbc.Row([dbc.Col(controls2, lg=12)], className="g-3"),
    dbc.Row([dbc.Col(plots_card, lg=12)], className="g-3"),
    dbc.Row([dbc.Col(logs_card, lg=12)], className="g-3"),

    dcc.Store(id="log-autoscroll-sentinel"),
    dcc.Store(id="csv-grid-store"),
    dcc.Store(id="csv-grid-json-path"),
    csv_store,
    grid3d_modal,
    csv_modal,
    readme_modal,
], fluid=True)




# =========================
# Callbacks
# =========================
# Unified driver: start, stream logs, enable/disable polling, update plots & download button

app.clientside_callback(
    """
    function(n){
        if(!n){ return window.dash_clientside.no_update; }
        window.open("https://doi.org/" + %s, "_blank");
        return null;
    }
    """ % json.dumps(ZENODO_DOI),
    Output("open-zenodo", "n_clicks"),
    Input("open-zenodo", "n_clicks"),
    prevent_initial_call=True,
    )

app.clientside_callback(
    """
    function(logValue) {
        // Run after the textarea has received new value
        const el = document.getElementById('log-box');
        if (!el) { return window.dash_clientside.no_update; }

        // Are we already near the bottom? (within 80px)
        const nearBottom = (el.scrollTop + el.clientHeight) >= (el.scrollHeight - 80);

        // If user hasn't scrolled up far, keep it pinned to bottom
        if (nearBottom) {
            el.scrollTop = el.scrollHeight;
        }
        // Touch a harmless store so Dash considers this a valid callback
        return Date.now();
    }
    """,
    dash.Output("log-autoscroll-sentinel", "data"),
    dash.Input("log-box", "value"),
)

@app.callback(
    Output("weights-status-text", "children"),
    Output("weights-progress", "value"),
    Output("weights-progress", "animated"),
    Output("weights-progress", "striped"),
    Input("weights-tick", "n_intervals"),
)
def update_weights_status(_n):
    s = WEIGHTS.snapshot()
    if s["status"] == "idle":
        return "Idle.", 0, False, False
    if s["status"] == "downloading":
        done = s["bytes_done"]; total = s["bytes_total"]
        if total > 0:
            txt = f"{s['msg']}  ({done/1e6:.1f}/{total/1e6:.1f} MB)"
        else:
            txt = f"{s['msg']}  ({done/1e6:.1f} MB)"
        return txt, s["progress"], True, True
    if s["status"] == "done":
        return s["msg"], 100, False, False
    # failed
    return s["msg"], 0, False, False


@app.callback(
    Output("weights-modal", "is_open"),
    Output("weights-modal-body", "children"),
    Input("startup-tick", "n_intervals"),
    State("weights-modal", "is_open"),
    prevent_initial_call=True,
)
def startup_auto_download(_n, is_open):
    # Auto-start download of DEFAULT model on first run IF URL is configured and file missing.
    mk = DEFAULT_MODEL_KEY
    if _weights_present(mk):
        return False, ""
    url = weights_url_for(mk)
    if not url:
        # No URL yet: show a friendly note in UI, but do not block startup.
        return True, (
            f"Default weights are missing: {_expected_weight_path(mk).name}. "
            f"Automatic download is not configured yet (URL not set)."
        )
    # start download immediately; no prompt for default
    WEIGHTS.start(mk, url)
    return False, ""

@app.callback(
    Output("weights-modal", "is_open", allow_duplicate=True),
    Output("weights-modal-body", "children", allow_duplicate=True),
    Input("model-select", "value"),
    State("weights-modal", "is_open"),
    prevent_initial_call=True,
)
def on_model_change_prompt(selected_model, is_open):
    mk = selected_model or DEFAULT_MODEL_KEY
    if _weights_present(mk):
        return False, ""
    url = weights_url_for(mk)
    label = MODEL_CONFIG.get(mk, {}).get("label", mk)
    fname = _expected_weight_path(mk).name

    if not url:
        return True, f"Weights for '{label}' are not present ({fname}). Automatic download is not configured yet."
    return True, f"Weights for '{label}' are missing ({fname}). Download now?"

@app.callback(
    Output("weights-modal", "is_open", allow_duplicate=True),
    Input("weights-confirm", "n_clicks"),
    Input("weights-cancel", "n_clicks"),
    State("model-select", "value"),
    prevent_initial_call=True,
)
def handle_weights_modal(n_yes, n_no, selected_model):
    trig = ctx.triggered_id
    if trig == "weights-cancel":
        return False
    mk = selected_model or DEFAULT_MODEL_KEY
    if _weights_present(mk):
        return False
    url = weights_url_for(mk)
    if url:
        WEIGHTS.start(mk, url)
    return False

@app.callback(
    Output("grid3d-modal", "is_open"),
    Output("grid3d-graph", "figure"),
    Output("grid3d-stats", "children"),
    Input("show-grid3d", "n_clicks"),
    Input("grid3d-close", "n_clicks"),
    State("grid3d-modal", "is_open"),
    State("input-mode", "value"),
    # grid-mode controls
    State("teff-min", "value"), State("teff-max", "value"), State("teff-n", "value"),
    State("logg-min", "value"), State("logg-max", "value"), State("logg-n", "value"),
    State("logz-min", "value"), State("logz-max", "value"), State("logz-n", "value"),
    # csv-mode store
    State("csv-grid-store", "data"),
    # NEW: single-mode controls
    State("teff-one", "value"), State("logg-one", "value"), State("logz-one", "value"),
    State("model-select", "value"),
    prevent_initial_call=True,
)
def show_grid3d(n_open, n_close, is_open, input_mode,
                tmin, tmax, tn, gmin, gmax, gn, zmin, zmax, zn,
                csv_grid_store,
                teff_one, logg_one, logz_one, selected_model):

    trig = ctx.triggered_id

    # If the Close button fired, just close the modal and leave figure/stats alone
    if trig == "grid3d-close":
        return False, no_update, no_update

    # Otherwise we were opened (or re-opened) via Show Parameter Grid
    ensure_limits_for_model(selected_model or DEFAULT_MODEL_KEY)
    mode = (input_mode or "grid").strip().lower()

    if mode == "grid":
        try:
            tmin = float(tmin); tmax = float(tmax); tn = int(tn)
            gmin = float(gmin); gmax = float(gmax); gn = int(gn)
            zmin = float(zmin); zmax = float(zmax); zn = int(zn)
        except Exception:
            return True, go.Figure(), "⚠️ Invalid numeric inputs."

        errs = _validate_params(tmin, tmax, tn, gmin, gmax, gn, zmin, zmax, zn)
        if errs:
            return True, go.Figure(), "🚫 " + "; ".join(errs)

        grid_np = build_grid(tmin, tmax, tn, gmin, gmax, gn, zmin, zmax, zn)
        title = f"3D Grid — linspace ({tn}×{gn}×{zn} = {grid_np.shape[0]} points)"

    elif mode == "csv":
        if not csv_grid_store or not csv_grid_store.get("rows"):
            return True, go.Figure(), "🚫 CSV mode selected but no valid CSV is loaded."

        # Refuse to use CSV from a different model/library
        csv_model = csv_grid_store.get("model_key")
        current_model = (selected_model or DEFAULT_MODEL_KEY)
        if csv_model is not None and csv_model != current_model:
            return (
                True,
                go.Figure(),
                "🚫 CSV was uploaded for a different model/library. "
                "Please re-upload the CSV for the current selection.",
            )

        grid_np = np.asarray(csv_grid_store["rows"], dtype=float)
        errs = _validate_csv_grid(grid_np)
        if errs:
            return True, go.Figure(), "🚫 " + "; ".join(errs)
        fname = csv_grid_store.get("filename") or "uploaded.csv"
        title = f"3D Grid — CSV ({grid_np.shape[0]} rows) • {fname}"


    else:  # single
        try:
            t = float(teff_one); g = float(logg_one); z = float(logz_one)
        except Exception:
            return True, go.Figure(), "⚠️ Invalid single triplet values."
        grid_np = np.array([[t, g, z]], dtype=float)
        errs = _validate_csv_grid(grid_np)
        if errs:
            return True, go.Figure(), "🚫 " + "; ".join(errs)
        title = "3D Grid — Single triplet (1 point)"

    fig, n_total, n_unique = _make_grid3d_figure(
        grid_np,
        title=title,
        model_key=(selected_model or DEFAULT_MODEL_KEY),
    )

    stats = f"{n_total} unique triples."
    return True, fig, stats





@app.callback(
    Output("input-mode", "value"),
    Output("grid-cards-row", "style"),
    Output("csv-card-row", "style"),
    Output("single-card-row", "style"),
    Input("mode-radio", "value"),
)
def choose_mode(val):
    mode = (val or "grid").strip().lower()
    show = {}
    hide = {"display": "none"}
    return (
        mode,
        (show if mode == "grid" else hide),
        (show if mode == "csv" else hide),
        (show if mode == "single" else hide),
    )



app.clientside_callback(
    """
    function(n_clicks){
        if(!n_clicks){ return window.dash_clientside.no_update; }
        const host = document.getElementById('csv-upload');
        if(!host){ return window.dash_clientside.no_update; }
        const input = host.querySelector('input[type="file"]');
        if(input){ input.click(); }
        return Date.now();  // touch a store so Dash considers this a valid callback
    }
    """,
    dash.Output("csv-browse-dummy", "data"),
    dash.Input("browse-csv", "n_clicks"),
    prevent_initial_call=True,
)

@app.callback(
    dash.Output("readme-modal", "is_open"),
    dash.Input("readme-btn", "n_clicks"),
    dash.Input("readme-close", "n_clicks"),
    dash.State("readme-modal", "is_open"),
)
def toggle_readme(n_open, n_close, is_open):
    if (n_open or 0) > 0 or (n_close or 0) > 0:
        return not (is_open or False)
    return is_open

@app.callback(
    Output("csv-preview", "children"),
    Input("csv-grid-store", "data"),
    State("model-select", "value"),
)
def tiny_preview(store, selected_model):
    if not store or not store.get("rows"):
        return "No CSV loaded."

    selected_model = (selected_model or DEFAULT_MODEL_KEY)

    # If CSV was uploaded under a different model, mark it as stale
    csv_model = store.get("model_key")
    if csv_model is not None and csv_model != selected_model:
        return (
            "CSV was uploaded for a different model/library. "
            "Please re-upload the CSV for the current selection."
        )

    # Normal path
    ensure_limits_for_model(selected_model)
    n = int(store.get("n", len(store["rows"])))
    dropped = int(store.get("dropped", 0))
    head = store["rows"][:3]
    head_txt = "; ".join(f"({t:.4g},{g:.4g},{z:.4g})" for t, g, z in head)
    msg = f"CSV ready • {n} row(s) kept • first 3: {head_txt}"
    if dropped > 0:
        msg += f" • {dropped} out-of-range row(s) removed"
    return msg


@app.callback(
    Output("csv-grid-store", "data"),
    Input("csv-upload", "contents"),
    Input("model-select", "value"),
    State("csv-upload", "filename"),
    prevent_initial_call=True,
)
def parse_csv(contents, selected_model, fname):
    """
    Parse the uploaded CSV and apply model-specific bounds.

    This callback now fires when EITHER:
      - the CSV contents change, OR
      - the model/library selection changes.

    On a model change, if a CSV is already selected (contents not None),
    we re-run the full parsing + bounds check with the new limits.
    """
    if not contents:
        # No CSV selected yet → nothing to do (for both upload + model-change)
        return None

    selected_model = (selected_model or DEFAULT_MODEL_KEY)
    ensure_limits_for_model(selected_model)

    try:
        b64 = contents.split(",", 1)[1]
        raw = base64.b64decode(b64)
    except Exception:
        return None

    grid = _parse_csv_bytes(raw)  # may return DF or ndarray

    # --- Coerce to ndarray (M,3) and dedupe ---
    if isinstance(grid, pd.DataFrame):
        lower = {c.lower(): c for c in grid.columns}
        if all(k in lower for k in ("teff", "logg", "logz")):
            grid = grid[[lower["teff"], lower["logg"], lower["logz"]]]
        grid = grid.drop_duplicates().to_numpy(dtype=float)
    else:
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 2 or grid.shape[1] != 3:
            return None
        grid = np.unique(grid, axis=0)

    # --- Split into kept vs dropped by hard ranges (using new model's limits) ---
    mask = _inrange_mask(grid)
    kept = grid[mask]
    dropped = grid[~mask]

    head_dropped = dropped[:5].tolist() if dropped.size else []
    return {
        "rows": kept.tolist(),
        "n": int(len(kept)),
        "filename": fname,
        "dropped": int(len(dropped)),
        "dropped_head": head_dropped,
    }



@app.callback(
    Output("csv-table", "data"),
    Output("csv-table", "columns"),
    Output("csv-stats", "children"),
    Output("csv-modal", "is_open"),
    Output("csv-data", "data"),
    Input("csv-upload", "contents"),
    Input("csv-ok", "n_clicks"),
    Input("csv-search", "value"),
    Input("model-select", "value"),
    State("csv-upload", "filename"),
    State("csv-modal", "is_open"),
    State("csv-data", "data"),
    prevent_initial_call=True,
)
def handle_csv_modal(contents, n_ok, q, selected_model, filename, is_open, store_rows):
    trig = ctx.triggered_id

    # If the model / library changed, clear the modal + stored rows
    if trig == "model-select":
        return [], [], "No CSV loaded.", False, None

    # Make sure HARD ranges match the current model
    ensure_limits_for_model(selected_model or DEFAULT_MODEL_KEY)

    # 1) OK → close modal
    if trig == "csv-ok":
        if n_ok:
            return no_update, no_update, no_update, False, no_update
        return no_update, no_update, no_update, is_open, no_update

    # 2) New upload → parse, filter, show modal
    if trig == "csv-upload":
        if not contents:
            return no_update, no_update, no_update, is_open, no_update

        try:
            b64 = contents.split(",", 1)[1]
            raw = base64.b64decode(b64)
        except Exception:
            return no_update, no_update, no_update, is_open, no_update

        # Parse to DF first (so the table has nice numeric columns)
        parsed = _parse_csv_bytes(raw)
        if isinstance(parsed, pd.DataFrame):
            df_all = parsed.drop_duplicates().reset_index(drop=True)
        else:
            df_all = pd.DataFrame(
                np.unique(np.asarray(parsed, dtype=float), axis=0),
                columns=["Teff", "logG", "logZ"],
            )

        # Apply the same in-range filter used for inference
        mask = _inrange_mask(df_all.to_numpy(dtype=float))
        df_kept = df_all.loc[mask].reset_index(drop=True)
        df_drop = df_all.loc[~mask].reset_index(drop=True)

        cols = [{"name": c, "id": c, "type": "numeric"} for c in df_kept.columns]
        data = df_kept.to_dict("records")

        kept_n = len(df_kept)
        drop_n = len(df_drop)
        fname = filename or "uploaded file"

        # Clear, friendly stats (and show a short sample of removed rows if any)
        if drop_n > 0:
            sample_txt = "; ".join(
                f"({t:.4g},{g:.4g},{z:.4g})"
                for t, g, z in df_drop.to_numpy(dtype=float)[:5]
            )
            stats = (
                f"Loaded {kept_n} rows from {fname}. "
                f"Removed {drop_n} out-of-range row(s). Sample removed: {sample_txt}"
            )
        else:
            stats = f"Loaded {kept_n} rows from {fname}. No rows were out of range."

        # Also push a minimal store for the search callback in this modal
        store = data
        return data, cols, stats, True, store

    # 3) Search → filter existing store_rows server-side
    if trig == "csv-search":
        if not store_rows:
            return no_update, no_update, no_update, no_update, no_update
        q = (q or "").strip().lower()
        if not q:
            return store_rows, no_update, no_update, no_update, no_update

        def _txt(r):
            if isinstance(r, dict):
                fields = [r.get("Teff"), r.get("logG"), r.get("logZ")]
            else:
                fields = r
            return " ".join(str(x) for x in fields).lower()

        filtered = [r for r in store_rows if q in _txt(r)]
        return filtered, no_update, no_update, no_update, no_update

    # Fallback
    return no_update, no_update, no_update, is_open, no_update



@app.callback(
    Output("run-status", "children"),
    Output("go", "disabled"),
    Output("log-box", "value"),
    Output("download-fits", "disabled"),
    Output("tick", "disabled"),
    Output("plot-first", "src"),
    Output("plot-last", "src"),
    Output("plot-first", "style"),
    Output("plot-last", "style"),
    Output("plot-note", "children"),
    Output("href-fits", "href"),
    Output("href-fits", "target"),
    Output("href-fits", "style"),
    Input("go", "n_clicks"),
    Input("tick", "n_intervals"),
    Input("clear-logs", "n_clicks"),
    State("teff-min", "value"), State("teff-max", "value"), State("teff-n", "value"),
    State("logg-min", "value"), State("logg-max", "value"), State("logg-n", "value"),
    State("logz-min", "value"), State("logz-max", "value"), State("logz-n", "value"),
    State("linear-flux", "value"),
    State("compress-fits", "value"),
    State("log-box", "value"),
    State("input-mode", "value"),
    State("csv-grid-store", "data"),
    State("teff-one", "value"),
    State("logg-one", "value"),
    State("logz-one", "value"),
    State("model-select", "value"),
    prevent_initial_call=True,
)
def driver(n_go, n_tick, n_clear,
           tmin, tmax, tn, gmin, gmax, gn, zmin, zmax, zn,
           linear_flags,
           compress_flags,
           current_log,
           input_mode,
           csv_grid_store,
           teff_one, logg_one, logz_one,
           selected_model):

    mode = (input_mode or "grid").strip().lower()
    is_csv_mode = (mode == "csv")
    is_single_mode = (mode == "single")

    ctx = dash.callback_context
    trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

    # Default link state: inert (prevents accidental nav while nothing ready)
    href_target = "_self"
    href_style = {"pointerEvents": "none"}

    # --- Clear Logs: DO NOT touch download state, href, or plots ---
    if trig == "clear-logs":
        return (
            no_update,       # run-status
            no_update,       # go.disabled
            "",              # log-box.value (clear)
            no_update,       # download-fits.disabled
            no_update,       # tick.disabled
            no_update,       # plot-first.src
            no_update,       # plot-last.src
            no_update,       # plot-first.style
            no_update,       # plot-last.style
            no_update,       # plot-note.children
            no_update,       # href-fits.href
            no_update,       # href-fits.target
            no_update,       # href-fits.style
        )

    # --- Baseline UI state (used in all other branches) ---
    status_out = "Failed – GPU busy" if getattr(RUN, "_oom", False) else RUN.status()
    go_disabled = (status_out == "Running")
    log_out = current_log or ""
    fits_dl_disabled = not RUN.ready() or (RUN.fits_path() is None)
    tick_disabled = status_out != "Running"  # poll only while running
    plot_first_src = no_update
    plot_last_src  = no_update
    plot_first_style = {"width": "100%", "display": "none"}
    plot_last_style  = {"width": "100%", "display": "none"}
    plot_note = ""
    fits_href = ""

    # --- Start run ---
    if trig == "go":
        freed = purge_gpu_memory()
        if freed:
            print(f"[GPU] Purged: {freed}")

        model_key = selected_model or DEFAULT_MODEL_KEY
        meta = ensure_limits_for_model(model_key)
        cfg = MODEL_CONFIG.get(model_key, MODEL_CONFIG[DEFAULT_MODEL_KEY])
        model_path = model_dir / cfg["model"]
        npz_path = meta_dir / cfg["npz"]
        meta_cache_path = meta_dir / cfg["meta"]

        # ---- Weights presence guard ----
        if not model_path.exists():
            # If URL is known, prompt user; otherwise instruct manual download
            url = weights_url_for(model_key)
            if url:
                # Start download automatically for default; otherwise open modal handled by model-select
                WEIGHTS.start(model_key, url) if (model_key == DEFAULT_MODEL_KEY) else None
                return (
                    f"Weights missing for '{MODEL_CONFIG[model_key]['label']}'. Download in progress…",
                    False, log_out,
                    True, True,
                    no_update, no_update,
                    plot_first_style, plot_last_style,
                    "", "",
                    href_target, href_style
                )
            else:
                return (
                    f"Missing weights file: {model_path.name}. "
                    f"Download from Zenodo (DOI {ZENODO_DOI}) and place it in {model_dir}/",
                    False, log_out,
                    True, True,
                    no_update, no_update,
                    plot_first_style, plot_last_style,
                    "", "",
                    href_target, href_style
                )
        # # Normalize mode
        # mode = (input_mode or "grid").strip().lower()
        # is_csv_mode = (mode == "csv")

        # Parse numeric inputs (still needed for grid mode; harmless for csv mode)
        try:
            tmin = float(tmin); tmax = float(tmax); tn = int(tn)
            gmin = float(gmin); gmax = float(gmax); gn = int(gn)
            zmin = float(zmin); zmax = float(zmax); zn = int(zn)
        except Exception:
            # If CSV mode, we can still proceed so long as CSV is valid; otherwise show error
            if not is_csv_mode:
                return (
                    "Invalid inputs.", False, log_out,
                    True, True,
                    no_update, no_update,
                    plot_first_style, plot_last_style,
                    "", "",
                    href_target, href_style
                )

        # Validate grid params only for grid mode
        if not is_csv_mode:
            errs = _validate_params(tmin, tmax, tn, gmin, gmax, gn, zmin, zmax, zn)
            if errs:
                return (
                    "🚫 " + "; ".join(errs), False, log_out,
                    True, True,
                    no_update, no_update,
                    plot_first_style, plot_last_style,
                    "", "",
                    href_target, href_style,
                )

        # Plan output filename
        if is_csv_mode:
            planned_fits = OUTPUT_DIR / (
                f"dash_phx_customM_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}.fits"
            )
        elif is_single_mode:
            planned_fits = OUTPUT_DIR / (
                f"dash_phx_single_T{teff_one:g}_g{logg_one:g}_Z{logz_one:g}_"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}.fits"
            )
        else:
            params = {
                "teff_min": tmin, "teff_max": tmax, "teff_n": tn,
                "logg_min": gmin, "logg_max": gmax, "logg_n": gn,
                "logz_min": zmin, "logz_max": zmax, "logz_n": zn
            }
            planned_fits = _unique_outname(params)

        # Base CLI args (common)
        cli_args = [
            "--save-fits", str(planned_fits),
            "--outdir", str(planned_fits.parent),
            "--model-path", str(model_path),
            "--npz-path", str(npz_path),
            "--meta-cache", str(meta_cache_path),
            "--precision", "fp32",
        ]

        if "linear" in (linear_flags or []):
            cli_args.append("--linear-flux")
        else:
            cli_args.append("--no-linear-flux")

        if "compress" in (compress_flags or []):
            cli_args.append("--compress-fits")
        else:
            cli_args.append("--no-compress-fits")

        # Supply inputs based on mode
        if is_csv_mode:
            # Must have parsed CSV rows
            if not csv_grid_store or not csv_grid_store.get("rows"):
                return (
                    "🚫 CSV mode selected but no valid CSV uploaded.",
                    False, log_out,
                    True, True,
                    no_update, no_update,
                    plot_first_style, plot_last_style,
                    "", "",
                    href_target, href_style
                )

            grid_np = np.asarray(csv_grid_store["rows"], dtype=np.float32)
            errs = _validate_csv_grid(grid_np)
            if errs:
                return (
                    "🚫 " + "; ".join(errs),
                    False, log_out,
                    True, True,
                    no_update, no_update,
                    plot_first_style, plot_last_style,
                    "", "",
                    href_target, href_style
                )

            # Write temp JSON for --csv-path
            tmp_json = OUTPUT_DIR / f"grid_{int(time.time())}_{uuid.uuid4().hex[:6]}.json"
            try:
                tmp_json.write_text(json.dumps(csv_grid_store["rows"]))
            except Exception as e:
                return (
                    f"🚫 Failed to write CSV grid JSON: {e}",
                    False, log_out,
                    True, True,
                    no_update, no_update,
                    plot_first_style, plot_last_style,
                    "", "",
                    href_target, href_style
                )
            cli_args.extend(["--csv-path", str(tmp_json)])
        elif is_single_mode:
            # Validate and write a one-row JSON, then use --csv-path
            try:
                t = float(teff_one); g = float(logg_one); z = float(logz_one)
            except Exception:
                return ("🚫 Invalid single triplet.", False, log_out, True, True,
                        no_update, no_update, {"width":"100%","display":"none"}, {"width":"100%","display":"none"},
                        "", "", href_target, href_style)

            grid_np = np.array([[t, g, z]], dtype=np.float32)
            errs = _validate_csv_grid(grid_np)
            if errs:
                return ("🚫 " + "; ".join(errs), False, log_out, True, True,
                        no_update, no_update, {"width":"100%","display":"none"}, {"width":"100%","display":"none"},
                        "", "", href_target, href_style)

            tmp_json = OUTPUT_DIR / f"grid_{int(time.time())}_{uuid.uuid4().hex[:6]}.json"
            tmp_json.write_text(json.dumps(grid_np.tolist()))
            cli_args.extend(["--csv-path", str(tmp_json)])
        else:
            # Standard linspace/grid mode
            cli_args.extend([
                "--teff-min", f"{tmin}", "--teff-max", f"{tmax}", "--teff-n", f"{tn}",
                "--logg-min", f"{gmin}", "--logg-max", f"{gmax}", "--logg-n", f"{gn}",
                "--logz-min", f"{zmin}", "--logz-max", f"{zmax}", "--logz-n", f"{zn}",
            ])

        started = RUN.start(cli_args)
        status_out = "Running…" if started else "Already running."
        go_disabled = True if started else go_disabled
        tick_disabled = not started

        return (
            status_out, go_disabled, "",
            True, tick_disabled,
            no_update, no_update,
            plot_first_style, plot_last_style,
            "", "",
            href_target, href_style
        )

    # --- Tick: stream logs + update status; stop polling when finished; push plots & href when ready ---
    if trig == "tick":
        new_chunks = RUN.pop_log_chunks()
        if new_chunks:
            log_out = (log_out or "") + new_chunks

        status_out = RUN.status()
        go_disabled = (status_out == "Running")
        tick_disabled = status_out != "Running"

        if RUN.ready():
            # Enable/disable download
            fits_path = RUN.fits_path()
            fits_dl_disabled = fits_path is None

            # Register token and set href
            if fits_path and fits_path.exists():
                token_fits = register_download(fits_path, kind="fits")
                fits_href = f"/api/download/{token_fits}"
                href_target = "_blank"
                href_style = {}  # enable pointer events

                cleanup_note = (
                    f"[CLEANUP] Generated files will be auto-deleted ~{DOWNLOAD_DELETE_AFTER_S}s "
                    f"after creation/token expiry."
                )
                if DELETE_ONCE_DOWNLOADED:
                    cleanup_note += " Completed downloads are deleted immediately."
                log_out = (log_out or "") + cleanup_note + "\n"

            # Plots
            p1, p2 = RUN.plot_paths()
            has_first = bool(p1 and p1.exists())
            has_last  = bool(p2 and p2.exists())

            if has_first:
                plot_first_src = _png_to_data_url(p1)
                plot_first_style = {"width": "100%", "display": "block"}
            if has_last:
                plot_last_src = _png_to_data_url(p2)
                plot_last_style = {"width": "100%", "display": "block"}

            # Dynamic note
            if has_first and not has_last:
                plot_note = "Single-triplet plot shown (log–log: x = wavelength, y = flux)."
            else:
                plot_note = "Plots are generated from the first and last label (log–log: x = wavelength, y = flux)."



        return (
            status_out, go_disabled, log_out,
            fits_dl_disabled, tick_disabled,
            plot_first_src, plot_last_src,
            plot_first_style, plot_last_style,
            plot_note, fits_href,
            href_target, href_style
        )

    # --- Fallback ---
    return (
        status_out, go_disabled, log_out,
        fits_dl_disabled, tick_disabled,
        plot_first_src, plot_last_src,
        plot_first_style, plot_last_style,
        "", fits_href,
        href_target, href_style
    )


@app.callback(
    # Grid range values
    Output("teff-min", "value"),
    Output("teff-max", "value"),
    Output("logg-min", "value"),
    Output("logg-max", "value"),
    Output("logz-min", "value"),
    Output("logz-max", "value"),
    # Single-triplet values
    Output("teff-one", "value"),
    Output("logg-one", "value"),
    Output("logz-one", "value"),
    # Grid range min/max props
    Output("teff-min", "min"),
    Output("teff-min", "max"),
    Output("teff-max", "min"),
    Output("teff-max", "max"),
    Output("logg-min", "min"),
    Output("logg-min", "max"),
    Output("logg-max", "min"),
    Output("logg-max", "max"),
    Output("logz-min", "min"),
    Output("logz-min", "max"),
    Output("logz-max", "min"),
    Output("logz-max", "max"),
    # Single-triplet min/max props
    Output("teff-one", "min"),
    Output("teff-one", "max"),
    Output("logg-one", "min"),
    Output("logg-one", "max"),
    Output("logz-one", "min"),
    Output("logz-one", "max"),
    Input("model-select", "value"),
)
def sync_limits_from_model(selected_model):
    """
    When the user changes the model dropdown:
    - make sure meta exists for that model (via ensure_limits_for_model)
    - update the visible limits and defaults in the UI.
    """
    model_key = selected_model or DEFAULT_MODEL_KEY
    meta = ensure_limits_for_model(model_key)

    tmin, tmax = float(meta["Teff_min"]), float(meta["Teff_max"])
    gmin, gmax = float(meta["logG_min"]), float(meta["logG_max"])
    zmin, zmax = float(meta["logZ_min"]), float(meta["logZ_max"])

    # Single-triplet defaults:
    #  - Teff/logG at midpoints
    #  - logZ at 0 if in-range, otherwise midpoint
    t_one = max(tmin, min(SUN_EXACT["Teff"], tmax))
    g_one = max(gmin, min(SUN_EXACT["logG"], gmax))

    z_pref = SUN_EXACT["logZ"]
    if zmin <= z_pref <= zmax:
        z_one = z_pref
    else:
        # if logZ=0 is not available in this model, fall back to midpoint
        z_one = zmax

    return (
        # Grid range values
        tmin, tmax, gmin, gmax, zmin, zmax,
        # Single-triplet values
        t_one, g_one, z_one,
        # Grid min/max props
        tmin, tmax, tmin, tmax,
        gmin, gmax, gmin, gmax,
        zmin, zmax, zmin, zmax,
        # Single-triplet min/max props
        tmin, tmax,
        gmin, gmax,
        zmin, zmax,
    )

# Download full log snapshot
@app.callback(
    Output("dl-log", "data"),
    Input("download-log", "n_clicks"),
    prevent_initial_call=True,
)
def do_download_log(_n):
    txt = RUN.full_log() or "No logs captured yet."
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return dcc.send_string(txt, filename=f"inference_{ts}.log")

@app.callback(
    Output("teff-preview", "children"),
    Output("logg-preview", "children"),
    Output("logz-preview", "children"),
    Input("teff-min", "value"), Input("teff-max", "value"), Input("teff-n", "value"),
    Input("logg-min", "value"), Input("logg-max", "value"), Input("logg-n", "value"),
    Input("logz-min", "value"), Input("logz-max", "value"), Input("logz-n", "value"),
    State("teff-preview", "children"),
    State("logg-preview", "children"),
    State("logz-preview", "children"),
)
def preview_grid(tmin, tmax, tn, gmin, gmax, gn, zmin, zmax, zn,
                 prev_teff, prev_logg, prev_logz):
    def fmt3(x):
        # format to 3 dp, then strip trailing zeros and dot
        return f"{x:.3f}".rstrip("0").rstrip(".")

    try:
        tmin, tmax, tn = float(tmin), float(tmax), int(tn)
        gmin, gmax, gn = float(gmin), float(gmax), int(gn)
        zmin, zmax, zn = float(zmin), float(zmax), int(zn)
    except Exception:
        # Keep previous previews instead of clearing while user is typing
        return prev_teff, prev_logg, prev_logz

    errs = _validate_params(tmin, tmax, tn, gmin, gmax, gn, zmin, zmax, zn)
    warn = ("\n" + "\n".join(f"- {e}" for e in errs)) if errs else ""

    def _ls(lo, hi, n):
        n = max(2, int(n))
        s = np.linspace(float(lo), float(hi), n)
        head_vals = ", ".join(fmt3(v) for v in s[:3])
        if n > 3:
            tail_vals = ", ".join(fmt3(v) for v in s[-3:])
            mid = " … " if n > 6 else "; "
            return f"[ {head_vals}{mid}{tail_vals} ]"
        else:
            return f"[ {head_vals} ]"

    return (
        f"Teff sample preview: {_ls(tmin, tmax, tn)}{warn}",
        f"logG sample preview: {_ls(gmin, gmax, gn)}{warn}",
        f"logZ sample preview: {_ls(zmin, zmax, zn)}{warn}",
    )



# ───────── Entrypoint ─────────
def _detect_lan_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        try: s.close()
        except Exception: pass
    return ip

# ---- Streaming route for large downloads (robust cleanup) ----
@app.server.route("/api/download/<token>")
def api_download(token: str):
    _purge_expired_downloads()

    with DOWNLOAD_LOCK:
        meta = DOWNLOAD_REGISTRY.get(token)

    if not meta:
        return Response("Invalid or expired token.", status=410)

    path = meta["path"]
    kind = meta.get("kind", "file")
    if not path.exists():
        # consume the token if file vanished
        with DOWNLOAD_LOCK:
            DOWNLOAD_REGISTRY.pop(token, None)
        return Response("File not found.", status=404)

    ts0 = time.time()
    start_iso = datetime.now(timezone.utc).isoformat()
    client = _client_ip()
    size_bytes = os.path.getsize(path)

    # state carried between generator and outer scope
    bytes_sent = 0
    completed = False

    def _gen():
        nonlocal bytes_sent, completed
        try:
            with path.open("rb") as f:
                chunk = f.read(1024 * 1024)  # 1 MiB
                while chunk:
                    bytes_sent += len(chunk)
                    yield chunk
                    chunk = f.read(1024 * 1024)
            # If we got here normally, the full file has been iterated
            completed = (bytes_sent == size_bytes)
        except GeneratorExit:
            # client disconnected; let finally block handle logging
            raise
        except Exception:
            # unexpected read error
            completed = False
            raise
        finally:
            # finalize logging
            ts1 = time.time()
            end_iso = datetime.now(timezone.utc).isoformat()
            status = "completed" if completed else "incomplete"
            try:
                _append_download_log({
                    "ts_start_iso": start_iso,
                    "ts_end_iso": end_iso,
                    "duration_s": round(ts1 - ts0, 3),
                    "client_ip": client,
                    "filename": str(path),
                    "size_bytes": size_bytes,
                    "kind": kind,
                    "bytes_sent": bytes_sent,
                    "status": status,
                    "token": token,
                })
            except Exception:
                pass

            # mark the token as used and optionally delete the file(s)
            with DOWNLOAD_LOCK:
                meta2 = DOWNLOAD_REGISTRY.pop(token, None)
            if DELETE_ONCE_DOWNLOADED and completed:
                try:
                    _delete_artifacts_and_previews(path)
                except Exception:
                    pass

    headers = {
        "Content-Length": str(size_bytes),
        "Content-Disposition": f'attachment; filename="{path.name}"',
        "X-Accel-Buffering": "no",
    }
    return Response(stream_with_context(_gen()),
                    headers=headers,
                    mimetype="application/octet-stream",
                    direct_passthrough=True)


def _filesystem_janitor_loop():
    """Periodically delete old artifacts directly from OUTPUT_DIR based on mtime."""
    # Only run if a timeout > 0 is set
    if DOWNLOAD_DELETE_AFTER_S <= 0:
        return
    while True:
        try:
            now = time.time()
            for patt in DELETE_FILE_PATTERNS:
                for p in OUTPUT_DIR.glob(f"*{patt}"):
                    try:
                        age = now - p.stat().st_mtime
                        if age > DOWNLOAD_DELETE_AFTER_S:
                            size_bytes = p.stat().st_size
                            os.remove(p)
                            for q in _preview_candidates_for(p):
                                try:
                                    if q.exists():
                                        os.remove(q)
                                except Exception:
                                    pass
                            _append_download_log({
                                "ts_start_iso": datetime.now(timezone.utc).isoformat(),
                                "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                                "duration_s": 0.0,
                                "client_ip": "janitor",
                                "filename": str(p),
                                "size_bytes": size_bytes,
                                "kind": "file",
                                "bytes_sent": 0,
                                "status": "mtime_deleted",
                                "token": "N/A",
                            })
                    except FileNotFoundError:
                        pass
                    except Exception:
                        pass
        except Exception:
            pass
        # Sleep a bit between scans
        time.sleep(30)

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8050))
    lan_ip = _detect_lan_ip()
    print(f"Local access:   http://127.0.0.1:{port}")
    print(f"Office network: http://{lan_ip}:{port}")
    _janitor = threading.Thread(target=_filesystem_janitor_loop, daemon=True)
    _janitor.start()
    app.run(debug=False, host=host, port=port)
