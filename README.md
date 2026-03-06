# SM-Net Local Dashboard (Dash/Plotly)

A local web interface for generating synthetic stellar spectra with SM-Net from input parameters:
- **Teff (K)**, **log g**, **log Z**

This repository contains:
- The Dash application (`scripts/dash_lookup.py`)
- Lightweight metadata caches (`data/processed_libraries/*.meta.npz`)
- A sample CSV (`sample_star_track.csv`)

**Model weights are NOT stored in this GitHub repo.**  
Weights are hosted on Zenodo and are downloaded automatically by the app when needed.

---

## Hardware requirements

- **NVIDIA GPU with at least 6 GB VRAM**
- Internet access (for first-time model weight downloads)
- Recommended RAM: 16 GB+

---

## Model weights (Zenodo)

Weights are hosted on Zenodo:

- **DOI:** 10.5281/zenodo.18883385

On first launch:
- The app will download the **default model weights** automatically (if not already present).
- If you select another model, the app will check if the weights exist in `models/`. If missing, it will ask whether you want to download them.

Manual download is also possible:
- Download the required `.pt` file(s) from Zenodo and place them in:
  - `models/`

---

## Quick start (all platforms)

### 1) Clone the repository
```bash
git clone https://github.com/ICRAR/SM_Net.git
cd SM_Net
