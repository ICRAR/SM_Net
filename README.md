# SM-Net Local Dashboard (Dash/Plotly)

A local web interface for generating synthetic stellar spectra with SM-Net from input parameters:
- **Teff (K)**, **log g**, **log Z**

This repository contains:
- The Dash application (`scripts/dash_lookup.py`)
- Lightweight metadata caches (`data/processed_libraries/*.meta.npz`)
- A sample CSV (`sample_star_track.csv`)

**Model weights are NOT stored in this GitHub repo.**  
Weights are hosted on Zenodo and are downloaded automatically by the web app when needed.

---

## Hardware requirements

- **NVIDIA GPU with at least 6 GB of GPU Memory** and latest Nvidia driver
- Internet access (for first-time model weight downloads)
- Recommended RAM: 32 GB+

---

## Model weights (Zenodo)

Weights are hosted on Zenodo:

- **DOI:** 10.5281/zenodo.18883385
- https://zenodo.org/records/18883385

On first launch:
- The app will download the **default model weights** automatically (if not already present).
- The download(s) may take some time as weights for each model are about 2.2 GB
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
```

### 2) Create a Python environment (recommended)
Option A — Windows (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
If PowerShell blocks activation, run this once (PowerShell as your user), then repeat the commands above:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
Option B — Windows (Command Prompt)
```bat
py -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```
Option C — Linux (Ubuntu) / macOS (Terminal)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```
### 3) Install NVIDIA driver + verify GPU

Run this (Windows PowerShell or Linux Terminal):
```bash
nvidia-smi
```
You should see your GPU listed and a driver version.
If nvidia-smi is not found, your NVIDIA driver is not installed correctly (or you need to reboot after installing it).

### 4) Install CUDA-enabled PyTorch (recommended)

Do not rely on pip install torch if you need GPU support. Use the official PyTorch selector to generate the correct command for your OS + CUDA:

https://pytorch.org/get-started/locally/

Then run the command it gives you inside your activated .venv.

In the selector choose:

PyTorch Build: Stable

Your OS: whichever is applicable

Package: Pip

Language: Python

Compute Platform: CUDA (pick the one compatible with your Nvidia driver version, see the link below)
https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html

Copy the command PyTorch gives you and run it inside your activated .venv.

### 5) Install the remaining Python packages

After PyTorch is installed, run:
```bash
pip install -r requirements.txt
```
### 6) Verify PyTorch can see the GPU
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```
Expected:

CUDA available: True

GPU name printed

### 7) Run the dashboard
```bash
python scripts/dash_lookup.py
```

Then open:

http://127.0.0.1:8050

On first launch, the app should start downloading the default model weights automatically (this can take time depending on your internet connection).

The “Read me” button on the top right of the dashboard provides an in-app Quick Start guide. Refer to the paper for technical details.