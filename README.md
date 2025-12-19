# CTPC-Lite (Phase A + B)

This repo currently implements:
- **Phase A**: reproducibility + environment logging
- **Phase B**: `SolverCallLogger` that records every UNet call timestep + best-effort logSNR/sigma, then postprocesses to compute normalized `u∈[-1,1]`.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```