from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_utils import ensure_dir


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    rows = _read_jsonl(in_path)

    # Collect all valid logsnr values
    lambdas: List[float] = []
    for r in rows:
        calls = r.get("calls", [])
        for c in calls:
            lam = c.get("logsnr", None)
            if isinstance(lam, (int, float)):
                lambdas.append(float(lam))

    if not lambdas:
        raise RuntimeError(
            "No logsnr values found in calls. "
            "Your scheduler mapping might not be exposing sigma/alphas correctly. "
            "We still logged t_raw/t_float; inspect the raw file."
        )

    lam_min = min(lambdas)
    lam_max = max(lambdas)
    denom = max(1e-12, lam_max - lam_min)

    # Add u to every call that has logsnr
    for r in rows:
        for c in r.get("calls", []):
            lam = c.get("logsnr", None)
            if isinstance(lam, (int, float)):
                u = 2.0 * (float(lam) - lam_min) / denom - 1.0
                c["u"] = u
            else:
                c["u"] = None

    out_path = Path(args.out_jsonl)
    _write_jsonl(out_path, rows)

    meta_path = out_path.with_suffix(".meta.json")
    ensure_dir(meta_path.parent)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"lam_min": lam_min, "lam_max": lam_max}, f, indent=2, sort_keys=True)

    print(f"[postprocess] wrote {out_path}")
    print(f"[postprocess] meta  {meta_path}")
    print(f"[postprocess] lam_min={lam_min:.6f}, lam_max={lam_max:.6f}")


if __name__ == "__main__":
    main()
