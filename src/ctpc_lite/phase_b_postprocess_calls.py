import argparse, json, math

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="calls_raw.jsonl")
    ap.add_argument("--outfile", default="calls.jsonl")
    args = ap.parse_args()

    rows = []
    lam_min = float("inf")
    lam_max = float("-inf")

    with open(args.infile, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append(obj)
            for c in obj["calls"]:
                lam = c.get("logsnr", None)
                if lam is None:
                    continue
                lam_min = min(lam_min, lam)
                lam_max = max(lam_max, lam)

    if not math.isfinite(lam_min) or not math.isfinite(lam_max) or abs(lam_max - lam_min) < 1e-12:
        raise RuntimeError("Cannot normalize: logsnr missing or degenerate. Check logger output.")

    def norm_u(lam):
        return 2.0 * (lam - lam_min) / (lam_max - lam_min) - 1.0

    with open(args.outfile, "w", encoding="utf-8") as f:
        for obj in rows:
            for c in obj["calls"]:
                lam = c.get("logsnr", None)
                c["u"] = norm_u(lam) if lam is not None else None
            obj["logsnr_min"] = lam_min
            obj["logsnr_max"] = lam_max
            f.write(json.dumps(obj) + "\n")

    print(f"Wrote {args.outfile}")
    print(f"logsnr_min={lam_min:.6f}, logsnr_max={lam_max:.6f}")

if __name__ == "__main__":
    main()
