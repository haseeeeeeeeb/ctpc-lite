import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", type=str, required=True, help="path to s0.pt")
    args = ap.parse_args()

    obj = torch.load(args.s0, map_location="cpu")
    meta = obj.get("meta", {})
    s0 = obj.get("s0_scale_step", {})
    clip = obj.get("s0_clip", {})

    print("=== Phase D priors sanity ===")
    for k, v in meta.items():
        print(f"{k}: {v}")

    print("\nsite priors:")
    for sid in sorted(s0.keys()):
        print(f"  {sid:12s} step={float(s0[sid]):.6g}  clip={float(clip[sid]):.6g}")

    # basic checks
    bad = [sid for sid in s0 if (not (float(s0[sid]) > 0.0)) or (float(s0[sid]) != float(s0[sid]))]
    if bad:
        print("\n[WARN] non-positive or NaN scales at:", bad)
    else:
        print("\n[OK] all step scales are positive.")


if __name__ == "__main__":
    main()
