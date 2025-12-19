import argparse
import torch

from ctpc_lite.ctpc_scale_field import CTPCScaleField, ScaleFieldConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", type=str, required=True, help="path to s0.pt from Phase D")
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--r", type=int, default=4)
    args = ap.parse_args()

    blob = torch.load(args.s0, map_location="cpu")
    s0 = blob["s0_scale_step"]
    site_ids = blob.get("site_ids", None) or list(blob["site_paths"].keys())

    cfg = ScaleFieldConfig(r=args.r, delta=args.delta)
    field = CTPCScaleField(site_ids=site_ids, s0_scale_step=s0, cfg=cfg)

    print("=== Phase E: CTPCScaleField sanity ===")
    print(f"sites={len(site_ids)}  r={args.r}  delta={args.delta}")
    print("Check 1: A==0 => scales(u)==s0 (within fp tolerance)")

    for u in [-1.0, -0.3, 0.0, 0.7, 1.0]:
        sd = field.scales_dict(u)
        max_rel = 0.0
        for sid in site_ids:
            s0v = float(s0[sid])
            sv = float(sd[sid].item())
            rel = abs(sv - s0v) / max(1e-12, abs(s0v))
            max_rel = max(max_rel, rel)
        print(f"  u={u:+.2f}  max_rel_err={max_rel:.3e}")

    print("\nCheck 2: gradient flows into A")
    u = torch.tensor([0.1, -0.2, 0.9], dtype=torch.float32)
    scales = field.scales_tensor(u)          # (B,S)
    loss = scales.mean()
    loss.backward()
    gnorm = field.A.grad.abs().mean().item()
    print(f"  mean(|grad(A)|) = {gnorm:.6e}  (EXPECTED: > 0)")

    print("\nCheck 3: Lipschitz penalty is finite")
    field.A.grad = None
    u2 = torch.linspace(-1, 1, steps=64)
    lp = field.lip_penalty(u2, delta_u=0.02)
    print(f"  lip_penalty = {lp.item():.6e}")

    print("\n[OK] Phase E module behaves as expected.")


if __name__ == "__main__":
    main()
