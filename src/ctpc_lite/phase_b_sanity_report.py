import argparse, json
import numpy as np


def _round6(x):
    return round(float(x), 6)


def _infer_unique_call_times(calls):
    seen = set()
    uniq = []
    for c in calls:
        t = c.get("t_float", None)
        if t is None:
            continue
        t = float(t)
        k = round(t, 8)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(t)
    return uniq


def _pick_macro_schedule(all_schedule, steps):
    n = len(all_schedule)
    if steps <= 0:
        return [], "invalid_steps"
    if n == 0:
        return [], "empty"
    if n == steps:
        return list(all_schedule), "identity"
    if n == 2 * steps - 1:
        return list(all_schedule[::2]), "every_other"
    if steps == 1:
        return [all_schedule[0]], "single"

    # evenly spaced indices fallback
    idxs = [round(i * (n - 1) / (steps - 1)) for i in range(steps)]
    idxs = [int(i) for i in idxs]
    # de-dup preserve order
    seen = set()
    idxs2 = []
    for i in idxs:
        if i not in seen:
            idxs2.append(i)
            seen.add(i)
    while len(idxs2) < steps:
        for j in range(len(idxs2) - 1):
            if len(idxs2) >= steps:
                break
            a, b = idxs2[j], idxs2[j + 1]
            mid = (a + b) // 2
            if mid not in seen:
                idxs2.append(mid)
                seen.add(mid)
        idxs2 = sorted(idxs2)
        if len(idxs2) < steps:
            for cand in range(n):
                if cand not in seen:
                    idxs2.append(cand)
                    seen.add(cand)
                if len(idxs2) >= steps:
                    break
            idxs2 = sorted(idxs2)
    idxs2 = idxs2[:steps]
    macro = [all_schedule[i] for i in idxs2]
    return macro, "evenly_spaced"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="calls_raw.jsonl")
    args = ap.parse_args()

    nfes = []
    off_fracs = []
    extra_call_fracs = []
    macro_lens = []
    all_lens = []
    any_multistage = False

    for line in open(args.infile, "r", encoding="utf-8"):
        obj = json.loads(line)
        steps = int(obj.get("num_inference_steps", 0))
        calls = obj["calls"]
        nfe = len(calls)
        nfes.append(nfe)

        # Prefer saved schedules; fallback to inference from calls if missing
        all_sched = obj.get("all_schedule") or []
        macro = obj.get("macro_schedule") or obj.get("outer_schedule") or []

        if len(all_sched) == 0:
            # infer "all_schedule" as unique call times (in order)
            all_sched = _infer_unique_call_times(calls)

        macro_rule = obj.get("macro_schedule_rule", None)
        if len(macro) == 0:
            macro, macro_rule = _pick_macro_schedule(all_sched, steps)

        all_set = set(_round6(x) for x in all_sched)
        macro_set = set(_round6(x) for x in macro)

        off_macro = 0
        uniq_call_ts = set()
        for c in calls:
            t = c.get("t_float", None)
            if t is None:
                continue
            t6 = _round6(t)
            uniq_call_ts.add(t6)
            if macro_set and t6 not in macro_set:
                off_macro += 1

        off_grid_frac = (off_macro / max(1, nfe)) if macro_set else float("nan")
        off_fracs.append(off_grid_frac)

        # Extra-eval-at-same-t diagnostic (useful for PNDM-like behavior)
        extra_calls_frac = (nfe - len(uniq_call_ts)) / max(1, nfe)
        extra_call_fracs.append(extra_calls_frac)

        macro_lens.append(len(macro))
        all_lens.append(len(all_sched))

        logsnrs = [c["logsnr"] for c in calls if c.get("logsnr") is not None]
        logsnr_min = min(logsnrs) if logsnrs else float("nan")
        logsnr_max = max(logsnrs) if logsnrs else float("nan")

        is_multistage = (steps > 0 and nfe > steps) or (len(all_sched) > steps)
        any_multistage = any_multistage or is_multistage

        print(
            f"off_grid_frac={off_grid_frac:.3f} (vs macro)  "
            f"extra_calls_frac={extra_calls_frac:.3f}  "
            f"sample={obj['sample_index']} solver={obj['solver']} steps={steps} "
            f"NFE={nfe} uniq_call_t={len(uniq_call_ts)} "
            f"macro_len={len(macro)} all_len={len(all_sched)} "
            f"macro_rule={macro_rule} "
            f"logsnr_range=[{logsnr_min:.3f},{logsnr_max:.3f}] "
            f"multistage={is_multistage}"
        )

    nfes = np.array(nfes, dtype=np.float32)
    macro_lens = np.array(macro_lens, dtype=np.float32)
    all_lens = np.array(all_lens, dtype=np.float32)
    off_fracs = np.array(off_fracs, dtype=np.float32)
    extra_call_fracs = np.array(extra_call_fracs, dtype=np.float32)

    print(f"\nNFE stats: mean={nfes.mean():.2f}, min={nfes.min():.0f}, max={nfes.max():.0f}")
    print(f"macro_schedule_len: mean={macro_lens.mean():.2f}, min={macro_lens.min():.0f}, max={macro_lens.max():.0f}")
    print(f"all_schedule_len:   mean={all_lens.mean():.2f}, min={all_lens.min():.0f}, max={all_lens.max():.0f}")
    print(f"off_grid_frac:      mean={off_fracs.mean():.3f}")
    print(f"extra_calls_frac:   mean={extra_call_fracs.mean():.3f}")
    print("Multi-stage detected?", bool(any_multistage))


if __name__ == "__main__":
    main()
