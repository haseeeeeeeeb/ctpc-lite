# ctpc_lite/phase_c_smoke_test.py
import torch
import torch.nn as nn

from ctpc_lite.phase_c_u_context import set_u
from ctpc_lite.phase_c_scale_field import CTPCScaleField, ScaleFieldConfig
from ctpc_lite.phase_c_wrappers import QuantActSiteWrapper


def main():
    site_ids = ["site0"]
    log_s0 = {"site0": 0.0}  # scale=1
    sf = CTPCScaleField(site_ids, log_s0, ScaleFieldConfig(rank=4, clamp_delta=2.0))

    # simple linear
    lin = nn.Linear(8, 8).cuda().half()
    w = QuantActSiteWrapper(lin, "site0", sf, enabled=True).cuda()

    x = torch.randn(2, 8, device="cuda", dtype=torch.float16)

    with set_u(torch.tensor(0.0, device="cuda")):
        y1 = w(x)

    # bump A so scale changes with u
    with torch.no_grad():
        sf.A[0, 1] = 1.0  # adds u term

    with set_u(torch.tensor(1.0, device="cuda")):
        y2 = w(x)

    print("ok: forward ran. mean|y1-y2| =", (y1 - y2).abs().mean().item())


if __name__ == "__main__":
    main()
