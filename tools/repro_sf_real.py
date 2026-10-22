"""Call DeepInfer's real entry point set_ue8m0_sf() on SM90.

This is the function that ran last before the warmup crash (weights.py:427, load phase
runtime_adapt). It builds sfs/outs itself from the params' attributes, so it reproduces
the real batching contract, including the n_experts 3D view that my earlier synthetic
calls got wrong.
"""
import sys
import torch

torch.cuda.init()
print("device", torch.cuda.get_device_name(0), flush=True)

from deepinfer.hugging_face.parameters import set_ue8m0_sf  # noqa: E402
from deepinfer.model_executor.layers.linear import get_fp8_casting_granularity  # noqa: E402

gran = get_fp8_casting_granularity()
print(f"granularity={gran}", flush=True)


def fp8_param(out_f, in_f, n_experts=None):
    p = torch.empty(out_f, in_f, dtype=torch.float8_e4m3fn, device="cuda")
    # sf must be a POWER OF TWO: the kernel turns the fp32 exponent into UE8M0, so a
    # non-power-of-two mantissa violates its contract (this invalidated the first attempt).
    exps = torch.randint(-12, 4, (out_f // gran, in_f // gran), device="cuda")
    p.sf = torch.pow(2.0, exps.float())
    if n_experts:
        p.n_experts = n_experts
    return p


cases = {
    "plain 2D fp8 [4608,5120]": lambda: [fp8_param(4608, 5120)],
    "expert n_experts=4 [9216,5120]": lambda: [fp8_param(4 * 2304, 5120, n_experts=4)],
    "two params (one plain, one expert)": lambda: [fp8_param(4608, 5120), fp8_param(4 * 2304, 5120, n_experts=4)],
}

fails = 0
for name, build in cases.items():
    try:
        params = build()
        set_ue8m0_sf(params)
        torch.cuda.synchronize()
        def _shapes(p):
            v = getattr(p, "ue8m0_sf", None)
            if v is None:
                return None
            # set_ue8m0_sf stores a (sf0_out, sf1_out) pair, not a single tensor
            return tuple(tuple(t.shape) for t in v) if isinstance(v, tuple) else tuple(v.shape)
        shapes = [_shapes(p) for p in params]
        print(f"  PASS  {name}  -> ue8m0_sf shapes {shapes}", flush=True)
    except Exception as exc:                          # noqa: BLE001
        fails += 1
        print(f"  FAIL  {name}  -> {type(exc).__name__}: {str(exc).splitlines()[0][:130]}", flush=True)
        break   # a device assert poisons the context; later cases would be meaningless

print(f"\nRESULT: {'FAILURES=' + str(fails) if fails else 'ALL OK'}", flush=True)
sys.exit(1 if fails else 0)
