"""Minimal repro for the warmup crash: deep_gemm.tf32_hc_prenorm_gemm on SM90.

The crash was `cuModuleLoadData(...) CUDA_ERROR_ILLEGAL_ADDRESS` from hc.py:129
(_hc_fused_post_pre -> _hc_pre -> mhc_pre_big_fuse -> mhc_prenorm_gemm ->
deep_gemm.tf32_hc_prenorm_gemm). That call is a DeepGEMM JIT kernel: it compiles
sm90_tf32_hc_prenorm_gemm at runtime, then loads it with cuModuleLoadData.

Tiny tensors only (a few MB), so it runs next to the production service.
"""
import sys
import torch

print("torch", torch.__version__, "device", torch.cuda.get_device_name(0), flush=True)

import deep_gemm

# real values from the checkpoint: hidden_size=5120, hc_mult=4,
# mhc_mult2 = hc_mult**2 = 16, mhc_mult3 = hc_mult*2 + mhc_mult2 = 24
n, mult3, hidden = 8, 24, 5120
x = torch.randn(n, hidden, dtype=torch.bfloat16, device="cuda")
fn = (torch.randn(mult3, hidden, dtype=torch.float32, device="cuda") * 0.01)
out = torch.empty(n, mult3, dtype=torch.float32, device="cuda")
sq = torch.empty(n, dtype=torch.float32, device="cuda")
print(f"calling tf32_hc_prenorm_gemm({tuple(x.shape)}, {tuple(fn.shape)}, {tuple(out.shape)}, {tuple(sq.shape)})", flush=True)

try:
    deep_gemm.tf32_hc_prenorm_gemm(x, fn, out, sq)
    torch.cuda.synchronize()
    print("RESULT: OK  out_abs_sum=%.6f  sq_sum=%.6f" % (out.abs().sum().item(), sq.sum().item()), flush=True)
except Exception as exc:                                    # noqa: BLE001
    print("RESULT: FAIL %s: %s" % (type(exc).__name__, exc), flush=True)
    sys.exit(1)
