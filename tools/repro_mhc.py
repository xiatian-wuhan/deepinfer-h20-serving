"""Walk the mHC op chain with the checkpoint's real shapes, one op at a time.

Target of the warmup crash:
  decoder_layer.py:143 attn_norm_forward -> hc.py:129 _hc_fused_post_pre
    -> (first layer) expand_to_mhc + _hc_pre
    -> _hc_pre -> mhc_pre_big_fuse -> mhc_prenorm_gemm (already verified OK on SM90)
                                     + mhc_pre_big_fuse_fwd (tile_kernels)

Shapes from the checkpoint: hidden_size=5120, hc_mult=4 -> mhc_mult2=16, mhc_mult3=24,
mhc_hidden_size=4*5120=20480, hc_sinkhorn_iters=20, rms_norm_eps=1e-20, hc_eps=1e-6.
Tiny token counts, so this runs next to the production service.
"""
import sys
import torch

torch.cuda.init()
print("device", torch.cuda.get_device_name(0), flush=True)

from deepinfer.model_executor.models.v4_modules.mhc_ops import expand_to_mhc   # noqa: E402
from deepinfer.model_executor.models.v4_modules.mhc_modeling import mhc_pre_big_fuse  # noqa: E402

N, HC, HID = 8, 4, 5120
MULT2, MULT3 = HC * HC, HC * 2 + HC * HC          # 16, 24
fails = 0


def step(name, fn):
    global fails
    try:
        out = fn()
        torch.cuda.synchronize()
        print(f"  PASS  {name}" + (f"  {out}" if out else ""), flush=True)
    except Exception as exc:                        # noqa: BLE001
        fails += 1
        print(f"  FAIL  {name}  ->  {type(exc).__name__}: {exc}", flush=True)


def s_expand():
    x = torch.randn(N, HID, dtype=torch.bfloat16, device="cuda") * 0.02
    y = expand_to_mhc(x, HC)
    return f"-> {tuple(y.shape)} {y.dtype}"


def s_pre_big_fuse():
    residual = torch.randn(1, N, HC, HID, dtype=torch.bfloat16, device="cuda") * 0.02
    fn = torch.randn(MULT3, HC * HID, dtype=torch.float32, device="cuda") * 1e-3
    scale = torch.rand(3, dtype=torch.float32, device="cuda") + 0.5
    base = torch.randn(MULT3, dtype=torch.float32, device="cuda") * 0.01
    post_mix, comb_mix, layer_input = mhc_pre_big_fuse(
        residual, fn, scale, base,
        1e-20,      # rms_eps   (config rms_norm_eps)
        1e-6,       # mhc_pre_eps      (config hc_eps)
        1e-6,       # mhc_sinkhorn_eps
        1.0,        # mhc_post_mult_value
        20,         # sinkhorn_repeat  (config hc_sinkhorn_iters)
    )
    ok = all(torch.isfinite(t).all().item() for t in (post_mix, comb_mix, layer_input))
    return f"-> post{tuple(post_mix.shape)} comb{tuple(comb_mix.shape)} in{tuple(layer_input.shape)} finite={ok}"


print("walking the mHC chain:", flush=True)
step("expand_to_mhc", s_expand)
step("mhc_pre_big_fuse (prenorm gemm + reduce/rmsnorm + split + sinkhorn + apply)", s_pre_big_fuse)

print(f"\nRESULT: {'FAILURES=' + str(fails) if fails else 'ALL OK'}", flush=True)
sys.exit(1 if fails else 0)
