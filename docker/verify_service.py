"""Build-time self-check for the service image. No GPU and no checkpoint required."""
import importlib.util
import sys

ok = True

def check(label, fn):
    global ok
    try:
        detail = fn()
        print(f"  PASS  {label}" + (f"  ({detail})" if detail else ""))
    except Exception as exc:                      # noqa: BLE001
        ok = False
        print(f"  FAIL  {label}  ->  {type(exc).__name__}: {exc}")

# torch must come first: deepinfer._C links libc10.so / libtorch.so, which only enter the
# loader search path once torch is initialised.
import torch                                   # noqa: E402

import deepinfer                               # noqa: E402
from deepinfer._C import engine                # noqa: E402

def pr195():
    missing = [n for n in ("free_finished_seqs", "add_prompted_sequence")
               if not hasattr(engine.Scheduler, n)]
    assert not missing, f"PR#195 bindings missing: {missing}"
    return "Scheduler.free_finished_seqs + add_prompted_sequence"

def pr144():
    assert hasattr(engine.Sequence, "block_table"), "PR#144 Sequence.block_table missing"
    # deepinfer.hicache -> deepinfer.worker.cache_engine -> attention_backend picks its SM90 vs
    # SM100 branch from PLATFORM.is_sm90, which reads NVML compute capability. With no GPU at
    # build time that is False, so the import would take the SM100 branch and need `flash_mla`
    # (absent by design on SM90). Verify that reason instead of assuming it, and leave the real
    # import to the --gpus runtime check.
    from deep_cluster.hardware.nvidia import compute_capability  # not in hardware.__all__
    cap = compute_capability()
    if cap is None:
        return "Sequence.block_table; hicache import deferred (no GPU: compute_capability()=None)"
    from deepinfer.hicache import HostPrefixCache  # noqa: F401
    return f"Sequence.block_table + deepinfer.hicache (capability {cap})"

def hicache_sources():
    import os
    d = "/opt/deepinfer/DeepInfer/deepinfer/hicache"
    need = ["__init__.py", "host_cache.py", "kernels.py", "page_copy.cpp", "page_copy.cu"]
    missing = [f for f in need if not os.path.exists(os.path.join(d, f))]
    assert not missing, f"missing hicache sources: {missing}"
    return f"{len(need)} files present"

def pr147():
    spec = importlib.util.spec_from_file_location("oa_server", "/opt/deepinfer/DeepInfer/serve/openai_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [n for n in dir(mod) if "app" in n.lower() or "engine" in n.lower()]
    return f"serve/openai_server.py imports, {len(fns)} app/engine symbols"

def recipe():
    import deepseek_recipe
    names = [n for n in ("DeepseekV41Encoding", "Tokenizer", "MessagesRequest") if hasattr(deepseek_recipe, n)]
    assert len(names) == 3, f"deepseek_recipe missing symbols, found {names}"
    return f"deepseek_recipe {getattr(deepseek_recipe, '__version__', '?')}"

def pool():
    import engine_pool                        # noqa: F401
    assert hasattr(engine_pool, "EnginePool")
    return "engine_pool.EnginePool"

def openai_api():
    spec = importlib.util.spec_from_file_location("oa2", "/opt/deepinfer/DeepInfer/serve/openai_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    build = getattr(mod, "build_app", None) or getattr(mod, "create_app", None)
    assert callable(build), "no app factory in serve/openai_server.py"
    return f"app factory = {build.__name__}"

print(f"torch {torch.__version__}, deepinfer {deepinfer.__file__}")
check("PR#195 scheduler bindings", pr195)
check("PR#144 prefix cache", pr144)
check("PR#144 hicache sources", hicache_sources)
check("PR#147 serve layer", pr147)
check("deepseek-recipe (Anthropic)", recipe)
check("deepinfer-server engine pool", pool)
check("serve/ app factory", openai_api)

if not ok:
    print("\nSELF-CHECK FAILED")
    sys.exit(1)
print("\nSELF-CHECK OK")
