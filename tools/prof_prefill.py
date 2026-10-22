"""在 latency_benchmark.py 的 single 套件上挂 torch.profiler，抓 rank0 的 N 个解码步。

不修改仓库文件：import 前设好 sys.argv，再 monkey-patch DeepInferDriver.step。
"""
import os
import sys

OUT = "/e2e-logs/prof"
os.makedirs(OUT, exist_ok=True)

sys.argv = [
    "latency_benchmark.py", "--ep", "8", "--cp", "1", "--dp", "8",
    "--suites", "ttft", "--ttft_lengths", "131072", "--ttft_warmup", "0", "--ttft_repeats", "1",
    "--prompt", "random",
    "--out_dir", os.path.join(OUT, "bench-out"),
]
sys.path.insert(0, "/opt/deepinfer/DeepInfer/examples")

import torch
from _launch import launch_dp

import latency_benchmark as lb

START = int(os.environ.get("PROF_START", "0"))
END = int(os.environ.get("PROF_END", "6"))

_orig = lb.DeepInferDriver.step


def step(self):
    dev = torch.cuda.current_device() if torch.cuda.is_available() else -1
    if dev != 0:
        return _orig(self)

    n = getattr(self, "_prof_n", 0)
    if n == START:
        self._prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
            record_shapes=False,
        )
        self._prof.__enter__()
        print(f"[prof] start at step {n}", flush=True)

    out = _orig(self)
    self._prof_n = n + 1

    if self._prof_n == END:
        p = self._prof
        p.__exit__(None, None, None)
        print(f"[prof] captured steps {START}..{END - 1}", flush=True)
        ka = p.key_averages()
        rows = [(e.self_device_time_total, e.count, e.key) for e in ka if e.self_device_time_total]
        rows.sort(reverse=True)
        total = sum(r[0] for r in rows)
        print(f"[prof] TOTAL_GPU_MS {total/1000:.1f} over {END - START} steps "
              f"=> {total/1000/(END - START):.2f} ms/step", flush=True)
        for us, n, k in rows[:30]:
            print(f"[prof] {us/1000:9.2f}ms {100*us/total:5.1f}% n={n:5d} {k}", flush=True)
    return out


lb.DeepInferDriver.step = step

if __name__ == "__main__":  # spawn 子进程会重新 import 本模块，必须加保护
    os.makedirs(lb.args.out_dir, exist_ok=True)
    launch_dp(lb.main, lb.DP_SIZE)
