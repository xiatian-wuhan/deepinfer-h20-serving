# run

layout: EP8 DP1 CP8 (8 GPUs), nextn 5, chunk 32768, commit 63b430a47c276fae2a62f38053b2a0f406e17d82, GPU NVIDIA H20-3e

| mode | output tok/s | TPOT (ms) | TTFT (s) | accept len | ok/measured |
|---|---:|---:|---:|---:|---:|
| single | 111.3 (per request) | 8.98 (median) | 0.408 | 1.83 | 3/3 |
| c32 | 1665.8 | 15.69 (mean), 12.45 (median) | 2.370 | 4.42 | 128/128 |

