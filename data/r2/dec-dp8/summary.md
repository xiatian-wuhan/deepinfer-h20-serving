# run

layout: EP8 DP8 CP1 (8 GPUs), nextn 5, chunk 32768, commit 63b430a47c276fae2a62f38053b2a0f406e17d82, GPU NVIDIA H20-3e

| mode | output tok/s | TPOT (ms) | TTFT (s) | accept len | ok/measured |
|---|---:|---:|---:|---:|---:|
| single | 223.5 (per request) | 4.47 (median) | 0.704 | 2.89 | 3/3 |
| c32 | 2605.7 | 7.80 (mean), 6.53 (median) | 2.956 | 4.91 | 128/128 |
| c64 | 3693.9 | 10.49 (mean), 8.70 (median) | 4.933 | 4.95 | 256/256 |

