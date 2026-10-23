# run

layout: EP8 DP1 CP8 (8 GPUs), nextn 5, chunk 32768, commit 63b430a47c276fae2a62f38053b2a0f406e17d82, GPU NVIDIA H20-3e

| mode | output tok/s | TPOT (ms) | TTFT (s) | accept len | ok/measured |
|---|---:|---:|---:|---:|---:|
| single | 243.8 (per request) | 4.10 (median) | 0.348 | 3.83 | 3/3 |
| c32 | 1614.3 | 15.96 (mean), 12.59 (median) | 2.402 | 4.58 | 128/128 |
| c64 | 1836.4 | 28.33 (mean), 21.74 (median) | 4.035 | 4.45 | 256/256 |

