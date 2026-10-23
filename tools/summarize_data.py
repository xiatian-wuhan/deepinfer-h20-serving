#!/usr/bin/env python3
"""Render the README performance tables from the frozen harness output in `data/`.

Every benchmark run leaves the harness' own `summary.json` next to its `summary.md`
(`examples/latency_benchmark.py --out_dir ...`). `data/` keeps those files so the
numbers in the README can be re-derived instead of copied by hand:

    python3 tools/summarize_data.py data/r2

Prints two markdown tables: TTFT per run, and decode throughput per run/concurrency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_runs(root: Path) -> list[tuple[str, dict]]:
    runs = []
    for summary in sorted(root.glob("*/summary.json")):
        runs.append((summary.parent.name, json.loads(summary.read_text())))
    return runs


def layout_of(meta: dict) -> str:
    args = meta.get("args") or {}
    dp, cp = args.get("dp"), args.get("cp")
    if cp and cp > 1:
        return f"CP{cp}"
    return f"DP{dp}" if dp else f"CP{cp or 1}"


def fmt(value, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def ttft_table(runs: list[tuple[str, dict]]) -> str:
    lines = ["| 运行 | 布局 | 输入 token | TTFT 中位 (s) | 最小 | 最大 | ok/测量 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name, data in runs:
        for row in data["results"].get("ttft", []):
            ok = f"{row['ok']}/{row['measured']}"
            lines.append(
                f"| `{name}` | {layout_of(data['meta'])} | {row['input_tokens']:,} | "
                f"{fmt(row['ttft_s_median'], 3)} | {fmt(row['ttft_s_min'], 3)} | "
                f"{fmt(row['ttft_s_max'], 3)} | {ok} |"
            )
    return "\n".join(lines)


def decode_table(runs: list[tuple[str, dict]]) -> str:
    lines = ["| 运行 | 布局 | 并发 | 总 tok/s | 单请求 tok/s | TPOT 中位 (ms) | TTFT 中位 (s) | accept |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, data in runs:
        for key, row in data["results"].items():
            if key == "single":
                lines.append(
                    f"| `{name}` | {layout_of(data['meta'])} | 1 | - | "
                    f"{fmt(row['decode_tok_s_median'], 1)} | {fmt(row['tpot_ms_median'], 2)} | "
                    f"{fmt(row['ttft_s_median'], 3)} | {fmt(row['accept_len_median'], 2)} |"
                )
            elif key.startswith("c") and key[1:].isdigit():
                lines.append(
                    f"| `{name}` | {layout_of(data['meta'])} | {row['concurrency']} | "
                    f"{fmt(row['total_output_tok_s'], 1)} | {fmt(row['per_request_decode_tok_s_mean'], 1)} | "
                    f"{fmt(row['tpot_ms_median'], 2)} | {fmt(row['ttft_s_median'], 3)} | "
                    f"{fmt(row['accept_len_mean'], 2)} |"
                )
    return "\n".join(lines)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"no <run>/summary.json under {root}")
    print("## TTFT\n")
    print(ttft_table(runs))
    print("\n## 解码\n")
    print(decode_table(runs))


if __name__ == "__main__":
    main()
