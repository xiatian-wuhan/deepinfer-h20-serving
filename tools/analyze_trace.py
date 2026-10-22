#!/usr/bin/env python3
"""Check the real Claude Code trace in serve.log for ordering invariants.

Reads the `[msgs] in=` digest the service logs (roles, content sizes, tool-call ids,
tool_call_id targets) and answers, for the longest request:

  * does every assistant tool_calls block get exactly the results it asked for?
  * did the gateway deliver the results in call order, or out of order?
  * where do the mid-conversation `system` messages sit?
"""

import re
import sys
from collections import Counter

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/serve.log"

ROLE_RE = re.compile(r"^(?P<role>[a-z_]+):(?P<size>\d+)(?:\((?P<calls>[^)]*)\))?(?:->(?P<target>.+))?$")


def parse(line: str) -> list[dict]:
    body = line.split("in=", 1)[1]
    out = []
    for chunk in body.split(" | "):
        m = ROLE_RE.match(chunk.strip())
        if not m:
            out.append({"role": "?", "raw": chunk})
            continue
        calls = m.group("calls")
        out.append(
            {
                "role": m.group("role"),
                "size": int(m.group("size")),
                "calls": [c for c in calls.split(",") if c] if calls else [],
                "target": m.group("target"),
            }
        )
    return out


def main() -> int:
    lines = [ln for ln in open(LOG, encoding="utf-8", errors="replace") if "[msgs] in=" in ln]
    if not lines:
        print("no [msgs] lines in the log")
        return 1
    longest = max(lines, key=len)
    msgs = parse(longest)
    print(f"requests logged: {len(lines)}   analysing the longest ({len(msgs)} messages)")
    print("role histogram:", dict(Counter(m["role"] for m in msgs)))

    # Invariant 1: every assistant tool_calls is answered by the immediately following tools.
    problems, out_of_order, rounds = [], 0, 0
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "assistant" and m["calls"]:
            rounds += 1
            want = m["calls"]
            j, got = i + 1, []
            while j < len(msgs) and msgs[j]["role"] == "tool":
                got.append(msgs[j]["target"])
                j += 1
            if sorted(got) != sorted(want):
                problems.append(f"round {rounds}: calls {want} vs results {got}")
            elif got != want:
                out_of_order += 1
            i = j
            continue
        i += 1

    # Invariant 2: no tool message without an assistant tool_calls right before it.
    stray = [k for k, m in enumerate(msgs) if m["role"] == "tool" and (k == 0 or msgs[k - 1]["role"] not in ("tool", "assistant"))]

    # Invariant 3: where do system messages sit?
    sys_pos = Counter()
    for k, m in enumerate(msgs):
        if m["role"] != "system":
            continue
        prev = msgs[k - 1]["role"] if k else "START"
        nxt = msgs[k + 1]["role"] if k + 1 < len(msgs) else "END"
        sys_pos[f"{prev} -> system -> {nxt}"] += 1

    print(f"\ntool rounds: {rounds}   mismatched: {len(problems)}   delivered out of call order: {out_of_order}")
    for p in problems:
        print("  PROBLEM", p)
    print(f"stray tool messages: {len(stray)}")
    print("\nmid-conversation system messages (position pattern -> count):")
    for pattern, n in sys_pos.most_common():
        print(f"  {n:3d}  {pattern}")
    print("\nfirst 12 messages:")
    for m in msgs[:12]:
        extra = f" calls={len(m['calls'])}" if m["calls"] else ""
        extra += f" ->{m['target'][:20]}" if m.get("target") else ""
        print(f"  {m['role']:<10} {m['size']:>7}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
