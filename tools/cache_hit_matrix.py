#!/usr/bin/env python3
"""Does every protocol surface report prefix-cache hits, and in the field its own spec uses?

For each of the three protocol endpoints: send the same long prompt twice (so the second one
hits the host prefix cache), then once more streamed, and dump every usage object that carries
a cache number. DeepSeek's own API reports chat hits as prompt_cache_hit_tokens; OpenAI as
prompt_tokens_details.cached_tokens; Anthropic as cache_read_input_tokens. All three should agree
on the same number for the same prompt.

    docker cp tools/cache_hit_matrix.py di-e2e:/tmp/ && docker exec di-e2e python /tmp/cache_hit_matrix.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

BASE = os.environ.get("DEEPINFER_SERVE_BASE", "http://127.0.0.1:8888")
MODEL = os.environ.get("DEEPINFER_SERVED_MODEL_NAME", "DeepSeek-V4.1-Flash-1017")
NONCE = f"cache-probe-{int(time.time())}"


def post(path: str, body: dict, stream: bool = False):
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    response = urllib.request.urlopen(request, timeout=1800)
    if not stream:
        return json.loads(response.read())
    frames = []
    for raw in response:
        line = raw.decode().strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload and payload != "[DONE]":
                try:
                    frames.append(json.loads(payload))
                except json.JSONDecodeError:
                    pass
    return frames


def filler() -> str:
    """~2k tokens, unique per run so no other test's blocks are reused."""
    return "\n".join(f"row {i}: the quick brown fox jumps over the lazy dog" for i in range(200))


def chat_body() -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"{NONCE}\n{filler()}\nReply with one word."}],
        "max_tokens": 4,
        "temperature": 0,
    }


def messages_body() -> dict:
    body = chat_body()
    return {"model": MODEL, "max_tokens": 4, "temperature": 0, "messages": body["messages"]}


def responses_body() -> dict:
    return {
        "model": MODEL,
        "input": f"{NONCE}\n{filler()}\nReply with one word.",
        "max_output_tokens": 16,
    }


def dig(obj, path: str):
    cur = obj
    for key in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def report(label: str, usage: dict, fields: list[str]) -> None:
    parts = []
    for field in fields:
        value = dig(usage, field)
        if value is not None:
            parts.append(f"{field}={value}")
    print(f"  {label:22s} " + "  ".join(parts) if parts else f"  {label:22s} (no cache field)")


CHAT_FIELDS = [
    "prompt_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "prompt_tokens_details.cached_tokens",
]
MESSAGES_FIELDS = ["input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"]
RESPONSES_FIELDS = ["input_tokens", "input_tokens_details.cached_tokens", "output_tokens"]


def main() -> int:
    print(f"nonce = {NONCE}\n")
    cases = [
        ("chat completions", "/v1/chat/completions", chat_body, CHAT_FIELDS, "usage"),
        ("messages", "/v1/messages", messages_body, MESSAGES_FIELDS, "usage"),
        ("responses", "/v1/responses", responses_body, RESPONSES_FIELDS, "usage"),
    ]
    for name, path, body_fn, fields, _ in cases:
        print(f"== {name} ==")
        first = post(path, body_fn())
        report("non-stream #1", dig(first, "usage") or {}, fields)
        second = post(path, body_fn())
        report("non-stream #2", dig(second, "usage") or {}, fields)
        streamed = post(path, {**body_fn(), "stream": True}, stream=True)
        for frame in streamed:
            usage = frame.get("usage") or dig(frame, "message.usage") or dig(frame, "response.usage")
            if isinstance(usage, dict) and any(dig(usage, f) for f in fields[1:]):
                kind = frame.get("type") or frame.get("object")
                report(f"stream ({kind})", usage, fields)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
