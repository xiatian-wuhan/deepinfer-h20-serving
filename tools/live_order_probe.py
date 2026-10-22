#!/usr/bin/env python3
"""Live ordering probe: does the model see, and emit, things in the same order?"""

import json
import urllib.request

BASE = "http://127.0.0.1:8888"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        },
    },
]

PROMPT = (
    "检查 /etc/hosts 的内容，同时跑 `uname -r`，再搜一下 H20 的显存规格。"
    "三个动作一次性并行发起，不要串行。"
)


def post(body: dict):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=900)


def stream_probe() -> None:
    print("== streaming: chunk order ==")
    body = {
        "model": "DeepSeek-V4.1-Flash-1017",
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": TOOLS,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 512,
        "temperature": 0.0,
    }
    content, calls, finish = [], {}, None
    events = []
    with post(body) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                events.append(("usage", obj["usage"]["prompt_tokens_details"]["cached_tokens"]))
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                if d.get("content"):
                    content.append(d["content"])
                    events.append(("content", d["content"]))
                for tc in d.get("tool_calls") or []:
                    idx = tc.get("index")
                    slot = calls.setdefault(idx, {"name": "", "args": ""})
                    if tc.get("function", {}).get("name"):
                        slot["name"] = tc["function"]["name"]
                        events.append(("tc_name", idx, tc["function"]["name"]))
                    if tc.get("function", {}).get("arguments"):
                        slot["args"] += tc["function"]["arguments"]
                        events.append(("tc_args", idx, len(tc["function"]["arguments"])))
    text = "".join(content)
    print(f"  content chars={len(text)}  tool_calls={len(calls)}  finish={finish}")
    print(f"  first 3 events: {events[:3]}")
    print(f"  last 4 events: {events[-4:]}")
    print("  reconstructed:")
    print(f"    content={text[:300]!r}")
    for idx in sorted(calls):
        print(f"    [{idx}] {calls[idx]['name']}({calls[idx]['args'][:160]})")
    # The block must never leak as text.
    leaked = "<｜DSML｜" in text or "DSML" in text
    print(f"  DSML leaked into content: {leaked}")
    print(f"  indices contiguous from 0: {sorted(calls) == list(range(len(calls)))}")


def nonstream_probe() -> None:
    print("\n== non-streaming ==")
    body = {
        "model": "DeepSeek-V4.1-Flash-1017",
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": TOOLS,
        "max_tokens": 512,
        "temperature": 0.0,
    }
    with post(body) as resp:
        obj = json.loads(resp.read())
    choice = obj["choices"][0]
    msg = choice["message"]
    print(f"  finish={choice['finish_reason']} content={msg.get('content','')[:200]!r}")
    for i, tc in enumerate(msg.get("tool_calls") or []):
        print(f"    [{i}] {tc['function']['name']}({tc['function']['arguments'][:160]})")


if __name__ == "__main__":
    stream_probe()
    nonstream_probe()
