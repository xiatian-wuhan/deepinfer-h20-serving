#!/usr/bin/env python3
"""Post-swap acceptance probe for the stop-sequence terminal path. Live HTTP, standard library only.

Run it right after the new `serve/deepseek_server.py` is in place and the service is healthy:

    python3 tools/stop_sequence_probe.py --base http://127.0.0.1:8888
    python3 tools/stop_sequence_probe.py --base http://127.0.0.1:8888/ --model DeepSeek-V4.1-Flash-1017

What it asserts
---------------
1. control: no stop sequence, plain request still returns 200 with normal content;
2. `/v1/messages` non-streaming with `stop_sequences=[" "]` returns **200** with
   `stop_reason == "stop_sequence"`, the matched string echoed in `stop_sequence`, and content that
   does not contain the stop string - this is the request shape that used to return a plain-text 500;
3. the same request with `stream: true` ends with exactly one `message_stop` and no broken stream
   (before the fix this was a silent truncation, invisible to the client);
4. repeating the stop request hits the prefix cache (`cache_read_input_tokens > 0`), proving that
   aborting the sequence mid-decode did not evict the stored prefix pages.

Exit code 0 means every case passed.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import urllib.error
import urllib.request

FILLER = "The quick brown fox jumps over the lazy dog. " * 60
# Greedy decoding plus a prompt that forces several words makes the trigger deterministic: the stop
# string is a single space, so it matches at the first word boundary. Sampling at temperature 1 can
# answer "Okay." and never emit a space at all, which is why this probe pins the temperature.
PROMPT = "Reply with one English sentence of about ten words describing the sea."


def _request(base: str, path: str, body: dict, stream: bool, timeout: float):
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "accept": "text/event-stream" if stream else "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if stream:
                try:
                    return response.status, response.read().decode()
                except http.client.IncompleteRead as exc:  # the pre-fix stream is cut after message_stop
                    return response.status, exc.partial.decode(errors="replace") + "\n[INCOMPLETE READ]"
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:  # a plain-text 500 is the failure we are hunting
        return exc.code, exc.read()[:400].decode(errors="replace")


def messages_body(model: str, stop: list[str] | None, stream: bool = False, long_prompt: bool = False) -> dict:
    content = (FILLER + "\n\n" if long_prompt else "") + PROMPT
    body = {
        "model": model,
        "max_tokens": 64,
        "stream": stream,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": content}],
    }
    if stop is not None:
        body["stop_sequences"] = stop
    return body


class Case:
    def __init__(self, name: str) -> None:
        self.name = name
        self.failures: list[str] = []

    def check(self, ok: bool, note: str) -> None:
        print(f"       {'ok  ' if ok else 'FAIL'}  {note}")
        if not ok:
            self.failures.append(note)

    @property
    def passed(self) -> bool:
        return not self.failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    cases: list[Case] = []

    control = Case("control: no stop sequence")
    status, payload = _request(args.base, "/v1/messages", messages_body(args.model, None), False, args.timeout)
    control.check(status == 200, f"HTTP {status}")
    if isinstance(payload, dict):
        text = "".join(block.get("text", "") for block in payload.get("content") or [])
        control.check(bool(text.strip()), f"content non-empty ({len(text)} chars)")
        control.check(payload.get("stop_reason") != "stop_sequence", f"stop_reason={payload.get('stop_reason')!r}")
    else:
        control.check(False, f"body: {payload}")
    cases.append(control)

    plain = Case("messages/non-stream: stop sequence returns 200 (was a plain-text 500)")
    status, payload = _request(args.base, "/v1/messages", messages_body(args.model, [" "]), False, args.timeout)
    plain.check(status == 200, f"HTTP {status}")
    if isinstance(payload, dict):
        text = "".join(block.get("text", "") for block in payload.get("content") or [])
        plain.check(payload.get("stop_reason") == "stop_sequence", f"stop_reason={payload.get('stop_reason')!r}")
        plain.check(payload.get("stop_sequence") == " ", f"stop_sequence={payload.get('stop_sequence')!r}")
        plain.check(" " not in text, "stop string not in the returned text")
        plain.check(bool(text.strip()), f"content kept before the match ({text!r})")
        usage = payload.get("usage") or {}
        plain.check(usage.get("output_tokens", 0) > 0, f"output_tokens={usage.get('output_tokens')}")
    else:
        plain.check(False, f"body: {payload}")
    cases.append(plain)

    streamed = Case("messages/stream: stop sequence ends cleanly, one message_stop")
    status, text = _request(args.base, "/v1/messages", messages_body(args.model, [" "], stream=True), True, args.timeout)
    streamed.check(status == 200, f"HTTP {status}")
    if isinstance(text, str):
        streamed.check(text.count("event: message_stop") == 1, f"message_stop x{text.count('event: message_stop')}")
        deltas, reported, reason = [], None, None
        for line in text.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except ValueError:
                continue
            delta = event.get("delta")
            if isinstance(delta, dict):
                if "text" in delta:
                    deltas.append(delta["text"])
                if delta.get("stop_reason"):
                    reason = delta["stop_reason"]
                if delta.get("stop_sequence"):
                    reported = delta["stop_sequence"]
        body = "".join(deltas)
        streamed.check(reason == "stop_sequence", f"stop_reason in message_delta: {reason!r}")
        streamed.check(" " not in body, "stop string not in the streamed text")
        streamed.check(reported == " ", f"stop_sequence in message_delta: {reported!r}")
    cases.append(streamed)

    cached = Case("cache: a stop-sequence abort does not evict the stored prefix")
    body = messages_body(args.model, [" "], long_prompt=True)
    _request(args.base, "/v1/messages", body, False, args.timeout)
    status, payload = _request(args.base, "/v1/messages", body, False, args.timeout)
    cached.check(status == 200, f"HTTP {status}")
    if isinstance(payload, dict):
        usage = payload.get("usage") or {}
        hit = usage.get("cache_read_input_tokens", 0)
        cached.check(hit > 0, f"cache_read_input_tokens={hit} (input_tokens={usage.get('input_tokens')})")
    else:
        cached.check(False, f"body: {payload}")
    cases.append(cached)

    print()
    failed = 0
    for case in cases:
        print(f"[{'PASS' if case.passed else 'FAIL'}] {case.name}")
        failed += 0 if case.passed else 1
    print(f"\n{len(cases) - failed}/{len(cases)} cases passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
