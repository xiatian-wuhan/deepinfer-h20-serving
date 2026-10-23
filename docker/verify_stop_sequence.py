#!/usr/bin/env python3
"""Offline regression test for the stop-sequence terminal path in `serve/deepseek_server.py`.

Why this exists
---------------
A client `stop_sequences` entry is only known to deepseek-recipe's Rust parser: the engine's
`SamplingParams` has no `stop` field, so the engine keeps decoding to EOS/max_tokens. When the
parser sees the stop string it emits the closing events (`content_block_stop`, `message_delta`,
`message_stop`) and completes the stream by itself. Every later `processor.push(...)` then raises

    RuntimeError: stream processor has finished

On the non-streaming path that exception escapes `recipe_response` and Starlette turns it into a
plain-text `500 Internal Server Error`; on the streaming path `message_stop` has already been
yielded, so the same bug shows up as a silently truncated stream instead of an error.

The test drives the real `recipe_response` with a scripted chunk queue instead of an engine, so it
needs no GPU, no service and no network. Run it against the module under test:

    python3 verify_stop_sequence.py --module /opt/deepinfer/DeepInfer/serve/deepseek_server.py

Exit code 0 means every case passed.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys

MODEL = "/model"
STOP = "</scratchpad>"
BEFORE = "VERDICT: safe. Reasons: none here. "
AFTER = " this tail must never reach the client "


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("ds_under_test", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ds_under_test"] = module
    spec.loader.exec_module(module)
    return module


class Harness:
    """A `Frontend` with a scripted queue instead of engines."""

    def __init__(self, module, num_engines: int = 1) -> None:
        argv = sys.argv
        sys.argv = [
            "deepseek_server.py",
            "--model",
            MODEL,
            "--protocol-layer",
            "recipe",
            "--served-model-name",
            "DeepSeek-V4.1-Flash-1017",
            "--dp",
            str(num_engines),
        ]
        try:
            args = module.parse_args()
        finally:
            sys.argv = argv
        args.num_engines = num_engines
        args.log_requests = False
        args.port = 0
        self.fe = module.Frontend(args)
        self.fe.ready = num_engines
        self.script: list[list[tuple]] = []
        self.releases: list[bool] = []
        self.fe.submit = self._submit
        self.fe.release = self._release

    def _submit(self, prompt_ids, sampling, key=None, use_cache=True, multi_modal=None):
        rid = "t%03d" % len(self.releases)
        q: asyncio.Queue = asyncio.Queue()
        for item in self.script.pop(0):
            q.put_nowait(item)
        self.fe.owner[rid] = 0
        self.fe.streams[rid] = q
        self.fe.stats["requests"] += 1
        return rid, q

    def _release(self, rid: str, aborted: bool) -> None:
        self.releases.append(aborted)
        self.fe.owner.pop(rid, None)
        self.fe.streams.pop(rid, None)
        self.fe.req_meta.pop(rid, None)
        self.fe.cached.pop(rid, 0)
        self.fe.stats["aborted" if aborted else "finished"] += 1

    def ids(self, text: str) -> list[int]:
        return self.fe.rtok.encode(text)

    def tokens(self, text: str, finished: bool = False) -> tuple[tuple]:
        """Chunk-queue items for `text`, the last one optionally marked finished."""
        ids = self.ids(text)
        items = [("token", token_id, False, None, 0) for token_id in ids]
        if finished:
            items.append(("token", None, True, "stop", 0))
        return tuple(items)


class Case:
    def __init__(self, name: str) -> None:
        self.name = name
        self.failures: list[str] = []
        self.notes: list[str] = []

    def check(self, ok: bool, note: str) -> None:
        if ok:
            self.notes.append(f"ok    {note}")
        else:
            self.failures.append(note)
            self.notes.append(f"FAIL  {note}")

    @property
    def passed(self) -> bool:
        return not self.failures


async def call(harness, module, protocol, body, items):
    harness.script.append(list(items))
    return await module.recipe_response(harness.fe, protocol, body)


def body_messages(model: str, stream: bool, stop: list[str] | None = None) -> dict:
    payload = {
        "model": model,
        "max_tokens": 256,
        "stream": stream,
        "messages": [{"role": "user", "content": "review this transcript"}],
    }
    if stop:
        payload["stop_sequences"] = stop
    return payload


async def case_stop_mid_stream(harness, module) -> Case:
    """Prod signature A: the stop string matches mid-output, more tokens follow."""
    case = Case("messages/stop mid-stream, tokens follow")
    body = body_messages(harness.fe.model_name, False, [STOP])
    items = harness.tokens(BEFORE + STOP, finished=False) + harness.tokens(AFTER, finished=True)
    try:
        resp = await call(harness, module, "messages", body, items)
    except Exception as exc:  # noqa: BLE001 - this is exactly what the bug did
        case.check(False, f"recipe_response raised {type(exc).__name__}: {exc}")
        return case
    case.check(getattr(resp, "status_code", 0) == 200, f"HTTP {getattr(resp, 'status_code', '?')}")
    payload = json.loads(resp.body)
    text = "".join(block.get("text", "") for block in payload.get("content", []))
    case.check(STOP not in text, "stop string stripped from content")
    case.check(AFTER.strip() not in text, "post-stop tail never reaches the client")
    case.check(payload.get("stop_reason") in ("end_turn", "stop_sequence"), f"stop_reason={payload.get('stop_reason')!r}")
    case.check(harness.releases[-1:] == [True], f"engine abort requested: releases={harness.releases}")
    case.check(harness.fe.stats.get("errors", 0) == 0, "counted as a finish, not an error")
    case.check(harness.fe.stats.get("stop_sequences", 0) == 1, "stop_sequences counter incremented")
    return case


async def case_stop_on_last_token(harness, module) -> Case:
    """Prod signature B: the stop string lands on the final token, so the failing push is `finish`."""
    case = Case("messages/stop on last token")
    body = body_messages(harness.fe.model_name, False, [STOP])
    items = harness.tokens(BEFORE + STOP, finished=True)
    try:
        resp = await call(harness, module, "messages", body, items)
    except Exception as exc:  # noqa: BLE001
        case.check(False, f"recipe_response raised {type(exc).__name__}: {exc}")
        return case
    case.check(getattr(resp, "status_code", 0) == 200, f"HTTP {getattr(resp, 'status_code', '?')}")
    payload = json.loads(resp.body)
    text = "".join(block.get("text", "") for block in payload.get("content", []))
    case.check(STOP not in text, "stop string stripped from content")
    case.check(harness.releases[-1:] == [True], f"engine abort requested: releases={harness.releases}")
    return case


async def case_no_stop_control(harness, module) -> Case:
    """Control: no stop sequence, the engine finishes normally. Must be unchanged by the fix."""
    case = Case("messages/no stop, normal finish (control)")
    body = body_messages(harness.fe.model_name, False)
    items = harness.tokens(BEFORE + AFTER, finished=True)
    try:
        resp = await call(harness, module, "messages", body, items)
    except Exception as exc:  # noqa: BLE001
        case.check(False, f"recipe_response raised {type(exc).__name__}: {exc}")
        return case
    case.check(getattr(resp, "status_code", 0) == 200, f"HTTP {getattr(resp, 'status_code', '?')}")
    payload = json.loads(resp.body)
    text = "".join(block.get("text", "") for block in payload.get("content", []))
    case.check(AFTER.strip() in text, "full text delivered")
    case.check(harness.releases[-1:] == [False], f"no engine abort: releases={harness.releases}")
    return case


async def case_stream(harness, module) -> Case:
    """Streaming path: the closing events are already on the wire, so nothing may raise afterwards."""
    case = Case("messages/stream, stop mid-stream")
    body = body_messages(harness.fe.model_name, True, [STOP])
    items = harness.tokens(BEFORE + STOP, finished=False) + harness.tokens(AFTER, finished=True)
    try:
        resp = await call(harness, module, "messages", body, items)
        events = []
        async for chunk in resp.body_iterator:
            events.append(chunk if isinstance(chunk, str) else chunk.decode())
    except Exception as exc:  # noqa: BLE001
        case.check(False, f"stream raised {type(exc).__name__}: {exc}")
        return case
    stream = "".join(events)
    case.check("event: message_stop" in stream, "message_stop reached the client")
    case.check(stream.count("event: message_stop") == 1, f"message_stop emitted once ({stream.count('event: message_stop')})")
    # The matched string legitimately shows up in the `stop_sequence` metadata; what must never reach
    # the client is the generated text at or after it.
    deltas, reported = [], None
    for line in stream.splitlines():
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
            if delta.get("stop_sequence"):
                reported = delta["stop_sequence"]
    content = "".join(deltas)
    case.check(STOP not in content, "stop string not in the streamed text")
    case.check(AFTER.strip() not in content, "post-stop tail not streamed")
    case.check(reported == STOP, f"stop_sequence reported in message_delta: {reported!r}")
    case.check(harness.releases[-1:] == [True], f"engine abort requested: releases={harness.releases}")
    return case


async def case_engine_error(harness, module) -> Case:
    """A real engine failure must still surface - the fix must not swallow errors."""
    case = Case("messages/engine error still fails")
    body = body_messages(harness.fe.model_name, False)
    try:
        await call(harness, module, "messages", body, [("error", "engine rank died: [3]")])
    except Exception as exc:  # noqa: BLE001
        case.check("engine rank died" in str(exc), f"propagated: {type(exc).__name__}: {exc}")
        return case
    case.check(False, "error item was swallowed")
    return case


async def case_chat_completions(harness, module) -> Case:
    """Same shared path, OpenAI surface: `stop` must terminate cleanly too."""
    case = Case("chat_completions/stop mid-stream")
    body = {
        "model": harness.fe.model_name,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "review this transcript"}],
        "stop": [STOP],
    }
    items = harness.tokens(BEFORE + STOP, finished=False) + harness.tokens(AFTER, finished=True)
    try:
        resp = await call(harness, module, "chat_completions", body, items)
    except Exception as exc:  # noqa: BLE001
        case.check(False, f"recipe_response raised {type(exc).__name__}: {exc}")
        return case
    case.check(getattr(resp, "status_code", 0) == 200, f"HTTP {getattr(resp, 'status_code', '?')}")
    payload = json.loads(resp.body)
    if "choices" not in payload:
        case.check(False, f"unexpected body: {resp.body[:300]!r}")
        return case
    text = payload["choices"][0]["message"].get("content") or ""
    case.check(STOP not in text, "stop string stripped from content")
    case.check(AFTER.strip() not in text, "post-stop tail never reaches the client")
    case.check(harness.releases[-1:] == [True], f"engine abort requested: releases={harness.releases}")
    return case


CASES = (
    case_stop_mid_stream,
    case_stop_on_last_token,
    case_no_stop_control,
    case_stream,
    case_engine_error,
    case_chat_completions,
)


async def amain(module_path: str) -> int:
    module = load_module(module_path)
    harness = Harness(module)
    print(f"\nmodule: {module_path}\n")
    failed = 0
    for run in CASES:
        result = await run(harness, module)
        print(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}")
        for note in result.notes:
            print(f"       {note}")
        failed += 0 if result.passed else 1
    print(f"\n{len(CASES) - failed}/{len(CASES)} cases passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--module", default="/opt/deepinfer/DeepInfer/serve/deepseek_server.py")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(amain(args.module)))
