#!/usr/bin/env python3
"""Offline test for the 400 boundary in `serve/deepseek_server.py`. No GPU, no service, no httpx.

`await request.json()` on a body that does not parse raises `json.decoder.JSONDecodeError`. FastAPI
has no handler for it, so it escapes to Starlette's `ServerErrorMiddleware`, which sends a plain
text `500 Internal Server Error` and then re-raises. A malformed body is the client's fault and has
to be a 400 in the documented shape:

    {"error": {"message": "...", "type": "invalid_request_error"}}

This script builds the real app with a real `Frontend` and drives the ASGI interface directly, so
it exercises the same middleware stack the server does (including the re-raise), without starting
uvicorn or touching an engine.

    python3 verify_request_errors.py --module /opt/deepinfer/DeepInfer/serve/deepseek_server.py
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys

MODEL = "/model"
SERVED = "DeepSeek-V4.1-Flash-1017"
ENDPOINTS = (
    "/v1/messages",
    "/v1/chat/completions",
    "/v1/responses",
    "/v1/messages/count_tokens",
)


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("ds_under_test", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ds_under_test"] = module
    spec.loader.exec_module(module)
    return module


def build_app(module):
    argv = sys.argv
    sys.argv = [
        "deepseek_server.py",
        "--model",
        MODEL,
        "--protocol-layer",
        "recipe",
        "--served-model-name",
        SERVED,
        "--dp",
        "1",
    ]
    try:
        args = module.parse_args()
    finally:
        sys.argv = argv
    args.num_engines = 1
    args.log_requests = False
    args.port = 0
    fe = module.Frontend(args)
    fe.ready = args.num_engines
    return module.build_app(fe)


async def post(app, path: str, body: bytes):
    """Drive one ASGI request; return (status, payload, exception-if-any).

    `ServerErrorMiddleware` sends its 500 and then re-raises, so the exception is part of the
    observable result - that is exactly what uvicorn logs and why the client sees a bare 500.
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "server": ("testserver", 80),
        "client": ("testclient", 123),
    }
    state = {"body": body, "sent": []}

    async def receive():
        chunk, state["body"] = state["body"], b""
        return {"type": "http.request", "body": chunk, "more_body": False}

    async def send(message):
        state["sent"].append(message)

    raised = None
    try:
        await app(scope, receive, send)
    except Exception as exc:  # noqa: BLE001 - the pre-fix 500 path re-raises by design
        raised = f"{type(exc).__name__}: {exc}"

    start = next((m for m in state["sent"] if m["type"] == "http.response.start"), None)
    payload = b"".join(m.get("body", b"") for m in state["sent"] if m["type"] == "http.response.body")
    return (start["status"] if start else 0), payload, raised


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


def parse(payload: bytes) -> dict:
    try:
        return json.loads(payload)
    except ValueError:
        return {}


async def run(app) -> list[Case]:
    cases: list[Case] = []

    for path in ENDPOINTS:
        case = Case(f"{path}: malformed JSON body -> 400")
        status, payload, raised = await post(app, path, b'{"messages": [}')
        case.check(status == 400, f"HTTP {status}")
        case.check(raised is None, f"no exception escaped ({raised})")
        body = parse(payload)
        message = (body.get("error") or {}).get("message", "")
        case.check(bool(message), f"error.message present: {message[:80]!r}")
        case.check(message.startswith("invalid JSON body"), "message names the parse failure")
        case.check(b"Internal Server Error" not in payload, "no plain-text 500 body")
        cases.append(case)

    empty = Case("empty body -> 400")
    status, payload, raised = await post(app, "/v1/messages", b"")
    empty.check(status == 400, f"HTTP {status}")
    empty.check(raised is None, f"no exception escaped ({raised})")
    cases.append(empty)

    not_object = Case("JSON array body -> 400 (not a 500 on .get)")
    status, payload, raised = await post(app, "/v1/messages", b"[1, 2, 3]")
    not_object.check(status == 400, f"HTTP {status}")
    message = (parse(payload).get("error") or {}).get("message", "")
    not_object.check("JSON object" in message, f"message: {message[:80]!r}")
    not_object.check(raised is None, f"no exception escaped ({raised})")
    cases.append(not_object)

    conversion = Case("valid JSON, bad request -> recipe's own 400 is untouched")
    status, payload, raised = await post(app, "/v1/messages", b"{}")
    conversion.check(status == 400, f"HTTP {status}")
    conversion.check(raised is None, f"no exception escaped ({raised})")
    message = (parse(payload).get("error") or {}).get("message", "")
    conversion.check(bool(message), f"error.message: {message[:100]!r}")
    conversion.check(not message.startswith("invalid JSON body"), "not attributed to the JSON boundary")
    cases.append(conversion)

    return cases


async def amain(module_path: str) -> int:
    module = load_module(module_path)
    app = build_app(module)
    print(f"\nmodule: {module_path}\n")
    cases = await run(app)
    failed = sum(1 for case in cases if not case.passed)
    for case in cases:
        print(f"[{'PASS' if case.passed else 'FAIL'}] {case.name}")
    print(f"\n{len(cases) - failed}/{len(cases)} cases passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", default="/opt/deepinfer/DeepInfer/serve/deepseek_server.py")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(amain(args.module)))
