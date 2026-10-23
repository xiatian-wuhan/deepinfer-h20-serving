#!/usr/bin/env python3
"""Differential audit of message ordering against the checkpoint's own encoder.

Oracle: `<model>/encoding/encoding.py` (DeepSeek-V4.1 reference encoder/decoder).
Subject: the deployed serving layer (`serve/deepseek_server.py`) - the code that
turns an OpenAI chat request into prompt tokens and model text back into
`tool_calls`.

Three independent checks:

  A  prompt parity      render_prompt(body) tokens == encode_messages(...) tokens
  B  prefix invariant   turn N's prompt is a prefix of turn N+1's prompt
                        (what the KV prefix cache relies on; a mid-history change
                        also means the model reads a different conversation)
  C  response parity    split_tool_calls / _ToolCallGate reconstruction ==
                        parse_message_from_completion_text

Run inside the serving container:

    docker cp check_encoding_parity.py di-e2e:/tmp/
    docker exec di-e2e python /tmp/check_encoding_parity.py
"""

from __future__ import annotations

import argparse
import base64
import copy
import importlib.util
import json
import sys
from pathlib import Path

MODEL_DIR = Path("/model")
SERVER_PATH = Path("/opt/deepinfer/DeepInfer/serve/deepseek_server.py")

# 1x1 PNG; enough to exercise the image path without shipping a fixture.
PNG_1X1 = base64.b64encode(
    base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
).decode()
DATA_URL = f"data:image/png;base64,{PNG_1X1}"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch",
            "description": "Fetch a URL",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]


def call(cid: str, name: str, args: dict) -> dict:
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def cases() -> list[dict]:
    """OpenAI-shaped requests: what a gateway hands the serving layer."""
    return [
        {
            "name": "plain_multiturn",
            "body": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "first answer"},
                    {"role": "user", "content": "second question"},
                ]
            },
        },
        {
            "name": "single_tool",
            "body": {
                "tools": TOOLS,
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "look it up"},
                    {
                        "role": "assistant",
                        "content": "checking",
                        "tool_calls": [call("call_a", "lookup", {"query": "x"})],
                    },
                    {"role": "tool", "tool_call_id": "call_a", "content": "42"},
                    {"role": "user", "content": "thanks, continue"},
                ],
            },
        },
        {
            # Parallel calls; the gateway delivered the results backwards.
            "name": "parallel_tools_reversed",
            "body": {
                "tools": TOOLS,
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "do two things"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            call("call_1", "lookup", {"query": "one"}),
                            call("call_2", "fetch", {"url": "http://b"}),
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_2", "content": "B result"},
                    {"role": "tool", "tool_call_id": "call_1", "content": "A result"},
                    {"role": "user", "content": "summarize"},
                ],
            },
        },
        {
            # `[tool_result, text]` inside one Claude message -> gateway: tool then user.
            "name": "tool_then_text",
            "body": {
                "tools": TOOLS,
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "go"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [call("call_x", "lookup", {"query": "q"})],
                    },
                    {"role": "tool", "tool_call_id": "call_x", "content": "R"},
                    {"role": "user", "content": "and now?"},
                ],
            },
        },
        {
            # `[text, tool_result]` inside one Claude message -> gateway: user then tool.
            "name": "text_then_tool",
            "body": {
                "tools": TOOLS,
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "go"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [call("call_y", "lookup", {"query": "q"})],
                    },
                    {"role": "user", "content": "meanwhile, a note"},
                    {"role": "tool", "tool_call_id": "call_y", "content": "R"},
                    {"role": "user", "content": "continue"},
                ],
            },
        },
        {
            # Mid-conversation system message (V4.1 feature).
            "name": "mid_conversation_system",
            "body": {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "one"},
                    {"role": "assistant", "content": "two"},
                    {"role": "system", "content": "policy update: be terse"},
                    {"role": "user", "content": "three"},
                ]
            },
        },
        {
            "name": "interleaved_images",
            "body": {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "before"},
                            {"type": "image_url", "image_url": {"url": DATA_URL}},
                            {"type": "text", "text": "after"},
                        ],
                    },
                ]
            },
        },
        {
            "name": "consecutive_user_messages",
            "body": {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "part one"},
                    {"role": "user", "content": "part two"},
                ]
            },
        },
        {
            "name": "no_system_message",
            "body": {"tools": TOOLS, "messages": [{"role": "user", "content": "hello"}]},
        },
        {
            "name": "assistant_text_plus_calls",
            "body": {
                "tools": TOOLS,
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": "I will look it up.",
                        "tool_calls": [call("call_z", "lookup", {"query": "z"})],
                    },
                    {"role": "tool", "tool_call_id": "call_z", "content": "Z"},
                    {"role": "user", "content": "ok"},
                ],
            },
        },
        {
            # Three-turn tool loop: the shape Claude Code produces all day long.
            "name": "claude_code_like",
            "body": {
                "tools": TOOLS,
                "messages": [
                    {"role": "system", "content": "sys prompt with tools"},
                    {"role": "user", "content": "fix the bug"},
                    {
                        "role": "assistant",
                        "content": "Reading the file.",
                        "tool_calls": [call("c1", "lookup", {"query": "a"})],
                    },
                    {"role": "tool", "tool_call_id": "c1", "content": "line 1"},
                    {
                        "role": "assistant",
                        "content": "Now fetching.",
                        "tool_calls": [call("c2", "fetch", {"url": "u"})],
                    },
                    {"role": "tool", "tool_call_id": "c2", "content": "page"},
                    {"role": "user", "content": "continue"},
                ],
            },
        },
    ]


def load_subject():
    sys.path.insert(0, str(MODEL_DIR / "encoding"))
    import encoding as enc

    spec = importlib.util.spec_from_file_location("df_server", SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from transformers import AutoTokenizer

    fe = object.__new__(mod.Frontend)
    fe.args = argparse.Namespace(
        model=str(MODEL_DIR),
        served_model_name="DeepSeek-V4.1-Flash-1017",
        thinking_mode="chat",
        log_requests=False,
        allow_image_urls=False,
        always_usage=False,
    )
    fe.model_name = fe.args.served_model_name
    fe.encode_messages = enc.encode_messages
    fe._parse_tool_calls = enc.parse_tool_calls
    fe._decode_dsml_to_arguments = enc.decode_dsml_to_arguments
    fe.dsml_token = enc.dsml_token
    fe.tool_calls_marker = f"<{enc.dsml_token}"
    fe.tool_calls_open = f"<{enc.dsml_token}{enc.tool_calls_block_name}"
    fe.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    vision = json.loads((MODEL_DIR / "config.json").read_text()).get("vision_config") or {}
    fe.vision = vision
    fe.has_vision = bool(vision)
    fe.stats = {"encode_fallbacks": 0}
    return mod, fe, enc


def reference_prompt(enc, body: dict) -> str:
    """Straight from the reference README: tools ride on the first message."""
    messages = copy.deepcopy(body["messages"])
    if body.get("tools"):
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": ""})
        messages[0]["tools"] = body["tools"]
    prompt, _media = enc.encode_messages(messages, thinking_mode="chat", return_multi_modal_data=True)
    return prompt


def check_prompt_parity(fe, enc) -> list[str]:
    failures = []
    print("\n== A. prompt parity (serving layer vs reference encoder) ==")
    for case in cases():
        body = case["body"]
        ids = fe.render_prompt(copy.deepcopy(body))[0]
        want_ids = fe.tokenizer.encode(reference_prompt(enc, body), add_special_tokens=False)
        ok = ids == want_ids
        print(f"  [{'ok ' if ok else 'FAIL'}] {case['name']:<28} tokens={len(ids)}")
        if not ok:
            failures.append(f"A/{case['name']}")
            first = next((i for i, (a, b) in enumerate(zip(ids, want_ids)) if a != b), None)
            print(f"         first divergence at token {first}")
            if first is not None:
                lo = max(0, first - 20)
                print(f"         serving: {fe.tokenizer.decode(ids[lo: first + 20])!r}")
                print(f"         oracle : {fe.tokenizer.decode(want_ids[lo: first + 20])!r}")
    return failures


def check_prefix_invariant(fe) -> list[str]:
    """Every turn must reuse the previous turn's prompt byte for byte."""
    failures = []
    print("\n== B. prefix invariance across turns (what the KV cache assumes) ==")
    history: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fix the bug"},
    ]
    renders: list[tuple[int, list[int]]] = []
    script = [
        {"role": "assistant", "content": "Reading.", "tool_calls": [call("c1", "lookup", {"query": "a"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "line 1"},
        {"role": "assistant", "content": "Fetching.", "tool_calls": [call("c2", "fetch", {"url": "u"})]},
        {"role": "tool", "tool_call_id": "c2", "content": "page"},
        {"role": "user", "content": "continue"},
    ]
    for step, message in enumerate(script):
        ids = fe.render_prompt({"tools": TOOLS, "messages": copy.deepcopy(history)})[0]
        renders.append((step, ids))
        if step:
            prev = renders[-2][1]
            ok = ids[: len(prev)] == prev
            print(
                f"  [{'ok ' if ok else 'FAIL'}] turn {step} prompt={len(ids):>5} "
                f"prev={len(prev):>5} +{len(ids) - len(prev)}"
            )
            if not ok:
                failures.append(f"B/turn{step}")
                first = next((i for i, (a, b) in enumerate(zip(ids, prev)) if a != b), len(prev))
                lo = max(0, first - 25)
                print(f"         diverges at token {first}")
                print(f"         prev: {fe.tokenizer.decode(prev[lo: first + 25])!r}")
                print(f"         next: {fe.tokenizer.decode(ids[lo: first + 25])!r}")
        else:
            print(f"  [ok ] turn 0 prompt={len(ids):>5}")
        history.append(message)
    return failures


def _blocks(enc) -> tuple[str, str, str]:
    """The exact DSML wrapper the model is trained to emit."""
    d, cb, ci, cp = enc.dsml_token, enc.tool_calls_block_name, enc.tool_call_tag_name, enc.tool_parameter_tag_name
    openning = f"<{d}{cb}>\n"
    closing = f"</{d}{cb}>"
    return openning, closing, (ci, cp)


def completions(enc) -> dict[str, str]:
    openning, closing, (ci, cp) = _blocks(enc)
    one = (
        openning
        + f'<{enc.dsml_token}{ci} name="lookup">\n'
        + f'<{enc.dsml_token}{cp} name="query" string="true">value</{enc.dsml_token}{cp}>\n'
        + f'<{enc.dsml_token}{cp} name="limit" string="false">2</{enc.dsml_token}{cp}>\n'
        + f'</{enc.dsml_token}{ci}>\n'
        + closing
    )
    two = (
        openning
        + f'<{enc.dsml_token}{ci} name="lookup">\n'
        + f'<{enc.dsml_token}{cp} name="query" string="true">one</{enc.dsml_token}{cp}>\n'
        + f'</{enc.dsml_token}{ci}>\n'
        + f'<{enc.dsml_token}{ci} name="fetch">\n'
        + f'<{enc.dsml_token}{cp} name="url" string="true">http://x</{enc.dsml_token}{cp}>\n'
        + f'</{enc.dsml_token}{ci}>\n'
        + closing
    )
    eos = enc.eos_token
    return {
        "text_only": "2 + 2 = 4." + eos,
        "one_call": "summary" + "\n\n" + one + eos,
        "two_calls": "doing both" + "\n\n" + two + eos,
        "calls_only": "\n\n" + one + eos,
        "arg_has_newline": "x" + "\n\n"
        + openning
        + f'<{enc.dsml_token}{ci} name="lookup">\n'
        + f'<{enc.dsml_token}{cp} name="query" string="true">line1\nline2</{enc.dsml_token}{cp}>\n'
        + f'</{enc.dsml_token}{ci}>\n'
        + closing
        + eos,
    }


def check_response_parity(mod, fe, enc) -> tuple[list[str], list[str]]:
    failures, notes = [], []
    print("\n== C1. response parity (stream/non-stream vs reference parser) ==")
    openning, closing, (ci, cp) = _blocks(enc)

    for name, text in completions(enc).items():
        want = enc.parse_message_from_completion_text(text, thinking_mode="chat")
        want_flat = [(c["function"]["name"], c["function"]["arguments"]) for c in want["tool_calls"]]

        # The serving layer never sees the EOS token: the detokenizer drops it. The
        # reference parser needs it as a stop marker, so strip it for the subject only.
        subject_text = text.removesuffix(enc.eos_token)

        got_content, got_calls = fe.split_tool_calls(subject_text)
        flat = [(c["function"]["name"], c["function"]["arguments"]) for c in got_calls]

        stream_content, stream_flat = _stream(fe, mod, subject_text, chunk=3)
        ok = (
            got_content == want["content"]
            and flat == want_flat
            and stream_content == want["content"]
            and stream_flat == want_flat
        )
        print(f"  [{'ok ' if ok else 'FAIL'}] {name:<16} calls={len(want_flat)}")
        if not ok:
            failures.append(f"C/{name}")
            print(f"         oracle  content={want['content']!r}\n                 calls={want_flat}")
            print(f"         batch   content={got_content!r}\n                 calls={flat}")
            print(f"         stream  content={stream_content!r}\n                 calls={stream_flat}")

    # C2. Layouts the model is not trained to emit but the checkpoint's own streaming parser
    # accepts (`NewlinesAndMarker`, and `ActionOnUnmatched::Skip` between the tag names).
    print("\n== C2. tolerant layouts (zero newlines, one newline, spaced tags) ==")
    args = '{"query": "value", "limit": 2}'
    lookup = ("lookup", args)
    near_misses = {
        "no_leading_newlines": "summary" + openning + _one_call(enc),
        "single_newline": "summary\n" + openning + _one_call(enc),
        "block_first": openning + _one_call(enc),
        "spaced_layout": (
            "summary\n\n"
            + f'<{enc.dsml_token}{enc.tool_calls_block_name}> '
            + f'<{enc.dsml_token}{ci} name="lookup"> '
            + f'<{enc.dsml_token}{cp} name="query" string="true">value</{enc.dsml_token}{cp}> '
            + f'<{enc.dsml_token}{cp} name="limit" string="false">2</{enc.dsml_token}{cp}> '
            + f'</{enc.dsml_token}{ci}>\n'
            + closing
        ),
    }
    for name, text in near_misses.items():
        want_content = "" if name == "block_first" else "summary"
        got_content, got_calls = fe.split_tool_calls(text)
        flat = [(c["function"]["name"], c["function"]["arguments"]) for c in got_calls]
        stream_content, stream_flat = _stream(fe, mod, text, chunk=2)
        ok = (
            got_content == want_content
            and flat == [lookup]
            and stream_content == want_content
            and stream_flat == [lookup]
            and "<" + enc.dsml_token not in got_content
        )
        print(f"  [{'ok ' if ok else 'FAIL'}] {name:<20} calls={len(flat)}")
        if not ok:
            failures.append(f"C2/{name}")
            print(f"         want content={want_content!r} calls={[lookup]}")
            print(f"         got  content={got_content!r} calls={flat}")
            print(f"         strm content={stream_content!r} calls={stream_flat}")

    # C3. Truncated block (max_tokens cut it): keep the raw text, never claim a tool call.
    truncated = "summary\n\n" + openning + f'<{enc.dsml_token}{ci} name="lookup">\n<{enc.dsml_token}{cp} name="query" string="true">val'
    got_content, got_calls = fe.split_tool_calls(truncated)
    ok = got_calls == [] and got_content == truncated
    print(f"  [{'ok ' if ok else 'FAIL'}] {'truncated_block':<20} calls={len(got_calls)} (raw text kept)")
    if not ok:
        failures.append("C3/truncated")
        print(f"         content={got_content[:120]!r} calls={got_calls}")

    # C4. Text after the closing block: the framework drops it and records
    # `InvalidTokenKind::ContentAfterFinished` (Stage::Finished), and so do we.
    tail_text = "summary\n\n" + openning + _one_call(enc) + closing + "\ntrailing words"
    got_content, got_calls = fe.split_tool_calls(tail_text)
    notes.append(
        f"text after the DSML block: dropped like deepseek-recipe (Stage::Finished => ContentAfterFinished); "
        f"serving content={got_content!r}"
    )
    print(f"  [note] text-after-block: content={got_content!r} calls={len(got_calls)}")
    return failures, notes


def _one_call(enc) -> str:
    openning, closing, (ci, cp) = _blocks(enc)
    return (
        f'<{enc.dsml_token}{ci} name="lookup">\n'
        f'<{enc.dsml_token}{cp} name="query" string="true">value</{enc.dsml_token}{cp}>\n'
        f'<{enc.dsml_token}{cp} name="limit" string="false">2</{enc.dsml_token}{cp}>\n'
        f'</{enc.dsml_token}{ci}>\n'
        + closing
    )


def _stream(fe, mod, text: str, chunk: int) -> tuple[str, list[tuple[str, str]]]:
    """Replay the SSE path: feed the text in `chunk`-sized pieces through the gate."""
    gate = mod._ToolCallGate(fe.tool_calls_marker)
    emitted: list[str] = []
    for i in range(0, len(text), chunk):
        out = gate.feed(text[i : i + chunk])
        if out:
            emitted.append(out)
    tail = gate.flush()
    if tail:
        emitted.append(tail)
    if gate.gated:
        _, calls = fe.split_tool_calls(gate.block)
        content = "".join(emitted)
    else:
        content, calls = fe.split_tool_calls("".join(emitted))
    return content, [(c["function"]["name"], c["function"]["arguments"]) for c in calls]


def dump_one(fe, enc, name: str) -> None:
    for case in cases():
        if case["name"] != name:
            continue
        prompt = reference_prompt(enc, case["body"])
        pretty = (
            prompt.replace("<｜User｜>", "\n<<<USER>>>\n")
            .replace("<｜Assistant｜>", "\n<<<ASST>>> ")
            .replace("<｜System｜>", "\n<<<SYS>>> ")
            .replace("<｜begin▁of▁sentence｜>", "<<<BOS>>>")
        )
        print(f"\n== rendered prompt: {case['name']} ==")
        print(pretty)
        print("\n== speaker order (serving layer log) ==")
        fe._log_message_order([dict(m) for m in case["body"]["messages"]])
        return


def main() -> int:
    mod, fe, enc = load_subject()
    failures = []
    failures += check_prompt_parity(fe, enc)
    failures += check_prefix_invariant(fe)
    resp_failures, notes = check_response_parity(mod, fe, enc)
    failures += resp_failures
    dump_one(fe, enc, "parallel_tools_reversed")
    print("\n== summary ==")
    for note in notes:
        print(f"note: {note}")
    if failures:
        print(f"FAIL: {len(failures)} check(s): {failures}")
        return 1
    print("all checks passed: serving layer == reference encoder, in order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
