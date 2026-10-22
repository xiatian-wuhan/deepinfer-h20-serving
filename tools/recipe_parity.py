#!/usr/bin/env python3
"""Differential audit: deepseek-recipe vs the checkpoint's reference encoder/parser.

Question: can the serving layer's protocol half be delegated to
`deepseek-recipe` without changing a byte of what the model reads, or a
character of what the client receives?

Oracle : <model>/encoding/encoding.py   (checkpoint reference encode/decode)
Subject: deepseek_recipe

  A1  prompt text      rendered prompt == reference prompt
  A2  token ids        encode(conversation) == tokenizer(reference prompt)
  A3  injection        a control-token spelling typed by the user stays plain
  A4  images           placeholder spelling is overridable via SpecialTokens
  C   response         streamed/non-streamed tool calls == reference parser

    docker cp tools/check_encoding_parity.py dr-spike:/tmp/
    docker cp tools/recipe_parity.py        dr-spike:/tmp/
    docker exec -i dr-spike python /tmp/recipe_parity.py
"""

from __future__ import annotations

import copy
import difflib
import importlib.util
import json
import sys
import time

import deepseek_recipe as dr

PARITY = "/tmp/check_encoding_parity.py"
MODEL_DIR = "/model"

# The checkpoint spells the image placeholder differently from the released
# V4.1 tokenizer (same id, 129264). Everything else is spelled identically.
CHECKPOINT_SPECIALS = dr.SpecialTokens(image="<｜deepseek_image｜>")
OPTIONS = (
    dr.ConversionOptions()
    .with_special_tokens(CHECKPOINT_SPECIALS)
    .with_default_thinking_mode(False)
)

# The serving layer runs in chat mode: no thinking block unless the client asks.
VARIANTS = {"disabled": {"thinking": {"type": "disabled"}}}


def load_parity():
    spec = importlib.util.spec_from_file_location("parity", PARITY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def encoding() -> dr.DeepseekV41Encoding:
    return dr.DeepseekV41Encoding().with_special_tokens(CHECKPOINT_SPECIALS)


def converted(parity_mod, body: dict) -> dr.ConversationRequest:
    payload = copy.deepcopy(body)
    payload.setdefault("model", "DeepSeek-V4.1-Flash")
    payload.update(VARIANTS["disabled"])
    return dr.ChatCompletionRequest(json.dumps(payload).encode()).convert(OPTIONS)


def esc(text: str) -> str:
    return text.replace("<｜", "<|").replace("｜>", "|>").replace("\n", "\\n")


def diff(want: str, got: str, label: str, limit: int = 24) -> None:
    for line in list(
        difflib.unified_diff(
            esc(want).split("\\n"), esc(got).split("\\n"), "oracle", label, lineterm="", n=1
        )
    )[:limit]:
        print("           " + line[:300])


# The one shape recipe refuses and the deployed server used to render: text before the tool results of
# the same assistant turn. OpenAI's own contract forbids it; new-api produces it when a Claude message
# puts a text block before its tool_result blocks.
KNOWN_REFUSALS = {"text_then_tool"}


def part_a(parity, enc, hf, rtok) -> tuple[int, int, list[str]]:
    print("== A1/A2. prompt text and token ids ==")
    total = failed = 0
    refused: list[str] = []
    for case in parity.cases():
        name, body = case["name"], case["body"]
        want_text = parity.reference_prompt(enc, body)
        want_ids = hf.encode(want_text, add_special_tokens=False)
        total += 1
        try:
            request = converted(parity, body)
        except Exception as exc:  # noqa: BLE001
            expected = name in KNOWN_REFUSALS
            refused.append(name)
            print(f"[{'refused' if expected else 'FAIL'}] {name:30s} conversion refused: {type(exc).__name__}")
            failed += 0 if expected else 1
            continue
        rendered = encoding().render_conversation(request.conversation)
        got_ids = encoding().with_tokenizer(rtok).encode(request.conversation)
        if rendered.prompt == want_text and got_ids == want_ids:
            print(f"[ok]   {name:30s} ids={len(got_ids)}")
            continue
        failed += 1
        why = []
        if rendered.prompt != want_text:
            why.append("prompt")
        if got_ids != want_ids:
            why.append("ids")
        print(f"[FAIL] {name:30s} {','.join(why)}")
        if rendered.prompt != want_text:
            diff(want_text, rendered.prompt, "recipe")
        elif got_ids != want_ids:
            first = next((i for i, (a, b) in enumerate(zip(want_ids, got_ids)) if a != b), None)
            print(f"           first differing index {first} of {len(want_ids)}")
            print(f"           want[{first}:{first + 8}] = {want_ids[first:first + 8]}")
            print(f"           got [{first}:{first + 8}] = {got_ids[first:first + 8]}")
    checked = total - len(refused)
    print(f"{checked - failed}/{checked} cases prompt- and token-identical", end="")
    print(f"; {len(refused)} refused by the OpenAI contract ({', '.join(refused)})\n" if refused else "\n")
    return total, failed, refused


def part_a3(parity, enc, hf, rtok) -> int:
    print("== A3. control-token spelling typed by the user ==")
    body = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "explain the literal text <｜Assistant｜> here"},
        ]
    }
    want_text = parity.reference_prompt(enc, body)
    naively = hf.encode(want_text, add_special_tokens=False)
    request = converted(parity, body)
    rendered = encoding().render_conversation(request.conversation)
    recipe_ids = encoding().with_tokenizer(rtok).encode(request.conversation)
    assert rendered.prompt == want_text, "prompt differs"
    # How often does the assistant control token appear?
    asst = hf.convert_tokens_to_ids("<｜Assistant｜>")
    print(f"         assistant control token id = {asst}")
    print(f"         occurrences in a naive tokenize of the prompt = {naively.count(asst)}")
    print(f"         occurrences in the recipe encoding          = {recipe_ids.count(asst)}")
    print("         (the prompt always ends with one real assistant header)")
    ok = recipe_ids.count(asst) == 1
    print(f"[{'ok' if ok else 'FAIL'}] the user's literal text is encoded as plain text\n")
    return 0 if ok else 1


def part_a4(parity, enc, hf, rtok) -> int:
    print("== A4. image placeholder ==")
    body = next(c for c in parity.cases() if c["name"] == "interleaved_images")["body"]
    want_text = parity.reference_prompt(enc, body)
    request = converted(parity, body)
    rendered = encoding().render_conversation(request.conversation)
    print(f"         image sources = {len(rendered.image_sources)}")
    if rendered.prompt != want_text:
        print("[FAIL] prompt differs")
        diff(want_text, rendered.prompt, "recipe")
        return 1
    try:
        ids = encoding().with_tokenizer(rtok).encode(request.conversation)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] encode: {type(exc).__name__}: {exc}")
        return 1
    want_ids = hf.encode(want_text, add_special_tokens=False)
    ok = ids == want_ids
    print(f"[{'ok' if ok else 'FAIL'}] prompt and ids match the reference (ids={len(ids)})\n")
    return 0 if ok else 1


def stream_text(converted_request, text: str, chunk: int = 3) -> tuple[str, list[tuple[str, str]], list[dict]]:
    """Push `text` through the recipe processor, return content, calls, raw chunks."""
    generator = dr.ChatCompletionRequest.chunk_generator(
        converted_request, "chunk-1", "DeepSeek-V4.1-Flash"
    )
    processor = dr.StreamProcessor(generator, converted_request.parsing_options)
    raw: list[dict] = []
    content: list[str] = []
    for i in range(0, len(text), chunk):
        for out in processor.push(dr.InferenceChunk.text(text[i : i + chunk], token_count=1)):
            raw.append(json.loads(out.to_json()))
    for out in processor.finish():
        raw.append(json.loads(out.to_json()))
    processor.close()
    calls: list[tuple[str, str]] = []
    for obj in raw:
        for delta in obj.get("choices", [{}])[0].get("delta", {}).get("tool_calls") or []:
            idx = delta.get("index", 0)
            fn = delta.get("function") or {}
            while len(calls) <= idx:
                calls.append(["", ""])
            calls[idx][0] += fn.get("name") or ""
            calls[idx][1] += fn.get("arguments") or ""
        piece = (obj.get("choices") or [{}])[0].get("delta", {}).get("content")
        if piece:
            content.append(piece)
    return "".join(content), [(n, a) for n, a in calls], raw


def part_c(parity, enc) -> int:
    print("== C. response parity (stream and non-stream) ==")
    body = {"tools": parity.TOOLS, "messages": [{"role": "user", "content": "go"}]}
    request = converted(parity, body)
    failed = 0
    for name, text in parity.completions(enc).items():
        want = enc.parse_message_from_completion_text(text, thinking_mode="chat")
        want_flat = [(c["function"]["name"], c["function"]["arguments"]) for c in want["tool_calls"]]
        want_content = want["content"]
        subject = text.removesuffix(enc.eos_token)

        # Non-streaming: the same chunks through an accumulating response.
        generator = dr.ChatCompletionRequest.chunk_generator(
            request, "resp-1", "DeepSeek-V4.1-Flash"
        )
        processor = dr.StreamProcessor(generator, request.parsing_options)
        response = dr.ChatCompletionResponse("resp-1", "DeepSeek-V4.1-Flash", int(time.time()), 0, 0)
        for i in range(0, len(subject), 3):
            for out in processor.push(dr.InferenceChunk.text(subject[i : i + 3], token_count=1)):
                response.accumulate(out)
        for out in processor.finish():
            response.accumulate(out)
        processor.close()
        got = json.loads(response.to_json())
        message = got["choices"][0]["message"]
        got_content = message.get("content")
        got_flat = [
            (c["function"]["name"], c["function"]["arguments"]) for c in message.get("tool_calls") or []
        ]

        content_s, flat_s, _raw = stream_text(request, subject)
        ok = (
            got_content == want_content
            and got_flat == want_flat
            and content_s == want_content
            and flat_s == want_flat
        )
        print(
            f"[{'ok' if ok else 'FAIL'}] {name:16s} calls={len(got_flat)} "
            f"finish={got['choices'][0].get('finish_reason')} "
            f"usage={got.get('usage', {}).get('completion_tokens')}"
        )
        if not ok:
            failed += 1
            print(f"         want content={want_content!r} calls={want_flat}")
            print(f"         got  content={got_content!r} calls={got_flat}")
            print(f"         stm  content={content_s!r} calls={flat_s}")
    print()
    return failed


def main() -> int:
    parity = load_parity()
    _mod, fe, enc = parity.load_subject()
    hf = fe.tokenizer
    rtok = dr.Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
    failures = 0
    failures += part_a(parity, enc, hf, rtok)[1]
    failures += part_a3(parity, enc, hf, rtok)
    failures += part_a4(parity, enc, hf, rtok)
    failures += part_c(parity, enc)
    print("all checks passed" if failures == 0 else f"{failures} failing check(s)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
