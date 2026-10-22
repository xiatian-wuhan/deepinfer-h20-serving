"""OpenAI 服务层功能验证：非流式、流式、usage 字段、cached_tokens、metrics。"""
import json, sys, time, urllib.error, urllib.request

BASE = "http://127.0.0.1:8888"
MODEL = None  # filled in from /v1/models: --served-model-name is configurable
ok = fail = 0

def check(name, cond, detail=""):
    global ok, fail
    if cond: ok += 1; print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
    else:    fail += 1; print(f"  FAIL  {name}  [{detail}]")

def post(body, stream=False, path="/v1/chat/completions"):
    req = urllib.request.Request(BASE + path,
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=1800)
    return r

def get(path):
    return urllib.request.urlopen(BASE + path, timeout=60).read().decode()

print("=== 1) /v1/models ===")
m = json.loads(get("/v1/models"))
MODEL = m["data"][0]["id"]
check("返回 model list", m.get("object") == "list" and bool(MODEL), MODEL)

print("\n=== 2) 非流式对话 ===")
t0 = time.perf_counter()
r = post({"model": MODEL, "messages": [{"role": "user", "content": "用一句话说明什么是 KV cache。"}],
          "max_tokens": 64, "temperature": 0})
d = json.loads(r.read()); dt = time.perf_counter() - t0
ch = d["choices"][0]
check("有文本输出", bool(ch["message"]["content"].strip()), repr(ch["message"]["content"][:80]))
check("finish_reason 合法", ch["finish_reason"] in ("stop", "length"), ch["finish_reason"])
u = d["usage"]
check("usage 三字段齐全", all(k in u for k in ("prompt_tokens", "completion_tokens", "total_tokens")), json.dumps(u))
check("有 prompt_tokens_details.cached_tokens（vLLM 兼容）",
      isinstance(u.get("prompt_tokens_details", {}).get("cached_tokens"), int), json.dumps(u.get("prompt_tokens_details")))
print(f"  用时 {dt*1e3:.0f} ms, usage={json.dumps(u)}")

print("\n=== 3) 流式对话 SSE ===")
r = post({"model": MODEL, "messages": [{"role": "user", "content": "数一下 1 到 5，只输出数字。"}],
          "max_tokens": 32, "temperature": 0, "stream": True,
          "stream_options": {"include_usage": True}}, stream=True)
t0 = time.perf_counter(); first = None; parts = []; nchunk = 0; usage = None; done = False; roles = 0
for raw in r:
    line = raw.decode().strip()
    if not line.startswith("data: "): continue
    p = line[6:]
    if p == "[DONE]": done = True; break
    o = json.loads(p); nchunk += 1
    if o.get("usage"): usage = o["usage"]
    if o.get("object") != "chat.completion.chunk": check("chunk.object", False, o.get("object"))
    for c in o.get("choices", []):
        if c["delta"].get("role") == "assistant": roles += 1
        t = c["delta"].get("content")
        if t:
            if first is None: first = time.perf_counter() - t0
            parts.append(t)
check("收到 [DONE]", done)
check("首帧带 role=assistant", roles >= 1, f"roles={roles}")
check("有增量文本", len(parts) > 0, f"chunks={nchunk}")
check("流式也返回 usage（include_usage）", usage is not None, json.dumps(usage))
if usage:
    check("流式 usage 含 cached_tokens", isinstance(usage.get("prompt_tokens_details", {}).get("cached_tokens"), int))
print(f"  chunks={nchunk}, TTFT={first*1e3:.0f} ms, text={''.join(parts)[:80]!r}")

print("\n=== 4) 同一长 prompt 连发两次 -> cached_tokens 应该 > 0 ===")
import uuid
# Unique marker FIRST: the prefix cache matches from the start, so a suffix would still hit.
doc = f"run{uuid.uuid4().hex} " + " ".join(f"tok{i%997}" for i in range(2000))
msg = [{"role": "system", "content": "You answer questions about this document.\n\n" + doc},
       {"role": "user", "content": "Reply with the single word: OK"}]
res = []
for i in range(2):
    r = post({"model": MODEL, "messages": msg, "max_tokens": 8, "temperature": 0})
    d = json.loads(r.read()); c = d["usage"]["prompt_tokens_details"]["cached_tokens"]; res.append(c)
    print(f"  第 {i+1} 次: prompt_tokens={d['usage']['prompt_tokens']} cached_tokens={c}")
check("首次请求 cached_tokens 为 0", res[0] == 0, str(res[0]))
check("第二次命中缓存 cached_tokens > 0", res[1] > 0, str(res[1]))

print("\n=== 5) /deepinfer/metrics ===")
txt = get("/deepinfer/metrics")
lines = txt.splitlines()
hic = [l for l in lines if "hicache" in l and ("hits" in l or "lookups" in l or "hit_rate" in l)]
check("有 hicache 指标族", len(hic) > 0, f"{len(hic)} 条")
check("有请求计数指标", any("requests_total" in l for l in lines))
for l in hic[:4]: print("   ", l)

print("\n=== 6) 图片输入（OpenAI image_url / Anthropic image）===")
import base64, glob, os
_IMG = os.environ.get("SERVE_TEST_IMAGE") or next(iter(sorted(glob.glob(
    "/home/ubuntu/data1/models/*/inference/examples/images/*.jpeg"))), None)
if not _IMG:
    print("  SKIP  找不到测试图片（用 SERVE_TEST_IMAGE 指定）")
else:
    _b64 = base64.b64encode(open(_IMG, "rb").read()).decode()
    _txt = [{"role": "user", "content": "Reply with one word."}]
    _base = json.loads(post({"model": MODEL, "messages": _txt, "max_tokens": 4, "temperature": 0}).read())
    _img = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64}"}},
        {"type": "text", "text": "Reply with one word."}]}]
    _r = json.loads(post({"model": MODEL, "messages": _img, "max_tokens": 8, "temperature": 0}).read())
    check("单图请求返回 200", _r.get("object") == "chat.completion")
    _n = _r["usage"]["prompt_tokens"]
    check("prompt_tokens 计入图片展开的 token", _n > _base["usage"]["prompt_tokens"] + 20,
          f"{_n} vs 纯文本 {_base['usage']['prompt_tokens']}")
    _anth = [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64}},
        {"type": "text", "text": "Reply with one word."}]}]
    check("Anthropic image 块可用（/v1/messages）",
          json.loads(post({"model": MODEL, "messages": _anth, "max_tokens": 8, "temperature": 0},
                          path="/v1/messages").read()).get("type") == "message")
    try:
        post({"model": MODEL, "messages": _anth, "max_tokens": 8, "temperature": 0})
        _anth_chat = False
    except urllib.error.HTTPError as exc:
        _anth_chat = exc.code == 400
    check("chat 端点拒绝 Anthropic image 块（严格 OpenAI schema）", _anth_chat)
    try:
        post({"model": MODEL, "max_tokens": 4, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
            {"type": "text", "text": "hi"}]}]})
        _rejected = False
    except urllib.error.HTTPError as exc:
        _rejected = exc.code == 400
    check("默认拒绝外部图片 URL（SSRF 防护）", _rejected)

print("\n=== 7) 工具调用（DSML -> OpenAI tool_calls）===")
_TOOLS = [{"type": "function", "function": {"name": "Bash", "description": "Run a shell command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
_ASK = [{"role": "user", "content": "用 Bash 工具列出 /tmp 目录，只调用工具。"}]
_r = json.loads(post({"model": MODEL, "max_tokens": 300, "temperature": 0,
                      "messages": _ASK, "tools": _TOOLS}).read())
_c = _r["choices"][0]
_tc = _c["message"].get("tool_calls") or []
check("非流式返回 tool_calls", len(_tc) == 1 and _tc[0]["function"]["name"] == "Bash", str(len(_tc)))
check("finish_reason 为 tool_calls", _c["finish_reason"] == "tool_calls", _c["finish_reason"])
try:
    _args = json.loads(_tc[0]["function"]["arguments"]) if _tc else {}
except json.JSONDecodeError:
    _args = {}
check("arguments 是合法 JSON 且含 command", "command" in _args, str(_args)[:80])
check("DSML 未泄漏到 content", "<\uff5cDSML\uff5c" not in (_c["message"].get("content") or ""))

# Streaming: accumulate delta.tool_calls the way an OpenAI client does.
_req = urllib.request.Request(BASE + "/v1/chat/completions",
    data=json.dumps({"model": MODEL, "max_tokens": 300, "temperature": 0, "stream": True,
                     "messages": _ASK, "tools": _TOOLS}).encode(),
    headers={"Content-Type": "application/json"})
_sfin, _scalls, _leak = None, {}, False
with urllib.request.urlopen(_req, timeout=300) as _resp:
    for _line in _resp:
        _line = _line.decode().strip()
        if not _line.startswith("data: {"):
            continue
        _o = json.loads(_line[6:])
        for _ch in _o.get("choices") or []:
            if _ch.get("finish_reason"):
                _sfin = _ch["finish_reason"]
            if "\uff5cDSML\uff5c" in ((_ch.get("delta") or {}).get("content") or ""):
                _leak = True
            for _d in (_ch.get("delta") or {}).get("tool_calls") or []:
                _e = _scalls.setdefault(_d["index"], {"name": "", "args": ""})
                _f = _d.get("function") or {}
                _e["name"] += _f.get("name") or ""
                _e["args"] += _f.get("arguments") or ""
check("流式返回 delta.tool_calls", len(_scalls) == 1 and _scalls[0]["name"] == "Bash", str(list(_scalls)))
check("流式 finish_reason 为 tool_calls", _sfin == "tool_calls", str(_sfin))
check("流式 DSML 未泄漏", not _leak)

print(f"\n===== 结果: {ok} passed, {fail} failed =====")
sys.exit(1 if fail else 0)
