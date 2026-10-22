"""OpenAI 服务层性能测量：TTFT / TPOT / 单流与聚合吞吐。

    python tools/serve_bench.py --prompt-tokens 4000 --max-tokens 128 --concurrency 1 --count 4
"""
import argparse, json, statistics, sys, threading, time, urllib.request

BASE = "http://127.0.0.1:8888"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
p = argparse.ArgumentParser()
p.add_argument("--prompt-tokens", type=int, default=4000)
p.add_argument("--max-tokens", type=int, default=128)
p.add_argument("--concurrency", type=int, default=1)
p.add_argument("--count", type=int, default=4, help="total requests (multiple of concurrency)")
p.add_argument("--tag", default="")
p.add_argument("--no-cache", action="store_true", help="send prefix_cache:false to measure cold prefill")
a = p.parse_args()

doc = " ".join(f"w{i % 991}" for i in range(a.prompt_tokens))
results = []
lock = threading.Lock()


def one(idx: int) -> None:
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": "Read the document.\n\n" + doc},
                         {"role": "user", "content": f"Reply with the single word OK. ({idx})"}],
            "max_tokens": a.max_tokens, "temperature": 0, "stream": True,
            "ignore_eos": True,  # generate exactly max_tokens so TPOT/throughput are real
            **({"prefix_cache": False} if a.no_cache else {}),
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None; ntok = 0; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            q = line[6:]
            if q == "[DONE]":
                break
            o = json.loads(q)
            if o.get("usage"):
                usage = o["usage"]
            for c in o.get("choices", []):
                if c["delta"].get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    ntok += 1
    total = time.perf_counter() - t0
    dec = total - (ttft or total)
    with lock:
        results.append(dict(ttft=ttft or 0.0, total=total, ntok=ntok, dec=dec,
                            pt=(usage or {}).get("prompt_tokens", 0),
                            cached=((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens", -1)))


def run_batch(n: int) -> float:
    ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    t0 = time.perf_counter()
    [t.start() for t in ts]; [t.join() for t in ts]
    return time.perf_counter() - t0


results.clear()
wall = 0.0
for i in range(a.count // a.concurrency):
    wall += run_batch(a.concurrency)

ttfts = sorted(r["ttft"] for r in results)
tpots = sorted(r["dec"] / max(r["ntok"] - 1, 1) for r in results)
tot_out = sum(r["ntok"] for r in results)
pt = results[0]["pt"]
print(f"[{a.tag or 'bench'}] prompt~{pt} tok | max_tokens={a.max_tokens} | concurrency={a.concurrency} | n={a.count}")
print(f"  TTFT        p50 {statistics.median(ttfts)*1e3:8.1f} ms   p90 {ttfts[int(len(ttfts)*0.9)-1 if len(ttfts)>1 else 0]*1e3:8.1f} ms   min {ttfts[0]*1e3:8.1f} ms")
print(f"  TPOT        p50 {statistics.median(tpots)*1e3:8.1f} ms   -> 单流 {1/statistics.median(tpots):6.1f} tok/s")
print(f"  预填吞吐    {(pt/statistics.median(ttfts)):8.1f} tok/s  (prompt / TTFT)")
print(f"  聚合输出    {tot_out/wall:8.1f} tok/s  ({tot_out} tok / {wall:.2f} s wall, c={a.concurrency})")
print(f"  cached_tokens 均值 {sum(r['cached'] for r in results)/len(results):.0f}")
