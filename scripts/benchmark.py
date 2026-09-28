"""End-to-end benchmark of the local AltContext service over HTTP.

For each model configuration, starts `python -m backend.app` as a subprocess and measures:
  - cold start: process spawn -> /health "ready" (imports + model load)
  - uncached brief requests on N test images with page context (client round trip and server generation time)
  - cached repeats of the same requests
  - detailed-mode requests (smaller sample)
  - a burst of concurrent requests (the service processes one at a time; measures queueing)
  - peak resident memory (RSS) of the server process

Usage:
  python scripts/benchmark.py                       # original, D, E adapters; n=100
  python scripts/benchmark.py --configs D --n 20    # quick run
Outputs results/benchmark/benchmark.json and results/benchmark/benchmark.md
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from altcontext.data import image_path, load_subset  # noqa: E402
from altcontext.model import environment_info  # noqa: E402

CONFIGS = {
    "original": "none",
    "D": "runs/D_context/best_adapter",
    "E": "runs/E_dropout30/best_adapter",
}
PORT = 8790
TOKEN = "benchmark-token"


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def summary(xs):
    return {"n": len(xs), "median": round(statistics.median(xs), 1), "p95": round(pct(xs, 0.95), 1), "mean": round(statistics.mean(xs), 1)}


class RssSampler(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.peak_mb, self.stop = pid, 0.0, False

    def run(self):
        while not self.stop:
            try:
                rss_kb = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(self.pid)]).strip() or 0)
                self.peak_mb = max(self.peak_mb, rss_kb / 1024)
            except Exception:
                pass
            time.sleep(0.25)


def metadata(r, mode="brief"):
    return json.dumps({"caption": r["caption"], "paragraph": r["context"], "page_title": r["page_title"], "mode": mode})


def post(client, r, mode="brief"):
    with open(image_path(r), "rb") as f:
        data = f.read()
    t0 = time.perf_counter()
    res = client.post(f"http://127.0.0.1:{PORT}/describe", headers={"X-AltContext-Token": TOKEN},
                      files={"image": ("i.jpg", data, "image/jpeg")}, data={"metadata": metadata(r, mode)}, timeout=180)
    rt = (time.perf_counter() - t0) * 1000
    res.raise_for_status()
    body = res.json()
    return rt, body


def bench_config(name, adapter, rows, n_detailed, burst):
    env = {**os.environ, "ALTCONTEXT_TOKEN": TOKEN, "PYTHONUNBUFFERED": "1"}
    t_spawn = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-m", "backend.app", "--port", str(PORT), "--adapter", adapter],
                            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sampler = RssSampler(proc.pid)
    sampler.start()
    out = {"config": name, "adapter": adapter}
    try:
        with httpx.Client() as client:
            while True:
                try:
                    h = client.get(f"http://127.0.0.1:{PORT}/health", timeout=2).json()
                    if h["status"] in ("ready", "error"):
                        break
                except httpx.HTTPError:
                    pass
                if proc.poll() is not None:
                    raise RuntimeError("backend exited during startup")
                time.sleep(0.2)
            if h["status"] != "ready":
                raise RuntimeError(h.get("error"))
            out["cold_start_s"] = round(time.perf_counter() - t_spawn, 2)
            out["model_load_s"] = h["load_seconds"]
            out["model_version"] = h["model_version"]

            for r in rows[-3:]:  # warm-up on images outside the measured set
                post(client, r)
            rows = rows[:-3]

            rt, gen, words = [], [], []
            for r in rows:
                t, b = post(client, r)
                assert not b["cached"]
                rt.append(t)
                gen.append(b["generation_latency_ms"])
                words.append(len(b["description"].split()))
            out["uncached_brief_roundtrip_ms"] = summary(rt)
            out["uncached_brief_generation_ms"] = summary(gen)
            out["brief_words_mean"] = round(statistics.mean(words), 1)
            out["throughput_req_per_min_sequential"] = round(60000 / statistics.mean(rt), 1)

            cached = []
            for r in rows:
                t, b = post(client, r)
                assert b["cached"]
                cached.append(t)
            out["cached_roundtrip_ms"] = summary(cached)

            det, det_words = [], []
            for r in rows[:n_detailed]:
                t, b = post(client, r, "detailed")
                det.append(t)
                det_words.append(len(b["description"].split()))
            out["uncached_detailed_roundtrip_ms"] = summary(det)
            out["detailed_words_mean"] = round(statistics.mean(det_words), 1)

        # Concurrency burst: distinct uncached inputs sent at once; the service runs one generation at a time.
        burst_rows = [dict(r, context=r["context"] + " (burst)") for r in rows[:burst]]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(burst) as ex, httpx.Client() as c:
            results = list(ex.map(lambda r: post(c, r)[0], burst_rows))
        wall = (time.perf_counter() - t0) * 1000
        out["burst"] = {"concurrent_requests": burst, "wall_ms": round(wall, 1), "per_request_roundtrip_ms": summary(results), "all_succeeded": len(results) == burst}
    finally:
        sampler.stop = True
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    out["peak_server_rss_mb"] = round(sampler.peak_mb, 1)
    return out


def write_markdown(res, path):
    L = [f"Hardware: {res['environment']['processor_name']}, {res['memory_gb']} GB RAM, device `{res['environment']['device']}`, fp32. "
         f"n = {res['n']} uncached brief requests (Concadia test images with caption + paragraph + page title), then the same {res['n']} repeated (cached), "
         f"{res['n_detailed']} detailed requests, and a burst of {res['burst']} concurrent requests. Latencies are client-side HTTP round trips unless noted.", "",
         "| Model | Cold start (spawn → ready) | Brief, uncached: median / p95 | Server generation median | Cached: median | Detailed: median | Brief words | Peak server RSS |",
         "|---|---|---|---|---|---|---|---|"]
    for r in res["results"]:
        L.append(f"| {r['config']} | {r['cold_start_s']:.1f} s | {r['uncached_brief_roundtrip_ms']['median'] / 1000:.2f} / {r['uncached_brief_roundtrip_ms']['p95'] / 1000:.2f} s | "
                 f"{r['uncached_brief_generation_ms']['median'] / 1000:.2f} s | {r['cached_roundtrip_ms']['median']:.1f} ms | {r['uncached_detailed_roundtrip_ms']['median'] / 1000:.2f} s | "
                 f"{r['brief_words_mean']} | {r['peak_server_rss_mb'] / 1024:.2f} GB |")
    L += ["", "| Model | Sequential throughput | Burst of concurrent requests: wall time | Burst: slowest request |", "|---|---|---|---|"]
    for r in res["results"]:
        b = r["burst"]
        L.append(f"| {r['config']} | {r['throughput_req_per_min_sequential']} req/min | {b['wall_ms'] / 1000:.1f} s for {b['concurrent_requests']} | {b['per_request_roundtrip_ms']['p95'] / 1000:.1f} s |")
    Path(path).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-detailed", type=int, default=10)
    ap.add_argument("--burst", type=int, default=8)
    args = ap.parse_args()
    rows = load_subset("test")[: args.n + 3]
    mem_gb = round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) / 2**30) if platform.system() == "Darwin" else None
    res = {"n": args.n, "n_detailed": args.n_detailed, "burst": args.burst, "memory_gb": mem_gb, "environment": environment_info("mps" if platform.system() == "Darwin" else "auto"),
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": []}
    for name in args.configs:
        print(f"benchmarking {name} ...", flush=True)
        r = bench_config(name, CONFIGS[name], rows, args.n_detailed, args.burst)
        print(json.dumps(r), flush=True)
        res["results"].append(r)
    out = ROOT / "results" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(res, indent=2))
    write_markdown(res, out / "benchmark.md")
    print((out / "benchmark.md").read_text())


if __name__ == "__main__":
    main()
