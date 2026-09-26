"""First milestone: run the original model on N examples with and without context.

Writes a per-example comparison CSV/JSONL and a small timing/memory report.

Usage:
  python scripts/run_baselines.py --subset milestone_100 --out results/milestone
  python scripts/run_baselines.py --subset milestone_100 --model-id HuggingFaceTB/SmolVLM-500M-Instruct --revision <sha>
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from altcontext.config import load_config  # noqa: E402
from altcontext.data import image_path, load_subset, record_context  # noqa: E402
from altcontext.model import environment_info, generate, load_model, peak_memory_mb, write_json  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--subset", default="milestone_100")
    ap.add_argument("--out", default="results/milestone")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model-id")
    ap.add_argument("--revision")
    ap.add_argument("--device")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model_id:
        cfg["model"]["base_model_id"] = args.model_id
        cfg["model"]["base_model_revision"] = args.revision or "main"
    rows = load_subset(args.subset)[: args.limit]
    lm = load_model(cfg, device=args.device)
    print(f"model {lm.version} loaded on {lm.device} in {lm.load_seconds:.1f}s")

    # One warm-up generation so the first measured example is not a cold start.
    generate(lm, image_path(rows[0]), "", "brief")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = cfg["model"]["base_model_id"].split("/")[-1]
    results = []
    for i, r in enumerate(rows):
        ctx = record_context(r, cfg)
        a = generate(lm, image_path(r), "", "brief")
        b = generate(lm, image_path(r), ctx, "brief")
        results.append({
            "example_id": r["id"],
            "split": r["split"],
            "image_path": r["image_path"],
            "context": ctx,
            "reference": r["target_description"],
            "output_image_only": a["text"],
            "output_with_context": b["text"],
            "model_revision": lm.version,
            "generation_settings": json.dumps(cfg["generation"]["brief"]),
            "latency_ms_image_only": a["latency_ms"],
            "latency_ms_with_context": b["latency_ms"],
            "input_tokens_image_only": a["input_tokens"],
            "input_tokens_with_context": b["input_tokens"],
        })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(rows)}  A={a['latency_ms']:.0f}ms B={b['latency_ms']:.0f}ms")

    with open(out_dir / f"comparison_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    leaks = sum(r["reference"].lower() in r["context"].lower() for r in results if len(r["reference"].split()) >= 4)
    la = [r["latency_ms_image_only"] for r in results]
    lb = [r["latency_ms_with_context"] for r in results]
    report = {
        "model": lm.version,
        "subset": args.subset,
        "n": len(results),
        "cold_load_seconds": round(lm.load_seconds, 2),
        "latency_ms": {
            "image_only": {"median": statistics.median(la), "p95": pct(la, 0.95)},
            "with_context": {"median": statistics.median(lb), "p95": pct(lb, 0.95)},
        },
        "mean_words": {
            "reference": statistics.mean(len(r["reference"].split()) for r in results),
            "image_only": statistics.mean(len(r["output_image_only"].split()) for r in results),
            "with_context": statistics.mean(len(r["output_with_context"].split()) for r in results),
        },
        "peak_device_memory_mb": peak_memory_mb(lm.device),
        "reference_found_in_context": leaks,
        "processor": cfg["processor"],
        "context_cfg": cfg["context"],
        "generation": cfg["generation"]["brief"],
        "environment": environment_info(lm.device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(out_dir / f"report_{tag}.json", report)
    print(json.dumps({k: report[k] for k in ("n", "cold_load_seconds", "latency_ms", "mean_words", "peak_device_memory_mb", "reference_found_in_context")}, indent=2))


if __name__ == "__main__":
    main()
