"""Generate per-example predictions for one evaluation condition.

Conditions (build guide, section 10):
  A  base model, image only             --condition A
  B  base model, image + context        --condition B
  C  image-only adapter, image only     --condition C --adapter runs/C_image_only/best_adapter
  D  context adapter, image + context   --condition D --adapter runs/D_context/best_adapter

Context robustness: --context-variant correct|none|shuffled (shuffled = context
from a different article, a fixed derangement of the subset).

Output: results/predictions/<name>.jsonl (resumable; finished IDs are skipped).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from altcontext.config import ROOT, load_config  # noqa: E402
from altcontext.data import image_path, load_subset, record_context  # noqa: E402
from altcontext.model import environment_info, generate, load_model, peak_memory_mb, write_json  # noqa: E402

CONDITION_USES_CONTEXT = {"A": False, "B": True, "C": False, "D": True}


def shuffled_sources(rows, seed=13):
    """Map each row index to a donor row from a different article."""
    rng = random.Random(seed)
    n = len(rows)
    order = list(range(n))
    rng.shuffle(order)
    donor = {}
    for k, i in enumerate(order):
        for step in range(1, n):
            j = order[(k + step) % n]
            if rows[j]["source_page"] != rows[i]["source_page"]:
                donor[i] = j
                break
    return donor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--condition", choices=list("ABCD"), required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--subset", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--context-variant", choices=["correct", "none", "shuffled"], default=None)
    ap.add_argument("--mode", choices=["brief", "detailed"], default="brief")
    ap.add_argument("--name", help="output name; defaults to <condition>_<subset>[_<variant>]")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "predictions"))
    ap.add_argument("--device")
    ap.add_argument("--model-id", help="override the base model (e.g. HuggingFaceTB/SmolVLM-500M-Instruct)")
    ap.add_argument("--revision", help="base model revision for --model-id")
    ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    args = ap.parse_args()

    if args.condition in "CD" and not args.adapter:
        raise SystemExit("conditions C and D need --adapter")
    if args.condition in "AB" and args.adapter:
        raise SystemExit("conditions A and B use the original model; do not pass --adapter")

    cfg = load_config(args.config)
    if args.model_id:
        cfg["model"]["base_model_id"] = args.model_id
        cfg["model"]["base_model_revision"] = args.revision or "main"
    if args.dtype:
        cfg["model"]["dtype"] = args.dtype
    rows = load_subset(args.subset)[: args.limit]
    variant = args.context_variant or ("correct" if CONDITION_USES_CONTEXT[args.condition] else "none")
    name = args.name or f"{args.condition}_{args.subset}" + (f"_{variant}" if args.context_variant else "") + ("_detailed" if args.mode == "detailed" else "")
    out_path = Path(args.out_dir) / f"{name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in open(out_path) if line.strip()}
    todo = [(i, r) for i, r in enumerate(rows) if r["id"] not in done]
    print(f"{name}: {len(done)} done, {len(todo)} to go")
    if not todo:
        return

    lm = load_model(cfg, adapter_path=args.adapter, device=args.device)
    print(f"loaded {lm.version} on {lm.device} in {lm.load_seconds:.1f}s")
    generate(lm, image_path(rows[0]), "", args.mode)  # warm-up

    donor = shuffled_sources(rows) if variant == "shuffled" else {}
    with open(out_path, "a") as f:
        for k, (i, r) in enumerate(todo):
            if variant == "none":
                ctx, ctx_src = "", None
            elif variant == "shuffled":
                ctx_src = rows[donor[i]]["id"]
                ctx = record_context(rows[donor[i]], cfg)
            else:
                ctx, ctx_src = record_context(r, cfg), r["id"]
            g = generate(lm, image_path(r), ctx, args.mode)
            f.write(json.dumps({
                "id": r["id"],
                "condition": args.condition,
                "context_variant": variant,
                "context_source_id": ctx_src,
                "context": ctx,
                "correct_context": record_context(r, cfg),
                "reference": r["target_description"],
                "output": g["text"],
                "latency_ms": g["latency_ms"],
                "input_tokens": g["input_tokens"],
                "output_tokens": g["output_tokens"],
                "model_version": lm.version,
                "mode": args.mode,
            }, ensure_ascii=False) + "\n")
            f.flush()
            if (k + 1) % 25 == 0:
                print(f"  {k + 1}/{len(todo)}  {g['latency_ms']:.0f}ms  {g['text'][:80]}")

    write_json(out_path.with_suffix(".meta.json"), {
        "name": name, "condition": args.condition, "adapter": args.adapter, "base_model": cfg["model"],  "subset": args.subset, "n": len(rows),
        "context_variant": variant, "mode": args.mode, "model_version": lm.version,
        "cold_load_seconds": round(lm.load_seconds, 2), "peak_device_memory_mb": peak_memory_mb(lm.device),
        "generation": cfg["generation"][args.mode], "processor": cfg["processor"], "context_cfg": cfg["context"],
        "environment": environment_info(lm.device), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


if __name__ == "__main__":
    main()
