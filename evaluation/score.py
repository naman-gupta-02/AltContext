"""Score prediction files and build the comparison tables.

Usage:
  python evaluation/score.py --preds A_test B_test C_test D_test --out results/test_metrics
  python evaluation/score.py --robustness D_test_correct D_test_none D_test_shuffled --out results/robustness_D

Per-example scores are written next to the aggregate so every number can be
traced back to its image, reference, context and model.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from altcontext.config import ROOT  # noqa: E402
from altcontext.data import DATA_DIR  # noqa: E402
from altcontext.model import open_image, write_json  # noqa: E402
from evaluation.metrics import BertScorer, ClipScorer, cider_d, context_copy_rate, paired_bootstrap, repetition_rate, tokenize  # noqa: E402

PRED_DIR = ROOT / "results" / "predictions"
STOPWORDS = set("""a an the and or of in on at to for with by from as is are was were be been being this that these those it its his her their
there here which who whom whose what when where while into onto over under above below near next between behind front left right
up down out off than then so such very can could may might will would should one two three four five six seven eight nine ten
image photo photograph picture shows showing shown depicts depicting view background foreground black white small large
some several many few other another each his her he she they them him man woman person people""".split())


def load_preds(name):
    p = Path(name)
    if not p.exists():
        p = PRED_DIR / f"{name}.jsonl"
    return [json.loads(l) for l in open(p) if l.strip()]


def image_for(rid):
    return open_image(DATA_DIR / "raw" / "images" / (rid.split("_", 1)[1] + ".jpg"))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


class Scorers:
    def __init__(self, use_bert=True, use_clip=True):
        self.bert = BertScorer() if use_bert else None
        self.clip = ClipScorer() if use_clip else None


def score_file(name, scorers: Scorers, image_cache: dict):
    preds = load_preds(name)
    outs = [p["output"] for p in preds]
    refs = [p["reference"] for p in preds]
    cider, cider_each = cider_d(outs, [[r] for r in refs])
    per = [{"id": p["id"], "cider_d": c} for p, c in zip(preds, cider_each)]
    agg = {"name": Path(name).stem, "n": len(preds), "cider_d": cider}
    if scorers.bert:
        b = scorers.bert.score(outs, refs)
        agg["bertscore_f1"] = statistics.mean(b["f1"])
        for row, f in zip(per, b["f1"]):
            row["bertscore_f1"] = f
    if scorers.clip:
        imgs = [image_cache.setdefault(p["id"], image_for(p["id"])) for p in preds]
        c = scorers.clip.score(imgs, outs)
        agg["clipscore"] = statistics.mean(c)
        for row, s in zip(per, c):
            row["clipscore"] = s
        agg["clipscore_reference"] = statistics.mean(scorers.clip.score(imgs, refs))
    words = [len(o.split()) for o in outs]
    lat = [p["latency_ms"] for p in preds]
    for row, p, w in zip(per, preds, words):
        row.update(words=w, repetition=repetition_rate(p["output"]), context_copy=context_copy_rate(p["output"], p.get("correct_context") or p["context"]), latency_ms=p["latency_ms"])
    agg.update(
        mean_words=statistics.mean(words), reference_mean_words=statistics.mean(len(r.split()) for r in refs),
        repetition_rate=statistics.mean(r["repetition"] for r in per),
        context_copy_rate=statistics.mean(r["context_copy"] for r in per),
        empty_outputs=sum(1 for o in outs if not o.strip()),
        latency_ms_median=statistics.median(lat), latency_ms_p95=pct(lat, 0.95),
        model_version=preds[0]["model_version"], context_variant=preds[0]["context_variant"],
    )
    meta = Path(PRED_DIR / f"{Path(name).stem}.meta.json")
    if meta.exists():
        m = json.load(open(meta))
        agg["peak_device_memory_mb"] = m.get("peak_device_memory_mb")
        agg["cold_load_seconds"] = m.get("cold_load_seconds")
    return agg, per


def unsupported_import(shuffled_preds, none_preds):
    """Lexical proxy for distraction: content words in the output that come from the
    donor (wrong) context but appear in neither the reference, the correct context,
    nor the same model's no-context output for that image."""
    none_by_id = {p["id"]: p for p in none_preds}
    flagged, rows = 0, []
    for p in shuffled_preds:
        donor = set(tokenize(p["context"])) - STOPWORDS
        allowed = set(tokenize(p["reference"])) | set(tokenize(p["correct_context"])) | set(tokenize(none_by_id.get(p["id"], {}).get("output", "")))
        imported = sorted({w for w in tokenize(p["output"]) if w in donor and w not in allowed and w not in STOPWORDS and len(w) > 2})
        flagged += bool(imported)
        rows.append({"id": p["id"], "imported_words": imported})
    return flagged / max(1, len(shuffled_preds)), rows


def fmt(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="*", default=[])
    ap.add_argument("--robustness", nargs="*", default=[], help="correct none shuffled prediction files for one model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-bert", action="store_true")
    ap.add_argument("--no-clip", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scorers = Scorers(not args.no_bert, not args.no_clip)
    cache = {}

    names = args.preds + args.robustness
    aggs, pers = {}, {}
    for n in names:
        a, p = score_file(n, scorers, cache)
        aggs[a["name"]], pers[a["name"]] = a, p
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in a.items()}))
        with open(out / f"per_example_{a['name']}.jsonl", "w") as f:
            for row in p:
                f.write(json.dumps(row) + "\n")

    report = {"aggregate": aggs, "comparisons": {}}
    md = ["| Run | n | CIDEr-D | BERTScore F1 | CLIPScore | Words (ref) | Context copy | Repetition | Latency median / p95 (ms) |", "|---|---|---|---|---|---|---|---|---|"]
    for a in aggs.values():
        md.append(f"| {a['name']} | {a['n']} | {fmt(a['cider_d'])} | {fmt(a.get('bertscore_f1'))} | {fmt(a.get('clipscore'))} | {a['mean_words']:.1f} ({a['reference_mean_words']:.1f}) | {a['context_copy_rate']:.3f} | {a['repetition_rate']:.3f} | {a['latency_ms_median']:.0f} / {a['latency_ms_p95']:.0f} |")

    if args.preds:
        stems = {Path(n).stem.split("_")[0]: Path(n).stem for n in args.preds}
        pairs = [("A", "B", "Does adding context to the original model help?"), ("A", "C", "Does task fine-tuning help without context?"),
                 ("B", "D", "Does contextual fine-tuning improve on contextual prompting?"), ("C", "D", "How do the two trained approaches compare?")]
        md += ["", "Paired bootstrap (2000 resamples over the same examples); difference = second minus first.", "", "| Comparison | Question | ΔCIDEr-D [95% CI] | ΔBERTScore F1 [95% CI] | ΔCLIPScore [95% CI] |", "|---|---|---|---|---|"]
        for x, y, q in pairs:
            if x in stems and y in stems:
                px, py = pers[stems[x]], pers[stems[y]]
                assert [r["id"] for r in px] == [r["id"] for r in py], "prediction files must cover identical examples in the same order"
                comp = {m: paired_bootstrap([r[m] for r in px], [r[m] for r in py]) for m in ("cider_d", "bertscore_f1", "clipscore") if m in px[0]}
                report["comparisons"][f"{y}_vs_{x}"] = {"question": q, **comp}
                cell = lambda m: f"{comp[m]['diff']:+.3f} [{comp[m]['ci95'][0]:+.3f}, {comp[m]['ci95'][1]:+.3f}]" if m in comp else "-"
                md.append(f"| {y} vs {x} | {q} | {cell('cider_d')} | {cell('bertscore_f1')} | {cell('clipscore')} |")

    if len(args.robustness) == 3:
        corr, none, shuf = (load_preds(n) for n in args.robustness)
        rate, rows = unsupported_import(shuf, none)
        rate_corr, _ = unsupported_import([dict(p, context=s["context"]) for p, s in zip(corr, shuf)], none)
        changed = sum(c["output"] != s["output"] for c, s in zip(corr, shuf)) / len(corr)
        report["robustness"] = {
            "outputs_with_donor_only_words_shuffled": rate,
            "outputs_with_donor_only_words_correct_context_control": rate_corr,
            "output_changed_correct_vs_shuffled": changed,
            "note": "Lexical proxy. A word counts as imported when it occurs in the donor context but not in the reference, the correct context, or the no-context output. The control applies the same donor vocabulary to correct-context outputs, estimating chance overlap.",
        }
        with open(out / "imported_words_shuffled.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        md += ["", f"**Context robustness** — outputs containing donor-only words: shuffled context {rate:.1%} vs chance-overlap control {rate_corr:.1%}; outputs that changed between correct and shuffled context: {changed:.1%}."]

    write_json(out / "metrics.json", report)
    (out / "table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
