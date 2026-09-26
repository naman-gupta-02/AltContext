"""Create the author error-review sheet: a fixed sample of outputs per condition.

The sheet is for YOUR manual review (author-reviewed analysis, not a user study).
Conditions are shown in a shuffled, blinded order per example so the reviewer
does not know which system produced which output; the key is saved separately.

Usage:
  python evaluation/make_error_review.py --preds A_test B_test C_test D_test --n 50
Outputs:
  evaluation/error_review/review_sheet.csv   fill in the 0/1 columns and notes
  evaluation/error_review/review_key.json    maps blinded labels back to conditions
  python evaluation/make_error_review.py --summarize   -> error_review/summary.md
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from altcontext.config import ROOT  # noqa: E402
from evaluation.score import load_preds  # noqa: E402

OUT = ROOT / "evaluation" / "error_review"
ERRORS = ["invented_object", "unsupported_attribute", "incorrect_relationship", "missing_important_detail", "copied_page_claim"]


def build(args):
    preds = {Path(n).stem.split("_")[0]: {p["id"]: p for p in load_preds(n)} for n in args.preds}
    ids = sorted(set.intersection(*[set(v) for v in preds.values()]))
    rng = random.Random(args.seed)
    sample = rng.sample(ids, min(args.n, len(ids)))
    OUT.mkdir(parents=True, exist_ok=True)
    key, rows = {}, []
    for ex in sample:
        conds = list(preds)
        rng.shuffle(conds)
        for j, c in enumerate(conds):
            label = f"{ex}#{'wxyz'[j]}"
            key[label] = c
            p = preds[c][ex]
            rows.append({"item": label, "example_id": ex, "image_path": f"data/raw/images/{ex.split('_', 1)[1]}.jpg", "reference": p["reference"], "page_context": p["correct_context"], "output": p["output"], **{e: "" for e in ERRORS}, "notes": ""})
    with open(OUT / "review_sheet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(key, open(OUT / "review_key.json", "w"), indent=1)
    print(f"wrote {len(rows)} rows ({len(sample)} examples x {len(preds)} conditions) to {OUT / 'review_sheet.csv'}")


def summarize(_args):
    key = json.load(open(OUT / "review_key.json"))
    counts, totals = defaultdict(lambda: defaultdict(int)), defaultdict(int)
    for row in csv.DictReader(open(OUT / "review_sheet.csv")):
        if not any(row[e].strip() for e in ERRORS):
            continue  # not reviewed yet
        c = key[row["item"]]
        totals[c] += 1
        for e in ERRORS:
            counts[c][e] += row[e].strip() in ("1", "y", "yes", "x", "true")
    lines = ["# Author error review (not a user study)", "", "| Condition | Reviewed | " + " | ".join(ERRORS) + " |", "|---|---|" + "---|" * len(ERRORS)]
    for c in sorted(totals):
        lines.append(f"| {c} | {totals[c]} | " + " | ".join(f"{counts[c][e]} ({counts[c][e] / totals[c]:.0%})" for e in ERRORS) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="*", default=["A_test", "B_test", "C_test", "D_test"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize(a) if a.summarize else build(a)
