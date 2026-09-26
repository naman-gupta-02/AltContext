"""Convert raw Concadia into a clean, normalized manifest plus fixed subsets.

Outputs (under data/processed/):
  manifest.jsonl          every raw record with accept/exclusion status
  subsets/<name>.jsonl    fixed experiment subsets (IDs saved for reproducibility)
  data_report.json / .md  counts, exclusions, duplicate audit, examples

Usage:
  python scripts/prepare_data.py --raw data/raw --out data/processed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from altcontext.text import normalize_text  # noqa: E402
from altcontext.config import load_config  # noqa: E402

try:
    import imagehash
except ImportError:  # near-duplicate detection becomes unavailable
    imagehash = None

MIN_TARGET_WORDS = 2
MAX_TARGET_WORDS = 120
PLACEHOLDER_TARGETS = {"image", "photo", "picture", "thumbnail", "logo", "icon", "see caption", "refer to caption", "caption", "alt text"}


def image_check(path: Path):
    """Return (ok, reason, md5, phash, size)."""
    if not path.exists():
        return False, "image_missing", None, None, None
    try:
        data = path.read_bytes()
        md5 = hashlib.md5(data).hexdigest()
        with Image.open(path) as im:
            im = im.convert("RGB")
            size = im.size
            ph = str(imagehash.phash(im)) if imagehash else None
        return True, None, md5, ph, size
    except Exception as e:  # corrupt or undecodable
        return False, f"image_unreadable:{type(e).__name__}", None, None, None


def target_exclusion(target: str, caption: str, context: str) -> str | None:
    words = target.split()
    if not target:
        return "empty_target"
    if target.lower().strip(" .") in PLACEHOLDER_TARGETS:
        return "placeholder_target"
    if len(words) < MIN_TARGET_WORDS:
        return "target_too_short"
    if len(words) > MAX_TARGET_WORDS:
        return "target_too_long"
    return None


def leakage_flags(target: str, caption: str, context: str) -> list[str]:
    flags = []
    t = target.lower().strip(" .")
    if t and t == caption.lower().strip(" ."):
        flags.append("target_equals_caption")
    elif len(t.split()) >= 4 and t in caption.lower():
        flags.append("target_in_caption")
    if len(t.split()) >= 4 and t in context.lower():
        flags.append("target_in_paragraph")
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--config", default="configs/default.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw, out = Path(args.raw), Path(args.out)
    (out / "subsets").mkdir(parents=True, exist_ok=True)

    records = json.load(open(raw / "wiki_split.json"))["images"]
    print(f"raw records: {len(records)}")

    manifest = []
    md5_index, phash_index = defaultdict(list), defaultdict(list)
    for i, r in enumerate(records):
        rid = Path(r["filename"]).stem
        caption = normalize_text(r["caption"]["raw"])
        context = normalize_text(r["context"]["raw"])
        target = normalize_text(r["description"]["raw"])
        split = {"val": "validation"}.get(r["split"], r["split"])
        rec = {
            "id": f"concadia_{rid}",
            "image_path": f"images/{r['filename']}",
            "source_page": r["article_id"],
            "page_title": r["article_id"].replace("_", " "),
            "source_url": r.get("orig_filename"),
            "caption": caption,
            "context": context,
            "target_description": target,
            "split": split,
            "status": "accepted",
            "exclusion_reason": None,
            "flags": [],
        }
        ok, reason, md5, ph, size = image_check(raw / rec["image_path"])
        if not ok:
            rec.update(status="excluded", exclusion_reason=reason)
        else:
            rec["image_md5"], rec["image_phash"], rec["image_size"] = md5, ph, size
            reason = target_exclusion(target, caption, context)
            if reason:
                rec.update(status="excluded", exclusion_reason=reason)
            rec["flags"] = leakage_flags(target, caption, context)
            md5_index[md5].append(len(manifest))
            if ph:
                phash_index[ph].append(len(manifest))
        manifest.append(rec)
        if (i + 1) % 10000 == 0:
            print(f"  processed {i + 1}")

    # Duplicate audit: exact (md5) and near (identical perceptual hash).
    def audit(index, kind):
        groups = [ix for ix in index.values() if len(ix) > 1]
        cross = 0
        for ix in groups:
            splits = {manifest[j]["split"] for j in ix}
            if len(splits) > 1:
                cross += 1
            for j in ix:
                manifest[j]["flags"].append(f"{kind}_duplicate" + ("_cross_split" if len(splits) > 1 else ""))
        return {"groups": len(groups), "records": sum(len(g) for g in groups), "cross_split_groups": cross}

    exact = audit(md5_index, "exact")
    near = audit(phash_index, "near")

    # Records whose image also appears in test must not be used for training/validation subsets.
    test_md5 = {m.get("image_md5") for m in manifest if m["split"] == "test"}
    test_ph = {m.get("image_phash") for m in manifest if m["split"] == "test"}
    for m in manifest:
        if m["status"] == "accepted" and m["split"] != "test" and (
            m.get("image_md5") in test_md5 or (m.get("image_phash") and m.get("image_phash") in test_ph)
        ):
            m["flags"].append("overlaps_test_image")

    with open(out / "manifest.jsonl", "w") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # ---- Fixed subsets --------------------------------------------------
    dcfg = cfg["data"]
    rng = random.Random(dcfg["seed"])
    blocked_flags = {"target_equals_caption", "target_in_caption", "target_in_paragraph", "overlaps_test_image"}

    def eligible(split, allow_dup=False):
        pool = []
        for m in manifest:
            if m["status"] != "accepted" or m["split"] != split:
                continue
            if blocked_flags & set(m["flags"]):
                continue
            if not m["context"] and not m["caption"]:
                continue
            pool.append(m)
        return pool

    train_pool, val_pool, test_pool = eligible("train"), eligible("validation"), eligible("test")
    # Within-subset duplicates are removed so no image appears twice in one subset.
    def sample(pool, n):
        pool = sorted(pool, key=lambda m: m["id"])
        rng.shuffle(pool)
        seen, chosen = set(), []
        for m in pool:
            key = m.get("image_phash") or m.get("image_md5")
            if key in seen:
                continue
            seen.add(key)
            chosen.append(m)
            if len(chosen) == n:
                break
        return chosen

    train = sample(train_pool, dcfg["train_n"])
    val = sample(val_pool, dcfg["val_n"])
    test = sample(test_pool, dcfg["test_n"])
    subsets = {
        "train_5k": train,
        "pilot": train[: dcfg["pilot_n"]],
        "tiny": train[: dcfg["tiny_n"]],
        "val": val,
        "milestone_100": val[: dcfg["milestone_n"]],
        "test": test,
    }
    keep = ["id", "image_path", "source_page", "page_title", "caption", "context", "target_description", "split"]
    for name, rows in subsets.items():
        with open(out / "subsets" / f"{name}.jsonl", "w") as f:
            for m in rows:
                f.write(json.dumps({k: m[k] for k in keep}, ensure_ascii=False) + "\n")

    # ---- Report -----------------------------------------------------------
    accepted = [m for m in manifest if m["status"] == "accepted"]
    flag_counts = Counter(fl for m in accepted for fl in m["flags"])
    tw = sorted(len(m["target_description"].split()) for m in accepted)
    cw = sorted(len(m["context"].split()) for m in accepted)
    report = {
        "raw_count": len(manifest),
        "accepted_count": len(accepted),
        "excluded_count": len(manifest) - len(accepted),
        "exclusions": dict(Counter(m["exclusion_reason"] for m in manifest if m["exclusion_reason"])),
        "accepted_by_split": dict(Counter(m["split"] for m in accepted)),
        "raw_by_split": dict(Counter(m["split"] for m in manifest)),
        "flags_among_accepted": dict(flag_counts),
        "duplicates": {"exact_md5": exact, "near_phash": near, "near_method": "imagehash.phash identical (Hamming 0)" if imagehash else "unavailable"},
        "target_words": {"median": tw[len(tw) // 2], "p95": tw[int(len(tw) * 0.95)]},
        "paragraph_words": {"median": cw[len(cw) // 2], "p95": cw[int(len(cw) * 0.95)]},
        "subset_sizes": {k: len(v) for k, v in subsets.items()},
        "subset_rules": "accepted, not flagged for target leakage or test-image overlap, at least one of caption/paragraph, unique image within subset; seed=%d" % dcfg["seed"],
        "inspected_examples": [
            {k: m[k] for k in ("id", "page_title", "caption", "target_description")} | {"context_head": " ".join(m["context"].split()[:40])}
            for m in val[:5]
        ],
    }
    json.dump(report, open(out / "data_report.json", "w"), indent=2, ensure_ascii=False)
    write_markdown_report(report, out / "data_report.md")
    print(json.dumps({k: report[k] for k in ("raw_count", "accepted_count", "exclusions", "subset_sizes")}, indent=2))


def write_markdown_report(r, path):
    lines = ["# Concadia data report", "", f"- Raw records: **{r['raw_count']}**", f"- Accepted: **{r['accepted_count']}**", f"- Excluded: **{r['excluded_count']}**", "", "## Exclusions", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(r["exclusions"].items(), key=lambda kv: -kv[1])]
    lines += ["", "## Accepted by split", ""] + [f"- {k}: {v} (raw {r['raw_by_split'].get(k)})" for k, v in r["accepted_by_split"].items()]
    lines += ["", "## Flags among accepted records", ""] + [f"- {k}: {v}" for k, v in sorted(r["flags_among_accepted"].items())]
    d = r["duplicates"]
    lines += ["", "## Duplicate audit", "", f"- Exact (MD5) groups: {d['exact_md5']['groups']} covering {d['exact_md5']['records']} records; {d['exact_md5']['cross_split_groups']} groups span splits",
              f"- Near ({d['near_method']}) groups: {d['near_phash']['groups']} covering {d['near_phash']['records']} records; {d['near_phash']['cross_split_groups']} groups span splits",
              "- Train/validation records whose image also occurs in test are flagged `overlaps_test_image` and kept out of experiment subsets."]
    lines += ["", "## Lengths (accepted)", "", f"- Target words: median {r['target_words']['median']}, p95 {r['target_words']['p95']}", f"- Paragraph words: median {r['paragraph_words']['median']}, p95 {r['paragraph_words']['p95']}"]
    lines += ["", "## Experiment subsets", "", f"Rule: {r['subset_rules']}", ""] + [f"- {k}: {v}" for k, v in r["subset_sizes"].items()]
    lines += ["", "## Inspected examples (validation)", ""]
    for e in r["inspected_examples"]:
        lines += [f"**{e['id']}** ({e['page_title']})", f"- Caption: {e['caption']}", f"- Target: {e['target_description']}", f"- Paragraph (start): {e['context_head']} ...", ""]
    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
