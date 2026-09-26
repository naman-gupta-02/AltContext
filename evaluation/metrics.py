"""Local captioning metrics: CIDEr-D, BERTScore, CLIPScore, length and repetition.

All metrics run offline once their scorer weights are downloaded.

- CIDEr-D: re-implementation of Vedantam et al. 2015 (sigma=6, clipped counts,
  length penalty, x10 scale) with document frequencies computed over the
  evaluated references, as in pycocoevalcap. Simple lowercase/punctuation
  tokenization instead of the Java PTB tokenizer, so absolute values are not
  directly comparable to COCO leaderboards, only across our own conditions.
- BERTScore: F1 using roberta-large layer 17 (the bert-score package default
  for English), no idf, no baseline rescaling. Implemented directly with
  transformers to avoid version pinning.
- CLIPScore: 2.5 * max(cos(image, text), 0) with openai/clip-vit-base-patch32
  (Hessel et al. 2021). Reference-free.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np
import torch

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(s: str) -> list[str]:
    return _TOKEN_RE.findall(s.lower())


# ------------------------------------------------------------------ CIDEr-D --
def _ngrams(tokens, n_max=4):
    c = Counter()
    for n in range(1, n_max + 1):
        for i in range(len(tokens) - n + 1):
            c[tuple(tokens[i : i + n])] += 1
    return c


def cider_d(candidates: list[str], references: list[list[str]], sigma: float = 6.0, n_max: int = 4) -> tuple[float, list[float]]:
    """Return (corpus score, per-example scores)."""
    cand_tok = [tokenize(c) for c in candidates]
    ref_tok = [[tokenize(r) for r in refs] for refs in references]
    ref_counts = [[_ngrams(r, n_max) for r in refs] for refs in ref_tok]
    df = defaultdict(float)
    for refs in ref_counts:
        for ng in set(ng for rc in refs for ng in rc):
            df[ng] += 1
    log_n = math.log(float(len(candidates)))

    def vec(counts):
        v = [defaultdict(float) for _ in range(n_max)]
        norm = [0.0] * n_max
        for ng, tf in counts.items():
            n = len(ng) - 1
            v[n][ng] = float(tf) * (log_n - math.log(max(1.0, df[ng])))
            norm[n] += v[n][ng] ** 2
        return v, [math.sqrt(x) for x in norm]

    def sim(vh, vr, nh, nr, lh, lr):
        delta = float(lh - lr)
        val = np.zeros(n_max)
        for n in range(n_max):
            for ng, x in vh[n].items():
                val[n] += min(x, vr[n].get(ng, 0.0)) * vr[n].get(ng, 0.0)
            if nh[n] != 0 and nr[n] != 0:
                val[n] /= nh[n] * nr[n]
            val[n] *= math.e ** (-(delta**2) / (2 * sigma**2))
        return val

    scores = []
    for ct, rts, rcs in zip(cand_tok, ref_tok, ref_counts):
        vh, nh = vec(_ngrams(ct, n_max))
        acc = np.zeros(n_max)
        for rt, rc in zip(rts, rcs):
            vr, nr = vec(rc)
            acc += sim(vh, vr, nh, nr, len(ct), len(rt))
        s = float(np.mean(acc)) / len(rts) * 10.0
        scores.append(s)
    return float(np.mean(scores)), scores


# ---------------------------------------------------------------- BERTScore --
class BertScorer:
    def __init__(self, model_name="roberta-large", layer=17, device=None, batch_size=16):
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, num_hidden_layers=layer).to(self.device).eval()
        self.batch_size = batch_size
        self.name = f"{model_name}_L{layer}_noidf_norescale"

    @torch.no_grad()
    def _embed(self, texts):
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(texts[i : i + self.batch_size], return_tensors="pt", padding=True, truncation=True, max_length=510).to(self.device)
            h = self.model(**enc).last_hidden_state
            h = torch.nn.functional.normalize(h, dim=-1)
            mask = enc["attention_mask"].clone()
            mask[:, 0] = 0  # drop <s>
            lengths = enc["attention_mask"].sum(1)
            for j in range(h.shape[0]):
                mask[j, lengths[j] - 1] = 0  # drop </s>
                out.append(h[j][mask[j].bool()].cpu())
        return out

    def score(self, candidates, references):
        c_emb, r_emb = self._embed([c or "." for c in candidates]), self._embed([r or "." for r in references])
        p_all, r_all, f_all = [], [], []
        for c, r in zip(c_emb, r_emb):
            if len(c) == 0 or len(r) == 0:
                p_all.append(0.0); r_all.append(0.0); f_all.append(0.0)
                continue
            sim = c @ r.T
            p = sim.max(1).values.mean().item()
            rc = sim.max(0).values.mean().item()
            f = 2 * p * rc / (p + rc) if p + rc > 0 else 0.0
            p_all.append(p); r_all.append(rc); f_all.append(f)
        return {"precision": p_all, "recall": r_all, "f1": f_all}


# ---------------------------------------------------------------- CLIPScore --
class ClipScorer:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None, batch_size=32):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)
        self.batch_size = batch_size
        self.name = model_name

    @torch.no_grad()
    def score(self, images, texts, w=2.5):
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.proc(text=[t or "." for t in texts[i : i + self.batch_size]], images=images[i : i + self.batch_size], return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
            ie = self.model.get_image_features(pixel_values=enc["pixel_values"])
            te = self.model.get_text_features(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            ie = getattr(ie, "pooler_output", ie)
            te = getattr(te, "pooler_output", te)
            cos = torch.nn.functional.cosine_similarity(ie, te)
            out.extend((w * cos.clamp(min=0)).tolist())
        return out


# ----------------------------------------------------------- length / misc --
def repetition_rate(text: str, n: int = 3) -> float:
    """Fraction of word n-grams that repeat an earlier n-gram in the same output."""
    t = tokenize(text)
    grams = [tuple(t[i : i + n]) for i in range(len(t) - n + 1)]
    if not grams:
        return 0.0
    return 1 - len(set(grams)) / len(grams)


def context_copy_rate(text: str, context: str, n: int = 5) -> float:
    """Fraction of output 5-grams that appear verbatim in the input context."""
    t, c = tokenize(text), tokenize(context)
    grams = [tuple(t[i : i + n]) for i in range(len(t) - n + 1)]
    if not grams:
        return 0.0
    cset = {tuple(c[i : i + n]) for i in range(len(c) - n + 1)}
    return sum(g in cset for g in grams) / len(grams)


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean difference b - a with a 95% paired bootstrap interval over examples."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = (b[idx] - a[idx]).mean(1)
    return {"diff": float((b - a).mean()), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))], "p_b_better": float((diffs > 0).mean())}
