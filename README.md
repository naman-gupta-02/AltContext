# AltContext

Context-aware alt text from a compact vision-language model that runs on your own computer, delivered through a Chrome extension.

AltContext fine-tunes **SmolVLM-256M-Instruct** with LoRA on **Concadia** (Wikipedia images with alt descriptions, captions and surrounding paragraphs). The fine-tuned model runs in a local FastAPI service. A Manifest V3 side-panel extension lets you pick an image on a page, collects nearby context (caption, section heading, closest paragraph, page title), and shows a description labelled as AI-generated. Existing alt text is never modified.

> **What the results show, and what they don't.** The results are captioning metrics on a held-out Concadia split, plus an author-run robustness analysis. No user study was run, so nothing here claims better outcomes for screen reader users.

---

## Quick start (inference only)

```bash
# 1. Python environment (Python 3.12 tested)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # exact versions of the measured setup: requirements.lock.txt

# 2. Start the local service (downloads the base model once, ~0.5 GB)
python -m backend.app                  # http://127.0.0.1:8765, prints a service token
# optional: --adapter runs/C_image_only/best_adapter | --adapter none | --fake
```

The service loads `runs/D_context/best_adapter` (the context-trained LoRA adapter, 3.6 MB) by default.

**Smallest inference example** (no browser):

```bash
TOKEN=$(cat ~/.altcontext/token)
curl -s -X POST http://127.0.0.1:8765/describe -H "X-AltContext-Token: $TOKEN" \
  -F image=@path/to/photo.jpg \
  -F 'metadata={"caption":"A small bowl of mixed nuts","paragraph":"...","page_title":"Snacks","mode":"brief"}'
# {"description":"A glass bowl with mixed nuts","model_version":"SmolVLM-256M-Instruct@7e3e67ed+D_context/best_adapter",
#  "mode":"brief","context_used":true,"ai_generated":true,"latency_ms":...,"cached":false}
```

### Load the extension

1. Open `chrome://extensions`, turn on **Developer mode**, click **Load unpacked**, and choose the `extension/` folder.
2. Open the extension's **Options** and paste the token printed by the backend (it is also stored in `~/.altcontext/token`).
3. On any page, open the panel with the toolbar icon or **Alt+Shift+A**, then choose **Scan this page for images**, then **Describe image N**.

To use the demo page: run `python demo/build_demo.py` with the backend running, then open `http://127.0.0.1:8765/demo/`. It includes existing, empty and missing alt text, an image inside a link, a lazy-loaded image and a broken image.

Startup order: backend first (check `GET /health` for `"status":"ready"`), then the extension.

---

## Results

All numbers below were measured on an **Apple M1 (8 GB unified memory), PyTorch MPS, float32**, using a single 512 px image view (64 image tokens), greedy decoding, `max_new_tokens=60` and `repetition_penalty=1.1`. The test set is a fixed 500-example subset of Concadia's test split. It was never used for training or checkpoint selection.

### Experiment design (build guide §10)

| Run | Model | Evaluation input |
|---|---|---|
| A | Original SmolVLM-256M-Instruct | Image only |
| B | Original SmolVLM-256M-Instruct | Image + context |
| C | LoRA adapter trained on image → description | Image only |
| D | LoRA adapter trained on image + context → description | Image + context |

C and D use the same 5,000 training examples, the same LoRA configuration, the same number of optimizer steps and the same seed. The only difference is whether the prompt includes context.

### Held-out test set (n = 500)

| Run | CIDEr-D | BERTScore F1 | CLIPScore | Mean words (reference 14.7) | Context 5-gram copy | Latency median / p95 |
|---|---|---|---|---|---|---|
| A | 0.117 | 0.848 | 0.698 | 33.0 | 0.000 | 1335 / 1844 ms |
| B | 0.144 | 0.843 | **0.736** | 39.0 | 0.227 | 1954 / 2374 ms |
| C | 0.207 | 0.865 | 0.716 | 16.1 | 0.001 | 977 / 1898 ms |
| D | **0.449** | **0.871** | 0.709 | 10.0 | 0.031 | 912 / 1993 ms |

Paired bootstrap over the same 500 examples (2,000 resamples, 95% CI; difference = second minus first):

| Comparison | Question | ΔCIDEr-D | ΔBERTScore F1 | ΔCLIPScore |
|---|---|---|---|---|
| B vs A | Does context help the original model? | +0.027 [−0.029, +0.087] | −0.005 [−0.009, −0.002] | +0.038 [+0.025, +0.051] |
| C vs A | Does fine-tuning help without context? | +0.090 [+0.052, +0.127] | +0.017 [+0.015, +0.019] | +0.018 [+0.009, +0.028] |
| D vs B | Does contextual fine-tuning beat contextual prompting? | **+0.305 [+0.209, +0.399]** | +0.028 [+0.025, +0.032] | −0.027 [−0.039, −0.016] |
| D vs C | How do the two trained approaches compare? | +0.242 [+0.167, +0.321] | +0.006 [+0.003, +0.009] | −0.008 [−0.018, +0.002] |

**Reading the table**

- On this split, contextual fine-tuning (D) roughly triples CIDEr-D compared with prompting the original model with the same context (B). It also cuts verbatim copying of page text from 22.7% to 3.1% of output 5-grams.
- Adding context to the original model (B vs A) makes no clear difference to CIDEr-D (the interval includes 0). It makes outputs longer and makes the model copy the caption and paragraph.
- **CLIPScore moves the other way:** B > D. CLIPScore favours longer, more visually detailed text, while D's outputs are short (10 words, below the 14.7-word reference average). So D matches Concadia's reference style much better, but these metrics cannot show that it describes more of the visible image.

### Context robustness (n = 200 fixed test examples)

The same images were described with the correct context, no context, and context taken from a different article (a fixed derangement).

| Model | Context | CIDEr-D | CLIPScore | Outputs with donor-only words* |
|---|---|---|---|---|
| B (original) | correct / none / shuffled | 0.204 / 0.167 / 0.028 | 0.736 / 0.703 / **0.485** | **81.5%** (chance control 16.0%) |
| D (fine-tuned) | correct / none / shuffled | 0.398 / 0.246 / 0.188 | 0.703 / 0.710 / **0.601** | **35.5%** (chance control 2.0%) |

\*A lexical proxy: an output content word that appears in the wrong ("donor") context but in neither the reference, the image's own context, nor the model's no-context output. The control applies the same donor vocabulary to correct-context outputs, which estimates chance overlap.

D is much less distracted than B, but it is **not grounded**. With irrelevant context it still pulls names and topics from the text into 35.5% of outputs. For example, a photo of an inscription in hieroglyphs described as *"A zonkey in Tijuana, Mexico"*, and a battlefield map described as *"A map of the lungs of a patient"*. With correct context, D sometimes states identities or places it cannot see: *"Former United States President Ronald Reagan is pictured smiling…"*, and *"A map of Wellington County, New Zealand"*, where "New Zealand" is invented. Prompt instructions alone do not prevent this, as the build guide anticipated.

### Training runs

| Adapter | Validation loss (100 fixed val examples), step 0 → 313 | Median step time | Peak MPS driver memory |
|---|---|---|---|
| C (image only) | 4.368 → 3.498 | 6.7 s / 16 examples | 1.8 GB |
| D (image + context) | 3.549 → 2.629 | 21.2 s / 16 examples | 2.1 GB |

Both adapters use the lowest validation loss as the selection rule, which fell at the final step (313) in both runs. The shape of each loss curve is in `runs/*/summary.json`.

### Latency and memory (Apple M1, MPS, fp32)

| Measure | Value |
|---|---|
| Cold model load (weights cached locally), base / with adapter | 3.3–4.8 s / 5–7.4 s |
| Warm uncached generation, brief mode, D | median 912 ms, p95 1,993 ms |
| Warm uncached generation, original model with context (B) | median 1,954 ms, p95 2,374 ms |
| Cached response via `/describe` | < 1 ms server time |
| Detailed mode (original model, 160 tokens) | ~5.7 s for one example |
| Peak MPS driver memory during generation | ~1.5 GB |

### First milestone (100 validation examples, original model)

`results/milestone/` contains the comparison file described in guide §18: example ID, split, image, context, reference, both outputs, model revision, generation settings and per-condition latency. Median latency was 1.49 s image-only and 2.14 s with context. Outputs averaged 33–38 words against 15-word references.

### Author error review

`evaluation/error_review/review_sheet.csv` holds 50 test examples × 4 conditions in a blinded, shuffled order. The columns cover invented object, unsupported attribute, incorrect relationship, missing important detail and copied page claim. **Not yet filled in.** After review, run `python evaluation/make_error_review.py --summarize`. It is an author-reviewed analysis, not a user study.

---

## How it works

```
OFFLINE TRAINING                                   BROWSING (local only)
Concadia ─► prepare_data.py ─► subsets             Page ─► content.js (caption, heading, paragraph, title)
          (clean, dedupe, leakage flags)                      │  user picks "Describe image N"
               │                                              ▼
               ▼                                   service_worker.js ─ fetches the element's image bytes
train_lora.py (LoRA on text attention,                        │  (only URLs read from the selected element)
 vision encoder + connector frozen)                           ▼
               │                                   backend/app.py (127.0.0.1, token, queue, cache)
               ▼                                              │  altcontext.text.build_context  ◄── same code as training
  runs/D_context/best_adapter  ─────────────────►  SmolVLM-256M + adapter ─► plain-text description
```

### Data (`scripts/prepare_data.py`)

Output is `data/processed/data_report.md`.

- There are 96,918 raw records, and 94,906 were accepted. The 2,012 exclusions are: target too short (1,745), placeholder (146) and too long (121).
- Images are hashed (MD5 plus perceptual hash). 1,110 exact-duplicate groups span splits, and 1,277 train/val records whose image also appears in test are flagged and kept out of the subsets.
- 8,511 targets equal the caption, and 1,846 appear inside the caption or paragraph. They are flagged and excluded from the experiment subsets so the target never enters the input.
- The fixed subsets use seed 13 with no duplicate image within a subset: `train_5k`, `pilot` (300), `tiny` (20), `val` (300), `milestone_100` and `test` (500).

### Input format (`altcontext/text.py`, shared by training, evaluation and the backend)

- **Priority order:** caption → section heading → nearby paragraph → page title.
- **Context budget:** 200 words and 1,000 characters. The character cap exists because some Wikipedia paragraphs are flattened tables with no spaces; one such example produced a 2,700-token context and crashed training with an out-of-memory error.
- **Prompt:** the one from the build guide, with context wrapped in `<context>` tags. A `</context>` inside page text is neutralized.

### Training (`training/train_lora.py`)

- **LoRA configuration:** r = 8, α = 16, dropout 0.05, on `q/k/v/o_proj` of the 30 **language-model** layers only (a regex excludes vision-encoder projections). That is 921,600 trainable parameters, or 0.36%. The script checks that no vision or connector parameter is trainable.
- **Loss:** answer-only. Prompt, image and padding tokens are masked, and the supervised span is decoded and checked against the target plus `<end_of_utterance>` before training (`runs/*/batch_inspection.json`).
- **Optimization:** AdamW with learning rate 1e-4, cosine schedule with 5% warmup, batch 1 × 16 accumulation, 1 epoch = 313 optimizer steps.
- **Frozen-encoder feature cache:** because the vision encoder and connector are frozen, their outputs are cached per image and passed back through `image_hidden_states`. The run asserts that the loss is identical with pixels and with cached features (1.81462 in both). This makes steps about 40% faster on MPS.
- **Checkpointing and resume:** checkpoints are resumable (adapter, optimizer, scheduler and RNG state, step), and `--resume` continues a run.
- **Debug stages from the guide were run:** a tiny fit (20 examples, validation loss 3.91 → 1.23) and a pilot (300 examples; save and reload verified).

### Backend (`backend/app.py`)

- **Endpoints:** `GET /health` and `POST /describe` (multipart image plus JSON metadata).
- **Access control:** binds to loopback only, requires the `X-AltContext-Token` header, and rejects web-page `Origin` headers.
- **Input validation:** 10 MB limit, a format allow-list (JPEG, PNG, WebP, GIF, BMP) and a decompression-bomb guard.
- **Execution:** a single-worker queue with a limit of 8 waiting requests and a 90 s timeout.
- **Cache:** in-memory LRU keyed by SHA-256 of image bytes + normalized context + mode + model/adapter version + processor settings + generation settings. Nothing is written to disk and request contents are not logged.
- **Detailed mode** uses the original model with a detailed prompt, because the adapter was trained only on brief descriptions. It is **unevaluated and prone to invented detail**; treat it as experimental.
- `ALTCONTEXT_FAKE_MODEL=1` returns fixed responses, for extension development.

### Extension (`extension/`)

- **Permissions:** `activeTab`, `scripting`, `sidePanel` and `storage`. Access to other image hosts is requested per origin, only when you ask to describe an image from that host.
- **Request-time re-read:** the service worker re-reads the selected element at request time, so it always uses the current `src` and context. It only fetches that element's URL, so it is not a general fetch proxy. If the fetch fails, it falls back to same-origin fetch or canvas inside the page for `blob:` and CORS-clean images. AVIF and very large images are re-encoded.
- **Alt-text cases:** existing, empty (treated as decorative, described only on request) and missing are shown distinctly. For images inside links or buttons, the functional label is shown and kept.
- **Accessibility:** native buttons and radio groups, a polite `role="status"` live region, `aria-busy`, and focus that is never moved to results. Output is inserted with `textContent` and labelled "AI-generated" with its mode, model and timing. Copy, Cancel and "Show on page" (scroll and outline) are available.
- **Errors:** backend stopped, missing token, model loading, blocked cross-origin fetch, lazy image not loaded, broken image, unsupported format, timeout and cancel each produce a specific message.

---

## Reproduce

```bash
# data (Google Drive links from the Concadia README; ~3.2 GB zip + 0.6 GB JSON)
mkdir -p data/raw && cd data/raw
gdown 1kiTSiqk7y7JdHssXjoLwcOomC7lhb5k8 -O wiki_split.json
gdown 1gDhVlOwcGcwBT5LWYwgn9xEElGlKVpFb -O resized.zip
python - <<'EOF'   # extract only referenced images (the zip has names that collide on macOS)
import json, zipfile, os
need = {i["filename"] for i in json.load(open("wiki_split.json"))["images"]}
z = zipfile.ZipFile("resized.zip"); os.makedirs("images", exist_ok=True)
for info in z.infolist():
    n = info.filename.split("/")[-1]
    if n in need and not os.path.exists("images/" + n):
        open("images/" + n, "wb").write(z.read(info))
EOF
cd ../..

scripts/run_experiments.sh    # prepare → milestone → train C and D → A/B/C/D → robustness → scoring → review sheet
```

Individual stages:

```bash
python scripts/prepare_data.py
python scripts/run_baselines.py --subset milestone_100
python training/train_lora.py --run-name tiny --train-subset tiny --val-subset tiny --val-loss-n 20 --epochs 15 --grad-accum 4 --lr 5e-4
python training/train_lora.py --run-name D_context --context-mode with_context [--context-dropout 0.3] [--resume]
python evaluation/generate.py --condition D --adapter runs/D_context/best_adapter [--context-variant shuffled --limit 200]
python evaluation/score.py --preds A_test B_test C_test D_test --out results/test_metrics
```

On the M1 used here, the full pipeline took about 7 hours: 1.9 h for D training, 0.6 h for C, and about 3 h for generation and scoring. On a CUDA GPU, pass `--device cuda` or let it auto-detect.

Tests: `pytest` runs backend and text-processing tests with the fake model. `node tests/extension_e2e.mjs` runs a headless Chrome check of scanning, describing, caching, the broken-image path and cancel; it needs `puppeteer-core` and the setup in the file header.

### Recorded with every run

`runs/<name>/run_config.json` records the full config, CLI arguments, base-model revision (`7e3e67edbbed1bf9888184d9df282b700a323964`), processor settings, SHA-256 of training and validation IDs, parameter counts, the feature-cache check, dependency versions, hardware and the checkpoint rule. Each `results/predictions/*.meta.json` records the model version, generation and context settings, cold-load time, peak memory and environment. Per-example scores are in `results/*/per_example_*.jsonl`.

### Repository layout

| Path | Contents |
|---|---|
| `altcontext/` | Shared text/context normalization, model loading and generation |
| `configs/default.json` | Model revision, processor, context budget, generation, LoRA and training settings |
| `scripts/` | `prepare_data.py`, `run_baselines.py`, `run_experiments.sh` |
| `training/train_lora.py` | LoRA training with masking checks, feature cache, checkpoints and resume |
| `evaluation/` | Metrics (CIDEr-D validated against pycocoevalcap, BERTScore, CLIPScore), generation, scoring, error-review sheet |
| `backend/app.py` | FastAPI service |
| `extension/` | Manifest V3 extension: service worker, content script, side panel, options |
| `demo/build_demo.py` | Controlled demo article from validation images |
| `data/processed/` | Data report and subset manifests (raw data and cache are git-ignored) |
| `runs/` | Adapters, configs, logs, summaries |
| `results/` | Predictions, metrics, comparison tables, milestone file |

---

## Limitations

- **Grounding:** context can inject identities, places and topics that are not visible (see the robustness section). A mitigation the build guide suggests, context dropout (`--context-dropout`), is implemented but was not run.
- **Metric fit:** D's outputs are shorter than the references and follow Concadia's Wikipedia style. CIDEr-D and BERTScore reward that style, while CLIPScore favours the longer outputs of the original model. No metric certifies accessibility quality or factual accuracy.
- **Resolution:** Concadia images are 256 × 256, so small text and fine detail are not recoverable.
- **Scope:** v1 handles ordinary `<img>` elements only. CSS backgrounds, canvas, SVG, cross-origin iframes, charts and dense text are not supported.
- **Hardware:** results come from one Apple M1 configuration, and latency depends on hardware, precision and input length.
- **Not run:** a single training seed was used, and neither the 500M model nor the DOCCI/VizWiz extensions were evaluated.

## Attribution

- **Concadia:** Kreiss et al., EMNLP 2022. Images come from Wikimedia Commons under their individual licenses; the demo page lists source URLs.
- **SmolVLM-256M-Instruct:** Hugging Face, Apache 2.0.
- **Evaluation models:** RoBERTa-large (BERTScore) and CLIP ViT-B/32 (CLIPScore).
- **Prior work:** *Context-Aware Image Descriptions for Web Accessibility* (arXiv:2409.03054) evaluated a context-aware extension with blind and low-vision participants. This project contributes the compact fine-tuned local model and the controlled offline comparison.
