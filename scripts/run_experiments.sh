#!/usr/bin/env bash
# Reproduce the main experiment end to end (after data/raw is downloaded).
# Each stage is resumable: training resumes with --resume, generation skips finished IDs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

$PY scripts/prepare_data.py
$PY scripts/run_baselines.py --subset milestone_100 --out results/milestone

# Adapters C (image -> description) and D (image + context -> description)
$PY training/train_lora.py --run-name C_image_only --context-mode image_only --resume
$PY training/train_lora.py --run-name D_context   --context-mode with_context --resume

# Held-out test predictions (500 examples, identical generation settings)
$PY evaluation/generate.py --condition A
$PY evaluation/generate.py --condition B
$PY evaluation/generate.py --condition C --adapter runs/C_image_only/best_adapter
$PY evaluation/generate.py --condition D --adapter runs/D_context/best_adapter

# Context robustness on a fixed 200-example subset: correct / none / shuffled
for V in correct none shuffled; do
  $PY evaluation/generate.py --condition B --limit 200 --context-variant $V
  $PY evaluation/generate.py --condition D --adapter runs/D_context/best_adapter --limit 200 --context-variant $V
done

$PY evaluation/score.py --preds A_test B_test C_test D_test --out results/test_metrics
$PY evaluation/score.py --robustness B_test_correct B_test_none B_test_shuffled --out results/robustness_B
$PY evaluation/score.py --robustness D_test_correct D_test_none D_test_shuffled --out results/robustness_D
$PY evaluation/make_error_review.py --preds A_test B_test C_test D_test --n 50
