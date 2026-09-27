#!/usr/bin/env bash
# Follow-up experiments: precision, context dropout, 500M baseline, second seed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python} -u"
mkdir -p logs
# Retry a resumable step if the OS kills it under memory pressure (training resumes
# from the last checkpoint; generation skips finished IDs).
retry() { for i in 1 2 3 4 5; do "$@" && return 0; echo "retry $i: $*" | cut -c1-120; sleep 30; done; return 1; }
R500=a7da5b986cb59b408707209984f360a5f4ad7e47

# 1. float16 inference with the D adapter (first 200 test examples)
retry bash -c "$PY evaluation/generate.py --condition D --adapter runs/D_context/best_adapter --limit 200 --dtype float16 --name Dfp16_test >> logs/gen_Dfp16.log 2>&1"
echo "fp16 done"

# 2. context dropout: 30% of training examples see the image-only prompt
retry bash -c "$PY training/train_lora.py --run-name E_dropout30 --context-mode with_context --context-dropout 0.3 --resume >> logs/train_E.log 2>&1"
echo "E trained"
retry bash -c "$PY evaluation/generate.py --condition D --adapter runs/E_dropout30/best_adapter --name E_test >> logs/gen_E.log 2>&1"
for V in correct none shuffled; do
  retry bash -c "$PY evaluation/generate.py --condition D --adapter runs/E_dropout30/best_adapter --limit 200 --context-variant $V --name E_test_$V >> logs/gen_E_$V.log 2>&1"
done
echo "E evaluated"

# 3. SmolVLM-500M original model, same prompts and settings
retry bash -c "$PY evaluation/generate.py --condition A --model-id HuggingFaceTB/SmolVLM-500M-Instruct --revision $R500 --name A500_test >> logs/gen_A500.log 2>&1"
retry bash -c "$PY evaluation/generate.py --condition B --model-id HuggingFaceTB/SmolVLM-500M-Instruct --revision $R500 --name B500_test >> logs/gen_B500.log 2>&1"
echo "500M evaluated"

# 4. second seed of D
retry bash -c "$PY training/train_lora.py --run-name D_context_seed2 --context-mode with_context --seed 14 --resume >> logs/train_Dseed2.log 2>&1"
retry bash -c "$PY evaluation/generate.py --condition D --adapter runs/D_context_seed2/best_adapter --name Dseed2_test >> logs/gen_Dseed2.log 2>&1"
echo "seed2 done"

# 5. scoring
retry bash -c "$PY evaluation/score.py --preds A_test B_test C_test D_test E_test Dseed2_test A500_test B500_test --out results/extended_metrics >> logs/score_ext.log 2>&1"
retry bash -c "$PY evaluation/score.py --robustness E_test_correct E_test_none E_test_shuffled --out results/robustness_E >> logs/score_rE.log 2>&1"
echo "ALL DONE"
