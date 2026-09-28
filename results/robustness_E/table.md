| Run | n | CIDEr-D | BERTScore F1 | CLIPScore | Words (ref) | Context copy | Repetition | Latency median / p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| E_test_correct | 200 | 0.441 | 0.871 | 0.715 | 11.3 (15.6) | 0.037 | 0.001 | 756 / 1277 |
| E_test_none | 200 | 0.277 | 0.867 | 0.703 | 14.4 (15.6) | 0.000 | 0.002 | 807 / 1212 |
| E_test_shuffled | 200 | 0.193 | 0.860 | 0.600 | 11.0 (15.6) | 0.000 | 0.000 | 762 / 1185 |

**Context robustness** — outputs containing donor-only words: shuffled context 34.5% vs chance-overlap control 2.5%; outputs that changed between correct and shuffled context: 99.0%.
