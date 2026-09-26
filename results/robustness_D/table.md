| Run | n | CIDEr-D | BERTScore F1 | CLIPScore | Words (ref) | Context copy | Repetition | Latency median / p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| D_test_correct | 200 | 0.398 | 0.870 | 0.703 | 10.4 (15.6) | 0.031 | 0.001 | 863 / 1749 |
| D_test_none | 200 | 0.246 | 0.867 | 0.710 | 15.7 (15.6) | 0.000 | 0.001 | 992 / 1860 |
| D_test_shuffled | 200 | 0.188 | 0.860 | 0.601 | 10.5 (15.6) | 0.000 | 0.000 | 877 / 1442 |

**Context robustness** — outputs containing donor-only words: shuffled context 35.5% vs chance-overlap control 2.0%; outputs that changed between correct and shuffled context: 98.0%.
