| Run | n | CIDEr-D | BERTScore F1 | CLIPScore | Words (ref) | Context copy | Repetition | Latency median / p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| B_test_correct | 200 | 0.204 | 0.846 | 0.736 | 38.2 (15.6) | 0.249 | 0.010 | 2123 / 2798 |
| B_test_none | 200 | 0.167 | 0.851 | 0.703 | 34.3 (15.6) | 0.000 | 0.004 | 1590 / 2125 |
| B_test_shuffled | 200 | 0.028 | 0.824 | 0.485 | 39.1 (15.6) | 0.000 | 0.012 | 2132 / 2548 |

**Context robustness** — outputs containing donor-only words: shuffled context 81.5% vs chance-overlap control 16.0%; outputs that changed between correct and shuffled context: 100.0%.
