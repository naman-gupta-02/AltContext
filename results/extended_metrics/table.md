| Run | n | CIDEr-D | BERTScore F1 | CLIPScore | Words (ref) | Context copy | Repetition | Latency median / p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| A_test | 500 | 0.117 | 0.848 | 0.698 | 33.0 (14.7) | 0.000 | 0.006 | 1335 / 1844 |
| B_test | 500 | 0.144 | 0.843 | 0.736 | 39.0 (14.7) | 0.227 | 0.013 | 1954 / 2374 |
| C_test | 500 | 0.207 | 0.865 | 0.716 | 16.1 (14.7) | 0.001 | 0.002 | 977 / 1898 |
| D_test | 500 | 0.449 | 0.871 | 0.709 | 10.0 (14.7) | 0.031 | 0.001 | 912 / 1993 |
| E_test | 500 | 0.454 | 0.871 | 0.717 | 10.9 (14.7) | 0.030 | 0.001 | 757 / 1317 |
| Dseed2_test | 500 | 0.453 | 0.870 | 0.706 | 10.3 (14.7) | 0.032 | 0.001 | 729 / 1258 |
| A500_test | 500 | 0.214 | 0.863 | 0.737 | 19.3 (14.7) | 0.001 | 0.004 | 1287 / 3338 |
| B500_test | 500 | 0.337 | 0.848 | 0.724 | 25.6 (14.7) | 0.147 | 0.005 | 2368 / 3579 |

Paired bootstrap (2000 resamples over the same examples); difference = second minus first.

| Comparison | Question | ΔCIDEr-D [95% CI] | ΔBERTScore F1 [95% CI] | ΔCLIPScore [95% CI] |
|---|---|---|---|---|
| B vs A | Does adding context to the original model help? | +0.027 [-0.029, +0.087] | -0.005 [-0.009, -0.002] | +0.038 [+0.025, +0.051] |
| C vs A | Does task fine-tuning help without context? | +0.090 [+0.052, +0.127] | +0.017 [+0.015, +0.019] | +0.018 [+0.009, +0.028] |
| D vs B | Does contextual fine-tuning improve on contextual prompting? | +0.305 [+0.209, +0.399] | +0.028 [+0.025, +0.032] | -0.027 [-0.039, -0.016] |
| D vs C | How do the two trained approaches compare? | +0.242 [+0.167, +0.321] | +0.006 [+0.003, +0.009] | -0.008 [-0.018, +0.002] |
| E vs D | Does context dropout (30%) change the contextual adapter? | +0.006 [-0.043, +0.052] | +0.000 [-0.001, +0.001] | +0.009 [+0.004, +0.014] |
| Dseed2 vs D | How much does a second training seed move the result? | +0.004 [-0.024, +0.033] | -0.001 [-0.002, +0.000] | -0.003 [-0.007, +0.001] |
| A500 vs A | Does the 500M original model beat 256M (image only)? | +0.098 [+0.056, +0.141] | +0.014 [+0.013, +0.016] | +0.039 [+0.029, +0.048] |
| B500 vs B | Does the 500M original model beat 256M (with context)? | +0.193 [+0.112, +0.278] | +0.005 [+0.001, +0.009] | -0.012 [-0.023, +0.001] |
