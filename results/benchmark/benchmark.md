Hardware: Apple M1, 8 GB RAM, device `mps`, fp32. n = 100 uncached brief requests (Concadia test images with caption + paragraph + page title), then the same 100 repeated (cached), 10 detailed requests, and a burst of 8 concurrent requests. Latencies are client-side HTTP round trips unless noted.

| Model | Cold start (spawn → ready) | Brief, uncached: median / p95 | Server generation median | Cached: median | Detailed: median | Brief words | Peak server RSS |
|---|---|---|---|---|---|---|---|
| original | 5.1 s | 1.90 / 1.96 s | 1.89 s | 1.1 ms | 3.85 s | 38.4 | 1.38 GB |
| D | 7.2 s | 0.71 / 1.11 s | 0.70 s | 1.1 ms | 3.92 s | 10.8 | 1.69 GB |
| E | 6.9 s | 0.76 / 1.42 s | 0.75 s | 1.1 ms | 3.94 s | 11.7 | 1.41 GB |

| Model | Sequential throughput | Burst of concurrent requests: wall time | Burst: slowest request |
|---|---|---|---|
| original | 34.4 req/min | 13.8 s for 8 | 13.7 s |
| D | 76.6 req/min | 5.7 s for 8 | 5.7 s |
| E | 73.4 req/min | 6.6 s for 8 | 6.6 s |
