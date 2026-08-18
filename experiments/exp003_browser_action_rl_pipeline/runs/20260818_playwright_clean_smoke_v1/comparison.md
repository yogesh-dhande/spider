# exp003a_playwright_coordinate_bias — clean-smoke-v1

Paired variants use identical task IDs and seeds.

| Variant | Episodes | Success rate | Mean reward | Mean steps | Parse error rate |
|---|---:|---:|---:|---:|---:|
| centered | 4 | 1.0000 | 1.6250 | 2.50 | 0.0000 |
| x_offset_120px | 4 | 0.2500 | 0.3000 | 5.00 | 0.0000 |

| Candidate vs control | Paired episodes | Success-rate delta | 95% paired bootstrap CI | Mean-reward delta |
|---|---:|---:|---:|---:|
| x_offset_120px vs centered | 4 | -0.7500 | [-1.0000, -0.2500] | -1.3250 |
