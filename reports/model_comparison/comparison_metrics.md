# Model comparison — same KPI, same temporal split, same horizon

KPI: `prb_dl_util` · Cell: `CELL_001`

| Model | RMSE | MAE | MAPE (%) |
| --- | ---: | ---: | ---: |
| `ridge_linear` | 0.8368 | 0.6954 | 0.8204 |
| `gradient_boosting` | 2.8755 | 2.6280 | 3.1641 |
| `mlp` | 22.5926 | 19.6399 | 26.9613 |

All three trained on the same non-shuffled temporal split. Zero new dependencies — all sklearn / NumPy. Reproduce: `make model-comparison`.
