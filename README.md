# Singapore Condominium Price Graph Model (HPGNet)

Forecasting models for Singapore condominium **rents** and **sale prices**, 1, 12 and
24 months ahead, built on a property graph of condominium projects and their
unit-size tiers. This repository is the delivery package: code, trained
checkpoints, documentation and an interactive map demo.

Two datasets are covered:

| Name | Coverage | Rent projects | Sale projects |
|---|---|---:|---:|
| **Full SG** | every condominium in Singapore | 2,497 | 2,371 |
| **CCR** | Core Central Region (postal districts 1, 2, 9, 10, 11) | 804 | 693 |

## What is in the package

| Folder | Contents |
|---|---|
| `checkpoints/` | 12 trained models: {Full SG, CCR} × {rent, sale} × {1, 12, 24 months}, each with a `meta.json` |
| `src/` | data loading, graph construction, model layers |
| `pipeline/` | node-table builder, CCR training runner, shared training utilities |
| `benchmark/` | Full SG cache builder and trainer |
| `scripts/evaluate_checkpoints.py` | load a checkpoint, predict, report metrics, export predictions |
| `results/` | the validation results behind every model choice |
| `demo/` | interactive map of the graph and the test predictions (Vue + FastAPI) |
| `docs/` | full documentation (below) |

The data (raw panels and processed graph tables, about 12 GB uncompressed) is
**not** in git. It is delivered separately as `pgim_condo_graph_data.zip`; unzip it
at the repository root to create `data/` and `demo/predictions/`. See
[docs/data.md](docs/data.md).

## Results of the delivered checkpoints

Test period: every observed month from **January 2020** onward (feature months
2020-01 and later). Relation sets, configurations and stopping epochs were chosen on validation data
(the last 24 months before 2020); the test period was evaluated once. RMSE and MAE are in SGD per
sq ft (per month for rent).

**Full SG**

| Target | Horizon | RMSE | MAE | MAPE | R² | Test points |
|---|---:|---:|---:|---:|---:|---:|
| Rent | 1 month | 0.400 | 0.279 | 6.8% | 0.916 | 177,754 |
| Rent | 12 months | 0.763 | 0.587 | 12.8% | 0.688 | 150,705 |
| Rent | 24 months | 1.121 | 0.937 | 19.6% | 0.289 | 119,839 |
| Sale | 1 month | 87.0 | 61.9 | 3.9% | 0.970 | 61,370 |
| Sale | 12 months | 219.2 | 170.7 | 10.0% | 0.803 | 53,863 |
| Sale | 24 months | 215.9 | 175.7 | 10.4% | 0.800 | 40,634 |

**CCR**

| Target | Horizon | RMSE | MAE | MAPE | R² | Test points |
|---|---:|---:|---:|---:|---:|---:|
| Rent | 1 month | 0.452 | 0.333 | 7.1% | 0.894 | 54,067 |
| Rent | 12 months | 0.905 | 0.738 | 14.7% | 0.566 | 45,778 |
| Rent | 24 months | 1.168 | 1.006 | 19.4% | 0.235 | 36,122 |
| Sale | 1 month | 116.4 | 81.6 | 4.1% | 0.926 | 11,535 |
| Sale | 12 months | 158.2 | 120.5 | 5.9% | 0.855 | 10,086 |
| Sale | 24 months | 207.9 | 163.1 | 7.6% | 0.731 | 7,381 |

Every number above was reproduced from the checkpoints in this repository with
`scripts/evaluate_checkpoints.py`. Per-checkpoint details, including how the Full SG
checkpoints relate to the figures in the final report, are in
[docs/checkpoints.md](docs/checkpoints.md).

**Read these with the comparison in mind.** In the controlled benchmark of the final
report (20 models on Full SG, 3 seeds each), this graph model was not the most
accurate in any setting: LightGBM was best for rent at 12 and 24 months and for sale
at 1 month, and models that see only price histories were best for sale at 12 and 24
months. The main reason is that the project-level market and macroeconomic covariates
that this model passes through the graph generalised poorly after 2020. See
[docs/checkpoints.md](docs/checkpoints.md#how-these-models-compare) and
`results/full_sg/summary/all_models.csv`.

## Quick start

```bash
# 1. Environment (Python 3.10, NVIDIA GPU with CUDA)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 2. Data: unzip the data package at the repository root
unzip pgim_condo_graph_data.zip        # creates data/ and demo/predictions/

# 3. Evaluate a CCR checkpoint (builds its inputs from data/processed, ~1 minute)
python scripts/evaluate_checkpoints.py --dataset ccr --task rent --horizon 12 --gpu 0 \
    --out predictions/ccr_rent_h12.csv

# 4. Full SG: build the input cache once per task (~2-5 minutes, ~6 GB each), then evaluate
export BENCH_CACHE=/path/with/space/full_sg_cache
python benchmark/build_cache.py --task rent --gpu 0
python scripts/evaluate_checkpoints.py --dataset full_sg --task rent --horizon 1 --gpu 0
```

The script prints the metrics next to the values recorded in the checkpoint's
`meta.json`; they should agree. With `--out` it writes one row per size tier and
month: `date, node_id, project_id, size_tier, actual, predicted`.

## Documentation

| Document | Covers |
|---|---|
| [docs/data.md](docs/data.md) | datasets, files, columns, splits, data quality |
| [docs/model.md](docs/model.md) | the property graph, relations, the HPGNet architecture, training protocol |
| [docs/checkpoints.md](docs/checkpoints.md) | each delivered model, its settings and metrics, how to load it in Python |
| [docs/reproduce.md](docs/reproduce.md) | rebuilding the data, caches and models from the raw panels |
| [demo/README.md](demo/README.md) | running and deploying the map demo |

## Important limitations

- **Back-test, not live forecasts.** Every prediction is for a month whose price is
  already known. Nothing here forecasts beyond the end of the data (early 2026).
- **Imputed price history.** The panels' imputed price per sq ft is used as an input
  and to build the price-peer relation. It was produced upstream and may have been
  fitted on the full period, which would make the inputs slightly optimistic.
- **Missing coordinates.** In Full SG, 1 rent project and 99 sale projects have no
  coordinates; they get no MRT links and are not drawn on the demo map.
- **Not bit-reproducible training.** GPU training is not deterministic. Retraining
  gives models within normal seed-to-seed variation, not identical weights.
- **Preprocessing statistics.** Feature normalisation statistics were computed over
  the full period of each table (a mild form of look-ahead).

More detail is in [docs/data.md](docs/data.md#data-quality) and
[docs/model.md](docs/model.md#training-protocol).
