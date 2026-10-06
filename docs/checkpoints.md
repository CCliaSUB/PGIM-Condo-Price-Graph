# Checkpoints

Twelve models: {Full SG, CCR} × {rent, sale} × {1, 12, 24 months ahead}, seed 42.

```
checkpoints/
  full_sg/{rent,sale}_h{1,12,24}/  model.pt  meta.json
  ccr/{rent,sale}_h{1,12,24}/      model.pt  meta.json
```

`meta.json` records everything needed to rebuild and score the model: relations,
temporal module, hyperparameters, target form and scaling, seed, and the validation
and test metrics. RMSE and MAE are in SGD per sq ft (per month for rent).

## Full SG

Relations: size-project, same-MRT, price-peer (k = 5). Target: log ratio to the current
imputed price. Hyperparameters and temporal module chosen on validation.

| Model | Temporal | Hidden / layers / dropout | Val RMSE | Test RMSE | Test MAPE | Test R² |
|---|---|---|---:|---:|---:|---:|
| `rent_h1` | Transformer | 128 / 2 / 0.05, mean fusion | 0.284 | 0.400 | 6.8% | 0.916 |
| `rent_h12` | Transformer | 128 / 3 / 0.15, cosine LR | 0.293 | 0.763 | 12.8% | 0.688 |
| `rent_h24` | Transformer | 128 / 2 / 0.10 | 0.332 | 1.121 | 19.6% | 0.289 |
| `sale_h1` | Transformer | 96 / 2 / 0.05 | 76.6 | 87.0 | 3.9% | 0.970 |
| `sale_h12` | Transformer | 96 / 2 / 0.05 | 97.8 | 219.2 | 10.0% | 0.803 |
| `sale_h24` | GRU | 64 / 2 / 0.10 | 110.8 | 215.9 | 10.4% | 0.800 |

### Relation to the final report

The final report's Full SG figures come from runs whose weights were not saved. These
checkpoints were **retrained** with the same code, data, configuration and seed, with
saving added. GPU training is not bit-reproducible, so the weights differ slightly:

| Model | Report, seed 42 | Report, mean of 3 seeds | This checkpoint |
|---|---:|---:|---:|
| `rent_h1` | 0.398 | 0.400 | 0.400 |
| `rent_h12` | 0.761 | 0.771 | 0.763 |
| `rent_h24` | 1.122 | 1.135 | 1.121 |
| `sale_h1` | 86.9 | 85.7 | 87.0 |
| `sale_h12` | 219.9 | 269.2 | 219.2 |
| `sale_h24` | 223.6 | 227.7 | 215.9 |

(test RMSE). All six are within the seed-to-seed range seen in the report. Both report
figures are also stored in each `meta.json` under `report_metrics`.

## CCR

The relation set is the best of all 64 combinations of the six optional relations on
validation RMSE, separately for each target and horizon (`results/ccr/`). Temporal
module: causal Transformer. Target: price level. These checkpoints are the original
study runs, unchanged.

| Model | Relations besides size-project | Hidden / layers / dropout | Val RMSE | Test RMSE | Test MAPE | Test R² |
|---|---|---|---:|---:|---:|---:|
| `rent_h1` | same-MRT | 64 / 2 / 0.10 | 0.340 | 0.452 | 7.1% | 0.894 |
| `rent_h12` | same-MRT, price-peer | 128 / 3 / 0.20 | 0.444 | 0.905 | 14.7% | 0.566 |
| `rent_h24` | none | 128 / 2 / 0.20, mean fusion | 0.408 | 1.168 | 19.4% | 0.235 |
| `sale_h1` | proximity, price-peer | 96 / 2 / 0.05 | 120.1 | 116.4 | 4.1% | 0.926 |
| `sale_h12` | proximity, school-distance | 128 / 2 / 0.05, mean fusion | 133.3 | 158.2 | 5.9% | 0.855 |
| `sale_h24` | adjacent-MRT, proximity, same-MRT, MRT-distance, school-distance | 64 / 2 / 0.10, mean fusion | 157.0 | 207.9 | 7.6% | 0.731 |

On validation, the best relation set improved RMSE over size-project alone by only
0.6–4.6%, and for rent at 24 months size-project alone was best. The relational signal
is real but small; price-peer was the most consistently useful relation.

The CCR test figures were not part of the final report, which used CCR only for
validation-based design studies. They were computed once for this delivery.

## How these models compare

The final report's controlled benchmark trained 20 models on Full SG under the same
protocol (3 seeds each). Test RMSE, mean of 3 seeds:

| Setting | Best model | Its RMSE | This graph model (3-seed mean) | Rank of 20 |
|---|---|---:|---:|---:|
| Rent, 1 month | homogeneous graph + Transformer | 0.385 | 0.400 | 15 |
| Rent, 12 months | LightGBM | 0.655 | 0.771 | 17 |
| Rent, 24 months | LightGBM with graph features | 1.011 | 1.135 | 13 |
| Sale, 1 month | LightGBM with graph features | 79.7 | 85.7 | 16 |
| Sale, 12 months | Transformer on own price history only | 115.8 | 269.2 | 18 |
| Sale, 24 months | graph model without project covariates | 172.5 | 227.7 | 17 |

The report traces most of the gap to the project-level covariates (market and
macroeconomic series) that this model receives through the graph: they did not hurt
validation accuracy but generalised poorly after 2020. Models that use only price
histories were much more robust at 12 and 24 months. All 20 models' results are in
`results/full_sg/summary/all_models.csv`.

## Using a checkpoint

From the command line (recommended; rebuilds the inputs exactly as in training):

```bash
python scripts/evaluate_checkpoints.py --dataset ccr --task sale --horizon 1 --gpu 0 \
    --split test --out predictions/ccr_sale_h1.csv
```

`--split validation` scores the validation months instead. `--demo` also writes the
predictions in the map demo's format to `demo/predictions/`.

In Python:

```python
import json, torch
from benchmark.models import GraphSequenceRegressor

meta = json.load(open("checkpoints/full_sg/rent_h1/meta.json"))
model = GraphSequenceRegressor(**meta["model_kwargs"])
model.load_state_dict(torch.load("checkpoints/full_sg/rent_h1/model.pt")["state_dict"])
# input: (batch, 12 months, groups, hops, features) slices of the Full SG cache
# output: (batch, 12, 1) standardised log ratios; see scripts/evaluate_checkpoints.py
```

CCR models are `GNNRegressor` instances built from the trial named in `meta.json`
(`pipeline.ccr_study.load_trial` and `build_model`); their state dict is under
`model_state_dict`. The inputs and target scaling must be rebuilt as in
`scripts/evaluate_checkpoints.py`, since the models expect precomputed graph context,
not raw features.
