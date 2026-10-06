# The model

HPGNet (Heterogeneous Property Graph Network) adapts RpHGNN — scalable heterogeneous
graph learning with random projection and precomputed propagation — to monthly
price forecasting. It splits the work into a **fixed graph stage**, computed once, and
a **learned sequence model**.

```
monthly panel ─► graph snapshot per month ─► fixed context precompute ─► spatial encoder ─► temporal encoder ─► price in h months
 (project and      (project + size-tier        (random projection,          (per-relation        (12-month window,
  size features)    nodes, relations)           mean propagation, 2 hops)    Conv1D + MLP)        Transformer / GRU)
```

## The property graph

Two node types:

- **project** nodes: one per condominium, carrying the project-level features;
- **size-tier** nodes: five per project (SZ1–SZ5), the nodes whose price is predicted.

A new graph snapshot is built for every month, so relations reflect only what was
true at that time.

### Relations

| Relation | Links | Rule | Code name |
|---|---|---|---|
| size-project | size tier – project | the tier belongs to the project (always used) | `size_project` |
| same-MRT | project – project | both within 250 m of the same MRT station that is already open | `same_mrt` |
| price-peer | size tier – size tier | same month, imputed prices within about 10% (\|Δ log price\| ≤ 0.1), at most 5 peers per tier | `similar_price` |
| proximity | project – project | within 250 m of each other | `dist` |
| MRT-distance | project – project | distances to nearest MRT differ by < 5 m | `same_mrt_dist` |
| adjacent-MRT | project – project | nearest stations are consecutive on one line | `adjacent_mrt` |
| school-distance | project – project | distances to nearest school differ by < 0.01 | `same_school_dist` |

Time validity:

- **MRT relations** use only stations open in that month (opening dates in
  `rail_transport.csv`); they are recomputed for each distinct set of open stations.
- **Projects** have no links before they are completed.
- **price-peer** is rebuilt every month from that month's imputed prices only.

The **Full SG** models use size-project, same-MRT and price-peer. The **CCR** models
use the relation set that scored best on validation for each target and horizon,
chosen from all 64 combinations of the six optional relations (see
[checkpoints.md](checkpoints.md)).

## Fixed stage: context precompute

1. Project and size-tier features are normalised and multiplied by a fixed random
   projection into a shared 32-dimensional space (seed 42).
2. For every month, features are averaged over each relation separately, for up to two
   hops. This produces, for every size-tier node and month, a stack of context vectors:
   one group per relation path (for example `self`, `rev_size_project`,
   `similar_price`) and one slot per hop (0, 1, 2).
3. The own imputed price is added as a direct size-tier input in the Full SG models
   and in the CCR models except rent at 1 and 24 months.

The result is a tensor of shape (months, groups, hops, size-tier nodes, features);
for Full SG rent it is (315, 3, 3, 12,485, 33). Nothing in this stage is trained, so it
is computed once (`benchmark/build_cache.py` for Full SG; on the fly for CCR).

## Learned stage

For each size-tier node, a 12-month window of context stacks is the input.

- **Spatial encoder** (per month): each relation group goes through a 1×1 convolution
  that mixes its hop slots, then an MLP; the groups are combined by concatenation or
  averaging.
- **Temporal encoder**: a causal Transformer, GRU or LSTM over the 12 months, so the
  output at each month uses only that month and earlier ones.
- **Head**: a linear layer predicts the price `h` months after each month in the window.

Targets:

- **CCR**: the price level, standardised with the training mean and standard deviation.
- **Full SG**: the log ratio log(price at t+h / imputed price at t), standardised;
  predictions are mapped back to price levels before scoring.

Loss: mean squared error over observed targets only. Where several windows predict
the same size tier and month, their predictions are averaged.

## Training protocol

- **Validation-only selection.** Every choice made for these models used validation
  RMSE (feature months Jan 2018 – Dec 2019): for Full SG the hyperparameters, target
  form and temporal module; for CCR the relation set; for both the stopping epoch.
  The CCR hyperparameter preset for each target and horizon was fixed beforehand
  from earlier tuning on CCR data (`TRIAL_BY_TASK_HORIZON` in `pipeline/ccr_study.py`).
  The test period (from Jan 2020) was evaluated once.
- **No refit.** Each checkpoint is the epoch with the best validation RMSE; models
  were not retrained on train + validation.
- **Leakage masking.** Training windows that reach into the test period have those
  target positions masked.
- **Optimiser.** AdamW with gradient clipping at 5, batch size 1,024 windows; up to 40
  epochs (Full SG, patience 8) or 20 epochs (CCR, patience 5).

## Code map

| File | Role |
|---|---|
| `src/graph/form_node.py`, `src/graph/form_edge.py` | node features and relation tables |
| `src/data/pgim.py` | loader: monthly graphs, dynamic relations, context precompute, windows |
| `src/models/gnn_regressor.py` | spatial encoder, causal Transformer and the CCR model (`GNNRegressor`) |
| `benchmark/models.py` | Full SG model (`GraphSequenceRegressor`: same spatial encoder, Transformer/GRU/LSTM) |
| `pipeline/ccr_study.py` | CCR training runner (one relation set, target and horizon per run) |
| `benchmark/build_cache.py`, `benchmark/train_nn.py` | Full SG context cache and trainer |
| `pipeline/common.py` | dataset registry, trial settings, windowing, scaling, training loop |
