# Singapore Condominium Price Map (demo)

An interactive Vue 2 + FastAPI web app that shows the property graph and the test
predictions of the delivered models, for **rent and sale**, on **Full SG** or **CCR**.
It:

- plots every condominium project on a map of Singapore;
- draws the relations around a selected project, one colour and dash pattern per
  relation;
- shows a schematic of the project's size-tier nodes (SZ1–SZ5, floor-area quintiles);
- compares observed prices with the model's predictions 1, 12 and 24 months ahead.

The demo only reads files: it does not train, and it does not predict for arbitrary
locations.

## Data it needs

| Dataset id (`?dataset=`) | Graph shown | Files |
|---|---|---|
| `full_sg_rent` (default), `full_sg_sale` | the graph the model used: size-project, same-MRT and price-peer, one snapshot per month | `data/processed/full_sg_<task>/`, `demo/predictions/full_sg_<task>_h{1,12,24}.csv`, `demo/predictions/full_sg_<task>_graph_snapshots.csv.gz` |
| `ccr_rent`, `ccr_sale` | the six static candidate relations from `edges/*.csv` | `data/processed/ccr_<task>/`, `demo/predictions/ccr_<task>_h{1,12,24}.csv` |

All of these are in the data package. The prediction files can be regenerated from
the checkpoints with `scripts/evaluate_checkpoints.py --demo` and the snapshots with
`benchmark/build_cache.py` (see [docs/reproduce.md](../docs/reproduce.md)).

For CCR the map shows the candidate relations, not the exact set each model used;
that set differs by target and horizon (see [docs/checkpoints.md](../docs/checkpoints.md)).

The price-peer relation links **size-tier nodes** whose imputed price per sq ft was
within about 10% in the same month (at most 5 per node). The map draws it between
the two projects, and the tooltip names the linked tiers (for example `SZ2 ↔ SZ5`).
Same-MRT uses only stations open in the selected month.

## What the predictions are

Every prediction is a held-out **test** prediction for a month that has an observed
price, from 2020 to early 2026, made by the delivered checkpoint. For a row with
target month *t* and horizon *h*, the model used data up to month *t − h*. Months
without an observed transaction have no row; the chart leaves a gap there.

Each prediction response carries a `provenance` block: dataset, model, horizon, the
file it came from, how the model was selected, and the whole-test-set RMSE, MAE, MAPE
and R². Predictions are model estimates, not market quotations.

Projects without coordinates (1 in `full_sg_rent`, 99 in `full_sg_sale`) can be
searched and show predictions, but cannot be drawn.

## Run it locally

Backend (Python 3.10+), from the repository root:

```bash
cd demo/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000        # API docs at http://localhost:8000/docs
```

Frontend (Node 18+), in another terminal:

```bash
cd demo
npm ci
PGIM_API_URL=http://localhost:8000 npm run serve   # open the URL it prints
```

Tables load in a background thread at startup; set `PGIM_SKIP_WARMUP=1` to disable
that. Tests: `pip install -r requirements-dev.txt && python -m pytest -q` in
`demo/backend` (51 tests, run against the delivered data).

## API

All endpoints take `dataset=` (default: `full_sg_rent`).

| Endpoint | Notes |
|---|---|
| `GET /health`, `GET /datasets` | which datasets are available and what is missing |
| `GET /metadata` | relations (colour, dash, level), horizons, snapshot `months`, horizon coverage |
| `GET /projects?q=` | one row per project, sorted by name |
| `GET /graph/{project_id}?relations=a,b&month=YYYY-MM-01&limit_per_relation=40` | omit `relations` for all, `relations=` for none; `month` applies to Full SG (default: latest) |
| `GET /predictions/{project_id}?horizon=12` | every size tier; tiers without test rows have `points: []` |

Errors: 400 for an unknown dataset, relation, month or horizon; 404 for an unknown
project; 422 for a non-integer id; 503 for a missing or inconsistent file.

## Hosting

`uvicorn app:site` serves the built frontend (`npm run build` → `dist/`) at `/` and
the API under `/api` from one process, so the app can run behind a single URL.

`scripts/build_deploy_bundle.py` assembles a self-contained Docker build context
(Dockerfile, API, built frontend and only the columns the API reads):

```bash
cd demo && npm ci && npm run build && cd ..
python demo/scripts/build_deploy_bundle.py --datasets full_sg_rent,full_sg_sale,ccr_rent,ccr_sale --out /tmp/site
```

The bundle with all four datasets is about 130 MB and uses about 450 MB of RAM;
`PGIM_DATASETS` limits which datasets are served on smaller hosts. `deploy/render.yaml`
deploys the bundle on Render from a GitHub repository; the `Dockerfile` works on any
container host.

A public copy of an earlier version runs at <https://sg-condo-graph-demo.onrender.com>.
It shows predictions from an earlier export, not these checkpoints; redeploy it from
this repository for the two to match.
