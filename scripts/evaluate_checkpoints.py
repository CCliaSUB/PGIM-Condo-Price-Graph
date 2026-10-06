"""Load a delivered checkpoint, predict its validation or test months, and report metrics.

Examples:

    python scripts/evaluate_checkpoints.py --dataset ccr --task rent --horizon 12 --gpu 0
    python scripts/evaluate_checkpoints.py --dataset full_sg --task sale --horizon 1 --gpu 0 \
        --split test --out predictions/full_sg_sale_h1.csv

CCR inputs are built from ``data/processed`` on the fly (a few minutes). Full SG
inputs come from the cache written by ``benchmark/build_cache.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.common import CACHE_ROOT  # noqa: E402
from benchmark.models import GraphSequenceRegressor  # noqa: E402
from benchmark.train_nn import build_splits, load_contexts, metrics_of, predict  # noqa: E402
from pipeline import ccr_study  # noqa: E402
from pipeline.common import DATASETS, apply_target_scale, compute_target_scale  # noqa: E402

CHECKPOINTS = PROJECT_ROOT / "checkpoints"


def ccr_inputs(meta, split, device, gpu):
    """Rebuild the CCR contexts exactly as in training and return (model, data, scale, node ids, dates)."""
    run_args = SimpleNamespace(
        dataset="ccr", task=meta["task"], horizon=meta["horizon"], seed=meta["seed"], feature_seed=meta["feature_seed"],
        gpu=gpu, batch_size=1024, similar_price_k=meta["similar_price_k"], context_mode="mixed_hops",
    )
    cfg = ccr_study.build_cfg(run_args, tuple(meta["edge_subset"]))
    loader, pretest, test = ccr_study.load_gpu_data(cfg, device)
    split_step = int(pretest["split_metadata"]["feature_split_timestep"])
    val_start = split_step - meta["validation_months"]
    inner = ccr_study.temporal_slice(pretest, max_feature=val_start)
    val = ccr_study.temporal_slice(pretest, min_feature=val_start, max_feature=split_step)
    scale = compute_target_scale(inner, 1024)
    data = apply_target_scale(val if split == "validation" else test, scale)

    trial = ccr_study.load_trial(SimpleNamespace(trial=meta["trial"], task=meta["task"], horizon=meta["horizon"], batch_size=1024))
    model = ccr_study.build_model(trial, loader.dataset.model_context_shape, device)
    state = torch.load(meta["dir"] / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    root = Path(DATASETS["ccr"][meta["task"]]["output_folder"])
    nodes = pd.read_csv(root / "node_id.csv")
    size_ids = nodes.loc[~nodes["size_tier"].eq(0), "node_id"].astype(int).sort_values().tolist()
    dates = pd.read_csv(root / "timestep_info.csv").set_index("timestep")["date"]
    return model, data, scale, size_ids, dates


def full_sg_inputs(meta, split, device):
    """Load the Full SG cache and rebuild the splits exactly as ``benchmark/train_nn.py`` does."""
    task, horizon = meta["task"], meta["horizon"]
    cache_meta = json.loads((CACHE_ROOT / task / "meta.json").read_text())
    masks_np = np.load(CACHE_ROOT / task / "masks.npy")
    shared = {
        "contexts": torch.from_numpy(np.ascontiguousarray(load_contexts(task, "hetero"))).to(device),
        "targets": torch.from_numpy(np.load(CACHE_ROOT / task / "targets.npy")).to(device),
        "masks": torch.from_numpy(masks_np).to(device),
    }
    data_h = dict(shared)
    if meta["target"] == "logratio":
        imp = torch.from_numpy(np.load(CACHE_ROOT / task / "tab_imp.npy")).to(device)
        base = torch.ones_like(shared["targets"])
        base[:, horizon:] = imp[: imp.shape[0] - horizon].T.clamp(min=1e-6)
        z = torch.log(shared["targets"].clamp(min=1e-6) / base)
        data_h["targets"] = torch.where(shared["masks"] > 0, z, torch.zeros_like(z))
    inner, val, test, _ = build_splits(data_h, masks_np, cache_meta["feature_split_timestep"], horizon)
    scale = {k: torch.tensor(v, device=device) for k, v in meta["target_scale"].items()}
    data = apply_target_scale(val if split == "validation" else test, scale)
    if meta["target"] == "logratio":
        data["base"], data["raw_targets"] = base, shared["targets"]

    model = GraphSequenceRegressor(**meta["model_kwargs"]).to(device)
    state = torch.load(meta["dir"] / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    dates = pd.Series(cache_meta["dates"])
    return model, data, scale, cache_meta["size_node_ids"], dates


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["ccr", "full_sg"], required=True)
    parser.add_argument("--task", choices=["rent", "sale"], required=True)
    parser.add_argument("--horizon", type=int, choices=[1, 12, 24], required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", type=Path, help="write one row per (size node, month) to this CSV")
    parser.add_argument("--demo", action="store_true",
                        help="also write the test predictions in the map demo's format to demo/predictions/")
    args = parser.parse_args()

    run_dir = CHECKPOINTS / args.dataset / f"{args.task}_h{args.horizon}"
    meta = json.loads((run_dir / "meta.json").read_text())
    meta["dir"] = run_dir
    device = torch.device(f"cuda:{args.gpu}")
    if args.dataset == "ccr":
        model, data, scale, size_ids, dates = ccr_inputs(meta, args.split, device, args.gpu)
    else:
        model, data, scale, size_ids, dates = full_sg_inputs(meta, args.split, device)

    frame = predict(model, data, scale)
    metrics = metrics_of(frame)
    recorded = meta["metrics"][args.split]
    print(json.dumps({"dataset": args.dataset, "task": args.task, "horizon": args.horizon, "split": args.split,
                      "metrics": metrics, "recorded": recorded}, indent=2))

    nodes = pd.read_csv(Path(DATASETS[args.dataset][args.task]["output_folder"]) / "node_id.csv").set_index("node_id")
    node_ids = np.asarray(size_ids)[frame["node"].to_numpy()]
    out = pd.DataFrame({
        "dataset": args.dataset, "task": args.task, "horizon": args.horizon,
        "date": dates.loc[frame["time"].to_numpy()].to_numpy(),
        "target_timestep": frame["time"].to_numpy(),
        "node_id": node_ids,
        "project_id": nodes.loc[node_ids, "project_id"].to_numpy(),
        "size_tier": nodes.loc[node_ids, "size_tier"].to_numpy(),
        "actual": frame["target"].to_numpy(), "predicted": frame["pred"].to_numpy(),
    })
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"wrote {len(out):,} rows to {args.out}")
    if args.demo:
        if args.split != "test":
            raise SystemExit("--demo exports test predictions only")
        demo = pd.DataFrame({
            "dataset": args.task, "seed": meta["seed"], "trial": meta["model"], "forecast_horizon": args.horizon,
            "target_timestep": out["target_timestep"], "date": out["date"], "node_id": out["node_id"],
            "project_id": out["project_id"], "size_tier": out["size_tier"],
            "ground_truth": out["actual"], "prediction": out["predicted"],
            "prediction_source": f"checkpoints/{args.dataset}/{args.task}_h{args.horizon}",
        })
        path = PROJECT_ROOT / "demo" / "predictions" / f"{args.dataset}_{args.task}_h{args.horizon}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        demo.to_csv(path, index=False)
        print(f"wrote {len(demo):,} demo rows to {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
