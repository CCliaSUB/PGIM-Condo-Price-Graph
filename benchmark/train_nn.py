"""Train and evaluate the neural models of the fair benchmark.

Protocol (identical for every model):
* inner training: feature months before the validation block;
* validation: the last ``VALIDATION_MONTHS`` pre-test feature months, used for
  early stopping and for choosing the hyperparameter configuration;
* test: feature months from the split onward, evaluated once with the
  checkpoint that had the best validation RMSE. No refit, no test-based choice.

``--phase search`` screens the configuration grid with one seed and records
validation metrics only. ``--phase final`` takes the validation-best
configuration per (model, horizon) and trains several seeds, reporting test.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.common import CACHE_ROOT, HORIZONS, RESULT_ROOT, VALIDATION_MONTHS, WINDOW
from benchmark.models import GraphSequenceRegressor
from pipeline.ccr_study import temporal_slice
from pipeline.common import apply_target_scale, compute_target_scale, inverse_target, iter_gpu_batches
from src.data.pgim import build_window_indices_feature_split
from src.utils.loss import MaskedMSELoss
from src.utils.seed import set_seed
from src.utils.training import compute_regression_metrics


@dataclass(frozen=True)
class Config:
    name: str
    lr: float
    weight_decay: float
    hidden: int
    layers: int
    dropout: float
    merge: str = "concat"
    cosine: bool = False


GRID = [
    Config("g1", 1e-3, 1e-4, 64, 2, 0.10),
    Config("g2", 5e-4, 1e-4, 128, 2, 0.10),
    Config("g3", 5e-4, 1e-4, 128, 2, 0.20, merge="mean"),
    Config("g4", 3e-4, 1e-4, 128, 3, 0.15, cosine=True),
    Config("g5", 8e-4, 5e-5, 96, 2, 0.05),
    Config("g6", 7e-4, 1e-5, 128, 2, 0.05, merge="mean"),
]
MAX_EPOCHS = 40
PATIENCE = 8
BATCH = 1024


def load_contexts(task: str, graph: str) -> np.ndarray:
    root = CACHE_ROOT / task
    if graph == "hetero":
        return np.load(root / "hetero.npy", mmap_mode="r")
    if graph == "homo":
        return np.load(root / "homo.npy", mmap_mode="r")
    if graph in ("hetero_np", "homo_np"):
        # Same graphs with project covariates zeroed: neighbours contribute price histories only.
        return np.load(root / f"{graph}.npy", mmap_mode="r")
    if graph == "struct":
        # Size-project hierarchy only: own features plus the parent project's features.
        return np.load(root / "struct.npy", mmap_mode="r")
    if graph == "none":
        # Own features only: the untouched self representation (hop 0).
        return np.ascontiguousarray(np.load(root / "hetero.npy", mmap_mode="r")[:, 0:1, 0:1])
    raise ValueError(graph)


def build_splits(shared, masks_np, split_step, horizon):
    (tr_nodes, tr_starts), (te_nodes, te_starts) = build_window_indices_feature_split(
        masks_np, WINDOW, split_step, horizon
    )
    device = shared["targets"].device

    def pack(nodes, starts, **bounds):
        data = dict(shared)
        data.update(
            node_indices=torch.from_numpy(nodes).long().to(device),
            target_start_times=torch.from_numpy(starts).long().to(device),
            window_size=WINDOW, target_shift=horizon, context_layout="time_group_major",
            min_target_time=None, min_feature_time=None, max_target_time=None, max_feature_time=None,
        )
        data.update(bounds)
        return data

    pretest = pack(tr_nodes, tr_starts)
    # Windows that straddle the split contain positions whose targets belong to
    # the test period; count them for the record and mask them out.
    straddle = temporal_slice(pretest, min_feature=split_step)
    leak_positions = 0
    for _, _, y_mask in iter_gpu_batches(straddle, 8192, shuffle=False, seed=0):
        leak_positions += int(y_mask.sum().item())
    val_start = split_step - VALIDATION_MONTHS
    inner = temporal_slice(pretest, max_feature=val_start)
    val = temporal_slice(pretest, min_feature=val_start, max_feature=split_step)
    test = temporal_slice(pack(te_nodes, te_starts), min_feature=split_step)
    return inner, val, test, leak_positions


@torch.no_grad()
def predict(model, data, scale):
    """One prediction per observed (node, target month), averaged over windows.

    With a log-ratio target, ``data`` carries ``raw_targets`` and ``base`` (the
    imputed price at each target's feature month) to map predictions back to
    price levels.
    """
    model.eval()
    nodes, times, preds, targets = [], [], [], []
    for inputs, y, y_mask, (node_idx, target_times) in iter_gpu_batches(
        data, 4096, shuffle=False, seed=0, include_meta=True
    ):
        out = model(inputs)
        valid = (y_mask > 0).squeeze(-1)
        if not valid.any():
            continue
        pred = inverse_target(out.squeeze(-1), scale, "none")
        if "base" in data:
            idx = (node_idx[:, None].expand_as(target_times), target_times)
            pred = torch.exp(pred) * data["base"][idx]
            true = data["raw_targets"][idx]
        else:
            true = inverse_target(y.squeeze(-1), scale, "none")
        preds.append(pred[valid].cpu().numpy())
        targets.append(true[valid].cpu().numpy())
        nodes.append(node_idx[:, None].expand_as(target_times)[valid].cpu().numpy())
        times.append(target_times[valid].cpu().numpy())
    frame = pd.DataFrame({
        "node": np.concatenate(nodes), "time": np.concatenate(times),
        "pred": np.concatenate(preds), "target": np.concatenate(targets),
    })
    return frame.groupby(["node", "time"], sort=True).agg(pred=("pred", "mean"), target=("target", "first")).reset_index()


def metrics_of(frame):
    m = compute_regression_metrics(frame["pred"].to_numpy(), frame["target"].to_numpy())
    return {k: float(v) for k, v in m.items()}


def train_one(cfg: Config, temporal, shape, inner, val, scale, seed, device, log, max_epochs=MAX_EPOCHS):
    set_seed(seed, deterministic=True)
    _, groups, hops, dim = shape
    model = GraphSequenceRegressor(
        groups, hops, dim, cfg.hidden, temporal=temporal, num_layers=cfg.layers, heads=4,
        dropout=cfg.dropout, merge_mode=cfg.merge,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs, 1e-5) if cfg.cosine else None
    loss_fn = MaskedMSELoss()
    best, best_state, stale, history = None, None, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for inputs, y, y_mask in iter_gpu_batches(inner, BATCH, shuffle=True, seed=seed * 1000 + epoch):
            loss = loss_fn(model(inputs), y, y_mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        if sched is not None:
            sched.step()
        vm = metrics_of(predict(model, val, scale))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in vm.items()}})
        log(f"    epoch {epoch:2d} train={np.mean(losses):.4f} val_rmse={vm['rmse']:.4f} val_r2={vm['r2']:.4f}")
        if not math.isfinite(vm["rmse"]):
            break
        if best is None or vm["rmse"] < best["rmse"]:
            best, best_state, stale = vm, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["rent", "sale"], required=True)
    parser.add_argument("--graph", choices=["none", "struct", "homo", "hetero", "homo_np", "hetero_np"], required=True)
    parser.add_argument("--temporal", choices=["transformer", "lstm", "gru"], required=True)
    parser.add_argument("--phase", choices=["search", "final"], required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--configs", nargs="+", default=None, help="subset of grid names (search phase)")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--target", choices=["level", "logratio"], default="level",
                        help="search phase: predict the price level or log(future / current imputed price)")
    parser.add_argument("--out-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    model_name = f"{args.graph}_{args.temporal}"
    out_dir = args.out_root / args.phase / args.task / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"

    def log(msg):
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    meta = json.loads((CACHE_ROOT / args.task / "meta.json").read_text())
    split_step = meta["feature_split_timestep"]
    masks_np = np.load(CACHE_ROOT / args.task / "masks.npy")
    contexts = load_contexts(args.task, args.graph)
    shared = {
        "contexts": torch.from_numpy(np.ascontiguousarray(contexts)).to(device),
        "targets": torch.from_numpy(np.load(CACHE_ROOT / args.task / "targets.npy")).to(device),
        "masks": torch.from_numpy(masks_np).to(device),
    }
    del contexts
    t_dim, groups, hops, n_nodes, dim = shared["contexts"].shape
    shape = (t_dim, groups, hops, dim)
    log(f"{args.task} {model_name} contexts={tuple(shared['contexts'].shape)} split={split_step}")

    results_path = out_dir / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            row = json.loads(line)
            done.add((row["horizon"], row["config"], row.get("target", "level"), row["seed"]))

    imp = torch.from_numpy(np.load(CACHE_ROOT / args.task / "tab_imp.npy")).to(device)  # (T, N)
    for horizon in args.horizons:
        if args.phase == "final":
            search = args.out_root / "search" / args.task / model_name / "results.jsonl"
            rows = [json.loads(l) for l in search.read_text().splitlines()]
            rows = [r for r in rows if r["horizon"] == horizon]
            best_row = min(rows, key=lambda r: r["val"]["rmse"])
            target_kind = best_row.get("target", "level")
        else:
            target_kind = args.target
        data_h = dict(shared)
        if target_kind == "logratio":
            base = torch.ones_like(shared["targets"])
            base[:, horizon:] = imp[: imp.shape[0] - horizon].T.clamp(min=1e-6)
            z = torch.log(shared["targets"].clamp(min=1e-6) / base)
            data_h["targets"] = torch.where(shared["masks"] > 0, z, torch.zeros_like(z))
        inner, val, test, leak = build_splits(data_h, masks_np, split_step, horizon)
        scale = compute_target_scale(inner, 8192)
        inner, val, test = (apply_target_scale(d, scale) for d in (inner, val, test))
        if target_kind == "logratio":
            for d in (val, test):
                d["base"], d["raw_targets"] = base, shared["targets"]
        log(f"h={horizon}: inner={inner['node_indices'].numel()} val={val['node_indices'].numel()} "
            f"test={test['node_indices'].numel()} masked_straddle_positions={leak}")
        if args.phase == "search":
            configs = [c for c in GRID if args.configs is None or c.name in args.configs]
        else:
            configs = [c for c in GRID if c.name == best_row["config"]]
        for cfg in configs:
            for seed in args.seeds:
                if (horizon, cfg.name, target_kind, seed) in done:
                    continue
                started = time.time()
                log(f"  h={horizon} {cfg.name} target={target_kind} seed={seed}")
                model, best_val, history = train_one(cfg, args.temporal, shape, inner, val, scale, seed, device, log, args.max_epochs)
                row = {
                    "task": args.task, "model": model_name, "graph": args.graph, "temporal": args.temporal,
                    "horizon": horizon, "config": cfg.name, "config_detail": asdict(cfg), "target": target_kind, "seed": seed,
                    "val": best_val, "epochs_run": len(history), "seconds": round(time.time() - started, 1),
                    "params": int(sum(p.numel() for p in model.parameters())),
                    "history": history,
                }
                if args.phase == "final":
                    frame = predict(model, test, scale)
                    row["test"] = metrics_of(frame)
                    frame.astype({"pred": "float32", "target": "float32"}).to_parquet(
                        out_dir / f"pred_h{horizon}_s{seed}.parquet", index=False)
                    torch.save({
                        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "model_kwargs": {
                            "num_groups": groups, "group_size": hops, "feat_dim": dim, "hidden_dim": cfg.hidden,
                            "temporal": args.temporal, "num_layers": cfg.layers, "heads": 4,
                            "dropout": cfg.dropout, "merge_mode": cfg.merge,
                        },
                        "task": args.task, "graph": args.graph, "horizon": horizon, "seed": seed,
                        "config": asdict(cfg), "target": target_kind,
                        "target_scale": {k: float(v) for k, v in scale.items()},
                        "window": WINDOW, "feature_split_timestep": split_step,
                        "group_names": meta["group_names"], "val": best_val, "test": row["test"],
                    }, out_dir / f"model_h{horizon}_s{seed}.pt")
                    log(f"  -> test rmse={row['test']['rmse']:.4f} r2={row['test']['r2']:.4f}")
                with results_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                del model
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
