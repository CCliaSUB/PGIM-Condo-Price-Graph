"""Validation-based edge-subset and exact-metapath experiments.

This runner deliberately separates model selection from the post-2020 test set.
Edge screening and metapath ablations use the final ``validation_months`` of the
pre-test period.  The untouched test split is evaluated only by the explicit
``--evaluate-test`` mode after a configuration has been selected.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.common import (
    DATASETS,
    TRIALS,
    _feature_columns,
    apply_target_scale,
    compute_target_scale,
    evaluate,
    prepare_gpu_windows,
    to_serializable,
    train_trial,
)
from src.data.pgim import get_dataloaders
from src.models.gnn_regressor import GNNRegressor
from src.utils.loss import MaskedMSELoss


OPTIONAL_EDGE_TYPES = (
    "dist",
    "same_mrt",
    "same_mrt_dist",
    "adjacent_mrt",
    "same_school_dist",
    "similar_price",
    "same_age",
    "same_planning_area",
)
MANDATORY_EDGE_TYPES = ("size_project",)

TRIAL_BY_TASK_HORIZON = {
    ("rent", 1): "c01_default_b1024",
    ("rent", 12): "c03_wide_deeper_reg",
    ("rent", 24): "c09_mean_reg_low_lr",
    ("sale", 1): "c08_small_dropout",
    ("sale", 12): "c14_mean_lowreg",
    ("sale", 24): "c13_small_mean_stable",
}


def parse_edge_subset(value: str):
    # "none" (or an empty string) means size-project only.
    subset = tuple(sorted({part.strip() for part in value.split(",") if part.strip() not in ("", "none")}))
    unknown = sorted(set(subset) - set(OPTIONAL_EDGE_TYPES))
    if unknown:
        raise ValueError(f"Unknown optional edge types: {unknown}")
    return subset


def subset_label(subset):
    return "structural" if not subset else "plus_" + "_".join(subset)


def expected_metapaths(edge_subset):
    paths = ["self", "size<-[size_project]-project"]
    project_relations = sorted(set(edge_subset) - {"similar_price"})
    if "similar_price" in edge_subset:
        paths.append("size<-[similar_price]-size")
    paths.extend(
        f"size<-[size_project]-project<-[{relation}]-project"
        for relation in project_relations
    )
    paths.append("size<-[size_project]-project<-[size_project]-size")
    if "similar_price" in edge_subset:
        paths.extend(
            [
                "size<-[similar_price]-size<-[size_project]-project",
                "size<-[similar_price]-size<-[similar_price]-size",
            ]
        )
    return paths


def selected_metapaths(args, edge_subset):
    if args.context_mode != "exact_metapaths":
        if args.metapath_setting != "full" or args.metapath:
            raise ValueError("Metapath selection requires --context-mode exact_metapaths.")
        return None
    paths = expected_metapaths(edge_subset)
    if args.metapath_setting == "full":
        return paths
    if args.metapath_setting == "self":
        return ["self"]
    if not args.metapath:
        raise ValueError(f"--metapath is required for setting {args.metapath_setting!r}.")
    if args.metapath not in paths:
        raise ValueError(f"Metapath {args.metapath!r} is not available; choices={paths}")
    if args.metapath_setting == "drop":
        return [path for path in paths if path != args.metapath]
    if args.metapath_setting == "only":
        return ["self"] if args.metapath == "self" else ["self", args.metapath]
    raise ValueError(f"Unknown metapath setting {args.metapath_setting!r}")


def safe_metapath_label(value):
    if not value:
        return "none"
    replacements = {"<-[": "_", "]-": "_", "<": "", ">": "", "[": "", "]": ""}
    label = value
    for old, new in replacements.items():
        label = label.replace(old, new)
    return "_".join(part for part in label.replace("-", "_").split("_") if part)


def build_cfg(args, edge_subset):
    dataset = DATASETS[args.dataset][args.task]
    graph_cfg = OmegaConf.load(dataset["graph_config_path"])
    graph_cfg.name = dataset["graph"]
    if "input_path" in dataset:
        graph_cfg.io_paths.input_path = dataset["input_path"]
    if "output_folder" in dataset:
        graph_cfg.io_paths.output_folder = dataset["output_folder"]
    enabled_csv = set(MANDATORY_EDGE_TYPES) | (set(edge_subset) - {"similar_price"})
    for key, value in graph_cfg.edges.items():
        if isinstance(value, dict) or OmegaConf.is_config(value):
            value.enable = key in enabled_csv

    project_features, size_features = _feature_columns(
        dataset, "target_history", include_ids=False
    )
    return OmegaConf.create(
        {
            "seed": args.seed,
            "deterministic": True,
            "device": f"cuda:{args.gpu}",
            "graph": graph_cfg,
            "data": {
                "graph_config_path": dataset["graph_config_path"],
                "ts_test": 73,
                "target_mask_mode": "observed_only",
                "target_col": "rent_per_sqft",
                "mask_col": "y_mask",
                "feat_norm": "norm1",
                "id_columns": ["node_id", "project_id", "timestep"],
                "target_columns": ["y_mask", "rent_per_sqft"],
                "extra_exclude_columns": [],
                "project_feature_columns": project_features,
                "size_feature_columns": size_features,
                "num_hops": 2,
                "window_size": 12,
                "target_shift": args.horizon,
                "rp_dim": 32,
                "batch_size": args.batch_size,
                "random_seed": args.feature_seed,
                "precompute_device": f"cuda:{args.gpu}",
                "direct_size_feature_columns": (
                    [] if (args.task, args.horizon) in {("rent", 1), ("rent", 24)}
                    else ["rent_per_sqft_imp"]
                ),
                "direct_feature_scale": (
                    "none" if (args.task, args.horizon) in {("rent", 1), ("rent", 24)}
                    else "standard"
                ),
                "split_on": "feature",
                "feature_split_date": "2020-01-01",
                "split_boundary_policy": "offset_mask",
                "dynamic_mrt_edges": True,
                "mrt_opening_date_column": "opening_date",
                "dynamic_condo_edges": True,
                "condo_age_column": "condo_age_at_transaction",
                "similar_price_edges": "similar_price" in edge_subset,
                "similar_price_column": "rent_per_sqft_imp",
                "similar_price_tolerance": 0.10,
                "similar_price_k": args.similar_price_k,
                "context_mode": args.context_mode,
                # Always materialize the complete path basis. Ablations zero
                # channels after loading so model dimensions/parameter counts
                # remain identical across full, drop-one, and single-path runs.
                "metapath_include": (
                    expected_metapaths(edge_subset)
                    if args.context_mode == "exact_metapaths" else None
                ),
            },
            "training": {"batch_size": args.batch_size},
        }
    )


def temporal_slice(data, *, min_feature=None, max_feature=None):
    """Mask and remove samples with no active offsets in a feature-time range."""
    result = dict(data)
    result["min_feature_time"] = min_feature
    result["max_feature_time"] = max_feature
    nodes = result["node_indices"]
    starts = result["target_start_times"]
    offsets = torch.arange(result["window_size"], device=nodes.device)
    target_times = starts[:, None] + offsets
    feature_times = target_times - result["target_shift"]
    active = result["masks"][nodes[:, None], target_times] > 0
    if min_feature is not None:
        active &= feature_times >= int(min_feature)
    if max_feature is not None:
        active &= feature_times < int(max_feature)
    keep = active.any(dim=1)
    result["node_indices"] = nodes[keep]
    result["target_start_times"] = starts[keep]
    return result


def load_trial(args):
    name = args.trial or TRIAL_BY_TASK_HORIZON[(args.task, args.horizon)]
    trial = next((trial for trial in TRIALS if trial.name == name), None)
    if trial is None:
        raise ValueError(f"Unknown trial {name!r}")
    return type(trial)(**{**trial.__dict__, "batch_size": args.batch_size})


def load_gpu_data(cfg, device):
    train_loader, test_loader = get_dataloaders(cfg)
    train_ds, test_ds = train_loader.dataset, test_loader.dataset
    shared = {
        "contexts": torch.from_numpy(train_ds.contexts).to(device),
        "targets": torch.from_numpy(train_ds.targets).to(device),
        "masks": torch.from_numpy(train_ds.masks).to(device),
    }
    train = prepare_gpu_windows(train_ds, tensors=shared, device=device)
    test = prepare_gpu_windows(test_ds, tensors=shared, device=device)
    return train_loader, train, test


def mask_metapath_channels(loader, train, test, active_paths):
    if active_paths is None:
        return train, test
    names = list(loader.dataset.group_names)
    unknown = sorted(set(active_paths) - set(names))
    if unknown:
        raise ValueError(f"Requested active metapaths are absent from contexts: {unknown}")
    disabled = [index for index, name in enumerate(names) if name not in set(active_paths)]
    if disabled:
        # train/test share this tensor by design; mutate it exactly once.
        train["contexts"][:, disabled, :, :, :] = 0
    return train, test


def evaluate_checkpoint(checkpoint_path, trial, shape, data, target_scale, device):
    _, _, num_groups, group_size, feat_dim = shape
    model = GNNRegressor(
        num_hops=group_size,
        num_graphs=num_groups,
        feat_dim=feat_dim,
        hidden_dim=trial.hidden_dim,
        out_dim=1,
        mlp_layers=trial.mlp_layers,
        num_transformer_layers=trial.num_layers,
        heads=trial.num_heads,
        dropout=trial.dropout,
        conv_filters=trial.conv_filters,
        merge_mode=trial.merge_mode,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return evaluate(
        model,
        data,
        MaskedMSELoss().to(device),
        device,
        trial.batch_size,
        target_scale,
        "none",
        False,
        "mean",
    )


def build_model(trial, shape, device):
    _, _, num_groups, group_size, feat_dim = shape
    return GNNRegressor(
        num_hops=group_size,
        num_graphs=num_groups,
        feat_dim=feat_dim,
        hidden_dim=trial.hidden_dim,
        out_dim=1,
        mlp_layers=trial.mlp_layers,
        num_transformer_layers=trial.num_layers,
        heads=trial.num_heads,
        dropout=trial.dropout,
        conv_filters=trial.conv_filters,
        merge_mode=trial.merge_mode,
    ).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ccr")
    parser.add_argument("--task", choices=["rent", "sale"], required=True)
    parser.add_argument("--horizon", choices=[1, 12, 24], type=int, required=True)
    parser.add_argument("--edge-subset", required=True)
    parser.add_argument("--context-mode", choices=["mixed_hops", "exact_metapaths"], default="mixed_hops")
    parser.add_argument("--metapath-setting", choices=["full", "drop", "only", "self"], default="full")
    parser.add_argument("--metapath", default=None)
    parser.add_argument("--validation-months", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--similar-price-k",
        type=int,
        default=5,
        help="Maximum dynamic similar-price neighbours per node and feature month.",
    )
    parser.add_argument("--trial", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/ccr_study"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment runner.")
    edge_subset = parse_edge_subset(args.edge_subset)
    label = subset_label(edge_subset)
    path_label = (
        "full" if args.metapath_setting == "full"
        else args.metapath_setting + "_" + safe_metapath_label(args.metapath)
    )
    run_name = (
        f"{args.dataset}_{args.task}_h{args.horizon}_{label}_{args.context_mode}_"
        f"{path_label}_{args.trial or 'task-default'}_k{args.similar_price_k}_s{args.seed}"
    )
    study_folder = "edge_screen" if args.context_mode == "mixed_hops" else "metapath_screen"
    run_dir = args.out_dir / study_folder / run_name
    result_path = run_dir / "result.json"
    if result_path.exists() and not args.force:
        print(result_path.read_text(encoding="utf-8"))
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    cfg = build_cfg(args, edge_subset)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    device = torch.device(f"cuda:{args.gpu}")
    loader, pretest, test = load_gpu_data(cfg, device)
    active_paths = selected_metapaths(args, edge_subset)
    pretest, test = mask_metapath_channels(loader, pretest, test, active_paths)
    trial = load_trial(args)
    split_step = int(pretest["split_metadata"]["feature_split_timestep"])
    validation_start = split_step - args.validation_months
    inner_train = temporal_slice(pretest, max_feature=validation_start)
    validation = temporal_slice(
        pretest, min_feature=validation_start, max_feature=split_step
    )
    target_scale = compute_target_scale(inner_train, args.batch_size)
    inner_train = apply_target_scale(inner_train, target_scale)
    validation = apply_target_scale(validation, target_scale)
    test = apply_target_scale(test, target_scale)
    checkpoint_path = run_dir / "best_validation.pt"
    trained = train_trial(
        trial,
        inner_train,
        validation,
        loader.dataset.model_context_shape,
        device=device,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        num_workers=0,
        target_scale=target_scale,
        target_transform="none",
        loss_name="mse",
        select_metric="loss",
        predict_last=False,
        aggregate_targets="mean",
        checkpoint_path=checkpoint_path,
    )
    scalar = {
        "dataset": args.dataset,
        "task": args.task,
        "horizon": args.horizon,
        "edge_subset": list(edge_subset),
        "edge_subset_label": label,
        "mandatory_edges": list(MANDATORY_EDGE_TYPES),
        "context_mode": args.context_mode,
        "metapath_setting": args.metapath_setting,
        "metapath": args.metapath,
        "active_metapaths": active_paths,
        "seed": args.seed,
        "feature_seed": args.feature_seed,
        "validation_months": args.validation_months,
        "validation_start_timestep": validation_start,
        "test_split_timestep": split_step,
        "trial": trial.name,
        "similar_price_k": args.similar_price_k,
        "best_epoch": trained["best_epoch"],
        "epochs_ran": trained["epochs_ran"],
        "validation_rmse": trained["rmse"],
        "validation_mae": trained["mae"],
        "validation_mape": trained["mape"],
        "validation_r2": trained["r2"],
        "inner_train_samples": int(inner_train["node_indices"].numel()),
        "validation_samples": int(validation["node_indices"].numel()),
        "group_names": list(loader.dataset.group_names),
        "seconds": round(time.time() - started, 1),
        "checkpoint_path": str(checkpoint_path),
    }
    if args.evaluate_test:
        test_metrics = evaluate_checkpoint(
            checkpoint_path,
            trial,
            loader.dataset.model_context_shape,
            test,
            target_scale,
            device,
        )
        scalar.update({f"test_{key}": value for key, value in test_metrics.items()})
    (run_dir / "history.json").write_text(
        json.dumps(to_serializable(trained["history"]), indent=2), encoding="utf-8"
    )
    result_path.write_text(
        json.dumps(to_serializable(scalar), indent=2), encoding="utf-8"
    )
    print(json.dumps(to_serializable(scalar), indent=2), flush=True)


if __name__ == "__main__":
    main()
