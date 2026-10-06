"""Shared pieces of the training pipelines: dataset registry, trial settings,
GPU windowing, target scaling, training and evaluation loops."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.gnn_regressor import GNNRegressor
from src.utils.loss import MaskedMSELoss
from src.utils.seed import set_seed
from src.utils.training import compute_regression_metrics

# Dataset registry. "ccr" is the Core Central Region panel (postal districts
# 1, 2, 9, 10, 11); "full_sg" covers every condominium in Singapore.
DATASETS = {
    "ccr": {
        "rent": {"graph": "ccr_rent", "graph_config_path": "src/config/graph/rent.yaml",
                 "input_path": "data/raw/ccr_rent.csv", "output_folder": "data/processed/ccr_rent"},
        "sale": {"graph": "ccr_sale", "graph_config_path": "src/config/graph/sale.yaml",
                 "input_path": "data/raw/ccr_sale.csv", "output_folder": "data/processed/ccr_sale"},
    },
    "full_sg": {
        "rent": {"graph": "full_sg_rent", "graph_config_path": "src/config/graph/rent.yaml",
                 "input_path": "data/raw/full_sg_rent.csv", "output_folder": "data/processed/full_sg_rent"},
        "sale": {"graph": "full_sg_sale", "graph_config_path": "src/config/graph/sale.yaml",
                 "input_path": "data/raw/full_sg_sale.csv", "output_folder": "data/processed/full_sg_sale"},
    },
}


@dataclass(frozen=True)
class TrialConfig:
    name: str
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    mlp_layers: int
    conv_filters: int
    merge_mode: str
    batch_size: int
    scheduler: str = "none"
    min_lr: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.999)
    optimizer: str = "adamw"


TRIALS = [
    TrialConfig("c01_default_b1024", 1e-3, 1e-4, 64, 2, 4, 0.1, 2, 2, "concat", 1024),
    TrialConfig("c02_wide_b1024", 1e-3, 1e-4, 128, 2, 4, 0.1, 2, 2, "concat", 1024),
    TrialConfig("c03_wide_deeper_reg", 5e-4, 1e-4, 128, 3, 4, 0.2, 2, 2, "concat", 1024),
    TrialConfig("c04_lowreg_nodrop", 5e-4, 1e-5, 64, 2, 4, 0.0, 2, 2, "concat", 1024),
    TrialConfig("c05_fast_wide_b2048", 2e-3, 1e-4, 128, 2, 4, 0.1, 2, 2, "concat", 2048),
    TrialConfig("c06_mean_merge", 1e-3, 1e-4, 128, 2, 4, 0.1, 2, 2, "mean", 1024),
    TrialConfig("c07_cosine_wide", 1e-3, 1e-4, 128, 2, 4, 0.15, 2, 2, "concat", 1024, "cosine"),
    TrialConfig("c08_small_dropout", 8e-4, 5e-5, 96, 2, 4, 0.05, 2, 2, "concat", 1024),
    TrialConfig("c09_mean_reg_low_lr", 3e-4, 1e-4, 128, 2, 4, 0.2, 2, 2, "mean", 1024),
    TrialConfig("c10_mean_deep_reg", 5e-4, 1e-4, 128, 3, 4, 0.2, 2, 2, "mean", 1024),
    TrialConfig("c11_concat_heads8", 5e-4, 1e-4, 128, 2, 8, 0.15, 2, 2, "concat", 1024),
    TrialConfig("c12_wide192_reg", 5e-4, 1e-4, 192, 2, 4, 0.15, 2, 2, "concat", 1024),
    TrialConfig("c13_small_mean_stable", 3e-4, 1e-4, 64, 2, 4, 0.1, 2, 2, "mean", 1024),
    TrialConfig("c14_mean_lowreg", 7e-4, 1e-5, 128, 2, 4, 0.05, 2, 2, "mean", 1024),
    TrialConfig("c15_concat_deep_cosine", 3e-4, 1e-4, 128, 3, 4, 0.15, 2, 2, "concat", 1024, "cosine"),
    TrialConfig("c16_mean_b2048", 1e-3, 1e-4, 128, 2, 4, 0.1, 2, 2, "mean", 2048),
    TrialConfig("c17_radam_mean", 5e-4, 1e-4, 128, 2, 4, 0.1, 2, 2, "mean", 1024, optimizer="radam"),
    TrialConfig("c18_radam_concat", 5e-4, 1e-4, 128, 2, 4, 0.1, 2, 2, "concat", 1024, optimizer="radam"),
]


def prepare_gpu_windows(dataset, *, tensors=None, device=None):
    if tensors is None:
        tensors = {
            "contexts": torch.from_numpy(dataset.contexts).to(device),
            "targets": torch.from_numpy(dataset.targets).to(device),
            "masks": torch.from_numpy(dataset.masks).to(device),
        }
    return {
        "contexts": tensors["contexts"],
        "targets": tensors["targets"],
        "masks": tensors["masks"],
        "node_indices": torch.from_numpy(dataset.node_indices).long().to(device),
        "target_start_times": torch.from_numpy(dataset.target_start_times).long().to(device),
        "window_size": int(dataset.window_size),
        "target_shift": int(dataset.target_shift),
        "min_target_time": dataset.min_target_time,
        "min_feature_time": getattr(dataset, "min_feature_time", None),
        "max_target_time": getattr(dataset, "max_target_time", None),
        "max_feature_time": getattr(dataset, "max_feature_time", None),
        "split_metadata": getattr(dataset, "split_metadata", {}),
        "context_layout": getattr(dataset, "context_layout", "node_major"),
    }


def iter_gpu_batches(
    data,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    predict_last: bool = False,
    include_meta: bool = False,
):
    num_samples = int(data["node_indices"].numel())
    if shuffle:
        generator = torch.Generator(device=data["node_indices"].device)
        generator.manual_seed(seed)
        order = torch.randperm(num_samples, device=data["node_indices"].device, generator=generator)
    else:
        order = torch.arange(num_samples, device=data["node_indices"].device)

    offsets = torch.arange(data["window_size"], device=data["node_indices"].device)
    for start in range(0, num_samples, batch_size):
        batch_idx = order[start : start + batch_size]
        node_idx = data["node_indices"][batch_idx]
        target_start = data["target_start_times"][batch_idx]
        input_times = target_start[:, None] - data["target_shift"] + offsets[None, :]
        target_times = target_start[:, None] + offsets[None, :]

        if data.get("context_layout", "node_major") == "time_major":
            inputs = data["contexts"][input_times, node_idx[:, None]]
        elif data.get("context_layout", "node_major") == "time_group_major":
            inputs = data["contexts"][input_times, :, :, node_idx[:, None], :]
        else:
            inputs = data["contexts"][node_idx[:, None], input_times]
        targets = data["targets"][node_idx[:, None], target_times].unsqueeze(-1)
        y_mask = data["masks"][node_idx[:, None], target_times].unsqueeze(-1).float()
        if data["min_target_time"] is not None:
            y_mask = y_mask * (target_times >= int(data["min_target_time"])).unsqueeze(-1)
        if data.get("min_feature_time") is not None:
            y_mask = y_mask * (input_times >= int(data["min_feature_time"])).unsqueeze(-1)
        if data.get("max_target_time") is not None:
            y_mask = y_mask * (target_times < int(data["max_target_time"])).unsqueeze(-1)
        if data.get("max_feature_time") is not None:
            y_mask = y_mask * (input_times < int(data["max_feature_time"])).unsqueeze(-1)
        meta = None
        if include_meta:
            meta = (node_idx, target_times)
        if predict_last:
            targets = targets[:, -1:, :]
            y_mask = y_mask[:, -1:, :]
            if include_meta:
                meta = (node_idx, target_times[:, -1:])
        if include_meta:
            yield inputs, targets, y_mask, meta
        else:
            yield inputs, targets, y_mask


def compute_target_scale(data, batch_size: int):
    total = torch.zeros((), device=data["targets"].device)
    total_sq = torch.zeros((), device=data["targets"].device)
    count = torch.zeros((), device=data["targets"].device)
    for _, targets, y_mask in iter_gpu_batches(data, batch_size, shuffle=False, seed=0):
        valid = y_mask > 0
        if valid.any():
            values = targets[valid]
            total += values.sum()
            total_sq += (values * values).sum()
            count += values.numel()
    mean = total / count.clamp(min=1)
    var = total_sq / count.clamp(min=1) - mean * mean
    std = torch.sqrt(var.clamp(min=1e-8))
    return {"mean": mean.detach(), "std": std.detach()}


def apply_target_scale(data, scale):
    data = dict(data)
    data["targets"] = (data["targets"] - scale["mean"]) / scale["std"]
    return data


def inverse_target(values, scale, transform: str):
    values = values * scale["std"] + scale["mean"]
    if transform == "none":
        return values
    if transform == "log":
        return torch.exp(values)
    raise ValueError(f"Unsupported target transform: {transform}")


def masked_train_loss(outputs, targets, y_mask, criterion, loss_name: str, target_scale, target_transform: str):
    if loss_name == "mse":
        return criterion(outputs, targets, y_mask)

    mask = y_mask.float()
    denom = mask.sum().clamp(min=1.0)
    raw_outputs = inverse_target(outputs, target_scale, target_transform)
    raw_targets = inverse_target(targets, target_scale, target_transform)

    if loss_name == "mae":
        loss = (raw_outputs - raw_targets).abs()
    elif loss_name == "mape":
        loss = (raw_outputs - raw_targets).abs() / raw_targets.abs().clamp(min=1e-8)
    elif loss_name == "mse_mape":
        mse = criterion(outputs, targets, y_mask)
        mape = ((raw_outputs - raw_targets).abs() / raw_targets.abs().clamp(min=1e-8) * mask).sum() / denom
        return mse + 0.01 * mape
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    return (loss * mask).sum() / denom


def is_better(metrics, best, select_metric: str):
    if best is None:
        return True
    value = metrics[select_metric]
    best_value = best[select_metric]
    if not np.isfinite(value):
        return False
    if select_metric == "r2":
        return value > best_value
    return value < best_value


def evaluate(
    model,
    data,
    criterion,
    device,
    batch_size: int,
    target_scale,
    target_transform: str,
    predict_last: bool,
    aggregate_targets: str = "none",
):
    model.eval()
    total_loss = 0.0
    preds = []
    targets_out = []
    meta_nodes = []
    meta_times = []
    num_batches = 0
    with torch.no_grad():
        for inputs, targets, y_mask, meta in iter_gpu_batches(
            data,
            batch_size,
            shuffle=False,
            seed=0,
            predict_last=predict_last,
            include_meta=True,
        ):
            outputs = model(inputs)
            if predict_last:
                outputs = outputs[:, -1:, :]
            total_loss += float(criterion(outputs, targets, y_mask).item())
            num_batches += 1
            valid = y_mask > 0
            if valid.any():
                raw_outputs = inverse_target(outputs[valid], target_scale, target_transform)
                raw_targets = inverse_target(targets[valid], target_scale, target_transform)
                preds.append(raw_outputs.detach().cpu().numpy().reshape(-1, 1))
                targets_out.append(raw_targets.detach().cpu().numpy().reshape(-1, 1))
                if aggregate_targets != "none":
                    node_idx, target_times = meta
                    valid_2d = valid.squeeze(-1)
                    meta_nodes.append(node_idx[:, None].expand_as(target_times)[valid_2d].detach().cpu().numpy().reshape(-1))
                    meta_times.append(target_times[valid_2d].detach().cpu().numpy().reshape(-1))

    if preds:
        pred_values = np.vstack(preds).reshape(-1)
        target_values = np.vstack(targets_out).reshape(-1)
        if aggregate_targets == "mean":
            meta_node_values = np.concatenate(meta_nodes)
            meta_time_values = np.concatenate(meta_times)
            frame = pd.DataFrame(
                {
                    "node": meta_node_values,
                    "time": meta_time_values,
                    "pred": pred_values,
                    "target": target_values,
                }
            )
            frame = frame.groupby(["node", "time"], sort=False).agg(
                pred=("pred", "mean"),
                target=("target", "first"),
            )
            metrics = compute_regression_metrics(
                frame["pred"].to_numpy().reshape(-1, 1),
                frame["target"].to_numpy().reshape(-1, 1),
            )
        elif aggregate_targets == "none":
            metrics = compute_regression_metrics(pred_values.reshape(-1, 1), target_values.reshape(-1, 1))
        else:
            raise ValueError(f"Unsupported aggregate_targets={aggregate_targets!r}")
    else:
        metrics = {"rmse": math.nan, "mae": math.nan, "mape": math.nan, "r2": math.nan, "num_samples": 0}
    metrics["loss"] = total_loss / max(num_batches, 1)
    return metrics


def predict_valid(model, data, batch_size: int, target_scale, target_transform: str, predict_last: bool):
    model.eval()
    preds = []
    targets_out = []
    node_ids = []
    target_times_out = []
    with torch.no_grad():
        num_samples = int(data["node_indices"].numel())
        offsets = torch.arange(data["window_size"], device=data["node_indices"].device)
        for start in range(0, num_samples, batch_size):
            batch_idx = torch.arange(start, min(start + batch_size, num_samples), device=data["node_indices"].device)
            node_idx = data["node_indices"][batch_idx]
            target_start = data["target_start_times"][batch_idx]
            input_times = target_start[:, None] - data["target_shift"] + offsets[None, :]
            target_times = target_start[:, None] + offsets[None, :]

            inputs = data["contexts"][node_idx[:, None], input_times]
            targets = data["targets"][node_idx[:, None], target_times].unsqueeze(-1)
            y_mask = data["masks"][node_idx[:, None], target_times].unsqueeze(-1).float()
            if data["min_target_time"] is not None:
                y_mask = y_mask * (target_times >= int(data["min_target_time"])).unsqueeze(-1)
            if data.get("min_feature_time") is not None:
                input_times = target_times - data["target_shift"]
                y_mask = y_mask * (input_times >= int(data["min_feature_time"])).unsqueeze(-1)
            if data.get("max_target_time") is not None:
                y_mask = y_mask * (target_times < int(data["max_target_time"])).unsqueeze(-1)
            if data.get("max_feature_time") is not None:
                input_times = target_times - data["target_shift"]
                y_mask = y_mask * (input_times < int(data["max_feature_time"])).unsqueeze(-1)
            if predict_last:
                targets = targets[:, -1:, :]
                y_mask = y_mask[:, -1:, :]
                target_times = target_times[:, -1:]

            outputs = model(inputs)
            if predict_last:
                outputs = outputs[:, -1:, :]
            valid = y_mask > 0
            if valid.any():
                preds.append(inverse_target(outputs[valid], target_scale, target_transform).detach().cpu().numpy().reshape(-1))
                targets_out.append(inverse_target(targets[valid], target_scale, target_transform).detach().cpu().numpy().reshape(-1))
                valid_2d = valid.squeeze(-1)
                node_ids.append(node_idx[:, None].expand_as(target_times)[valid_2d].detach().cpu().numpy().reshape(-1))
                target_times_out.append(target_times[valid_2d].detach().cpu().numpy().reshape(-1))
    if not preds:
        empty_float = np.empty((0,), dtype=np.float32)
        empty_int = np.empty((0,), dtype=np.int64)
        return empty_float, empty_float, empty_int, empty_int
    return (
        np.concatenate(preds).astype(np.float32),
        np.concatenate(targets_out).astype(np.float32),
        np.concatenate(node_ids).astype(np.int64),
        np.concatenate(target_times_out).astype(np.int64),
    )


def train_trial(
    trial: TrialConfig,
    train_data,
    test_data,
    shape,
    *,
    device,
    seed: int,
    epochs: int,
    patience: int,
    num_workers: int,
    target_scale,
    target_transform: str,
    loss_name: str,
    select_metric: str,
    predict_last: bool,
    aggregate_targets: str,
    save_pred_path: Path | None = None,
    checkpoint_path: Path | None = None,
    epoch_callback=None,
):
    set_seed(seed, deterministic=True)
    _, _, num_graphs, num_hops, feat_dim = shape
    model = GNNRegressor(
        num_hops=num_hops,
        num_graphs=num_graphs,
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
    criterion = MaskedMSELoss().to(device)
    if trial.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=trial.learning_rate,
            weight_decay=trial.weight_decay,
            betas=trial.betas,
        )
    elif trial.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=trial.learning_rate,
            weight_decay=trial.weight_decay,
            betas=trial.betas,
        )
    elif trial.optimizer == "radam":
        optimizer = torch.optim.RAdam(
            model.parameters(),
            lr=trial.learning_rate,
            weight_decay=trial.weight_decay,
            betas=trial.betas,
        )
    else:
        raise ValueError(f"Unsupported optimizer={trial.optimizer}")
    scheduler = None
    if trial.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=trial.min_lr,
        )

    best = None
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for inputs, targets, y_mask in iter_gpu_batches(
            train_data,
            trial.batch_size,
            shuffle=True,
            seed=seed + epoch,
            predict_last=predict_last,
        ):
            outputs = model(inputs)
            if predict_last:
                outputs = outputs[:, -1:, :]
            loss = masked_train_loss(outputs, targets, y_mask, criterion, loss_name, target_scale, target_transform)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))

        if scheduler is not None:
            scheduler.step()

        metrics = evaluate(
            model,
            test_data,
            criterion,
            device,
            trial.batch_size,
            target_scale,
            target_transform,
            predict_last,
            aggregate_targets,
        )
        train_loss = float(np.mean(losses)) if losses else math.nan
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": metrics["loss"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "mape": metrics["mape"],
                "r2": metrics["r2"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if epoch_callback is not None:
            epoch_callback(history[-1])
        print(
            f"{trial.name} epoch={epoch} train={train_loss:.6f} "
            f"rmse={metrics['rmse']:.6f} mae={metrics['mae']:.6f} "
            f"mape={metrics['mape']:.4f} r2={metrics['r2']:.6f}",
            flush=True,
        )

        if is_better(metrics, best, select_metric):
            best = dict(metrics)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if patience and stale >= patience:
                break

        if not np.isfinite(metrics["loss"]):
            break

    if best is None:
        best = {"loss": math.nan, "rmse": math.nan, "mae": math.nan, "mape": math.nan, "r2": math.nan, "num_samples": 0}
    if checkpoint_path is not None and best_state is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "best_epoch": best_epoch,
                "best_metrics": best,
                "trial": asdict(trial),
                "seed": seed,
            },
            checkpoint_path,
        )
    pred_path = None
    if save_pred_path is not None and best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
        preds, targets, node_ids, target_times = predict_valid(
            model,
            test_data,
            trial.batch_size,
            target_scale,
            target_transform,
            predict_last,
        )
        save_pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_pred_path,
            preds=preds,
            targets=targets,
            node_ids=node_ids,
            target_times=target_times,
        )
        pred_path = str(save_pred_path)
    return {
        "trial": trial.name,
        **asdict(trial),
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "seconds": round(time.time() - start, 1),
        "loss": best["loss"],
        "rmse": best["rmse"],
        "mae": best["mae"],
        "mape": best["mape"],
        "r2": best["r2"],
        "num_samples": best["num_samples"],
        "pred_path": pred_path,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "history": history,
    }


def to_serializable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    return value


ID_FEATURES = {
    "project_idx",
    "month_idx",
    "node_id_feat",
    "project_id_feat",
    "timestep_feat",
}


TARGET_DERIVED_FEATURES = {
    "rent_per_sqft",
    "rent_per_sqft_feat",
    "rent_per_sqft_imp",
    "rent_psf_mu",
    "rent_psf_pi_lo",
    "rent_psf_pi_hi",
    "sale_psf_mu",
    "sale_psf_pi_lo",
    "sale_psf_pi_hi",
    "avg_psf_hist",
    "decoder_sigma",
}


def _feature_columns(dataset: dict, setting: str, include_ids: bool):
    root = Path(dataset["output_folder"])
    project_cols = pd.read_csv(root / "project_level_node.csv", nrows=0).columns.tolist()
    size_cols = pd.read_csv(root / "size_level_node.csv", nrows=0).columns.tolist()

    always_excluded = {"node_id", "project_id", "timestep", "y_mask", "rent_per_sqft"}
    excluded = set(always_excluded)
    if setting == "leakage_safe":
        excluded |= TARGET_DERIVED_FEATURES
    elif setting != "target_history":
        raise ValueError(f"Unknown feature setting: {setting}")
    if not include_ids:
        excluded |= ID_FEATURES

    project = [col for col in project_cols if col not in excluded]
    size = [col for col in size_cols if col not in excluded]
    return project, size
