import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.graph.form_edge import (
    FormEdgeProcessor,
    adjacent_mrt_pairs,
    pairwise_haversine_m,
)
from src.graph.form_node import FormNodeProcessor

DEFAULT_GRAPH_CONFIG_PATH = "src/config/graph/rent.yaml"
logger = logging.getLogger(__name__)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@dataclass
class HeteroGraphSpec:
    edge_index: dict
    num_nodes: dict

    @property
    def canonical_etypes(self):
        return list(self.edge_index.keys())

    @property
    def ntypes(self):
        return list(self.num_nodes.keys())

    def num_nodes_of(self, ntype):
        return self.num_nodes[ntype]

    def __repr__(self):
        edge_counts = {
            f"{src}:{etype}:{dst}": int(edges[0].numel())
            for (src, etype, dst), edges in self.edge_index.items()
        }
        return f"HeteroGraphSpec(num_nodes={self.num_nodes}, edge_counts={edge_counts})"


def infer_feature_columns(frame: pd.DataFrame, id_columns=None, target_columns=None, extra_exclude=None):
    excluded = set(extra_exclude or set()).union(id_columns or set()).union(target_columns or set())
    return [
        col
        for col in frame.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
    ]


def to_plain_config(config):
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    if config is None:
        return {}
    return dict(config)


def normalize_loader_config(config):
    config = to_plain_config(config)
    if "data" in config:
        data_config = dict(config["data"])
        training_config = config.get("training", {})
        data_config.setdefault("batch_size", training_config.get("batch_size", 32))
        data_config.setdefault("random_seed", config.get("seed", 42))
    else:
        data_config = config

    data_config.setdefault("graph_config_path", DEFAULT_GRAPH_CONFIG_PATH)
    data_config.setdefault("ts_test", 25)
    data_config.setdefault("num_hops", 2)
    data_config.setdefault("batch_size", 32)
    data_config.setdefault("target_col", "rent_per_sqft_imp")
    data_config.setdefault("mask_col", "y_mask")
    data_config.setdefault("feat_norm", True)
    data_config.setdefault("window_size", 12)
    data_config.setdefault("target_shift", 1)
    data_config.setdefault("target_mask_mode", "train_all_test_observed")
    data_config.setdefault("id_columns", ["node_id", "project_id", "timestep"])
    data_config.setdefault("target_columns", ["y_mask", "rent_per_sqft", "rent_per_sqft_imp"])
    data_config.setdefault("extra_exclude_columns", [])
    data_config.setdefault("rp_dim", 32)
    data_config.setdefault("random_seed", 42)
    data_config.setdefault("precompute_device", "auto")
    data_config.setdefault("direct_size_feature_columns", [])
    data_config.setdefault("direct_feature_scale", "none")
    data_config.setdefault("project_feature_columns", None)
    data_config.setdefault("size_feature_columns", None)
    data_config.setdefault("split_on", "target")
    data_config.setdefault("feature_split_date", None)
    data_config.setdefault("feature_split_timestep", None)
    data_config.setdefault("split_boundary_policy", "offset_mask")
    data_config.setdefault("dynamic_mrt_edges", False)
    data_config.setdefault("mrt_opening_date_column", "opening_date")
    data_config.setdefault("dynamic_condo_edges", False)
    data_config.setdefault("condo_age_column", "condo_age_at_transaction")
    data_config.setdefault("similar_price_edges", False)
    data_config.setdefault("similar_price_column", "rent_per_sqft_imp")
    data_config.setdefault("similar_price_tolerance", 0.10)
    data_config.setdefault("similar_price_k", 5)
    data_config.setdefault("context_mode", "mixed_hops")
    data_config.setdefault("metapath_include", None)
    return data_config


def feature_normalize(frame: pd.DataFrame, mode, stats_folder, feature_cols):
    """Apply saved preprocessing normalization statistics to selected feature columns."""
    if mode in (None, False, "false", "False", "none", "None", "off", "raw"):
        return frame.copy()

    mode = "norm1" if mode is True else str(mode)
    stats_filename = {
        "norm0": "norm0.json",
        "norm1": "norm1.json",
        "stand": "stand.json",
        "standard": "stand.json",
        "standardize": "stand.json",
        "standardise": "stand.json",
    }.get(mode)
    if stats_filename is None:
        raise ValueError(
            "feat_norm must be one of norm0, norm1, stand, false/raw, "
            f"got {mode!r}."
        )

    stats_path = Path(stats_folder) / stats_filename
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing normalization stats file: {stats_path}")

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    normalized = frame.copy()
    missing_stats = [col for col in feature_cols if col not in stats]
    if missing_stats:
        raise KeyError(
            f"Missing {mode} stats for feature columns in {stats_path}: {missing_stats}"
        )

    for col in feature_cols:
        values = pd.to_numeric(normalized[col], errors="coerce").astype(float)
        col_stats = stats[col]
        if stats_filename == "stand.json":
            std = float(col_stats.get("std", 1.0)) or 1.0
            normalized[col] = (values - float(col_stats["mean"])) / std
        else:
            denominator = float(col_stats.get("denominator", 1.0)) or 1.0
            target_min = float(col_stats.get("target_min", 0.0))
            target_max = float(col_stats.get("target_max", 1.0))
            normalized[col] = (
                (values - float(col_stats["min"]))
                / denominator
                * (target_max - target_min)
                + target_min
            )

    normalized[feature_cols] = normalized[feature_cols].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return normalized


def build_time_node_tensor(frame, node_ids, timesteps, feature_cols):
    frame = frame.copy()
    for col in ["node_id", "timestep", *feature_cols]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.sort_values(["timestep", "node_id"]).drop_duplicates(["timestep", "node_id"], keep="last")

    time_index = {int(t): i for i, t in enumerate(timesteps)}
    node_index = {int(n): i for i, n in enumerate(node_ids)}
    tensor = np.zeros((len(timesteps), len(node_ids), len(feature_cols)), dtype=np.float32)

    rows = frame[["timestep", "node_id", *feature_cols]].copy()
    rows["time_idx"] = rows["timestep"].map(time_index)
    rows["node_idx"] = rows["node_id"].map(node_index)
    rows = rows.dropna(subset=["time_idx", "node_idx"])
    if rows.empty:
        return tensor

    time_idx = rows["time_idx"].to_numpy(dtype=np.int64)
    node_idx = rows["node_idx"].to_numpy(dtype=np.int64)
    values = rows[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    tensor[time_idx, node_idx] = values
    return tensor


def build_target_tensor(size_frame, size_node_ids, timesteps, target_col, mask_col, target_mask_mode, train_steps):
    frame = size_frame.copy()
    for col in ["node_id", "timestep", target_col, mask_col]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.sort_values(["timestep", "node_id"]).drop_duplicates(["timestep", "node_id"], keep="last")

    time_index = {int(t): i for i, t in enumerate(timesteps)}
    node_index = {int(n): i for i, n in enumerate(size_node_ids)}
    targets = np.zeros((len(size_node_ids), len(timesteps)), dtype=np.float32)
    observed = np.zeros_like(targets)

    rows = frame[["timestep", "node_id", target_col, mask_col]].copy()
    rows["time_idx"] = rows["timestep"].map(time_index)
    rows["node_idx"] = rows["node_id"].map(node_index)
    rows = rows.dropna(subset=["time_idx", "node_idx"])
    time_idx = rows["time_idx"].to_numpy(dtype=np.int64)
    node_idx = rows["node_idx"].to_numpy(dtype=np.int64)
    targets[node_idx, time_idx] = rows[target_col].fillna(0.0).to_numpy(dtype=np.float32)
    observed[node_idx, time_idx] = rows[mask_col].fillna(0.0).to_numpy(dtype=np.float32)

    if target_mask_mode == "all":
        masks = np.ones_like(targets, dtype=np.float32)
    elif target_mask_mode == "observed_only":
        masks = observed
    elif target_mask_mode == "train_all_test_observed":
        masks = np.ones_like(targets, dtype=np.float32)
        masks[:, train_steps:] = observed[:, train_steps:]
    else:
        raise ValueError(f"Unknown target_mask_mode={target_mask_mode}")
    return targets, masks


def random_project_features(features: np.ndarray, out_dim: int, seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.from_numpy(features.astype(np.float32))
    if x.shape[-1] != out_dim:
        scale = np.sqrt(2.0 / max(x.shape[-1] + out_dim, 1))
        kernel = torch.randn((x.shape[-1], out_dim), generator=generator, dtype=torch.float32) * scale
        x = x @ kernel
    x = torch.nn.functional.normalize(x, dim=-1)
    return x.numpy().astype(np.float32)


def append_direct_size_features(
    project_features,
    size_features,
    size_frame,
    size_node_ids,
    timesteps,
    direct_feature_cols,
    train_steps,
    scale="none",
):
    if not direct_feature_cols:
        return project_features, size_features

    missing = [col for col in direct_feature_cols if col not in size_frame.columns]
    if missing:
        raise KeyError(f"Missing direct size feature columns: {missing}")

    direct_features = build_time_node_tensor(size_frame, size_node_ids, timesteps, direct_feature_cols)
    if scale == "standard":
        train_values = direct_features[:train_steps].reshape(-1, direct_features.shape[-1])
        mean = train_values.mean(axis=0, keepdims=True)
        std = train_values.std(axis=0, keepdims=True)
        std = np.where(std > 1e-8, std, 1.0)
        direct_features = (direct_features - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    elif scale not in (None, "none", "raw"):
        raise ValueError(f"Unsupported direct_feature_scale={scale!r}")

    direct_features = direct_features.astype(np.float32)
    project_pad = np.zeros(
        (project_features.shape[0], project_features.shape[1], direct_features.shape[-1]),
        dtype=np.float32,
    )
    return (
        np.concatenate([project_features, project_pad], axis=-1),
        np.concatenate([size_features, direct_features], axis=-1),
    )


def edge_pairs_from_frame(edges: pd.DataFrame, source):
    if not {"node_id", "neig_node_id"}.issubset(edges.columns):
        raise ValueError(f"Expected node_id/neig_node_id columns in {source}")
    edges = edges[["node_id", "neig_node_id"]].copy()
    edges["node_id"] = pd.to_numeric(edges["node_id"], errors="coerce")
    edges["neig_node_id"] = pd.to_numeric(edges["neig_node_id"], errors="coerce")
    return edges.dropna().astype({"node_id": int, "neig_node_id": int}).drop_duplicates()


def build_heterograph_from_edge_frames(edge_frames, project_node_ids, size_node_ids):
    project_index = {int(node_id): i for i, node_id in enumerate(project_node_ids)}
    size_index = {int(node_id): i for i, node_id in enumerate(size_node_ids)}
    graph_data = {}

    def add_edges(src_type, etype, dst_type, src, dst):
        src_tensor = torch.tensor(src, dtype=torch.int64)
        dst_tensor = torch.tensor(dst, dtype=torch.int64)
        graph_data[(src_type, etype, dst_type)] = (src_tensor, dst_tensor)
        graph_data[(dst_type, f"rev_{etype}", src_type)] = (dst_tensor, src_tensor)

    for edge_name, edge_frame in edge_frames.items():
        pairs = edge_pairs_from_frame(edge_frame, edge_name)
        stem = Path(edge_name).stem
        if stem == "size_project":
            src = pairs["node_id"].map(size_index)
            dst = pairs["neig_node_id"].map(project_index)
            valid = src.notna() & dst.notna()
            add_edges("size", stem, "project", src[valid].astype(int).tolist(), dst[valid].astype(int).tolist())
        else:
            src = pairs["node_id"].map(project_index)
            dst = pairs["neig_node_id"].map(project_index)
            valid = src.notna() & dst.notna()
            add_edges("project", stem, "project", src[valid].astype(int).tolist(), dst[valid].astype(int).tolist())

    return HeteroGraphSpec(
        edge_index=graph_data,
        num_nodes={"project": len(project_node_ids), "size": len(size_node_ids)},
    )


def _similar_price_pairs(node_ids, prices, tolerance, max_neighbors):
    """Return unique same-timestep pairs of size nodes with similar log prices."""
    values = pd.DataFrame({"node_id": node_ids, "price": prices})
    values["node_id"] = pd.to_numeric(values["node_id"], errors="coerce")
    values["price"] = pd.to_numeric(values["price"], errors="coerce")
    values = values.dropna()
    values = values.loc[values["price"].gt(0)].drop_duplicates("node_id", keep="last")
    if len(values) < 2:
        return []

    values["log_price"] = np.log(values["price"].to_numpy(dtype=float))
    values = values.sort_values(["log_price", "node_id"]).reset_index(drop=True)
    ids = values["node_id"].to_numpy(dtype=np.int64)
    logs = values["log_price"].to_numpy(dtype=float)
    candidates = set()
    # In sorted log-price order the closest candidate is always immediately to
    # the left or right. This two-pointer walk avoids materialising dense
    # all-pairs differences.
    for idx, node_id in enumerate(ids):
        left, right = idx - 1, idx + 1
        selected = 0
        while selected < max_neighbors and (left >= 0 or right < len(ids)):
            left_diff = logs[idx] - logs[left] if left >= 0 else np.inf
            right_diff = logs[right] - logs[idx] if right < len(ids) else np.inf
            if min(left_diff, right_diff) > tolerance:
                break
            neighbor_idx = left if left_diff <= right_diff else right
            pair = tuple(sorted((int(node_id), int(ids[neighbor_idx]))))
            candidates.add((float(abs(logs[neighbor_idx] - logs[idx])), pair))
            selected += 1
            if neighbor_idx == left:
                left -= 1
            else:
                right += 1

    # Greedy degree-capped symmetrisation makes ``max_neighbors`` an actual
    # maximum degree rather than only a maximum number selected by each source.
    degrees = defaultdict(int)
    pairs = []
    for _, pair in sorted(candidates, key=lambda item: (item[0], item[1])):
        src, dst = pair
        if degrees[src] >= max_neighbors or degrees[dst] >= max_neighbors:
            continue
        pairs.append(pair)
        degrees[src] += 1
        degrees[dst] += 1
    return sorted(pairs)


def build_similar_price_graphs(
    base_graphs,
    size_frame,
    size_node_ids,
    timesteps,
    price_column="rent_per_sqft_imp",
    tolerance=0.10,
    max_neighbors=5,
):
    """Overlay a contemporaneous size-to-size price relation on graph snapshots.

    Each snapshot uses only rows whose ``timestep`` equals the graph timestep.
    No edge connects records from different timesteps.
    """
    if tolerance <= 0:
        raise ValueError(f"similar_price_tolerance must be positive, got {tolerance}.")
    if max_neighbors <= 0:
        raise ValueError(f"similar_price_k must be positive, got {max_neighbors}.")
    required = {"timestep", "node_id", price_column}
    missing = sorted(required - set(size_frame.columns))
    if missing:
        raise KeyError(f"Missing columns for similar-price edges: {missing}")
    if len(base_graphs) != len(timesteps):
        raise ValueError(
            f"Expected one base graph per timestep, got {len(base_graphs)} for {len(timesteps)} timesteps."
        )

    size_index = {int(node_id): idx for idx, node_id in enumerate(size_node_ids)}
    frame = size_frame[["timestep", "node_id", price_column]].copy()
    frame["timestep"] = pd.to_numeric(frame["timestep"], errors="coerce")
    grouped = {int(step): group for step, group in frame.dropna(subset=["timestep"]).groupby("timestep")}

    graphs = []
    counts = []
    for timestep, base_graph in zip(timesteps, base_graphs):
        current = grouped.get(int(timestep), frame.iloc[0:0])
        pairs = _similar_price_pairs(
            current["node_id"],
            current[price_column],
            tolerance=float(tolerance),
            max_neighbors=int(max_neighbors),
        )
        src = torch.tensor([size_index[a] for a, _ in pairs], dtype=torch.int64)
        dst = torch.tensor([size_index[b] for _, b in pairs], dtype=torch.int64)
        edge_index = dict(base_graph.edge_index)
        # Similarity is symmetric. Store both message directions under one
        # canonical edge type so arbitrary node ordering does not create two
        # independently learned relation groups.
        edge_index[("size", "similar_price", "size")] = (
            torch.cat([src, dst]),
            torch.cat([dst, src]),
        )
        graphs.append(HeteroGraphSpec(edge_index=edge_index, num_nodes=dict(base_graph.num_nodes)))
        counts.append(len(pairs))

    count_array = np.asarray(counts, dtype=np.int64)
    metadata = {
        "enabled": True,
        "price_column": str(price_column),
        "tolerance": float(tolerance),
        "max_neighbors": int(max_neighbors),
        "timesteps": int(len(timesteps)),
        "undirected_edges_total": int(count_array.sum()),
        "directed_message_edges_total": int(2 * count_array.sum()),
        "undirected_edges_min": int(count_array.min()) if count_array.size else 0,
        "undirected_edges_max": int(count_array.max()) if count_array.size else 0,
        "undirected_edges_mean": float(count_array.mean()) if count_array.size else 0.0,
        "undirected_edges_by_timestep": {
            str(int(timestep)): int(count)
            for timestep, count in zip(timesteps, counts)
        },
    }
    return graphs, metadata


def _project_mrt_distance_data(project_frame, node_map, mrt_frame, opening_date_column):
    project_required = {"project_id", "latitude", "longitude"}
    mrt_required = {"latitude", "longitude", opening_date_column}
    if not project_required.issubset(project_frame.columns):
        raise KeyError(
            "Dynamic MRT edges require project columns: "
            f"{sorted(project_required - set(project_frame.columns))}"
        )
    if not mrt_required.issubset(mrt_frame.columns):
        raise KeyError(
            "Dynamic MRT edges require rail columns: "
            f"{sorted(mrt_required - set(mrt_frame.columns))}"
        )

    project_nodes = (
        node_map.loc[node_map["size_tier"].eq(0), ["project_id", "node_id"]]
        .drop_duplicates("project_id")
    )
    projects = project_frame[["project_id", "latitude", "longitude"]].drop_duplicates("project_id").copy()
    projects = projects.merge(project_nodes, on="project_id", how="inner")
    projects["latitude"] = pd.to_numeric(projects["latitude"], errors="coerce")
    projects["longitude"] = pd.to_numeric(projects["longitude"], errors="coerce")
    projects = projects.dropna(subset=["latitude", "longitude"]).sort_values("node_id").reset_index(drop=True)

    stations = mrt_frame.copy()
    stations["latitude"] = pd.to_numeric(stations["latitude"], errors="coerce")
    stations["longitude"] = pd.to_numeric(stations["longitude"], errors="coerce")
    stations[opening_date_column] = pd.to_datetime(stations[opening_date_column], errors="coerce")
    stations = stations.dropna(subset=["latitude", "longitude", opening_date_column]).reset_index(drop=True)
    if stations.empty:
        raise ValueError("No MRT stations have valid coordinates and opening dates.")

    distances = pairwise_haversine_m(
        projects["latitude"].to_numpy(dtype=float),
        projects["longitude"].to_numpy(dtype=float),
        stations["latitude"].to_numpy(dtype=float),
        stations["longitude"].to_numpy(dtype=float),
    )
    return projects, stations, distances


def build_dynamic_mrt_graphs(
    edge_frames,
    project_node_ids,
    size_node_ids,
    project_frame,
    node_map,
    mrt_frame,
    timestep_dates,
    edge_config,
    opening_date_column="opening_date",
):
    """Build one graph per feature timestep using only operational MRT stations.

    Graph objects are shared between consecutive timesteps in the same station-opening
    regime. Non-MRT relations remain static. A station is available only when its
    exact opening date is on or before the feature timestamp.
    """
    dynamic_names = {
        name
        for name in ("same_mrt", "same_mrt_dist", "adjacent_mrt")
        if name in edge_frames
    }
    if not dynamic_names:
        graph = build_heterograph_from_edge_frames(edge_frames, project_node_ids, size_node_ids)
        return [graph] * len(timestep_dates), {
            "enabled": True,
            "dynamic_relations": [],
            "num_regimes": 1,
            "opening_date_column": opening_date_column,
        }

    projects, stations, distance_m = _project_mrt_distance_data(
        project_frame, node_map, mrt_frame, opening_date_column
    )
    project_ids = projects["project_id"].to_numpy()
    node_ids = projects["node_id"].to_numpy(dtype=int)
    opening_dates = stations[opening_date_column].to_numpy(dtype="datetime64[ns]")
    dates = pd.to_datetime(pd.Series(timestep_dates), errors="raise")

    same_mrt_threshold = float(edge_config.get("same_mrt", {}).get("threshold", 250))
    same_mrt_dist_epsilon = float(edge_config.get("same_mrt_dist", {}).get("epsilon", 5))
    adjacent_mrt_config = edge_config.get("adjacent_mrt", {})
    adjacent_transport_type = adjacent_mrt_config.get("transport_type", "MRT")
    if "adjacent_mrt" in dynamic_names:
        required_line_cols = {
            "alphanumeric_code",
            "station_name_english",
            "line_english",
            "transport_type",
        }
        missing_line_cols = sorted(required_line_cols - set(stations.columns))
        if missing_line_cols:
            raise KeyError(
                "Dynamic adjacent_mrt edges require rail columns: "
                f"{missing_line_cols}"
            )
    static_frames = {name: frame for name, frame in edge_frames.items() if name not in dynamic_names}
    graphs_by_open_count = {}
    graph_sequence = []
    regime_edge_counts = []

    for feature_date in dates:
        open_mask = opening_dates <= np.datetime64(feature_date.to_datetime64())
        regime_key = tuple(np.flatnonzero(open_mask).tolist())
        if regime_key not in graphs_by_open_count:
            frames = dict(static_frames)
            opened = np.flatnonzero(open_mask)

            if "same_mrt" in dynamic_names:
                pair_set = set()
                for station_idx in opened:
                    nearby = np.flatnonzero(distance_m[:, station_idx] < same_mrt_threshold)
                    if nearby.size < 2:
                        continue
                    left, right = np.triu_indices(nearby.size, k=1)
                    pair_set.update(zip(nearby[left].tolist(), nearby[right].tolist()))
                ordered_pairs = sorted(pair_set)
                if ordered_pairs:
                    src_idx, dst_idx = map(np.asarray, zip(*ordered_pairs))
                    frames["same_mrt"] = pd.DataFrame({
                        "node_id": node_ids[src_idx],
                        "project_id": project_ids[src_idx],
                        "neig_node_id": node_ids[dst_idx],
                        "neig_project_id": project_ids[dst_idx],
                    })
                else:
                    frames["same_mrt"] = pd.DataFrame(columns=["node_id", "neig_node_id"])

            if "same_mrt_dist" in dynamic_names:
                if opened.size:
                    nearest_distance = distance_m[:, opened].min(axis=1)
                    distance_diff = np.abs(nearest_distance[:, None] - nearest_distance[None, :])
                    src_idx, dst_idx = np.where(
                        (distance_diff < same_mrt_dist_epsilon)
                        & np.triu(np.ones(distance_diff.shape, dtype=bool), k=1)
                    )
                    frames["same_mrt_dist"] = pd.DataFrame({
                        "node_id": node_ids[src_idx],
                        "project_id": project_ids[src_idx],
                        "neig_node_id": node_ids[dst_idx],
                        "neig_project_id": project_ids[dst_idx],
                        "nearest_mrt_distance_m": nearest_distance[src_idx],
                        "neig_nearest_mrt_distance_m": nearest_distance[dst_idx],
                        "nearest_mrt_distance_diff_m": distance_diff[src_idx, dst_idx],
                    })
                else:
                    frames["same_mrt_dist"] = pd.DataFrame(columns=["node_id", "neig_node_id"])

            if "adjacent_mrt" in dynamic_names:
                if opened.size:
                    opened_stations = stations.iloc[opened]
                    if adjacent_transport_type:
                        opened_stations = opened_stations.loc[
                            opened_stations["transport_type"].eq(adjacent_transport_type)
                        ]
                else:
                    opened_stations = stations.iloc[0:0]

                if not opened_stations.empty:
                    physical_open_stations = (
                        opened_stations.groupby("station_name_english", as_index=False)
                        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
                        .sort_values("station_name_english")
                        .reset_index(drop=True)
                    )
                    physical_distance_m = pairwise_haversine_m(
                        projects["latitude"].to_numpy(dtype=float),
                        projects["longitude"].to_numpy(dtype=float),
                        physical_open_stations["latitude"].to_numpy(dtype=float),
                        physical_open_stations["longitude"].to_numpy(dtype=float),
                    )
                    nearest_station_position = physical_distance_m.argmin(axis=1)
                    nearest_station_names = physical_open_stations.loc[
                        nearest_station_position, "station_name_english"
                    ].to_numpy()
                    pairs = adjacent_mrt_pairs(
                        nearest_station_names,
                        opened_stations["station_name_english"],
                        opened_stations["alphanumeric_code"],
                        opened_stations["line_english"],
                    )
                    if pairs:
                        src_idx = np.asarray([pair[0] for pair in pairs], dtype=int)
                        dst_idx = np.asarray([pair[1] for pair in pairs], dtype=int)
                        frames["adjacent_mrt"] = pd.DataFrame({
                            "node_id": node_ids[src_idx],
                            "project_id": project_ids[src_idx],
                            "neig_node_id": node_ids[dst_idx],
                            "neig_project_id": project_ids[dst_idx],
                            "nearest_mrt_name": nearest_station_names[src_idx],
                            "neig_nearest_mrt_name": nearest_station_names[dst_idx],
                            "adjacent_mrt_codes": [pair[2] for pair in pairs],
                        })
                    else:
                        frames["adjacent_mrt"] = pd.DataFrame(
                            columns=["node_id", "neig_node_id"]
                        )
                else:
                    frames["adjacent_mrt"] = pd.DataFrame(
                        columns=["node_id", "neig_node_id"]
                    )

            graph = build_heterograph_from_edge_frames(frames, project_node_ids, size_node_ids)
            counts = {
                name: int(len(frames[name]))
                for name in sorted(dynamic_names)
            }
            graphs_by_open_count[regime_key] = graph
            regime_edge_counts.append({
                "feature_date": feature_date.date().isoformat(),
                "open_stations": int(opened.size),
                **counts,
            })
        graph_sequence.append(graphs_by_open_count[regime_key])

    metadata = {
        "enabled": True,
        "dynamic_relations": sorted(dynamic_names),
        "opening_date_column": opening_date_column,
        "opening_policy": "opening_date <= feature_timestamp",
        "num_regimes": len(graphs_by_open_count),
        "first_regime": regime_edge_counts[0],
        "last_regime": regime_edge_counts[-1],
        "regimes": regime_edge_counts,
    }
    return graph_sequence, metadata


def infer_project_built_years(
    project_frame,
    timestep_dates,
    age_column="condo_age_at_transaction",
):
    """Infer project completion years from a time-varying age column.

    A project is accepted only when its positive ages vary over the panel and
    every ``calendar_year - age`` observation gives the same year. Constant,
    null, zero-only, or inconsistent age histories deliberately remain
    unresolved; callers treat those projects as existing for the full panel.
    """
    required = {"project_id", "timestep", age_column}
    missing = sorted(required - set(project_frame.columns))
    if missing:
        raise KeyError(f"Dynamic condo edges require project columns: {missing}")

    date_lookup = pd.DataFrame(
        {
            "timestep": list(timestep_dates.keys()),
            "feature_date": list(timestep_dates.values()),
        }
    )
    date_lookup["timestep"] = pd.to_numeric(date_lookup["timestep"], errors="raise").astype(int)
    date_lookup["feature_date"] = pd.to_datetime(date_lookup["feature_date"], errors="raise")

    ages = project_frame[["project_id", "timestep", age_column]].copy()
    ages["timestep"] = pd.to_numeric(ages["timestep"], errors="coerce")
    ages[age_column] = pd.to_numeric(ages[age_column], errors="coerce")
    ages = ages.merge(date_lookup, on="timestep", how="left", validate="many_to_one")
    ages["calendar_year"] = ages["feature_date"].dt.year

    age_counts = ages.groupby("project_id", dropna=False)[age_column].nunique(dropna=True)
    positive = ages.loc[
        ages[age_column].gt(0) & ages["calendar_year"].notna(),
        ["project_id", age_column, "calendar_year"],
    ].copy()
    positive["candidate_built_year"] = positive["calendar_year"] - positive[age_column]
    integral = np.isclose(
        positive["candidate_built_year"],
        np.rint(positive["candidate_built_year"]),
    )
    positive = positive.loc[integral].copy()
    positive["candidate_built_year"] = np.rint(positive["candidate_built_year"]).astype(int)

    candidate_counts = positive.groupby("project_id")["candidate_built_year"].nunique()
    candidates = positive.groupby("project_id")["candidate_built_year"].first()
    computable_ids = age_counts[age_counts.gt(1)].index.intersection(
        candidate_counts[candidate_counts.eq(1)].index
    )
    return candidates.loc[computable_ids].sort_index().astype(int)


def build_dynamic_condo_graphs(
    base_graphs,
    project_frame,
    node_map,
    project_node_ids,
    size_node_ids,
    timesteps,
    timestep_dates,
    age_column="condo_age_at_transaction",
):
    """Remove graph edges before projects with recoverable built years exist.

    Projects whose built year cannot be recovered are intentionally always
    active, preserving their original static edges for the complete panel.
    """
    if len(base_graphs) != len(timesteps) or len(timesteps) != len(timestep_dates):
        raise ValueError(
            "Dynamic condo edges require one base graph and date per timestep: "
            f"graphs={len(base_graphs)} timesteps={len(timesteps)} dates={len(timestep_dates)}"
        )

    timestep_date_map = {
        int(timestep): pd.Timestamp(date)
        for timestep, date in zip(timesteps, timestep_dates)
    }
    built_years = infer_project_built_years(
        project_frame,
        timestep_date_map,
        age_column=age_column,
    )

    node_projects = node_map[["node_id", "project_id"]].drop_duplicates("node_id").copy()
    node_projects["node_id"] = pd.to_numeric(node_projects["node_id"], errors="raise").astype(int)
    project_by_node = node_projects.set_index("node_id")["project_id"]

    def start_years(node_ids):
        project_ids = project_by_node.reindex(node_ids)
        if project_ids.isna().any():
            missing_ids = list(np.asarray(node_ids)[project_ids.isna().to_numpy()][:10])
            raise ValueError(f"Missing project mapping for node ids: {missing_ids}")
        return pd.to_numeric(project_ids.map(built_years), errors="coerce").to_numpy(dtype=float)

    project_start_years = start_years(project_node_ids)
    size_start_years = start_years(size_node_ids)
    start_year_by_type = {
        "project": project_start_years,
        "size": size_start_years,
    }

    graph_cache = {}
    graphs = []
    regimes = []
    for timestep, feature_date, base_graph in zip(timesteps, timestep_dates, base_graphs):
        year = int(pd.Timestamp(feature_date).year)
        cache_key = (id(base_graph), year)
        if cache_key not in graph_cache:
            active_by_type = {
                node_type: torch.from_numpy(np.isnan(years) | (years <= year))
                for node_type, years in start_year_by_type.items()
            }
            edge_index = {}
            for etype, (src, dst) in base_graph.edge_index.items():
                src_type, _, dst_type = etype
                keep = active_by_type[src_type][src] & active_by_type[dst_type][dst]
                edge_index[etype] = (src[keep], dst[keep])
            graph_cache[cache_key] = HeteroGraphSpec(
                edge_index=edge_index,
                num_nodes=dict(base_graph.num_nodes),
            )
            regimes.append(
                {
                    "timestep": int(timestep),
                    "feature_date": pd.Timestamp(feature_date).date().isoformat(),
                    "active_computable_projects": int((project_start_years <= year).sum()),
                    "directed_message_edges": int(
                        sum(src.numel() for src, _ in edge_index.values())
                    ),
                }
            )
        graphs.append(graph_cache[cache_key])

    total_projects = int(node_map["project_id"].nunique())
    metadata = {
        "enabled": True,
        "age_column": age_column,
        "activation_policy": "built_year <= feature_calendar_year",
        "unknown_policy": "static_from_first_to_last_timestep",
        "computable_projects": int(len(built_years)),
        "static_fallback_projects": int(total_projects - len(built_years)),
        "built_year_min": int(built_years.min()) if len(built_years) else None,
        "built_year_max": int(built_years.max()) if len(built_years) else None,
        "num_regimes": int(len(graph_cache)),
        "regimes": regimes,
    }
    return graphs, metadata


def mean_aggregate(src_h, src_idx, dst_idx, num_dst):
    out = torch.zeros((num_dst, src_h.shape[-1]), dtype=src_h.dtype, device=src_h.device)
    counts = torch.zeros((num_dst, 1), dtype=src_h.dtype, device=src_h.device)
    if src_idx.numel() == 0:
        return out
    out.index_add_(0, dst_idx, src_h[src_idx])
    counts.index_add_(0, dst_idx, torch.ones((dst_idx.numel(), 1), dtype=src_h.dtype, device=src_h.device))
    return out / counts.clamp(min=1.0)


def mean_aggregate_time(src_h, src_idx, dst_idx, num_dst):
    """Mean aggregate all timesteps at once.

    src_h shape: (T, num_src, D)
    returns: (T, num_dst, D)
    """
    out = torch.zeros(
        (src_h.shape[0], num_dst, src_h.shape[-1]),
        dtype=src_h.dtype,
        device=src_h.device,
    )
    if src_idx.numel() == 0:
        return out
    out.index_add_(1, dst_idx, src_h.index_select(1, src_idx))
    counts = torch.zeros((num_dst, 1), dtype=src_h.dtype, device=src_h.device)
    counts.index_add_(
        0,
        dst_idx,
        torch.ones((dst_idx.numel(), 1), dtype=src_h.dtype, device=src_h.device),
    )
    return out / counts.clamp(min=1.0).view(1, num_dst, 1)


def rphgnn_precompute_contexts(
    graph,
    project_features,
    size_features,
    num_hops,
    device=None,
    timestep_graphs=None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    target_type = "size"
    incoming_target_etypes = [etype for etype in graph.canonical_etypes if etype[-1] == target_type]
    group_names = ["self", *[etype[1] for etype in incoming_target_etypes]]
    group_index = {name: idx for idx, name in enumerate(group_names)}

    num_timesteps, num_size_nodes, rp_dim = size_features.shape
    # Store as (time, group, hop, node, dim) so every group/hop block write is
    # contiguous over all nodes and feature dimensions.
    contexts_group_major = np.zeros(
        (num_timesteps, len(group_names), num_hops + 1, num_size_nodes, rp_dim),
        dtype=np.float32,
    )
    graph_sequence = timestep_graphs or [graph] * num_timesteps
    if len(graph_sequence) != num_timesteps:
        raise ValueError(
            f"Expected one graph per timestep, got {len(graph_sequence)} graphs for {num_timesteps} timesteps."
        )
    device_edge_cache = {}

    def device_edges(current_graph):
        cache_key = id(current_graph)
        if cache_key not in device_edge_cache:
            device_edge_cache[cache_key] = {
                etype: (src_idx.to(device), dst_idx.to(device))
                for etype, (src_idx, dst_idx) in current_graph.edge_index.items()
            }
        return device_edge_cache[cache_key]
    project_features_t = torch.from_numpy(project_features).float().to(device)
    size_features_t = torch.from_numpy(size_features).float().to(device)

    # The all-timestep vectorized aggregation looks attractive for small edge
    # sets, but on the shared training host it becomes CPU-bound during large
    # advanced-index materialization.  The timestep loop is faster and more
    # memory-stable once context writes use the group-major layout above.
    use_time_vectorized = False
    if use_time_vectorized:
        edge_index = device_edges(graph)
        h_dict = {
            "project": project_features_t,
            "size": size_features_t,
        }
        contexts_group_major[:, group_index["self"], 0, :, :] = (
            size_features_t.detach().cpu().numpy()
        )
        for hop in tqdm(range(1, num_hops + 1), desc="RpHGNN precompute"):
            next_parts = defaultdict(list)
            for etype in graph.canonical_etypes:
                src_type, _, dst_type = etype
                src_idx, dst_idx = edge_index[etype]
                msg = mean_aggregate_time(
                    h_dict[src_type],
                    src_idx,
                    dst_idx,
                    graph.num_nodes_of(dst_type),
                )
                next_parts[dst_type].append(msg)
                if dst_type == target_type:
                    contexts_group_major[:, group_index[etype[1]], hop, :, :] = (
                        msg.detach().cpu().numpy()
                    )

            next_h = {}
            for ntype in graph.ntypes:
                if next_parts[ntype]:
                    mixed = torch.stack(next_parts[ntype], dim=0).mean(dim=0)
                    next_h[ntype] = torch.nn.functional.normalize(
                        0.5 * h_dict[ntype] + 0.5 * mixed,
                        dim=-1,
                    )
                else:
                    next_h[ntype] = h_dict[ntype]
            h_dict = next_h
            contexts_group_major[:, group_index["self"], hop, :, :] = (
                h_dict[target_type].detach().cpu().numpy()
            )
        return contexts_group_major, group_names, "time_group_major"

    for t in tqdm(range(num_timesteps), desc="RpHGNN precompute"):
        current_graph = graph_sequence[t]
        edge_index = device_edges(current_graph)
        h_dict = {
            "project": project_features_t[t],
            "size": size_features_t[t],
        }
        contexts_group_major[t, group_index["self"], 0, :, :] = (
            size_features_t[t].detach().cpu().numpy()
        )

        for hop in range(1, num_hops + 1):
            next_parts = defaultdict(list)
            for etype in current_graph.canonical_etypes:
                src_type, _, dst_type = etype
                src_idx, dst_idx = edge_index[etype]
                msg = mean_aggregate(
                    h_dict[src_type],
                    src_idx,
                    dst_idx,
                    current_graph.num_nodes_of(dst_type),
                )
                next_parts[dst_type].append(msg)
                if dst_type == target_type:
                    contexts_group_major[t, group_index[etype[1]], hop, :, :] = (
                        msg.detach().cpu().numpy()
                    )

            next_h = {}
            for ntype in graph.ntypes:
                if next_parts[ntype]:
                    mixed = torch.stack(next_parts[ntype], dim=0).mean(dim=0)
                    next_h[ntype] = torch.nn.functional.normalize(0.5 * h_dict[ntype] + 0.5 * mixed, dim=-1)
                else:
                    next_h[ntype] = h_dict[ntype]
            h_dict = next_h
            contexts_group_major[t, group_index["self"], hop, :, :] = (
                h_dict[target_type].detach().cpu().numpy()
            )

    return contexts_group_major, group_names, "time_group_major"


def _semantic_edge_index(graph, device):
    """Return direction-aware operators with reverse-name artifacts removed.

    Project-project CSV relations are stored once and mirrored as ``rev_*`` by
    the lightweight graph builder.  Treating those two storage directions as
    different semantics would make metapath interpretation depend on arbitrary
    node ordering, so they are merged into one bidirectional operator.  Cross-
    type directions remain distinct through their source/destination types.
    """
    parts = defaultdict(list)
    for (src_type, relation, dst_type), (src, dst) in graph.edge_index.items():
        semantic_relation = relation[4:] if relation.startswith("rev_") else relation
        parts[(src_type, semantic_relation, dst_type)].append(
            (src.to(device), dst.to(device))
        )

    operators = {}
    for etype, edge_parts in parts.items():
        src = torch.cat([part[0] for part in edge_parts])
        dst = torch.cat([part[1] for part in edge_parts])
        if src.numel():
            pairs = torch.stack([src, dst], dim=1).unique(dim=0)
            src, dst = pairs[:, 0], pairs[:, 1]
        operators[etype] = (src, dst)
    return operators


def enumerate_size_metapaths(graph, max_hops=2):
    """Enumerate exact typed relation sequences ending at a size node."""
    if max_hops not in {0, 1, 2}:
        raise ValueError(f"Exact metapath mode supports 0, 1, or 2 hops, got {max_hops}.")

    semantic_etypes = sorted(
        {
            (src, relation[4:] if relation.startswith("rev_") else relation, dst)
            for src, relation, dst in graph.canonical_etypes
        }
    )
    incoming = defaultdict(list)
    for etype in semantic_etypes:
        incoming[etype[2]].append(etype)

    paths = [("self", ())]
    if max_hops >= 1:
        for first in incoming["size"]:
            name = f"size<-[{first[1]}]-{first[0]}"
            paths.append((name, (first,)))
    if max_hops >= 2:
        for first in incoming["size"]:
            for second in incoming[first[0]]:
                name = (
                    f"size<-[{first[1]}]-{first[0]}"
                    f"<-[{second[1]}]-{second[0]}"
                )
                paths.append((name, (first, second)))
    return paths


def rphgnn_precompute_metapaths(
    graph,
    project_features,
    size_features,
    num_hops=2,
    device=None,
    timestep_graphs=None,
    include=None,
):
    """Precompute exact length-0/1/2 metapaths without relation mixing.

    Output layout is ``(time, metapath, 1, size_node, feature)``.  The singleton
    axis preserves the model's existing group/hop input contract while making
    each group an independently ablatable semantic path.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    all_paths = enumerate_size_metapaths(graph, max_hops=num_hops)
    if include:
        requested = set(include)
        unknown = sorted(requested - {name for name, _ in all_paths})
        if unknown:
            raise ValueError(f"Unknown metapaths requested: {unknown}")
        paths = [path for path in all_paths if path[0] in requested]
    else:
        paths = all_paths
    if not paths:
        raise ValueError("Metapath selection produced no context groups.")

    graph_sequence = timestep_graphs or [graph] * len(size_features)
    if len(graph_sequence) != len(size_features):
        raise ValueError(
            f"Expected one graph per timestep, got {len(graph_sequence)} graphs "
            f"for {len(size_features)} timesteps."
        )
    features = {
        "project": torch.from_numpy(project_features).float().to(device),
        "size": torch.from_numpy(size_features).float().to(device),
    }
    contexts = np.zeros(
        (len(size_features), len(paths), 1, size_features.shape[1], size_features.shape[2]),
        dtype=np.float32,
    )
    operator_cache = {}
    for timestep, current_graph in enumerate(
        tqdm(graph_sequence, desc="Exact metapath precompute")
    ):
        cache_key = id(current_graph)
        if cache_key not in operator_cache:
            operator_cache[cache_key] = _semantic_edge_index(current_graph, device)
        operators = operator_cache[cache_key]
        current_features = {key: value[timestep] for key, value in features.items()}

        for path_index, (_, steps) in enumerate(paths):
            if not steps:
                value = current_features["size"]
            elif len(steps) == 1:
                first = steps[0]
                src, dst = operators[first]
                value = mean_aggregate(
                    current_features[first[0]], src, dst, current_graph.num_nodes_of(first[2])
                )
            else:
                first, second = steps
                second_src, second_dst = operators[second]
                intermediate = mean_aggregate(
                    current_features[second[0]],
                    second_src,
                    second_dst,
                    current_graph.num_nodes_of(second[2]),
                )
                first_src, first_dst = operators[first]
                value = mean_aggregate(
                    intermediate,
                    first_src,
                    first_dst,
                    current_graph.num_nodes_of(first[2]),
                )
            contexts[timestep, path_index, 0] = value.detach().cpu().numpy()

    return contexts, [name for name, _ in paths], "time_group_major"


class RpHGNNWindowDataset(Dataset):
    def __init__(
        self,
        contexts,
        targets,
        masks,
        node_indices,
        target_start_times,
        window_size,
        target_shift=0,
        min_target_time=None,
        min_feature_time=None,
        context_layout="node_major",
    ):
        self.contexts = contexts.astype(np.float32, copy=False)
        self.targets = targets.astype(np.float32, copy=False)
        self.masks = masks.astype(np.float32, copy=False)
        self.node_indices = np.asarray(node_indices, dtype=np.int64)
        self.target_start_times = np.asarray(target_start_times, dtype=np.int64)
        self.window_size = window_size
        self.target_shift = target_shift
        self.min_target_time = min_target_time
        self.min_feature_time = min_feature_time
        self.context_layout = context_layout
        if context_layout == "time_group_major":
            self.model_context_shape = (
                self.contexts.shape[3],
                self.contexts.shape[0],
                self.contexts.shape[1],
                self.contexts.shape[2],
                self.contexts.shape[4],
            )
            self.context_window_shape = self.contexts.shape[1:3] + self.contexts.shape[4:]
        elif context_layout == "time_major":
            self.model_context_shape = (
                self.contexts.shape[1],
                self.contexts.shape[0],
                self.contexts.shape[2],
                self.contexts.shape[3],
                self.contexts.shape[4],
            )
            self.context_window_shape = self.contexts.shape[2:]
        else:
            self.model_context_shape = self.contexts.shape
            self.context_window_shape = self.contexts.shape[2:]

    def __len__(self):
        return len(self.node_indices)

    def __getitem__(self, idx):
        node_idx = int(self.node_indices[idx])
        target_start = int(self.target_start_times[idx])
        target_end = target_start + self.window_size
        input_start = target_start - self.target_shift
        input_end = input_start + self.window_size

        valid_input_start = max(input_start, 0)
        context_timesteps = self.contexts.shape[0] if self.context_layout in {"time_major", "time_group_major"} else self.contexts.shape[1]
        valid_input_end = min(input_end, context_timesteps)
        if valid_input_end < valid_input_start:
            valid_input_end = valid_input_start

        if self.context_layout == "time_major":
            input_window = self.contexts[valid_input_start:valid_input_end, node_idx]
        elif self.context_layout == "time_group_major":
            input_window = np.transpose(
                self.contexts[valid_input_start:valid_input_end, :, :, node_idx, :],
                (0, 1, 2, 3),
            )
        else:
            input_window = self.contexts[node_idx, valid_input_start:valid_input_end]
        left_pad = min(self.window_size, max(-input_start, 0))
        right_pad = self.window_size - left_pad - input_window.shape[0]
        if left_pad > 0:
            input_window = np.concatenate(
                [
                    np.zeros((left_pad, *self.context_window_shape), dtype=np.float32),
                    input_window,
                ],
                axis=0,
            )
        if right_pad > 0:
            input_window = np.concatenate(
                [
                    input_window,
                    np.zeros((right_pad, *self.context_window_shape), dtype=np.float32),
                ],
                axis=0,
            )

        target_mask_window = self.masks[node_idx, target_start:target_end].copy()
        if self.min_target_time is not None:
            target_mask_window[: max(int(self.min_target_time) - target_start, 0)] = 0.0
        if self.min_feature_time is not None:
            feature_start = target_start - self.target_shift
            offsets = np.arange(self.window_size, dtype=np.int64)
            target_mask_window[feature_start + offsets < int(self.min_feature_time)] = 0.0
        return (
            torch.from_numpy(input_window),
            torch.from_numpy(self.targets[node_idx, target_start:target_end, None]),
            torch.from_numpy(target_mask_window[:, None]),
            torch.tensor(node_idx, dtype=torch.long),
            torch.tensor(target_start, dtype=torch.long),
        )


def build_window_indices(masks, window_size, train_steps, target_shift=0):
    num_nodes, num_timesteps = masks.shape
    max_target_start = num_timesteps - window_size + 1
    if max_target_start <= 0:
        raise ValueError(
            f"window_size must be <= number of timesteps, got "
            f"window_size={window_size}, T={num_timesteps}."
        )

    masks = np.asarray(masks, dtype=np.float32)
    prefix = np.concatenate(
        [np.zeros((num_nodes, 1), dtype=np.float32), np.cumsum(masks, axis=1)],
        axis=1,
    )
    starts = np.arange(max_target_start, dtype=np.int64)
    ends = starts + window_size
    window_sums = prefix[:, ends] - prefix[:, starts]

    train_start_valid = (ends <= train_steps) & (starts >= target_shift)
    train_valid = window_sums[:, train_start_valid] > 0
    train_nodes, train_start_pos = np.nonzero(train_valid)
    train_target_starts = starts[train_start_valid][train_start_pos]

    test_start_valid = ends > train_steps
    test_starts = starts[test_start_valid]
    test_ends = ends[test_start_valid]
    eval_starts = np.maximum(test_starts, train_steps)
    eval_sums = prefix[:, test_ends] - prefix[:, eval_starts]
    test_valid = eval_sums > 0
    test_nodes, test_start_pos = np.nonzero(test_valid)
    test_target_starts = test_starts[test_start_pos]

    return (
        (train_nodes, train_target_starts),
        (test_nodes, test_target_starts),
    )


def build_window_indices_feature_split(masks, window_size, feature_split_step, target_shift=0):
    num_nodes, num_timesteps = masks.shape
    max_feature_start = num_timesteps - target_shift - window_size + 1
    if max_feature_start <= 0:
        raise ValueError(
            "No valid feature windows: "
            f"window_size={window_size}, target_shift={target_shift}, T={num_timesteps}."
        )
    if feature_split_step <= 0 or feature_split_step >= num_timesteps:
        raise ValueError(
            f"feature_split_step must be in [1, T-1], got "
            f"feature_split_step={feature_split_step}, T={num_timesteps}."
        )

    masks = np.asarray(masks, dtype=np.float32)
    feature_starts = np.arange(max_feature_start, dtype=np.int64)
    target_starts = feature_starts + target_shift
    target_ends = target_starts + window_size
    offsets = np.arange(window_size, dtype=np.int64)
    feature_times = feature_starts[:, None] + offsets[None, :]
    target_times = target_starts[:, None] + offsets[None, :]

    # Per-start target mask sums, separated by feature-time membership.
    gathered = masks[:, target_times.reshape(-1)].reshape(num_nodes, len(feature_starts), window_size)
    train_offsets = feature_times < feature_split_step
    test_offsets = feature_times >= feature_split_step
    train_sums = (gathered * train_offsets[None, :, :]).sum(axis=2)
    test_sums = (gathered * test_offsets[None, :, :]).sum(axis=2)

    train_valid = train_sums > 0
    test_valid = test_sums > 0
    train_nodes, train_start_pos = np.nonzero(train_valid)
    test_nodes, test_start_pos = np.nonzero(test_valid)

    return (
        (train_nodes, target_starts[train_start_pos]),
        (test_nodes, target_starts[test_start_pos]),
    )


def resolve_feature_split_step(loader_config, timesteps, node_processor):
    if loader_config.get("feature_split_timestep") is not None:
        split_step = int(loader_config["feature_split_timestep"])
        if split_step not in set(map(int, timesteps)):
            raise ValueError(f"feature_split_timestep={split_step} is not present in timesteps.")
        return split_step

    split_date = loader_config.get("feature_split_date")
    if split_date in (None, "", "null", "None"):
        return None

    timestep_info_path = node_processor.out_paths.get("timestep_info_path")
    if timestep_info_path is None or not Path(timestep_info_path).exists():
        raise FileNotFoundError(
            "feature_split_date requires timestep_info.csv, missing "
            f"{timestep_info_path}."
        )
    timestep_info = pd.read_csv(timestep_info_path)
    if not {"timestep", "date"}.issubset(timestep_info.columns):
        raise ValueError(f"Expected timestep/date columns in {timestep_info_path}.")
    dates = pd.to_datetime(timestep_info["date"])
    target_date = pd.Timestamp(split_date)
    rows = timestep_info.loc[dates.eq(target_date), "timestep"]
    if rows.empty:
        raise ValueError(
            f"feature_split_date={split_date!r} not found in {timestep_info_path}. "
            f"Available range: {dates.min().date()} to {dates.max().date()}."
        )
    return int(rows.iloc[0])


def split_date_ranges(timesteps, split_step, target_shift):
    timesteps = list(map(int, timesteps))
    latest = timesteps[-1]
    last_feature = latest - target_shift
    return {
        "feature_split_timestep": int(split_step),
        "first_test_feature_timestep": int(split_step) if split_step <= last_feature else None,
        "last_test_feature_timestep": int(last_feature) if split_step <= last_feature else None,
        "first_test_target_timestep": int(split_step + target_shift) if split_step <= last_feature else None,
        "last_test_target_timestep": int(latest) if split_step <= last_feature else None,
        "test_feature_timesteps": int(max(0, last_feature - split_step + 1)),
    }


def add_split_metadata(dataset, metadata):
    dataset.split_metadata = dict(metadata)
    for key, value in metadata.items():
        setattr(dataset, key, value)


def get_dataloaders(config):
    started_at = time.time()

    def mark(message):
        logger.info("get_dataloaders %.1fs %s", time.time() - started_at, message)

    loader_config = normalize_loader_config(config)
    ts_test = loader_config["ts_test"]
    num_hops = loader_config["num_hops"]
    batch_size = loader_config["batch_size"]
    target_col = loader_config["target_col"]
    mask_col = loader_config["mask_col"]
    feat_norm = loader_config["feat_norm"]
    window_size = loader_config["window_size"]
    target_shift = loader_config["target_shift"]
    target_mask_mode = loader_config["target_mask_mode"]
    rp_dim = loader_config["rp_dim"]
    random_seed = loader_config["random_seed"]
    precompute_device = loader_config["precompute_device"]
    direct_size_feature_cols = loader_config["direct_size_feature_columns"]
    direct_feature_scale = loader_config["direct_feature_scale"]
    split_on = str(loader_config["split_on"])
    split_boundary_policy = str(loader_config["split_boundary_policy"])
    dynamic_mrt_edges = bool(loader_config["dynamic_mrt_edges"])
    mrt_opening_date_column = str(loader_config["mrt_opening_date_column"])
    dynamic_condo_edges = bool(loader_config["dynamic_condo_edges"])
    condo_age_column = str(loader_config["condo_age_column"])
    similar_price_edges = bool(loader_config["similar_price_edges"])
    similar_price_column = str(loader_config["similar_price_column"])
    similar_price_tolerance = float(loader_config["similar_price_tolerance"])
    similar_price_k = int(loader_config["similar_price_k"])
    context_mode = str(loader_config["context_mode"])
    metapath_include = loader_config["metapath_include"]
    if precompute_device == "auto":
        precompute_device = "cuda" if torch.cuda.is_available() else "cpu"

    if target_shift < 0:
        raise ValueError(f"target_shift must be >= 0, got {target_shift}.")
    if split_on not in {"target", "feature"}:
        raise ValueError(f"split_on must be 'target' or 'feature', got {split_on!r}.")
    if split_boundary_policy != "offset_mask":
        raise ValueError(
            "Only split_boundary_policy='offset_mask' is currently implemented, "
            f"got {split_boundary_policy!r}."
        )

    node_processor = FormNodeProcessor(config.graph)
    node_processor.process()
    mark("node processor ready")
    edge_processor = FormEdgeProcessor(config.graph)
    edge_processor.process()
    mark("edge processor ready")
    
    node_map, project_frame, size_frame = node_processor.load_node_tables()
    mark(f"loaded node tables node={node_map.shape} project={project_frame.shape} size={size_frame.shape}")
    edge_dfs = edge_processor.load_enabled_edge_dfs()
    mark("loaded edges")

    project_node_ids = node_map.loc[node_map["size_tier"].eq(0), "node_id"].astype(int).sort_values().tolist()
    size_node_ids = node_map.loc[~node_map["size_tier"].eq(0), "node_id"].astype(int).sort_values().tolist()
    logger.info(
        "Project nodes in each timestep: %s, Size nodes: %s",
        len(project_node_ids),
        len(size_node_ids),
    )
    timesteps = sorted(size_frame["timestep"].dropna().astype(int).unique().tolist())
    logger.info("Total Timesteps: %s", len(timesteps))
    if ts_test <= 0 or ts_test >= len(timesteps):
        raise ValueError(f"ts_test must be in [1, T-1], got ts_test={ts_test}, T={len(timesteps)}")
    train_steps = len(timesteps) - ts_test
    feature_split_step = resolve_feature_split_step(loader_config, timesteps, node_processor)
    if feature_split_step is None:
        feature_split_step = train_steps
    logger.info(
        "Train steps: %s, Test steps: %s, split_on=%s, feature_split_step=%s",
        train_steps,
        ts_test,
        split_on,
        feature_split_step,
    )

    extra_exclude = loader_config["extra_exclude_columns"]
    target_cols = loader_config["target_columns"]
    id_cols = loader_config["id_columns"]
    inferred_project_cols = infer_feature_columns(project_frame, id_cols, target_cols, extra_exclude)
    inferred_size_cols = infer_feature_columns(size_frame, id_cols, target_cols, extra_exclude)
    project_feature_cols = loader_config["project_feature_columns"] or inferred_project_cols
    size_feature_cols = loader_config["size_feature_columns"] or inferred_size_cols

    missing_project_cols = sorted(set(project_feature_cols) - set(project_frame.columns))
    missing_size_cols = sorted(set(size_feature_cols) - set(size_frame.columns))
    if missing_project_cols or missing_size_cols:
        raise KeyError(
            "Configured feature columns are missing: "
            f"project={missing_project_cols}, size={missing_size_cols}"
        )

    if not project_feature_cols or not size_feature_cols:
        raise ValueError("Project and size node tables must each have at least one numeric feature column.")

    logger.info(
        "Using database_260519 RpHGNN loader: "
        "project_nodes=%s, size_nodes=%s, timesteps=%s, project_feat=%s, "
        "size_feat=%s, rp_dim=%s, target_shift=%s",
        len(project_node_ids),
        len(size_node_ids),
        len(timesteps),
        len(project_feature_cols),
        len(size_feature_cols),
        rp_dim,
        target_shift,
    )

    raw_project_frame = project_frame.copy()
    raw_size_frame = size_frame.copy()
    project_frame = feature_normalize(
        project_frame,
        feat_norm,
        node_processor.out_paths["project_node_folder"],
        project_feature_cols,
    )
    mark("normalized project features")
    size_frame = feature_normalize(
        size_frame,
        feat_norm,
        node_processor.out_paths["size_node_folder"],
        size_feature_cols,
    )
    mark("normalized size features")

    project_features = build_time_node_tensor(project_frame, project_node_ids, timesteps, project_feature_cols)
    mark(f"built project tensor {project_features.shape}")
    size_features = build_time_node_tensor(size_frame, size_node_ids, timesteps, size_feature_cols)
    mark(f"built size tensor {size_features.shape}")


    project_features = random_project_features(project_features, rp_dim, random_seed + 17)
    mark("random projected project features")
    size_features = random_project_features(size_features, rp_dim, random_seed + 29)
    mark("random projected size features")
    project_features, size_features = append_direct_size_features(
        project_features,
        size_features,
        raw_size_frame,
        size_node_ids,
        timesteps,
        direct_size_feature_cols,
        train_steps,
        scale=direct_feature_scale,
    )

    graph = build_heterograph_from_edge_frames(edge_dfs, project_node_ids, size_node_ids)
    logger.info("Heterograph: %s", graph)
    mark("built heterograph")
    timestep_graphs = None
    timestep_dates = None
    dynamic_graph_metadata = {"enabled": False}
    if dynamic_mrt_edges or dynamic_condo_edges:
        timestep_info = pd.read_csv(node_processor.out_paths["timestep_info_path"])
        timestep_to_date = dict(
            zip(
                pd.to_numeric(timestep_info["timestep"], errors="raise").astype(int),
                pd.to_datetime(timestep_info["date"], errors="raise"),
            )
        )
        missing_timestep_dates = [step for step in timesteps if step not in timestep_to_date]
        if missing_timestep_dates:
            raise ValueError(f"Missing dates for timesteps: {missing_timestep_dates[:10]}")
        timestep_dates = [timestep_to_date[step] for step in timesteps]
    if dynamic_mrt_edges:
        timestep_graphs, dynamic_graph_metadata = build_dynamic_mrt_graphs(
            edge_dfs,
            project_node_ids,
            size_node_ids,
            edge_processor.project_df,
            node_map,
            edge_processor.mrt_df,
            timestep_dates,
            to_plain_config(config.graph)["edges"],
            opening_date_column=mrt_opening_date_column,
        )
        graph = timestep_graphs[-1]
        logger.info("Dynamic MRT graph metadata: %s", dynamic_graph_metadata)
        mark(f"built {dynamic_graph_metadata['num_regimes']} dynamic MRT regimes")
    similar_price_metadata = {"enabled": False}
    if similar_price_edges:
        base_graphs = timestep_graphs or [graph] * len(timesteps)
        timestep_graphs, similar_price_metadata = build_similar_price_graphs(
            base_graphs,
            raw_size_frame,
            size_node_ids,
            timesteps,
            price_column=similar_price_column,
            tolerance=similar_price_tolerance,
            max_neighbors=similar_price_k,
        )
        graph = timestep_graphs[-1]
        logger.info("Similar-price graph metadata: %s", similar_price_metadata)
        mark(
            "built contemporaneous similar-price graphs "
            f"total_edges={similar_price_metadata['undirected_edges_total']}"
        )
    dynamic_condo_metadata = {"enabled": False}
    if dynamic_condo_edges:
        resolved_condo_age_column = condo_age_column
        if (
            resolved_condo_age_column not in raw_project_frame.columns
            and resolved_condo_age_column == "condo_age_at_transaction"
            and "Condo_Age_2026" in raw_project_frame.columns
        ):
            logger.warning(
                "Using legacy generated age column Condo_Age_2026; rebuild the "
                "graph database to retain condo_age_at_transaction."
            )
            resolved_condo_age_column = "Condo_Age_2026"
        base_graphs = timestep_graphs or [graph] * len(timesteps)
        timestep_graphs, dynamic_condo_metadata = build_dynamic_condo_graphs(
            base_graphs,
            raw_project_frame,
            node_map,
            project_node_ids,
            size_node_ids,
            timesteps,
            timestep_dates,
            age_column=resolved_condo_age_column,
        )
        graph = timestep_graphs[-1]
        logger.info("Dynamic condo graph metadata: %s", dynamic_condo_metadata)
        mark(
            "built dynamic condo-existence graphs "
            f"computable={dynamic_condo_metadata['computable_projects']} "
            f"static_fallback={dynamic_condo_metadata['static_fallback_projects']}"
        )
    if context_mode == "mixed_hops":
        precompute_result = rphgnn_precompute_contexts(
            graph,
            project_features,
            size_features,
            num_hops,
            device=precompute_device,
            timestep_graphs=timestep_graphs,
        )
    elif context_mode == "exact_metapaths":
        precompute_result = rphgnn_precompute_metapaths(
            graph,
            project_features,
            size_features,
            num_hops,
            device=precompute_device,
            timestep_graphs=timestep_graphs,
            include=metapath_include,
        )
    else:
        raise ValueError(
            f"context_mode must be 'mixed_hops' or 'exact_metapaths', got {context_mode!r}."
        )
    if len(precompute_result) == 2:
        contexts, group_names = precompute_result
        context_layout = "node_major"
    else:
        contexts, group_names, context_layout = precompute_result
    mark(f"precomputed contexts {contexts.shape}")
    logger.info("RpHGNN groups: %s", group_names)

    targets, masks = build_target_tensor(
        size_frame=raw_size_frame,
        size_node_ids=size_node_ids,
        timesteps=timesteps,
        target_col=target_col,
        mask_col=mask_col,
        target_mask_mode=target_mask_mode,
        train_steps=train_steps,
    )
    mark(f"built target tensors {targets.shape}")
    if split_on == "feature":
        (train_nodes, train_target_starts), (test_nodes, test_target_starts) = build_window_indices_feature_split(
            masks,
            window_size,
            feature_split_step,
            target_shift,
        )
        min_target_time = None
        min_feature_time = feature_split_step
    else:
        (train_nodes, train_target_starts), (test_nodes, test_target_starts) = build_window_indices(
            masks,
            window_size,
            train_steps,
            target_shift,
        )
        min_target_time = train_steps
        min_feature_time = None
    mark(f"built window indices train={len(train_nodes)} test={len(test_nodes)}")
    split_metadata = {
        "split_on": split_on,
        "split_boundary_policy": split_boundary_policy,
        "train_steps": int(train_steps),
        "ts_test": int(ts_test),
        "target_shift": int(target_shift),
        "window_size": int(window_size),
        **split_date_ranges(timesteps, feature_split_step, target_shift),
        "dynamic_mrt_edges": dynamic_mrt_edges,
        "dynamic_mrt_num_regimes": int(dynamic_graph_metadata.get("num_regimes", 0)),
        "dynamic_condo_edges": dynamic_condo_edges,
        "dynamic_condo_metadata": dynamic_condo_metadata,
        "similar_price_edges": similar_price_edges,
        "similar_price_metadata": similar_price_metadata,
    }
    split_metadata["dynamic_graph_metadata"] = dynamic_graph_metadata

    train_ds = RpHGNNWindowDataset(
        contexts,
        targets,
        masks,
        train_nodes,
        train_target_starts,
        window_size,
        target_shift=target_shift,
        context_layout=context_layout,
    )
    test_ds = RpHGNNWindowDataset(
        contexts,
        targets,
        masks,
        test_nodes,
        test_target_starts,
        window_size,
        target_shift=target_shift,
        min_target_time=min_target_time,
        min_feature_time=min_feature_time,
        context_layout=context_layout,
    )
    add_split_metadata(train_ds, split_metadata | {"split": "train"})
    add_split_metadata(test_ds, split_metadata | {"split": "test"})
    train_ds.group_names = list(group_names)
    test_ds.group_names = list(group_names)
    logger.info(
        "Train samples: %s | Test samples: %s | "
        "Per-sample input: (window=%s, groups=%s, K+1=%s, rp_dim=%s), target_shift=%s",
        len(train_ds),
        len(test_ds),
        window_size,
        len(group_names),
        contexts.shape[2],
        rp_dim,
        target_shift,
    )
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
            worker_init_fn=seed_worker,
        ),
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            worker_init_fn=seed_worker,
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description="Debug database_260519 RpHGNN dataloader construction")
    parser.add_argument("--graph-config", type=str, default=DEFAULT_GRAPH_CONFIG_PATH)
    parser.add_argument("--num-hops", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--target-shift", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rp-dim", type=int, default=32)
    args = parser.parse_args()
    graph_config = OmegaConf.load(args.graph_config)
    cfg = OmegaConf.create(
        {
            "graph": graph_config,
            "data": {
                "graph_config_path": args.graph_config,
                "num_hops": args.num_hops,
                "window_size": args.window_size,
                "target_shift": args.target_shift,
                "rp_dim": args.rp_dim,
            },
            "training": {
                "batch_size": args.batch_size,
            },
            "seed": 42,
        }
    )
    train_loader, test_loader = get_dataloaders(cfg)
    batch = next(iter(train_loader))
    logger.info("First train batch: %s", [item.shape if hasattr(item, "shape") else item for item in batch])
