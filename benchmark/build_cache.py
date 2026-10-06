"""Build shared inputs for the Full SG models (rent and sale panels).

One pass per task produces everything every model needs, so all models see the
same features, graphs, targets, masks and split:

* ``hetero.npy``  relation-wise context (time, group, hop, size node, dim),
                  identical to the existing RpHGNN precompute;
* ``homo.npy``    relation-agnostic context over the union of the same edges;
* ``targets.npy`` / ``masks.npy`` observed targets and masks (size node, time);
* ``tab_*.npy``   raw per-month inputs for the tabular (LightGBM) baselines,
                  including neighbour-price aggregates over the same graphs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.data.pgim as pgim
from benchmark.common import CACHE_ROOT, EDGE_SUBSET, SIMILAR_PRICE_K
from pipeline.ccr_study import build_cfg


def homogeneous_contexts(graph_sequence, project_features, size_features, num_hops, device):
    """Relation-agnostic propagation: all edge types merged into one relation.

    Project and size nodes share one index space (projects first). The update
    rule matches the heterogeneous precompute, except that messages are averaged
    over the union of neighbours instead of per relation.
    """
    num_t, num_size, dim = size_features.shape
    num_project = project_features.shape[1]
    out = np.zeros((num_t, 2, num_hops + 1, num_size, dim), dtype=np.float32)
    cache = {}

    def merged(graph):
        key = id(graph)
        if key not in cache:
            src_all, dst_all = [], []
            for (src_type, _, dst_type), (src, dst) in graph.edge_index.items():
                src_off = 0 if src_type == "project" else num_project
                dst_off = 0 if dst_type == "project" else num_project
                src_all.append(src.to(device) + src_off)
                dst_all.append(dst.to(device) + dst_off)
            src = torch.cat(src_all)
            dst = torch.cat(dst_all)
            # Duplicate pairs (e.g. a project pair linked by two relations) count once.
            pair = torch.unique(src * (num_project + num_size) + dst)
            cache[key] = (pair // (num_project + num_size), pair % (num_project + num_size))
        return cache[key]

    pf = torch.from_numpy(project_features).float().to(device)
    sf = torch.from_numpy(size_features).float().to(device)
    for t in range(num_t):
        src, dst = merged(graph_sequence[t])
        h = torch.cat([pf[t], sf[t]], dim=0)
        out[t, 0, 0] = sf[t].cpu().numpy()
        for hop in range(1, num_hops + 1):
            msg = pgim.mean_aggregate(h, src, dst, h.shape[0])
            out[t, 1, hop] = msg[num_project:].cpu().numpy()
            h = torch.nn.functional.normalize(0.5 * h + 0.5 * msg, dim=-1)
            out[t, 0, hop] = h[num_project:].cpu().numpy()
    return out


def neighbour_mean(values, src, dst, num_dst):
    """Mean of ``values[src]`` per destination; NaN where there is no neighbour."""
    total = np.zeros(num_dst)
    count = np.zeros(num_dst)
    np.add.at(total, dst, values[src])
    np.add.at(count, dst, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def tabular_inputs(cfg, captured, size_node_ids, project_node_ids, timesteps):
    """Raw inputs for tabular models: project features, own prices, graph aggregates."""
    root = Path(cfg.graph.io_paths.output_folder)
    node_map = pd.read_csv(root / "node_id.csv")
    size_frame = pd.read_csv(root / "size_level_node.csv")
    project_frame = pd.read_csv(root / "project_level_node.csv")

    proj_cols = list(cfg.data.project_feature_columns)
    project_tensor = pgim.build_time_node_tensor(project_frame, project_node_ids, timesteps, proj_cols)
    imp = pgim.build_time_node_tensor(size_frame, size_node_ids, timesteps, ["rent_per_sqft_imp"])[..., 0]
    obs = pgim.build_time_node_tensor(size_frame, size_node_ids, timesteps, ["rent_per_sqft"])[..., 0]
    mask = pgim.build_time_node_tensor(size_frame, size_node_ids, timesteps, ["y_mask"])[..., 0]
    obs = np.where(mask > 0, obs, np.nan).astype(np.float32)

    node_info = node_map.set_index("node_id")
    project_index = {int(n): i for i, n in enumerate(project_node_ids)}
    # node_id.csv stores project_id for both levels; map a size node to its project node.
    proj_node_by_pid = {
        int(pid): project_index[int(nid)]
        for nid, pid in node_map.loc[node_map["size_tier"].eq(0), ["node_id", "project_id"]].itertuples(index=False)
    }
    size_parent = np.array([proj_node_by_pid[int(node_info.loc[n, "project_id"])] for n in size_node_ids])
    size_tier = np.array([int(node_info.loc[n, "size_tier"]) for n in size_node_ids])

    num_t, num_size = imp.shape
    num_project = len(project_node_ids)
    sibling = np.full((num_t, num_size), np.nan, dtype=np.float32)
    peer = np.full((num_t, num_size), np.nan, dtype=np.float32)
    mrt = np.full((num_t, num_size), np.nan, dtype=np.float32)
    sizes = np.arange(num_size)
    for t in range(num_t):
        graph = captured["graphs"][t]
        project_price = neighbour_mean(imp[t], sizes, size_parent, num_project)
        # Sibling mean excludes the node itself.
        total = np.bincount(size_parent, weights=imp[t], minlength=num_project)
        count = np.bincount(size_parent, minlength=num_project)
        with np.errstate(invalid="ignore", divide="ignore"):
            sibling[t] = np.where(count[size_parent] > 1,
                                  (total[size_parent] - imp[t]) / (count[size_parent] - 1), np.nan)
        key = ("size", "similar_price", "size")
        if key in graph.edge_index:
            src, dst = (x.numpy() for x in graph.edge_index[key])
            peer[t] = neighbour_mean(imp[t], src, dst, num_size)
        key = ("project", "same_mrt", "project")
        if key in graph.edge_index:
            src, dst = (x.numpy() for x in graph.edge_index[key])
            s2, d2 = np.concatenate([src, dst]), np.concatenate([dst, src])
            mrt_project = neighbour_mean(np.nan_to_num(project_price), s2, d2, num_project)
            mrt[t] = mrt_project[size_parent]
    return {
        "tab_project": project_tensor.astype(np.float32),
        "tab_imp": imp.astype(np.float32),
        "tab_obs": obs,
        "tab_sibling": sibling,
        "tab_peer": peer,
        "tab_mrt": mrt,
        "tab_size_parent": size_parent.astype(np.int64),
        "tab_size_tier": size_tier.astype(np.int64),
    }, proj_cols


SNAPSHOT_RELATIONS = {
    ("size", "size_project", "project"): "size_project",
    ("project", "same_mrt", "project"): "same_mrt",
    ("size", "similar_price", "size"): "similar_price",
}
SNAPSHOT_START = "2020-01-01"  # first held-out feature month


def graph_snapshots(graphs, project_node_ids, size_node_ids, timesteps, dates):
    """Unique undirected edges per held-out month (graph node IDs), for the map demo."""
    ids = {"project": np.asarray(project_node_ids), "size": np.asarray(size_node_ids)}
    rows, cache = [], {}
    for timestep, date, graph in zip(timesteps, dates, graphs):
        if date < SNAPSHOT_START:
            continue
        key = id(graph)
        if key not in cache:
            parts = []
            for etype, name in SNAPSHOT_RELATIONS.items():
                if etype not in graph.edge_index:
                    continue
                src, dst = (np.asarray(t.cpu() if torch.is_tensor(t) else t) for t in graph.edge_index[etype])
                a, b = ids[etype[0]][src], ids[etype[2]][dst]
                frame = pd.DataFrame({"relation": name, "node_id": np.minimum(a, b), "neig_node_id": np.maximum(a, b)})
                parts.append(frame.loc[frame["node_id"].ne(frame["neig_node_id"])].drop_duplicates())
            cache[key] = pd.concat(parts, ignore_index=True)
        rows.append(cache[key].assign(timestep=timestep, date=date))
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["rent", "sale"], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--no-project-features", action="store_true",
                        help="only build hetero_np.npy / homo_np.npy: same graph, project covariates zeroed, "
                             "so neighbours contribute price histories only")
    parser.add_argument("--struct-only", action="store_true",
                        help="only build struct.npy: the size-project hierarchy without optional relations")
    args = parser.parse_args()
    started = time.time()
    out_dir = CACHE_ROOT / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")

    runner_args = SimpleNamespace(
        dataset="full_sg", task=args.task, horizon=1, seed=42, feature_seed=42,
        gpu=args.gpu, batch_size=1024, similar_price_k=SIMILAR_PRICE_K, context_mode="mixed_hops",
    )
    cfg = build_cfg(runner_args, () if args.struct_only else EDGE_SUBSET)
    # Every model receives the same inputs: the own imputed price is a direct
    # size-level input in every setting.
    cfg.data.direct_size_feature_columns = ["rent_per_sqft_imp"]
    cfg.data.direct_feature_scale = "standard"

    captured = {}
    original = pgim.rphgnn_precompute_contexts

    def capture(graph, project_features, size_features, num_hops, device=None, timestep_graphs=None):
        if args.no_project_features:
            project_features = np.zeros_like(project_features)
        captured.update(
            graphs=timestep_graphs or [graph] * size_features.shape[0],
            project_features=project_features,
            size_features=size_features,
            num_hops=num_hops,
        )
        return original(graph, project_features, size_features, num_hops,
                        device=device, timestep_graphs=timestep_graphs)

    pgim.rphgnn_precompute_contexts = capture
    train_loader, _ = pgim.get_dataloaders(cfg)
    pgim.rphgnn_precompute_contexts = original
    ds = train_loader.dataset
    print(f"[{time.time() - started:.0f}s] loader ready, contexts {ds.contexts.shape}", flush=True)

    if args.no_project_features:
        np.save(out_dir / "hetero_np.npy", ds.contexts)
        np.save(out_dir / "homo_np.npy", homogeneous_contexts(
            captured["graphs"], captured["project_features"], captured["size_features"], captured["num_hops"], device))
        print(f"[{time.time() - started:.0f}s] saved no-project-feature contexts groups={ds.group_names}", flush=True)
        return
    if args.struct_only:
        np.save(out_dir / "struct.npy", ds.contexts)
        print(f"[{time.time() - started:.0f}s] saved struct contexts {ds.contexts.shape} groups={ds.group_names}", flush=True)
        return
    np.save(out_dir / "hetero.npy", ds.contexts)
    np.save(out_dir / "targets.npy", ds.targets)
    np.save(out_dir / "masks.npy", ds.masks)
    homo = homogeneous_contexts(captured["graphs"], captured["project_features"],
                                captured["size_features"], captured["num_hops"], device)
    np.save(out_dir / "homo.npy", homo)
    del homo
    print(f"[{time.time() - started:.0f}s] saved hetero/homo contexts", flush=True)

    root = Path(cfg.graph.io_paths.output_folder)
    node_map = pd.read_csv(root / "node_id.csv")
    project_node_ids = node_map.loc[node_map["size_tier"].eq(0), "node_id"].astype(int).sort_values().tolist()
    size_node_ids = node_map.loc[~node_map["size_tier"].eq(0), "node_id"].astype(int).sort_values().tolist()
    size_frame_steps = pd.read_csv(root / "timestep_info.csv")
    timesteps = size_frame_steps["timestep"].astype(int).tolist()
    tab, proj_cols = tabular_inputs(cfg, captured, size_node_ids, project_node_ids, timesteps)
    for name, value in tab.items():
        np.save(out_dir / f"{name}.npy", value)
    snapshots = graph_snapshots(captured["graphs"], project_node_ids, size_node_ids, timesteps,
                                size_frame_steps["date"].tolist())
    snapshots.to_csv(out_dir / "graph_snapshots.csv.gz", index=False, compression="gzip")

    meta = {
        "task": args.task,
        "group_names": list(ds.group_names),
        "feature_split_timestep": int(ds.split_metadata["feature_split_timestep"]),
        "num_timesteps": int(ds.targets.shape[1]),
        "dates": size_frame_steps["date"].tolist(),
        "size_node_ids": size_node_ids,
        "project_feature_columns": proj_cols,
        "edge_subset": list(EDGE_SUBSET),
        "similar_price_k": SIMILAR_PRICE_K,
        "split_metadata": {k: v for k, v in ds.split_metadata.items() if isinstance(v, (int, float, str, bool))},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{time.time() - started:.0f}s] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
