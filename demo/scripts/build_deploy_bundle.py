"""Assemble a self-contained Docker build context for hosting the map demo.

The bundle mirrors the repo layout the backend expects (``PGIM_REPO_ROOT``) but keeps
only what the API reads:

* ``node_id.csv`` / ``project_info.csv``: just the columns used.
* CCR static edges: canonical project pairs (no self-loops, no duplicates), which
  is what ``relation_neighbours`` reduces them to anyway (hundreds of MB -> a few MB).
* prediction CSVs: just the columns ``prediction_table`` loads.
* Full SG graph snapshots: copied unchanged.

Usage (from the repo root, after ``npm run build`` in demo/):

    python demo/scripts/build_deploy_bundle.py            # -> demo/build/space
    python demo/scripts/build_deploy_bundle.py --out /tmp/space
    python demo/scripts/build_deploy_bundle.py --datasets full_sg_rent,full_sg_sale   # 512 MB hosts
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO / "backend"))
import app  # noqa: E402  (reuses DATASETS, HORIZONS, column lists)

PROJECT_COLUMNS = ["project_id", "project_name", "latitude", "longitude", "Planning Area",
                   "condo_age_at_transaction", "Condo_Age_2026"]


def _target(out_root: Path, path: Path) -> Path:
    dest = out_root / path.relative_to(app.REPO_ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def _write(frame: pd.DataFrame, out_root: Path, path: Path) -> None:
    frame.to_csv(_target(out_root, path), index=False)


def bundle_dataset(ds: app.Dataset, out_root: Path) -> None:
    nodes = pd.read_csv(ds.data_root / "node_id.csv", usecols=["project_id", "size_tier", "node_id"])
    _write(nodes, out_root, ds.data_root / "node_id.csv")

    info_path = ds.data_root / "project_info.csv"
    header = pd.read_csv(info_path, nrows=0).columns
    _write(pd.read_csv(info_path, usecols=[c for c in PROJECT_COLUMNS if c in header]), out_root, info_path)

    if ds.graph_mode == "static":
        project_nodes = set(nodes.loc[nodes["size_tier"].eq(0), "node_id"].tolist())
        for relation in ds.relations:
            path = ds.data_root / "edges" / f"{relation}.csv"
            edges = pd.read_csv(path, usecols=["node_id", "neig_node_id"])
            edges = edges.loc[edges["node_id"].isin(project_nodes) & edges["neig_node_id"].isin(project_nodes)]
            edges = edges.loc[edges["node_id"].ne(edges["neig_node_id"])]
            pairs = pd.DataFrame({
                "node_id": np.minimum(edges["node_id"], edges["neig_node_id"]),
                "neig_node_id": np.maximum(edges["node_id"], edges["neig_node_id"]),
            }).drop_duplicates().sort_values(["node_id", "neig_node_id"])
            _write(pairs, out_root, path)
            print(f"  {ds.id} {relation}: {len(edges):,} -> {len(pairs):,} edges")
    else:
        shutil.copy2(ds.snapshot_path, _target(out_root, ds.snapshot_path))
        if ds.manifest_file:
            manifest = app.PREDICTIONS / ds.manifest_file
            shutil.copy2(manifest, _target(out_root, manifest))

    for horizon in app.HORIZONS:
        path = ds.prediction_path(horizon)
        header = pd.read_csv(path, nrows=0).columns
        keep = app.BASE_COLUMNS + [c for c in app.OPTIONAL_COLUMNS if c in header]
        _write(pd.read_csv(path, usecols=keep, dtype=str, keep_default_na=False), out_root, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEMO / "build" / "space")
    parser.add_argument("--datasets", default=",".join(app.DATASETS), help="comma-separated dataset IDs to include")
    args = parser.parse_args()

    dist = DEMO / "dist"
    if not (dist / "index.html").exists():
        sys.exit("dist/ is missing: run `npm run build` in demo/ first.")
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    for name in ("Dockerfile", ".dockerignore", "render.yaml"):
        shutil.copy2(DEMO / "deploy" / name, args.out / name)
    shutil.copy2(DEMO / "deploy" / "space_README.md", args.out / "README.md")
    shutil.copy2(DEMO / "backend" / "app.py", args.out / "app.py")
    shutil.copy2(DEMO / "backend" / "requirements.txt", args.out / "requirements.txt")
    shutil.copytree(dist, args.out / "static", ignore=shutil.ignore_patterns("*.map"))

    data_root = args.out / "data"
    selected = args.datasets.split(",")
    unknown = set(selected) - set(app.DATASETS)
    if unknown:
        sys.exit(f"unknown datasets: {sorted(unknown)}")
    with open(args.out / "Dockerfile", "a") as dockerfile:
        dockerfile.write(f"ENV PGIM_DATASETS={','.join(selected)}\n")
    for ds in (app.DATASETS[key] for key in selected):
        if not app.dataset_available(ds)["available"]:
            print(f"skip {ds.id}: missing {app.dataset_available(ds)['missing']}")
            continue
        print(f"bundling {ds.id}")
        bundle_dataset(ds, data_root)

    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"wrote {args.out} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
