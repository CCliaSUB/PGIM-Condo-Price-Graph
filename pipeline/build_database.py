from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

PROJECT_KEEP_PREFIXES = (
    "glob_",
    "dist_",
    "condo_level_",
)

COMMON_PROJECT_KEEP_COLS = {
    "district",
    "area_sqft_med",
    "post_covid",
    "tenure_freehold_like",
    "tenure_medium_lease_60_80",
    "tenure_more_than_80",
    "tenure_short_lease_lt60",
    "tenure_unknown",
    "tenure_remaining_years",
    "project_size_small",
    "project_size_medium",
    "project_size_large",
    "Large_Dev_200plus",
    "Postal District",
    "Planning Area",
    "nearest_mrt_dist",
    "nearest_bus_stops_dist",
    "n_bus_stops_within_500m",
    "nearest_supermarkets_dist",
    "n_supermarkets_within_500m",
    "nearest_parks_dist",
    "n_parks_within_500m",
    "nearest_clinics_dist",
    "n_clinics_within_500m",
    "nearest_bank_dist",
    "n_bank_within_500m",
    "nearest_atms_dist",
    "n_atms_within_500m",
    "nearest_school_dist",
    "month_of_year",
    "ura_private_rental_index_yoy_pct",
    "sora_compounded_3m_pct",
    "gva_yoy_growth_pct",
    "cpi_all_items_infl_yoy_pct",
    "unemployment_rate_sa_pct",
    "sg_govt_bond_yield_10y_pct",
    "sgd_per_usd_logret_12m_pct",
    "ura_private_price_index_avg_3seg_yoy_pct",
    "hdb_resale_price_index_yoy_pct",
}


def normalize_project_name(value) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def read_csv(path: Path, usecols=None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=usecols, engine="pyarrow")
    except Exception:
        return pd.read_csv(path, usecols=usecols)


def parse_yyyymm(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.astype(str).str.strip(), format="%Y%m", errors="raise")


def add_locations(
    frame: pd.DataFrame,
    project_location_path: Path,
    *,
    allow_missing: bool = False,
) -> pd.DataFrame:
    loc = pd.read_csv(project_location_path)
    loc = loc.drop_duplicates("project_name").copy()
    direct_lat = loc.set_index("project_name")["latitude"]
    direct_lon = loc.set_index("project_name")["longitude"]
    norm_loc = loc.assign(_norm=loc["project_name"].map(normalize_project_name))
    norm_loc = norm_loc.loc[~norm_loc["_norm"].duplicated(keep=False)]
    norm_lat = norm_loc.set_index("_norm")["latitude"]
    norm_lon = norm_loc.set_index("_norm")["longitude"]

    names = pd.Index(frame["project_name"].unique())
    lat = names.to_series(index=names).map(direct_lat)
    lon = names.to_series(index=names).map(direct_lon)
    missing = lat.isna() | lon.isna()
    if missing.any():
        missing_names = names[missing]
        lat.loc[missing_names] = pd.Series(missing_names, index=missing_names).map(
            lambda value: norm_lat.get(normalize_project_name(value))
        )
        lon.loc[missing_names] = pd.Series(missing_names, index=missing_names).map(
            lambda value: norm_lon.get(normalize_project_name(value))
        )
    if (lat.isna().any() or lon.isna().any()) and not allow_missing:
        unresolved = sorted(names[lat.isna() | lon.isna()].tolist())
        raise KeyError(f"Missing locations for projects: {unresolved}")

    frame["latitude"] = frame["project_name"].map(lat).astype(float)
    frame["longitude"] = frame["project_name"].map(lon).astype(float)
    return frame


def stats_for(frame: pd.DataFrame, feature_cols: list[str]) -> tuple[dict, dict, dict]:
    stand, norm0, norm1 = {}, {}, {}
    for col in feature_cols:
        values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = values.dropna()
        if finite.empty:
            mean = min_value = max_value = 0.0
            std = denom = 1.0
        else:
            mean = float(finite.mean())
            std = float(finite.std(ddof=0)) or 1.0
            min_value = float(finite.min())
            max_value = float(finite.max())
            denom = max_value - min_value or 1.0
        stand[col] = {"mean": mean, "std": std}
        norm0[col] = {
            "min": min_value,
            "max": max_value,
            "denominator": denom,
            "target_min": 0.0,
            "target_max": 1.0,
        }
        norm1[col] = {
            "min": min_value,
            "max": max_value,
            "denominator": denom,
            "target_min": -1.0,
            "target_max": 1.0,
        }
    return stand, norm0, norm1


def write_stats(folder: Path, frame: pd.DataFrame, feature_cols: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    stand, norm0, norm1 = stats_for(frame, feature_cols)
    for name, payload in [("stand.json", stand), ("norm0.json", norm0), ("norm1.json", norm1)]:
        (folder / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_database(
    config_path: Path,
    *,
    input_path_override: Path | None = None,
    output_folder_override: Path | None = None,
    allow_missing_locations: bool = False,
) -> None:
    started = time.time()

    def mark(message: str) -> None:
        print(f"{time.time() - started:8.1f}s {message}", flush=True)

    cfg = OmegaConf.load(config_path)
    if input_path_override is not None:
        cfg.io_paths.input_path = str(input_path_override)
    if output_folder_override is not None:
        cfg.io_paths.output_folder = str(output_folder_override)
    input_path = Path(cfg.io_paths.input_path)
    output = Path(cfg.io_paths.output_folder)
    output.mkdir(parents=True, exist_ok=True)

    raw_cols = pd.read_csv(input_path, nrows=0).columns.tolist()
    forecast_cols = [col for col in raw_cols if col.startswith("consulting_")]
    keep_cols = set(COMMON_PROJECT_KEEP_COLS) | set(forecast_cols)
    keep_cols |= {col for col in raw_cols if col.startswith(PROJECT_KEEP_PREFIXES)}
    keep_cols = {col for col in keep_cols if col in raw_cols}
    required = {
        "condo_name",
        "yyyymm",
        "size_tier",
        "rent_psf_obs",
        "rent_psf_imp",
        "was_observed",
        "condo_age_at_transaction",
        *keep_cols,
    }
    mark(f"reading {input_path}")
    df = read_csv(input_path, usecols=sorted(required))
    mark(f"read raw shape={df.shape}")
    df = df.rename(
        columns={
            "condo_name": "project_name",
            "yyyymm": "date",
            "rent_psf_obs": "rent_per_sqft",
            "rent_psf_imp": "rent_per_sqft_imp",
            "was_observed": "y_mask",
        }
    )
    df["date"] = parse_yyyymm(df["date"])
    project_codes, _ = pd.factorize(df["project_name"].to_numpy(), sort=True)
    timestep_codes, _ = pd.factorize(df["date"].to_numpy(), sort=True)
    df["project_id"] = project_codes.astype(int)
    df["timestep"] = timestep_codes.astype(int)
    df["size_tier"] = df["size_tier"].astype(str).str.upper().str.replace("SZ", "", regex=False).astype(int)
    df["y_mask"] = df["y_mask"].astype(bool).astype(int)
    df["rent_per_sqft"] = pd.to_numeric(df["rent_per_sqft"], errors="coerce")
    df["rent_per_sqft_imp"] = pd.to_numeric(df["rent_per_sqft_imp"], errors="coerce")
    df = add_locations(
        df,
        Path(cfg.io_paths.full_project_location_path),
        allow_missing=allow_missing_locations,
    )
    missing_location_projects = int(
        df.loc[df["latitude"].isna() | df["longitude"].isna(), "project_name"].nunique()
    )
    if missing_location_projects:
        mark(
            f"warning: {missing_location_projects} projects lack coordinates; "
            "they will have no coordinate-derived project edges"
        )
    mark("standardized ids/dates/locations")

    project_info_cols = [
        "project_id",
        "project_name",
        "latitude",
        "longitude",
        "condo_age_at_transaction",
        "Planning Area",
        "nearest_school_dist",
    ]
    df[project_info_cols].drop_duplicates("project_id").sort_values("project_id").to_csv(
        output / "project_info.csv", index=False
    )
    df[["timestep", "date"]].drop_duplicates("timestep").sort_values("timestep").to_csv(
        output / "timestep_info.csv", index=False
    )

    projects = np.sort(df["project_id"].unique())
    size_tiers = np.sort(df["size_tier"].unique())
    node_rows = []
    node_id = 0
    for project_id in projects:
        node_rows.append({"project_id": int(project_id), "size_tier": 0, "node_id": node_id})
        node_id += 1
        for tier in size_tiers:
            node_rows.append({"project_id": int(project_id), "size_tier": int(tier), "node_id": node_id})
            node_id += 1
    node_id_frame = pd.DataFrame(node_rows)
    node_id_frame.to_csv(output / "node_id.csv", index=False)
    mark(f"wrote metadata/node ids nodes={len(node_id_frame)}")

    size_node_ids = node_id_frame[node_id_frame["size_tier"].ne(0)].set_index(["project_id", "size_tier"])["node_id"]
    size_keys = pd.MultiIndex.from_frame(df[["project_id", "size_tier"]])
    size_frame = df[["project_id", "timestep", "size_tier", "rent_per_sqft_imp", "rent_per_sqft", "y_mask"]].copy()
    size_frame["node_id"] = size_node_ids.reindex(size_keys).to_numpy()
    size_frame = size_frame[
        ["node_id", "project_id", "timestep", "size_tier", "rent_per_sqft_imp", "rent_per_sqft", "y_mask"]
    ].copy()
    for tier in size_tiers:
        size_frame[f"size_tier_{tier}"] = (size_frame["size_tier"] == tier).astype(int)
    size_frame["project_id_feat"] = size_frame["project_id"]
    size_frame["timestep_feat"] = size_frame["timestep"]
    size_frame["rent_per_sqft_feat"] = size_frame["rent_per_sqft"]
    size_frame = size_frame.sort_values(["node_id", "project_id", "timestep"])
    size_frame.to_csv(output / "size_level_node.csv", index=False)
    mark(f"wrote size nodes shape={size_frame.shape}")

    project_feature_cols = [
        "project_id",
        "timestep",
        "project_name",
        "date",
        "rent_per_sqft",
        "condo_age_at_transaction",
        *sorted(keep_cols - {"condo_age_at_transaction"}),
    ]
    project_feature_cols = [col for col in project_feature_cols if col in df.columns]
    mark("building project first-row mask")
    first_mask = ~df.duplicated(["project_id", "timestep"], keep="first")
    mark(f"project first rows={int(first_mask.sum())}")
    first_rows = df.loc[first_mask, project_feature_cols].copy()
    mark(f"copied project first rows shape={first_rows.shape}")
    target_mean = df.groupby(["project_id", "timestep"], as_index=False, dropna=False)["rent_per_sqft"].mean()
    mark(f"computed project target mean shape={target_mean.shape}")
    project_frame = first_rows.drop(columns=["rent_per_sqft"], errors="ignore").merge(
        target_mean, on=["project_id", "timestep"], how="left"
    )
    mark(f"merged project frame shape={project_frame.shape}")
    project_nodes = node_id_frame[node_id_frame["size_tier"].eq(0)].set_index("project_id")["node_id"]
    project_frame["node_id"] = project_frame["project_id"].map(project_nodes)
    project_frame["size_tier"] = 0
    project_frame["y_mask"] = project_frame["rent_per_sqft"].notna().astype(int)
    project_frame["timestep_feat"] = project_frame["timestep"]
    project_frame["node_id_feat"] = project_frame["node_id"]

    project_frame = project_frame.drop(columns=["project_name", "date", "district", "Planning Area"], errors="ignore")
    project_frame = project_frame.sort_values(["node_id", "project_id", "timestep"])
    ordered = ["timestep", "node_id", "project_id"]
    rest = [col for col in project_frame.columns if col not in ordered + ["rent_per_sqft", "y_mask"]]
    project_frame = project_frame[ordered + rest + ["rent_per_sqft", "y_mask"]]
    mark("writing project node csv")
    project_frame.to_csv(output / "project_level_node.csv", index=False)
    mark(f"wrote project nodes shape={project_frame.shape}")

    project_feature_stats = [
        col
        for col in project_frame.columns
        if col not in {"node_id", "project_id", "timestep", "rent_per_sqft", "y_mask"}
        and pd.api.types.is_numeric_dtype(project_frame[col])
    ]
    size_feature_stats = [
        col
        for col in size_frame.columns
        if col not in {"node_id", "project_id", "timestep", "rent_per_sqft", "y_mask"}
        and pd.api.types.is_numeric_dtype(size_frame[col])
    ]
    write_stats(output / "project_level_node", project_frame, project_feature_stats)
    write_stats(output / "size_level_node", size_frame, size_feature_stats)
    mark("wrote stats")
    print(
        json.dumps(
            {
                "output": str(output),
                "project_shape": project_frame.shape,
                "size_shape": size_frame.shape,
                "node_id_shape": node_id_frame.shape,
                "project_features": len(project_feature_stats),
                "size_features": len(size_feature_stats),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", type=Path, help="Override io_paths.input_path.")
    parser.add_argument("--output", type=Path, help="Override io_paths.output_folder.")
    parser.add_argument(
        "--allow-missing-locations",
        action="store_true",
        help="Keep unmatched projects with null coordinates instead of failing.",
    )
    args = parser.parse_args()
    build_database(
        args.config,
        input_path_override=args.input,
        output_folder_override=args.output,
        allow_missing_locations=args.allow_missing_locations,
    )


if __name__ == "__main__":
    main()
