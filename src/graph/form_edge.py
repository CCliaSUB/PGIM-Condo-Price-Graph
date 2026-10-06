from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd

# from src.utils.utils import check_config_keys

logger = logging.getLogger(__name__)


def pairwise_haversine_m(lat_a, lon_a, lat_b, lon_b):
    lat_a = np.radians(np.asarray(lat_a, dtype=float))
    lon_a = np.radians(np.asarray(lon_a, dtype=float))
    lat_b = np.radians(np.asarray(lat_b, dtype=float))
    lon_b = np.radians(np.asarray(lon_b, dtype=float))

    lat_diff = lat_a[:, None] - lat_b[None, :]
    lon_diff = lon_a[:, None] - lon_b[None, :]
    haversine = (
        np.sin(lat_diff / 2) ** 2
        + np.cos(lat_a[:, None])
        * np.cos(lat_b[None, :])
        * np.sin(lon_diff / 2) ** 2
    )
    return 6_371_000 * 2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))


def adjacent_mrt_pairs(
    nearest_station_names,
    station_names,
    alphanumeric_codes,
    line_names,
):
    """Return project pairs assigned to consecutive stations on one MRT line.

    Station numbers are ordered within each code prefix and ``line_english``.
    Consecutive *available* codes are treated as neighbors, so NS5--NS7 is
    retained when NS6 is absent/reserved. Codes without a numeric suffix are
    ignored. Condo pairs assigned to the same physical station are excluded;
    that case belongs to the existing ``same_mrt`` relation.
    """
    station_rows = []
    for station_name, code, line_name in zip(
        station_names,
        alphanumeric_codes,
        line_names,
    ):
        if pd.isna(station_name) or pd.isna(code) or pd.isna(line_name):
            continue
        station_name = str(station_name).strip()
        code = str(code).strip().upper()
        line_name = str(line_name).strip()
        match = re.fullmatch(r"([A-Z]+)(\d+)", code)
        if not station_name or not line_name or match is None:
            continue
        station_rows.append(
            {
                "station_name": station_name,
                "code": code,
                "prefix": match.group(1),
                "number": int(match.group(2)),
                "line_name": line_name,
            }
        )

    adjacency_labels = defaultdict(set)
    if station_rows:
        station_frame = pd.DataFrame(station_rows).drop_duplicates(
            ["station_name", "code", "line_name"]
        )
        for _, line_frame in station_frame.groupby(
            ["line_name", "prefix"],
            sort=True,
        ):
            ordered = line_frame.sort_values(["number", "code", "station_name"])
            ordered = ordered.drop_duplicates("code", keep="first").reset_index(drop=True)
            for left_idx in range(len(ordered) - 1):
                left = ordered.iloc[left_idx]
                right = ordered.iloc[left_idx + 1]
                if left["station_name"] == right["station_name"]:
                    continue
                station_pair = tuple(sorted((left["station_name"], right["station_name"])))
                adjacency_labels[station_pair].add(f"{left['code']}-{right['code']}")

    projects_by_station = defaultdict(list)
    for project_idx, station_name in enumerate(nearest_station_names):
        if pd.isna(station_name):
            continue
        projects_by_station[str(station_name).strip()].append(project_idx)

    labels_by_project_pair = defaultdict(set)
    for station_pair, labels in adjacency_labels.items():
        left_projects = projects_by_station.get(station_pair[0], ())
        right_projects = projects_by_station.get(station_pair[1], ())
        for left_idx in left_projects:
            for right_idx in right_projects:
                project_pair = tuple(sorted((left_idx, right_idx)))
                labels_by_project_pair[project_pair].update(labels)

    return [
        (src_idx, dst_idx, "|".join(sorted(labels)))
        for (src_idx, dst_idx), labels in sorted(labels_by_project_pair.items())
    ]


class FormEdgeProcessor:
    def __init__(self, config):
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        self.config = config
        self.in_paths, self.out_paths = self.set_io_paths(config["io_paths"], config["io_node_paths"], config["edges"])
        self.project_df = self.load_project_data(self.in_paths["project_info_path"])
        self.node_id_df = self.load_node_id_data(self.in_paths["node_id_path"])
        self.mrt_df = self.load_mrt_data(self.in_paths["mrt_info_path"])

    def load_project_data(self, node_path):
        project_df = pd.read_csv(node_path)
        return project_df

    def load_node_id_data(self, node_id_path):
        node_id_df = pd.read_csv(node_id_path)
        return node_id_df

    def load_mrt_data(self, mrt_info_path):
        mrt_df = pd.read_csv(mrt_info_path)
        return mrt_df

    def get_project_node_id_map(self):
        node_id_required_cols = ["project_id", "size_tier", "node_id"]
        node_id_missing_cols = [
            col
            for col in node_id_required_cols
            if col not in self.node_id_df.columns
        ]
        if node_id_missing_cols:
            raise KeyError(
                f"Missing columns for project-level node ids: {node_id_missing_cols}"
            )

        return (
            self.node_id_df[self.node_id_df["size_tier"].eq(0)]
            .drop_duplicates(subset=["project_id"])
            .set_index("project_id")["node_id"]
        )

    def add_project_node_ids(self, project_df):
        project_df = project_df.copy()
        project_node_ids = self.get_project_node_id_map()
        project_df["node_id"] = project_df["project_id"].map(project_node_ids)
        if project_df["node_id"].isna().any():
            missing_project_ids = project_df.loc[
                project_df["node_id"].isna(),
                "project_id",
            ].tolist()
            raise ValueError(
                "Missing project-level node_id for project_id values: "
                f"{missing_project_ids}"
            )
        project_df["node_id"] = project_df["node_id"].astype(int)
        return project_df

    def pairwise_haversine_m(self, lat_a, lon_a, lat_b, lon_b):
        return pairwise_haversine_m(lat_a, lon_a, lat_b, lon_b)

    def prepare_project_mrt_distance_df(self):
        project_required_cols = ["project_id", "latitude", "longitude"]
        mrt_required_cols = [
            "alphanumeric_code",
            "station_name_english",
            "latitude",
            "longitude",
        ]
        project_missing_cols = [
            col
            for col in project_required_cols
            if col not in self.project_df.columns
        ]
        if project_missing_cols:
            raise KeyError(
                "Missing project columns for MRT distance edges: "
                f"{project_missing_cols}"
            )
        mrt_missing_cols = [
            col
            for col in mrt_required_cols
            if col not in self.mrt_df.columns
        ]
        if mrt_missing_cols:
            raise KeyError(
                f"Missing MRT columns for MRT distance edges: {mrt_missing_cols}"
            )

        project_df = self.project_df[project_required_cols].copy()
        project_df["latitude"] = pd.to_numeric(
            project_df["latitude"],
            errors="coerce",
        )
        project_df["longitude"] = pd.to_numeric(
            project_df["longitude"],
            errors="coerce",
        )
        project_df = project_df.dropna(subset=["latitude", "longitude"])
        project_df = self.add_project_node_ids(project_df)

        mrt_df = self.mrt_df[mrt_required_cols].copy()
        mrt_df["latitude"] = pd.to_numeric(mrt_df["latitude"], errors="coerce")
        mrt_df["longitude"] = pd.to_numeric(mrt_df["longitude"], errors="coerce")
        mrt_df = mrt_df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

        project_to_mrt_dist = self.pairwise_haversine_m(
            project_df["latitude"].to_numpy(dtype=float),
            project_df["longitude"].to_numpy(dtype=float),
            mrt_df["latitude"].to_numpy(dtype=float),
            mrt_df["longitude"].to_numpy(dtype=float),
        )
        nearest_mrt_idx = project_to_mrt_dist.argmin(axis=1)
        nearest_mrt_df = mrt_df.iloc[nearest_mrt_idx].reset_index(drop=True)
        project_df = project_df.reset_index(drop=True)
        project_df["nearest_mrt_code"] = nearest_mrt_df["alphanumeric_code"]
        project_df["nearest_mrt_name"] = nearest_mrt_df["station_name_english"]
        project_df["nearest_mrt_distance_m"] = project_to_mrt_dist[
            np.arange(len(project_df)),
            nearest_mrt_idx,
        ]
        return project_df, mrt_df, project_to_mrt_dist
        
    def set_io_paths(self, io_paths, io_node_paths, config_edges):
        input_path = Path(io_paths["input_path"])
        mrt_info_path = Path(io_paths["mrt_info_path"])
        output_folder = Path(io_paths["output_folder"])
        edge_output_folder = output_folder / "edges"
        output_folder.mkdir(parents=True, exist_ok=True)
        edge_output_folder.mkdir(parents=True, exist_ok=True)
        data_file_ext = config_edges.get("data_file_ext", ".csv")
        if data_file_ext and not data_file_ext.startswith("."):
            data_file_ext = f".{data_file_ext}"

        def output_data_path(filename, folder=output_folder):
            filename = Path(filename)
            if filename.suffix:
                return folder / filename
            return folder / f"{filename}{data_file_ext}"

        def format_param_value(value):
            return str(value).replace(".", "p")

        def edge_output_name(edge_config):
            filename = Path(edge_config["name"])
            suffix = filename.suffix
            stem = filename.with_suffix("").name if suffix else filename.name
            param_parts = []
            if "threshold" in edge_config:
                param_parts.append(format_param_value(edge_config["threshold"]))
            if "epsilon" in edge_config:
                param_parts.append(f"eps_{format_param_value(edge_config['epsilon'])}")
            if param_parts:
                stem = "_".join([stem, *param_parts])
            return f"{stem}{suffix}" if suffix else stem

        project_info_path = output_data_path(io_node_paths["project_info_filename"])
        node_id_path = output_data_path(io_node_paths["node_id_filename"])

        in_paths = {
            "input_path": input_path,
            "mrt_info_path": mrt_info_path,
            "project_info_path": project_info_path,
            "node_id_path": node_id_path,
        }
        out_paths = {
            edge_key: output_data_path(edge_output_name(edge_config), edge_output_folder)
            for edge_key, edge_config in config_edges.items()
            if isinstance(edge_config, dict) and "name" in edge_config
        }

        self.data_file_ext = data_file_ext
        for name, path in {**in_paths, **out_paths}.items():
            setattr(self, name, path)
        return in_paths, out_paths

    def create_dist_edges(self):
        edge_config = self.config["edges"]["dist"]
        threshold = edge_config["threshold"]
        required_cols = ["project_id", "latitude", "longitude"]
        missing_cols = [
            col
            for col in required_cols
            if col not in self.project_df.columns
        ]
        if missing_cols:
            raise KeyError(f"Missing columns for dist edges: {missing_cols}")

        project_df = self.project_df[required_cols].copy()
        project_df["latitude"] = pd.to_numeric(
            project_df["latitude"],
            errors="coerce",
        )
        project_df["longitude"] = pd.to_numeric(
            project_df["longitude"],
            errors="coerce",
        )
        project_df = project_df.dropna(subset=["latitude", "longitude"])
        project_df = self.add_project_node_ids(project_df)

        project_ids = project_df["project_id"].to_numpy()
        node_ids = project_df["node_id"].to_numpy(dtype=int)
        lat = project_df["latitude"].to_numpy(dtype=float)
        lon = project_df["longitude"].to_numpy(dtype=float)

        distance_m = self.pairwise_haversine_m(lat, lon, lat, lon)
        src_idx, dst_idx = np.where(
            (distance_m < threshold)
            & (np.triu(np.ones(distance_m.shape, dtype=bool), k=1))
        )

        edges = pd.DataFrame(
            {
                "node_id": node_ids[src_idx],
                "project_id": project_ids[src_idx],
                "neig_node_id": node_ids[dst_idx],
                "neig_project_id": project_ids[dst_idx],
                "distance_m": distance_m[src_idx, dst_idx],
            }
        )
        edges.to_csv(self.out_paths["dist"], index=False)
        logger.info(
            "Saved dist edges: path=%s threshold=%s rows=%s",
            self.out_paths["dist"],
            threshold,
            len(edges),
        )
    def create_same_mrt_edges(self):
        edge_config = self.config["edges"]["same_mrt"]
        threshold = edge_config["threshold"]
        project_required_cols = ["project_id", "latitude", "longitude"]
        mrt_required_cols = [
            "alphanumeric_code",
            "station_name_english",
            "latitude",
            "longitude",
        ]
        project_missing_cols = [
            col
            for col in project_required_cols
            if col not in self.project_df.columns
        ]
        if project_missing_cols:
            raise KeyError(
                f"Missing project columns for same_mrt edges: {project_missing_cols}"
            )
        mrt_missing_cols = [
            col
            for col in mrt_required_cols
            if col not in self.mrt_df.columns
        ]
        if mrt_missing_cols:
            raise KeyError(f"Missing MRT columns for same_mrt edges: {mrt_missing_cols}")

        project_df = self.project_df[project_required_cols].copy()
        project_df["latitude"] = pd.to_numeric(
            project_df["latitude"],
            errors="coerce",
        )
        project_df["longitude"] = pd.to_numeric(
            project_df["longitude"],
            errors="coerce",
        )
        project_df = project_df.dropna(subset=["latitude", "longitude"])
        project_df = self.add_project_node_ids(project_df)

        mrt_df = self.mrt_df[mrt_required_cols].copy()
        mrt_df["latitude"] = pd.to_numeric(mrt_df["latitude"], errors="coerce")
        mrt_df["longitude"] = pd.to_numeric(mrt_df["longitude"], errors="coerce")
        mrt_df = mrt_df.dropna(subset=["latitude", "longitude"])

        project_to_mrt_dist = self.pairwise_haversine_m(
            project_df["latitude"].to_numpy(dtype=float),
            project_df["longitude"].to_numpy(dtype=float),
            mrt_df["latitude"].to_numpy(dtype=float),
            mrt_df["longitude"].to_numpy(dtype=float),
        )

        edge_parts = []
        for mrt_idx, mrt_row in mrt_df.reset_index(drop=True).iterrows():
            project_idx = np.where(project_to_mrt_dist[:, mrt_idx] < threshold)[0]
            if len(project_idx) < 2:
                continue

            nearby_projects = project_df.iloc[project_idx]
            src_local, dst_local = np.triu_indices(len(nearby_projects), k=1)
            node_ids = nearby_projects["node_id"].to_numpy(dtype=int)
            project_ids = nearby_projects["project_id"].to_numpy()
            distances = project_to_mrt_dist[project_idx, mrt_idx]
            edge_parts.append(
                pd.DataFrame(
                    {
                        "node_id": node_ids[src_local],
                        "project_id": project_ids[src_local],
                        "neig_node_id": node_ids[dst_local],
                        "neig_project_id": project_ids[dst_local],
                        "mrt_code": mrt_row["alphanumeric_code"],
                        "mrt_name": mrt_row["station_name_english"],
                        "project_mrt_distance_m": distances[src_local],
                        "neig_project_mrt_distance_m": distances[dst_local],
                    }
                )
            )

        if edge_parts:
            edges = pd.concat(edge_parts, ignore_index=True)
        else:
            edges = pd.DataFrame(
                columns=[
                    "node_id",
                    "project_id",
                    "neig_node_id",
                    "neig_project_id",
                    "mrt_code",
                    "mrt_name",
                    "project_mrt_distance_m",
                    "neig_project_mrt_distance_m",
                ]
            )

        edges = edges.drop_duplicates(
            subset=["node_id", "neig_node_id", "mrt_code"],
        ).reset_index(drop=True)
        edges.to_csv(self.out_paths["same_mrt"], index=False)
        logger.info(
            "Saved same_mrt edges: path=%s threshold=%s rows=%s",
            self.out_paths["same_mrt"],
            threshold,
            len(edges),
        )
    def create_same_mrt_dist_edges(self):
        edge_config = self.config["edges"]["same_mrt_dist"]
        epsilon = edge_config["epsilon"]
        project_df, _, _ = self.prepare_project_mrt_distance_df()

        project_ids = project_df["project_id"].to_numpy()
        node_ids = project_df["node_id"].to_numpy(dtype=int)
        nearest_mrt_codes = project_df["nearest_mrt_code"].to_numpy()
        nearest_mrt_names = project_df["nearest_mrt_name"].to_numpy()
        nearest_mrt_dist = project_df["nearest_mrt_distance_m"].to_numpy(dtype=float)

        distance_diff = np.abs(
            nearest_mrt_dist[:, None] - nearest_mrt_dist[None, :]
        )
        src_idx, dst_idx = np.where(
            (distance_diff < epsilon)
            & (np.triu(np.ones(distance_diff.shape, dtype=bool), k=1))
        )

        edges = pd.DataFrame(
            {
                "node_id": node_ids[src_idx],
                "project_id": project_ids[src_idx],
                "neig_node_id": node_ids[dst_idx],
                "neig_project_id": project_ids[dst_idx],
                "nearest_mrt_code": nearest_mrt_codes[src_idx],
                "nearest_mrt_name": nearest_mrt_names[src_idx],
                "nearest_mrt_distance_m": nearest_mrt_dist[src_idx],
                "neig_nearest_mrt_code": nearest_mrt_codes[dst_idx],
                "neig_nearest_mrt_name": nearest_mrt_names[dst_idx],
                "neig_nearest_mrt_distance_m": nearest_mrt_dist[dst_idx],
                "nearest_mrt_distance_diff_m": distance_diff[src_idx, dst_idx],
            }
        )
        edges.to_csv(self.out_paths["same_mrt_dist"], index=False)
        logger.info(
            "Saved same_mrt_dist edges: path=%s epsilon=%s rows=%s",
            self.out_paths["same_mrt_dist"],
            epsilon,
            len(edges),
        )
    def create_adjacent_mrt_edges(self):
        edge_config = self.config["edges"]["adjacent_mrt"]
        transport_type = edge_config.get("transport_type", "MRT")
        project_required_cols = ["project_id", "latitude", "longitude"]
        mrt_required_cols = [
            "alphanumeric_code",
            "station_name_english",
            "line_english",
            "transport_type",
            "latitude",
            "longitude",
        ]
        project_missing_cols = [
            col for col in project_required_cols if col not in self.project_df.columns
        ]
        if project_missing_cols:
            raise KeyError(
                "Missing project columns for adjacent_mrt edges: "
                f"{project_missing_cols}"
            )
        mrt_missing_cols = [
            col for col in mrt_required_cols if col not in self.mrt_df.columns
        ]
        if mrt_missing_cols:
            raise KeyError(
                f"Missing MRT columns for adjacent_mrt edges: {mrt_missing_cols}"
            )

        project_df = self.project_df[project_required_cols].copy()
        project_df["latitude"] = pd.to_numeric(project_df["latitude"], errors="coerce")
        project_df["longitude"] = pd.to_numeric(project_df["longitude"], errors="coerce")
        project_df = project_df.dropna(subset=["latitude", "longitude"])
        project_df = self.add_project_node_ids(project_df).reset_index(drop=True)

        mrt_df = self.mrt_df[mrt_required_cols].copy()
        mrt_df["latitude"] = pd.to_numeric(mrt_df["latitude"], errors="coerce")
        mrt_df["longitude"] = pd.to_numeric(mrt_df["longitude"], errors="coerce")
        if transport_type:
            mrt_df = mrt_df.loc[mrt_df["transport_type"].eq(transport_type)]
        mrt_df = mrt_df.dropna(
            subset=[
                "alphanumeric_code",
                "station_name_english",
                "line_english",
                "latitude",
                "longitude",
            ]
        ).reset_index(drop=True)
        if mrt_df.empty:
            raise ValueError(
                "No rail stations have valid codes, names, lines, and coordinates "
                f"for transport_type={transport_type!r}."
            )

        # Collapse station-line rows to physical stations before finding each
        # condo's nearest station. The full rows below still preserve every
        # line served by interchange stations.
        physical_stations = (
            mrt_df.groupby("station_name_english", as_index=False)
            .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
            .sort_values("station_name_english")
            .reset_index(drop=True)
        )
        distance_m = self.pairwise_haversine_m(
            project_df["latitude"].to_numpy(dtype=float),
            project_df["longitude"].to_numpy(dtype=float),
            physical_stations["latitude"].to_numpy(dtype=float),
            physical_stations["longitude"].to_numpy(dtype=float),
        )
        nearest_station_idx = distance_m.argmin(axis=1)
        nearest_station_names = physical_stations.loc[
            nearest_station_idx, "station_name_english"
        ].to_numpy()
        nearest_station_distances = distance_m[
            np.arange(len(project_df)), nearest_station_idx
        ]
        pairs = adjacent_mrt_pairs(
            nearest_station_names,
            mrt_df["station_name_english"],
            mrt_df["alphanumeric_code"],
            mrt_df["line_english"],
        )

        lines_by_station = (
            mrt_df.groupby("station_name_english")["line_english"]
            .agg(lambda values: "|".join(sorted(set(map(str, values)))))
            .to_dict()
        )
        codes_by_station = (
            mrt_df.groupby("station_name_english")["alphanumeric_code"]
            .agg(lambda values: "|".join(sorted(set(map(str, values)))))
            .to_dict()
        )
        project_ids = project_df["project_id"].to_numpy()
        node_ids = project_df["node_id"].to_numpy(dtype=int)

        if pairs:
            src_idx = np.asarray([pair[0] for pair in pairs], dtype=int)
            dst_idx = np.asarray([pair[1] for pair in pairs], dtype=int)
            adjacent_codes = [pair[2] for pair in pairs]
            edges = pd.DataFrame(
                {
                    "node_id": node_ids[src_idx],
                    "project_id": project_ids[src_idx],
                    "neig_node_id": node_ids[dst_idx],
                    "neig_project_id": project_ids[dst_idx],
                    "nearest_mrt_name": nearest_station_names[src_idx],
                    "nearest_mrt_codes": [
                        codes_by_station[name] for name in nearest_station_names[src_idx]
                    ],
                    "nearest_mrt_lines": [
                        lines_by_station[name] for name in nearest_station_names[src_idx]
                    ],
                    "nearest_mrt_distance_m": nearest_station_distances[src_idx],
                    "neig_nearest_mrt_name": nearest_station_names[dst_idx],
                    "neig_nearest_mrt_codes": [
                        codes_by_station[name] for name in nearest_station_names[dst_idx]
                    ],
                    "neig_nearest_mrt_lines": [
                        lines_by_station[name] for name in nearest_station_names[dst_idx]
                    ],
                    "neig_nearest_mrt_distance_m": nearest_station_distances[dst_idx],
                    "adjacent_mrt_codes": adjacent_codes,
                }
            )
        else:
            edges = pd.DataFrame(
                columns=[
                    "node_id",
                    "project_id",
                    "neig_node_id",
                    "neig_project_id",
                    "nearest_mrt_name",
                    "nearest_mrt_codes",
                    "nearest_mrt_lines",
                    "nearest_mrt_distance_m",
                    "neig_nearest_mrt_name",
                    "neig_nearest_mrt_codes",
                    "neig_nearest_mrt_lines",
                    "neig_nearest_mrt_distance_m",
                    "adjacent_mrt_codes",
                ]
            )

        edges.to_csv(self.out_paths["adjacent_mrt"], index=False)
        logger.info(
            "Saved adjacent_mrt edges: path=%s transport_type=%s rows=%s",
            self.out_paths["adjacent_mrt"],
            transport_type,
            len(edges),
        )
    def create_same_school_dist_edges(self):
        edge_config = self.config["edges"]["same_school_dist"]
        epsilon = edge_config["epsilon"]
        school_dist_col = "nearest_school_dist"
        required_cols = ["project_id", school_dist_col]
        missing_cols = [
            col
            for col in required_cols
            if col not in self.project_df.columns
        ]
        if missing_cols:
            raise KeyError(
                f"Missing columns for same_school_dist edges: {missing_cols}"
            )

        project_df = self.project_df[required_cols].copy()
        project_df[school_dist_col] = pd.to_numeric(
            project_df[school_dist_col],
            errors="coerce",
        )
        project_df = project_df.dropna(subset=[school_dist_col])
        project_df = self.add_project_node_ids(project_df)

        project_ids = project_df["project_id"].to_numpy()
        node_ids = project_df["node_id"].to_numpy(dtype=int)
        nearest_school_dist = project_df[school_dist_col].to_numpy(dtype=float)

        distance_diff = np.abs(
            nearest_school_dist[:, None] - nearest_school_dist[None, :]
        )
        src_idx, dst_idx = np.where(
            (distance_diff < epsilon)
            & (np.triu(np.ones(distance_diff.shape, dtype=bool), k=1))
        )

        edges = pd.DataFrame(
            {
                "node_id": node_ids[src_idx],
                "project_id": project_ids[src_idx],
                "neig_node_id": node_ids[dst_idx],
                "neig_project_id": project_ids[dst_idx],
                "nearest_school_dist": nearest_school_dist[src_idx],
                "neig_nearest_school_dist": nearest_school_dist[dst_idx],
                "nearest_school_dist_diff": distance_diff[src_idx, dst_idx],
            }
        )
        edges.to_csv(self.out_paths["same_school_dist"], index=False)
        logger.info(
            "Saved same_school_dist edges: path=%s epsilon=%s rows=%s",
            self.out_paths["same_school_dist"],
            epsilon,
            len(edges),
        )
    def create_same_age_edges(self):
        age_col = self.config["edges"]["same_age"].get("age_column")
        if age_col is None:
            # Older graph databases genuinely use Condo_Age_2026. New
            # transaction-age datasets configure their original column name
            # explicitly, without renaming it downstream.
            age_col = (
                "condo_age_at_transaction"
                if "condo_age_at_transaction" in self.project_df.columns
                else "Condo_Age_2026"
            )
        elif age_col not in self.project_df.columns and age_col == "condo_age_at_transaction":
            # Compatibility for already-generated CCR databases. A clean
            # rebuild writes the original name and no longer takes this path.
            if "Condo_Age_2026" in self.project_df.columns:
                logger.warning(
                    "Using legacy generated age column Condo_Age_2026; rebuild "
                    "the graph database to retain condo_age_at_transaction."
                )
                age_col = "Condo_Age_2026"
        required_cols = ["project_id", age_col]
        missing_cols = [
            col
            for col in required_cols
            if col not in self.project_df.columns
        ]
        if missing_cols:
            raise KeyError(f"Missing columns for same_age edges: {missing_cols}")

        project_df = self.project_df[required_cols].copy()
        project_df[age_col] = pd.to_numeric(project_df[age_col], errors="coerce")
        project_df = project_df.dropna(subset=[age_col])
        project_df = self.add_project_node_ids(project_df)

        edge_parts = []
        for age, age_df in project_df.groupby(age_col, dropna=False):
            if len(age_df) < 2:
                continue

            node_ids = age_df["node_id"].to_numpy(dtype=int)
            project_ids = age_df["project_id"].to_numpy()
            src_idx, dst_idx = np.triu_indices(len(age_df), k=1)
            edge_parts.append(
                pd.DataFrame(
                    {
                        "node_id": node_ids[src_idx],
                        "project_id": project_ids[src_idx],
                        "neig_node_id": node_ids[dst_idx],
                        "neig_project_id": project_ids[dst_idx],
                        age_col: age,
                    }
                )
            )

        if edge_parts:
            edges = pd.concat(edge_parts, ignore_index=True)
        else:
            edges = pd.DataFrame(
                columns=[
                    "node_id",
                    "project_id",
                    "neig_node_id",
                    "neig_project_id",
                    age_col,
                ]
            )

        edges.to_csv(self.out_paths["same_age"], index=False)
        logger.info(
            "Saved same_age edges: path=%s rows=%s",
            self.out_paths["same_age"],
            len(edges),
        )
    def create_same_planning_area_edges(self):
        planning_area_col = "Planning Area"
        required_cols = ["project_id", planning_area_col]
        missing_cols = [
            col
            for col in required_cols
            if col not in self.project_df.columns
        ]
        if missing_cols:
            raise KeyError(
                f"Missing columns for same_planning_area edges: {missing_cols}"
            )

        project_df = self.project_df[required_cols].copy()
        project_df = project_df.dropna(subset=[planning_area_col])
        project_df = self.add_project_node_ids(project_df)

        edge_parts = []
        for planning_area, area_df in project_df.groupby(
            planning_area_col,
            dropna=False,
        ):
            if len(area_df) < 2:
                continue

            node_ids = area_df["node_id"].to_numpy(dtype=int)
            project_ids = area_df["project_id"].to_numpy()
            src_idx, dst_idx = np.triu_indices(len(area_df), k=1)
            edge_parts.append(
                pd.DataFrame(
                    {
                        "node_id": node_ids[src_idx],
                        "project_id": project_ids[src_idx],
                        "neig_node_id": node_ids[dst_idx],
                        "neig_project_id": project_ids[dst_idx],
                        planning_area_col: planning_area,
                    }
                )
            )

        if edge_parts:
            edges = pd.concat(edge_parts, ignore_index=True)
        else:
            edges = pd.DataFrame(
                columns=[
                    "node_id",
                    "project_id",
                    "neig_node_id",
                    "neig_project_id",
                    planning_area_col,
                ]
            )

        edges.to_csv(self.out_paths["same_planning_area"], index=False)
        logger.info(
            "Saved same_planning_area edges: path=%s rows=%s",
            self.out_paths["same_planning_area"],
            len(edges),
        )

    def create_size_project_edges(self):
        required_cols = ["project_id", "size_tier", "node_id"]
        missing_cols = [
            col
            for col in required_cols
            if col not in self.node_id_df.columns
        ]
        if missing_cols:
            raise KeyError(f"Missing columns for size_project edges: {missing_cols}")

        node_df = self.node_id_df[required_cols].copy()
        node_df = node_df.dropna(subset=required_cols)
        node_df["node_id"] = node_df["node_id"].astype(int)

        edge_parts = []
        for project_id, project_nodes in node_df.groupby("project_id", dropna=False):
            project_level_nodes = project_nodes[project_nodes["size_tier"].eq(0)]
            size_level_nodes = project_nodes[~project_nodes["size_tier"].eq(0)]
            if project_level_nodes.empty or size_level_nodes.empty:
                continue

            if len(project_level_nodes) > 1:
                raise ValueError(
                    "Expected one project-level node with size_tier=0 for "
                    f"project_id={project_id}, got {len(project_level_nodes)}"
                )

            project_node = project_level_nodes.iloc[0]
            edge_parts.append(
                pd.DataFrame(
                    {
                        "node_id": size_level_nodes["node_id"].to_numpy(dtype=int),
                        "project_id": size_level_nodes["project_id"].to_numpy(),
                        "size_tier": size_level_nodes["size_tier"].to_numpy(),
                        "neig_node_id": int(project_node["node_id"]),
                        "neig_project_id": project_id,
                        "neig_size_tier": project_node["size_tier"],
                    }
                )
            )

        if edge_parts:
            edges = pd.concat(edge_parts, ignore_index=True)
        else:
            edges = pd.DataFrame(
                columns=[
                    "node_id",
                    "project_id",
                    "size_tier",
                    "neig_node_id",
                    "neig_project_id",
                    "neig_size_tier",
                ]
            )

        edges.to_csv(self.out_paths["size_project"], index=False)
        logger.info(
            "Saved size_project edges: path=%s rows=%s",
            self.out_paths["size_project"],
            len(edges),
        )

    def process(self):
        for edge_key, edge_config in self.config["edges"].items():
            if not isinstance(edge_config, dict):
                continue
            if not edge_config["enable"]:
                logger.info("Skipping disabled edge: %s", edge_key)
                continue

            out_path = self.out_paths[edge_key]
            if out_path.exists():
                logger.info("Skipping existing edge: %s path=%s", edge_key, out_path)
                continue

            create_edges = getattr(self, f"create_{edge_key}_edges", None)
            if create_edges is None:
                raise AttributeError(
                    f"Missing edge creation function: create_{edge_key}_edges"
                )

            logger.info("Creating edge: %s path=%s", edge_key, out_path)
            create_edges()

    def load_enabled_edge_dfs(self):
        edge_dfs = {}
        for edge_key, edge_config in self.config["edges"].items():
            if not isinstance(edge_config, dict):
                continue
            if not edge_config["enable"]:
                continue

            out_path = self.out_paths[edge_key]
            if not out_path.exists():
                logger.warning(
                    "Enabled edge file does not exist: %s path=%s",
                    edge_key,
                    out_path,
                )
                continue

            edge_df = pd.read_csv(out_path)
            edge_dfs[edge_key] = edge_df
            logger.info(
                "Loaded edge: %s path=%s rows=%s",
                edge_key,
                out_path,
                len(edge_df),
            )

        return edge_dfs
    
@hydra.main(config_path="../config/graph", config_name="rent", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = OmegaConf.to_container(cfg, resolve=True)
    logger.info("Loaded Hydra graph config: name=%s", config.get("name"))
    
    edge_processor = FormEdgeProcessor(config)
    edge_processor.process()


if __name__ == "__main__":
    main()
