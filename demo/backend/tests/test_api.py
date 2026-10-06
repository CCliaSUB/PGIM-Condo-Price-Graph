"""Lightweight endpoint tests against the real artifacts.

Run from demo/backend:  python -m pytest -q
Datasets whose artifacts are missing are skipped, not failed.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("PGIM_SKIP_WARMUP", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as api  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(api.app)
AVAILABLE = [key for key, ds in api.DATASETS.items() if api.dataset_available(ds)["available"]]
SNAPSHOT = [key for key in AVAILABLE if api.DATASETS[key].graph_mode == "snapshot"]


def nodes(ds_id):
    return pd.read_csv(api.DATASETS[ds_id].data_root / "node_id.csv")


def project_with_predictions(ds_id) -> int:
    counts = api.prediction_table(ds_id, 12).groupby("project_id")["size_tier"].nunique()
    return int(counts.sort_values(ascending=False, kind="mergesort").index[0])  # most room-size tiers


def project_without_predictions(ds_id):
    table = api.project_table(ds_id)
    missing = table.loc[~table["has_predictions"]]
    return None if missing.empty else int(missing.index[0])


@pytest.fixture(params=AVAILABLE)
def ds(request):
    return request.param


def test_health_and_datasets():
    assert client.get("/health").json()["status"] == "ok"
    body = client.get("/datasets").json()
    assert {d["id"] for d in body["datasets"]} == set(api.DATASETS)
    assert body["default"] in AVAILABLE
    assert client.get("/metadata", params={"dataset": "nope"}).status_code == 400


def test_metadata(ds):
    meta = client.get("/metadata", params={"dataset": ds}).json()
    assert meta["dataset"] == ds and meta["horizons"] == [1, 12, 24]
    assert {r["id"] for r in meta["relations"]} == set(api.DATASETS[ds].relations)
    assert all(r["color"] and "dash" in r for r in meta["relations"])
    assert bool(meta["months"]) == (api.DATASETS[ds].graph_mode == "snapshot")


def test_projects_one_row_each_and_json_safe(ds):
    body = client.get("/projects", params={"dataset": ds}).json()
    ids = [p["project_id"] for p in body["projects"]]
    assert len(ids) == len(set(ids)) == int((nodes(ds)["size_tier"] == 0).sum())
    json.dumps(body, allow_nan=False)
    assert all(p["planning_area"] != "nan" for p in body["projects"])


def test_project_search_and_stable_order(ds):
    name = client.get("/projects", params={"dataset": ds}).json()["projects"][0]["name"]
    query = name.split()[0].lower()
    first = client.get("/projects", params={"dataset": ds, "q": query}).json()["projects"]
    assert first and all(query in p["name"].lower() for p in first)
    assert first == client.get("/projects", params={"dataset": ds, "q": query}).json()["projects"]


def test_graph_edges_unique_and_consistent(ds):
    pid = project_with_predictions(ds)
    body = client.get(f"/graph/{pid}", params={"dataset": ds}).json()
    keys = [(e["source"], e["target"], e["relation"]) for e in body["edges"]]
    assert len(keys) == len(set(keys))
    assert all(e["source"] == pid and e["target"] != pid for e in body["edges"])
    assert {e["target"] for e in body["edges"]} == {n["project_id"] for n in body["nodes"]}
    for item in body["relation_summary"]:
        shown = sum(1 for e in body["edges"] if e["relation"] == item["id"])
        assert shown == item["neighbours_shown"] <= item["neighbours_total"]
    assert body["selected"]["node_id"] == api.project_node_maps(ds)[0][pid]
    expected = sorted(nodes(ds).query("project_id == @pid and size_tier > 0")["size_tier"])
    assert [n["size_tier"] for n in body["size_nodes"]] == expected
    assert body == client.get(f"/graph/{pid}", params={"dataset": ds}).json()


def test_empty_relations_means_none(ds):
    pid = project_with_predictions(ds)
    body = client.get(f"/graph/{pid}", params={"dataset": ds, "relations": ""}).json()
    assert body["edges"] == [] and body["nodes"] == [] and body["relations"] == []
    assert body["size_nodes"]


def test_graph_errors(ds):
    pid = project_with_predictions(ds)
    assert client.get(f"/graph/{pid}", params={"dataset": ds, "relations": "bogus"}).status_code == 400
    assert client.get("/graph/999999", params={"dataset": ds}).status_code == 404
    assert client.get("/graph/abc", params={"dataset": ds}).status_code == 422


@pytest.mark.parametrize("ds_id", SNAPSHOT)
def test_snapshot_graph_matches_file(ds_id):
    """Neighbours come from the exported monthly snapshot the model consumed."""
    pid = project_with_predictions(ds_id)
    months = client.get("/metadata", params={"dataset": ds_id}).json()["months"]
    month = months[len(months) // 2]
    body = client.get(f"/graph/{pid}", params={"dataset": ds_id, "month": month, "limit_per_relation": 200}).json()
    assert body["month"] == month
    snap = pd.read_csv(api.DATASETS[ds_id].snapshot_path)
    snap = snap.loc[snap["date"].eq(month)]
    lookup = nodes(ds_id).set_index("node_id")["project_id"]
    size_ids = set(nodes(ds_id).query("project_id == @pid and size_tier > 0")["node_id"])
    sp = snap.loc[snap["relation"].eq("similar_price") & (snap["node_id"].isin(size_ids) | snap["neig_node_id"].isin(size_ids))]
    expected = set(lookup[sp["node_id"]]) | set(lookup[sp["neig_node_id"]])
    expected.discard(pid)
    shown = {e["target"] for e in body["edges"] if e["relation"] == "similar_price"}
    total = next(r for r in body["relation_summary"] if r["id"] == "similar_price")["neighbours_total"]
    assert total == len(expected) and shown <= expected
    assert client.get(f"/graph/{pid}", params={"dataset": ds_id, "month": "1990-01-01"}).status_code == 400


@pytest.mark.parametrize("horizon", [1, 12, 24])
def test_predictions_match_artifact(ds, horizon):
    pid = project_with_predictions(ds)
    body = client.get(f"/predictions/{pid}", params={"dataset": ds, "horizon": horizon}).json()
    prov = body["provenance"]
    assert body["horizon"] == horizon and prov["horizon_months"] == horizon and prov["dataset"] == ds
    assert prov["artifact"] == api._rel(api.DATASETS[ds].prediction_path(horizon))
    raw = pd.read_csv(api.DATASETS[ds].prediction_path(horizon))
    raw = raw.loc[raw["project_id"].eq(pid)]
    assert sorted(s["node_id"] for s in body["series"]) == sorted(nodes(ds).query("project_id == @pid and size_tier > 0")["node_id"])
    for series in body["series"]:
        expected = raw.loc[raw["node_id"].eq(series["node_id"])].sort_values("date")
        assert [p["date"] for p in series["points"]] == expected["date"].tolist()
        if len(expected):
            assert series["points"][0]["prediction"] == pytest.approx(expected["prediction"].iloc[0], abs=1e-3)
            assert (expected["size_tier"] == series["size_tier"]).all()
    json.dumps(body, allow_nan=False)


def test_project_without_predictions(ds):
    pid = project_without_predictions(ds)
    if pid is None:
        pytest.skip("every project has predictions")
    body = client.get(f"/predictions/{pid}", params={"dataset": ds}).json()
    assert body["has_predictions"] is False
    assert all(s["points"] == [] and s["last_evaluated"] is None for s in body["series"])
    assert client.get(f"/graph/{pid}", params={"dataset": ds}).status_code == 200


def test_prediction_errors(ds):
    pid = project_with_predictions(ds)
    assert client.get(f"/predictions/{pid}", params={"dataset": ds, "horizon": 6}).status_code == 400
    assert client.get("/predictions/999999", params={"dataset": ds}).status_code == 404


def test_pct_error_is_null_safe(ds):
    assert api.json_safe({"x": float("nan")}) == {"x": None}
    body = client.get(f"/predictions/{project_with_predictions(ds)}", params={"dataset": ds}).json()
    for series in body["series"]:
        for p in series["points"]:
            if p["actual"] in (None, 0):
                assert p["pct_error"] is None
