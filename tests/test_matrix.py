"""Vectorized distances, cross distances, and related helpers (0.9.0)."""

import random
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import taxodist
from taxodist.distance import _compute_distance, _distance_block, _prefix_node_ids


def random_lineages(n, depth=8, branching=3, seed=1):
    rng = random.Random(seed)
    lineages = {}
    for i in range(n):
        length = rng.randint(2, depth)
        path = [rng.randint(1, branching) for _ in range(length)]
        lineages[f"t{i + 1}"] = ["Biota"] + [
            "N" + "".join(map(str, path[:d + 1])) for d in range(length)
        ]
    return lineages


def test_vectorized_matrix_matches_pairwise_compute_distance():
    lineages = random_lineages(40)
    lineages["dup"] = list(lineages["t1"])
    lineages["ancestor"] = lineages["t2"][:2]
    lineages["foreign"] = ["Natura", "Mineralia"]
    matrix = taxodist.distance_matrix(
        taxodist.taxo_from_lineages(lineages), progress=False
    )
    names = list(lineages)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            expected = _compute_distance(lineages[a], lineages[b])["distance"]
            assert matrix.iloc[i, j] == expected


def test_prefix_ids_and_blocks_handle_empty_and_missing_lineages():
    ids, lengths = _prefix_node_ids([])
    assert ids.shape == (0, 0) and len(lengths) == 0
    assert np.isinf(_distance_block([[]], [[]])[0, 0])
    assert np.isnan(_distance_block([None], [["Biota"]])[0, 0])
    assert _distance_block([], []).shape == (0, 0)


def test_distance_matrix_rejects_a_single_string_and_tolerates_missing_names():
    with pytest.raises(TypeError, match="single string"):
        taxodist.distance_matrix("Alpha")
    with patch(
        "taxodist.distance.get_lineage",
        side_effect=lambda taxon, **kwargs: ["Biota", taxon],
    ):
        matrix = taxodist.distance_matrix(["Alpha", None, float("nan"), "Beta"], progress=False)
    assert np.isnan(matrix.iloc[0, 1]) and np.isnan(matrix.iloc[0, 2])
    assert matrix.iloc[0, 3] == 1
    assert matrix.iloc[1, 1] == 0


def test_distance_matrix_rejects_malformed_resolution():
    resolution = taxodist.taxo_from_lineages({"A": ["Biota", "A"]})
    broken = resolution.drop(columns=["lineage"])
    with pytest.raises(ValueError, match="Invalid TaxodistResolution"):
        taxodist.distance_matrix(broken)


def test_cross_distances_is_the_rectangular_block_of_distance_matrix():
    lineages = random_lineages(12, seed=7)
    names = list(lineages)
    x = taxodist.taxo_from_lineages({k: lineages[k] for k in names[:5]})
    y = taxodist.taxo_from_lineages({k: lineages[k] for k in names[5:]})
    full = taxodist.distance_matrix(taxodist.taxo_from_lineages(lineages), progress=False)

    cross = taxodist.cross_distances(x, y, progress=False)
    assert cross.shape == (5, 7)
    pd.testing.assert_frame_equal(cross, full.loc[names[:5], names[5:]])


def test_cross_distances_with_names_bundles_and_missing_taxa(capsys):
    with patch(
        "taxodist.distance.get_lineage",
        side_effect=lambda taxon, **kwargs: None if taxon == "Gone" else ["Biota", "Animalia", taxon],
    ):
        result = taxodist.cross_distances(["Alpha", "Gone"], ["Beta", "Alpha"], progress=True)
    assert list(result.loc["Alpha"]) == [0.5, 0.0]
    assert result.loc["Gone"].isna().all()
    assert "Fetching" in capsys.readouterr().out

    bundle = taxodist.taxo_bundle(
        taxodist.taxo_from_lineages({"A": ["Biota", "A"], "B": ["Biota", "B"]}),
        progress=False,
    )
    assert taxodist.cross_distances(bundle, bundle, progress=False).loc["A", "B"] == 1
    assert taxodist.cross_distances([], bundle, progress=False).shape == (0, 2)


def test_focal_distances_reports_the_lineage_node_for_identical_taxa():
    with patch(
        "taxodist.distance.get_lineage",
        return_value=["Biota", "Dinosauria", "Tyrannosaurus"],
    ):
        result = taxodist.focal_distances("tyrannosaurus", ["Tyrannosaurus"], progress=False)
    assert result["distance"].iloc[0] == 0
    assert result["mrca"].iloc[0] == "Tyrannosaurus"
    assert result["mrca_depth"].iloc[0] == 3


@patch("taxodist.utils.sns.clustermap")
@patch("matplotlib.pyplot.show")
def test_taxo_heatmap_accepts_a_resolution(mock_show, mock_sns):
    resolution = taxodist.taxo_from_lineages({
        "A": ["Biota", "X", "A"], "B": ["Biota", "X", "B"],
    })
    result = taxodist.taxo_heatmap(resolution, progress=False)
    assert result.loc["A", "B"] == 0.5
    mock_sns.assert_called_once()
    assert mock_sns.call_args.kwargs == {"cmap": "YlGnBu_r", "annot": True}
