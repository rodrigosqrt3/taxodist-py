"""Vectorized distances, cross distances, and related helpers (0.9.0)."""

import random
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import taxodist
from taxodist.distance import (
    _common_prefix_depth,
    _compute_distance,
    _distance_block,
    _metric_values,
    _mrca_depths_from,
    _mrca_index,
    _pairwise_block,
    _prefix_node_ids,
)


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


def _edge_case_lineages(seed=20261124, n_random=40):
    rng = random.Random(seed)

    def random_lineage():
        depth = rng.randint(1, 12)
        return [f"n{d + 1}_{rng.randint(0, 1)}" for d in range(depth)]

    return [random_lineage() for _ in range(n_random)] + [
        None,                                     # unresolved taxon
        [],                                       # empty lineage
        ["n1_0", "n2_0"],                         # ancestor of others
        ["n1_0", "n2_0"],                         # duplicate
        ["Other", "root"],                        # disconnected hierarchy
        ["n1_0", "n2_0", "n3_1", "n4_0", "n5_1"],
    ]


def _brute_force(a, b, metric):
    if a is None or b is None:
        return float("nan")
    return float(
        _metric_values(_common_prefix_depth(a, b), len(a), len(b), metric)
    )


@pytest.mark.parametrize("metric", taxodist.distance.METRICS)
def test_sorted_prefix_mrca_index_agrees_with_pairwise_comparison(metric):
    # Mirrors R's "sorted-prefix MRCA index agrees with pairwise comparison".
    # For "ranked" the block functions receive already reduced lineages, so
    # the brute force applies the same formula to the same lineages.
    lineages = _edge_case_lineages()
    n = len(lineages)
    fast = _pairwise_block(lineages, metric)
    for i in range(n):
        for j in range(n):
            expected = _brute_force(lineages[i], lineages[j], metric)
            np.testing.assert_equal(fast[i, j], expected, err_msg=f"{metric} {i} {j}")
    np.testing.assert_array_equal(fast, fast.T)

    index = _mrca_index(*_prefix_node_ids(lineages))
    for j in (0, 19, n - 2, n - 1, n - 6):
        expected = [
            0 if lineage is None or lineages[j] is None
            else _common_prefix_depth(lineage, lineages[j])
            for lineage in lineages
        ]
        assert list(_mrca_depths_from(index, j)) == expected


@pytest.mark.parametrize("seed", range(5))
def test_distance_matrix_equals_brute_force_on_random_resolutions(seed):
    rng = random.Random(seed)
    lineages = {}
    ranks = {}
    for i in range(60):
        depth = rng.randint(1, 10)
        root = "Biota" if rng.random() < 0.9 else "Natura"
        lineage = [root] + [f"d{d}_{rng.randint(0, 2)}" for d in range(depth)]
        lineages[f"t{i}"] = lineage
        ranks[f"t{i}"] = [rng.choice(["genus", "family", None, "clade"]) for _ in lineage[:-1]]
        ranks[f"t{i}"].append("species")
    lineages["copy"] = list(lineages["t0"])
    ranks["copy"] = list(ranks["t0"])
    resolution = taxodist.taxo_from_lineages(lineages, ranks=ranks)
    names = list(lineages)
    for metric in taxodist.distance.METRICS:
        matrix = taxodist.distance_matrix(resolution, progress=False, metric=metric)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                expected = 0.0 if i == j else _compute_distance(
                    lineages[a], lineages[b], metric=metric,
                    ranks_a=ranks[a], ranks_b=ranks[b],
                )["distance"]
                np.testing.assert_equal(matrix.iloc[i, j], expected)


def test_cross_distances_uses_the_same_index_as_distance_matrix():
    rng = random.Random(7)
    lineages = {}
    for i in range(25):
        nodes = [f"c{d + 1}{rng.randint(0, 1)}" for d in range(8)]
        lineages[f"t{i + 1}"] = nodes[:rng.randint(2, 8)]
    names = list(lineages)
    full = taxodist.distance_matrix(
        taxodist.taxo_from_lineages(lineages), progress=False, metric="wu_palmer"
    )
    x = taxodist.taxo_from_lineages({k: lineages[k] for k in names[:10]})
    y = taxodist.taxo_from_lineages({k: lineages[k] for k in names[10:]})
    cross = taxodist.cross_distances(x, y, progress=False, metric="wu_palmer")
    np.testing.assert_array_equal(cross.values, full.values[:10, 10:])

    edge = _edge_case_lineages(seed=3, n_random=10)
    block = _distance_block(edge[:8], edge[8:], "path")
    for i, a in enumerate(edge[:8]):
        for j, b in enumerate(edge[8:]):
            np.testing.assert_equal(block[i, j], _brute_force(a, b, "path"))


def test_mrca_index_handles_degenerate_inputs():
    empty = _mrca_index(*_prefix_node_ids([]))
    assert empty["n"] == 0 and len(empty["adj"]) == 0
    single = _mrca_index(*_prefix_node_ids([["a", "b"]]))
    assert list(_mrca_depths_from(single, 0)) == [2]
    flat = _mrca_index(*_prefix_node_ids([[], []]))
    assert list(_mrca_depths_from(flat, 0)) == [0, 0]
    assert _pairwise_block([]).shape == (0, 0)
    assert _distance_block([["a"]], []).shape == (1, 0)
    assert np.isnan(_pairwise_block([None, None])).all()


@patch("seaborn.clustermap")
@patch("matplotlib.pyplot.show")
def test_taxo_heatmap_accepts_a_resolution(mock_show, mock_sns):
    resolution = taxodist.taxo_from_lineages({
        "A": ["Biota", "X", "A"], "B": ["Biota", "X", "B"],
    })
    result = taxodist.taxo_heatmap(resolution, progress=False)
    assert result.loc["A", "B"] == 0.5
    mock_sns.assert_called_once()
    assert mock_sns.call_args.kwargs == {"cmap": "YlGnBu_r", "annot": True}
