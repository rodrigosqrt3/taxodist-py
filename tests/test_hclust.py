"""taxo_cluster() reproduces R's stats::hclust() exactly (1.1.0)."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import dendrogram, is_valid_linkage, leaves_list

import taxodist
from taxodist._hclust import HCLUST_METHODS, match_hclust_method, merge_to_linkage, r_hclust

FIXTURE = Path(__file__).parent / "fixtures" / "hclust_r.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def _matrix(case):
    return np.array([[float.fromhex(value) for value in row] for row in case["dist"]])


def _reference(result):
    # jsonlite writes length-one vectors as scalars.
    merge = np.array(result["merge"], dtype=np.int64).reshape(-1, 2)
    height = np.array([float.fromhex(h) for h in np.atleast_1d(result["height"])])
    order = np.atleast_1d(np.array(result["order"], dtype=np.int64))
    return merge, height, order


def test_fixture_covers_every_method_and_tied_distances():
    assert len(CASES) >= 10
    assert all(set(case["results"]) == set(HCLUST_METHODS) for case in CASES)
    tied = [case for case in CASES if len(np.unique(_matrix(case))) < 6]
    assert tied


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
@pytest.mark.parametrize("method", HCLUST_METHODS)
def test_merge_height_and_order_equal_r(case, method):
    matrix = _matrix(case)
    labels = case["labels"]
    expected_merge, expected_height, expected_order = _reference(case["results"][method])

    result = taxodist.taxo_cluster(
        pd.DataFrame(matrix, index=labels, columns=labels), method=method
    )
    np.testing.assert_array_equal(result["merge"], expected_merge)
    np.testing.assert_array_equal(result["height"], expected_height)
    np.testing.assert_array_equal(result["order"], expected_order)
    assert result["method"] == method
    assert result["labels"] == labels

    # The SciPy linkage describes the same tree, with R's leaf order.
    linkage = result["hclust"]
    assert is_valid_linkage(linkage)
    np.testing.assert_array_equal(linkage[:, 2], expected_height)
    assert linkage[-1, 3] == len(labels)
    np.testing.assert_array_equal(leaves_list(linkage), expected_order - 1)
    assert dendrogram(linkage, no_plot=True)["leaves"] == list(expected_order - 1)


@pytest.mark.parametrize(
    "case", [case for case in CASES if case["lineages"]],
    ids=[case["name"] for case in CASES if case["lineages"]],
)
def test_distance_matrix_equals_r_on_the_fixture_lineages(case):
    resolution = taxodist.taxo_from_lineages(case["lineages"])
    matrix = taxodist.distance_matrix(resolution, progress=False, metric=case["metric"])
    np.testing.assert_array_equal(matrix.to_numpy(), _matrix(case))

    cluster = taxodist.taxo_cluster(resolution, progress=False, metric=case["metric"])
    _, height, order = _reference(case["results"]["average"])
    np.testing.assert_array_equal(cluster["height"], height)
    np.testing.assert_array_equal(cluster["order"], order)


def test_method_names_are_matched_as_in_r(capsys):
    assert match_hclust_method("ave") == "average"
    assert match_hclust_method("ward.D2") == "ward.D2"
    assert match_hclust_method("mc") == "mcquitty"
    assert match_hclust_method("ward") == "ward.D"
    assert 'renamed to "ward.D"' in capsys.readouterr().err
    for invalid in ("c", "ward.", "", "upgma"):
        with pytest.raises(ValueError, match="invalid clustering method"):
            match_hclust_method(invalid)
    with pytest.raises(TypeError, match="single string"):
        match_hclust_method(["average"])


def test_taxo_cluster_reports_the_full_method_name_and_rejects_bad_input():
    d = pd.DataFrame(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [2.0, 2.0, 0.0]],
        index=list("ABC"), columns=list("ABC"),
    )
    assert taxodist.taxo_cluster(d, method="cen")["method"] == "centroid"
    with pytest.raises(ValueError, match="invalid clustering method"):
        taxodist.taxo_cluster(d, method="nope")
    with pytest.raises(ValueError, match="square"):
        r_hclust(np.zeros(3))
    with pytest.raises(ValueError, match="square"):
        r_hclust(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="n >= 2"):
        r_hclust(np.zeros((1, 1)))


def test_skipped_clustering_keeps_the_new_fields():
    d = pd.DataFrame([[0.0]], index=["A"], columns=["A"])
    with pytest.warns(UserWarning, match="At least two taxa"):
        result = taxodist.taxo_cluster(d, method="single")
    assert result["hclust"] is None and result["merge"] is None
    assert result["height"] is None and result["order"] is None
    assert result["labels"] == ["A"] and result["method"] == "single"


def test_ties_follow_r_rather_than_scipy():
    # Every distance tied: R merges (1, 2), then 3, 4, ... in turn.
    d = np.full((5, 5), 2.0)
    np.fill_diagonal(d, 0.0)
    result = r_hclust(d, "single")
    np.testing.assert_array_equal(
        result["merge"], [[-1, -2], [-3, 1], [-4, 2], [-5, 3]]
    )
    np.testing.assert_array_equal(result["order"], [5, 4, 3, 1, 2])


def test_merge_to_linkage_numbers_clusters_like_scipy():
    linkage = merge_to_linkage([[-1, -2], [-3, -4], [1, 2]], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        linkage, [[0, 1, 1, 2], [2, 3, 2, 2], [4, 5, 3, 4]]
    )


@patch("matplotlib.pyplot.show")
def test_dendrogram_of_a_non_monotone_clustering_can_be_plotted(mock_show):
    import matplotlib.pyplot as plt

    # The centroid method gives an inversion on this matrix.
    d = pd.DataFrame(
        [[0, 2, 2.2, 5], [2, 0, 2.2, 5], [2.2, 2.2, 0, 5], [5, 5, 5, 0]],
        index=list("ABCD"), columns=list("ABCD"), dtype=float,
    )
    result = taxodist.taxo_cluster(d, method="centroid")
    assert np.any(np.diff(result["height"]) < 0)
    taxodist.plot_taxodist_cluster(result)
    assert plt.gcf()._suptitle.get_text() == "Method: centroid"
    mock_show.assert_called_once()
    plt.close("all")
