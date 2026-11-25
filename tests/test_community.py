"""
Taxonomic dissimilarity (Clarke, Somerfield & Chapman 2006) and standardized
effect sizes of MPD and MNTD (Webb et al. 2002).
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

import taxodist
import taxodist.community as community

COMMUNITY_LINEAGES = {
    "A": ["Biota", "Animalia", "Chordata", "Mammalia", "Carnivora", "A"],
    "B": ["Biota", "Animalia", "Chordata", "Mammalia", "Carnivora", "B"],
    "C": ["Biota", "Animalia", "Chordata", "Mammalia", "Primates", "C"],
    "D": ["Biota", "Animalia", "Arthropoda", "Insecta", "Diptera", "D"],
    "E": ["Biota", "Animalia", "Arthropoda", "Insecta", "Coleoptera", "E"],
    "F": ["Biota", "Plantae", "Tracheophyta", "Magnoliopsida", "Fagales", "F"],
}


def community_d(metric="path"):
    return taxodist.distance_matrix(
        taxodist.taxo_from_lineages(COMMUNITY_LINEAGES), metric=metric, progress=False
    )


def lower(frame):
    """Lower triangle in the column-major order of R's ``dist``."""
    values = frame.to_numpy(dtype=float)
    cols, rows = np.triu_indices(len(values), 1)
    return values[rows, cols]


# Presence/absence Sorensen (Bray-Curtis) and Kulczynski, as
# vegan::vegdist(binary = TRUE).
def sorensen(x, y):
    shared = np.sum(x & y)
    return 1 - 2 * shared / (x.sum() + y.sum())


def kulczynski(x, y):
    shared = np.sum(x & y)
    return 1 - 0.5 * (shared / x.sum() + shared / y.sum())


def test_gamma_and_theta_reduce_to_sorensen_and_kulczynski_on_a_flat_classification():
    rng = np.random.default_rng(11)
    taxa = [f"sp{i}" for i in range(1, 13)]
    flat = pd.DataFrame(1 - np.eye(12), index=taxa, columns=taxa)
    comm = rng.binomial(1, 0.4, size=(6, 12)).astype(float)
    comm[:, 0] = 1  # no empty sample
    comm[comm == 1] = rng.integers(1, 10, size=int(comm.sum()))  # abundances are ignored
    comm = pd.DataFrame(comm, index=[f"s{i}" for i in range(1, 7)], columns=taxa)
    gamma = taxodist.taxo_dissimilarity(comm, flat, "gamma_plus")
    theta = taxodist.taxo_dissimilarity(comm, flat, "theta_plus")
    present = comm.to_numpy() > 0
    n = len(comm)
    for k in range(n):
        for l in range(n):
            if k == l:
                continue
            assert gamma.iloc[k, l] == pytest.approx(sorensen(present[k], present[l]))
            assert theta.iloc[k, l] == pytest.approx(kulczynski(present[k], present[l]))
    assert list(gamma.index) == list(comm.index)
    assert list(gamma.columns) == list(comm.index)
    assert np.all(np.diag(gamma.to_numpy()) == 0)
    assert gamma.attrs["method"] == "gamma_plus"
    assert theta.attrs["method"] == "theta_plus"
    assert gamma.attrs["metric"] is None


def test_gamma_and_theta_agree_with_scipy_on_a_flat_classification():
    from scipy.spatial.distance import dice, pdist

    rng = np.random.default_rng(4)
    taxa = [f"t{i}" for i in range(10)]
    flat = pd.DataFrame(1 - np.eye(10), index=taxa, columns=taxa)
    comm = rng.binomial(1, 0.5, size=(5, 10)).astype(bool)
    comm[:, 0] = True
    gamma = taxodist.taxo_dissimilarity(pd.DataFrame(comm, columns=taxa), flat)
    np.testing.assert_allclose(lower(gamma), pdist(comm, dice))


def test_taxonomic_dissimilarity_follows_its_definition():
    d = community_d()
    comm = pd.DataFrame(
        [[1, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0]],
        index=["s1", "s2", "s3"], columns=list("ABCDEF"),
    )
    m = d
    gamma = taxodist.taxo_dissimilarity(comm, d)
    theta = taxodist.taxo_dissimilarity(comm, d, "theta_plus")
    assert gamma.loc["s1", "s2"] == pytest.approx(
        (m.loc["A", "B"] + m.loc["C", "B"] + m.loc["B", "A"]) / 3
    )
    assert theta.loc["s1", "s2"] == pytest.approx(
        ((m.loc["A", "B"] + m.loc["C", "B"]) / 2 + m.loc["B", "A"]) / 2
    )
    # Equal sample sizes give Gamma+ = Theta+.
    assert gamma.loc["s1", "s3"] == pytest.approx(theta.loc["s1", "s3"])
    assert gamma.attrs["metric"] == "path"
    # Identical samples have zero dissimilarity.
    same = pd.DataFrame([[1, 1], [1, 1]], index=["a", "b"], columns=["A", "B"])
    assert lower(taxodist.taxo_dissimilarity(same, d)).tolist() == [0]


def test_taxonomic_dissimilarity_handles_empty_samples_and_missing_distances():
    d = community_d()
    comm = pd.DataFrame(
        [[1, 0], [0, 0], [0, 1]], index=["s1", "s2", "s3"], columns=["A", "F"]
    )
    values = lower(taxodist.taxo_dissimilarity(comm, d))
    # Pairs (s2, s1) and (s3, s2) involve the empty sample s2.
    assert np.isnan(values[0]) and np.isnan(values[2])
    assert values[1] == pytest.approx(d.loc["A", "F"])
    disconnected = taxodist.distance_matrix(
        taxodist.taxo_from_lineages({"X": ["Biota", "X"], "Y": ["Other", "Y"]}),
        progress=False,
    )
    comm2 = pd.DataFrame([[1, 0], [0, 1]], index=["a", "b"], columns=["X", "Y"])
    assert np.isnan(lower(taxodist.taxo_dissimilarity(comm2, disconnected))[0])
    single = taxodist.taxo_dissimilarity({"A": 1}, d)
    assert single.shape == (1, 1) and lower(single).size == 0
    with pytest.raises(ValueError, match="gamma_plus"):
        taxodist.taxo_dissimilarity(comm, d, "jaccard")
    with pytest.raises(ValueError, match="gamma_plus"):
        taxodist.taxo_dissimilarity(comm, d, ["gamma_plus"])
    with pytest.raises(ValueError, match="no taxa"):
        taxodist.taxo_dissimilarity([], d)
    with pytest.raises(ValueError, match="no taxa"):
        taxodist.taxo_dissimilarity(pd.DataFrame(index=["a"]), d)
    with pytest.raises(ValueError, match="Missing"):
        taxodist.taxo_dissimilarity({"Z": 1}, d)


def test_taxonomic_dissimilarity_computes_distances_when_d_is_none(monkeypatch):
    calls = []

    def fake_distance_matrix(taxa, verbose=False, progress=True, metric="reciprocal"):
        calls.append((list(taxa), metric))
        return taxodist.distance_matrix(
            taxodist.taxo_from_lineages({t: COMMUNITY_LINEAGES[t] for t in taxa}),
            progress=False, metric=metric,
        )

    monkeypatch.setattr(community, "distance_matrix", fake_distance_matrix)
    comm = pd.DataFrame([[1, 0], [0, 1]], index=["a", "b"], columns=["A", "B"])
    out = taxodist.taxo_dissimilarity(comm, metric="path")
    assert calls == [(["A", "B"], "path")]
    assert lower(out).tolist() == [2]
    assert out.attrs["metric"] == "path"
    with pytest.raises(ValueError, match="Unknown distance metric"):
        taxodist.taxo_dissimilarity(comm, metric="bogus")


# Reference implementations written from picante's definitions.
def reference_mpd(x, m, weighted):
    p = x > 0
    w = m[np.ix_(p, p)]
    if weighted:
        return np.average(w, weights=np.outer(x[p], x[p]))
    cols, rows = np.triu_indices(p.sum(), 1)
    return w[rows, cols].mean()


def reference_mntd(x, m, weighted):
    p = x > 0
    w = m[np.ix_(p, p)].copy()
    np.fill_diagonal(w, np.nan)
    nearest = np.nanmin(w, axis=0)
    return np.average(nearest, weights=x[p]) if weighted else nearest.mean()


def test_mpd_and_mntd_follow_picantes_definitions():
    d = community_d()
    comm = pd.DataFrame(
        [[3, 1, 0, 2, 0, 1], [0, 5, 2, 0, 1, 0]], columns=list("ABCDEF")
    )
    m = d.loc[comm.columns, comm.columns].to_numpy()
    stat = community._community_statistic
    for x in comm.to_numpy(dtype=float):
        for weighted in (False, True):
            assert stat(x, m, "mpd", weighted) == pytest.approx(reference_mpd(x, m, weighted))
            assert stat(x, m, "mntd", weighted) == pytest.approx(reference_mntd(x, m, weighted))
    ab = d.loc[["A", "B"], ["A", "B"]].to_numpy()
    assert np.isnan(stat([1, 0], ab, "mpd", False))
    inf = ab.copy()
    inf[0, 1] = inf[1, 0] = np.inf
    assert np.isnan(stat([1, 1], inf, "mpd", False))


def test_the_pool_null_model_has_the_pool_mean_as_expected_mpd(capsys):
    d = community_d()
    res = taxodist.taxo_ses(["A", "B", "C"], d, n_iter=4000, seed=3)
    m = d.to_numpy()
    assert res["null_mean"][0] == pytest.approx(lower(d).mean(), rel=0.02)
    sub = d.loc[list("ABC"), list("ABC")].to_numpy()
    assert res["observed"][0] == pytest.approx(reference_mpd(np.ones(3), sub, False))
    assert res["NRI"][0] == pytest.approx(
        -(res["observed"][0] - res["null_mean"][0]) / res["null_sd"][0]
    )
    # Three closely related mammals are clustered relative to the pool.
    assert res["NRI"][0] > 0
    assert res.attrs["pool_size"] == 6
    assert res.attrs["index"] == "mpd"
    assert res.attrs["null_model"] == "pool"
    assert res.attrs["abundance_weighted"] is False
    assert res.attrs["metric"] == "path"
    assert m.shape == (6, 6)
    assert list(res.columns) == [
        "sample", "S", "observed", "null_mean", "null_sd", "z", "NRI",
        "p_lower", "p_upper", "n_iter",
    ]
    assert res["sample"][0] == "sample"
    assert res["n_iter"][0] == 4000
    assert taxodist.print_taxodist_ses(res) is res
    out = capsys.readouterr().out
    assert "Net relatedness index (MPD)" in out
    assert "Null model: pool, pool of 6 taxa; metric: path" in out
    assert "sample" in out
    # Reproducible with a seed.
    again = taxodist.taxo_ses(["A", "B", "C"], d, n_iter=4000, seed=3)
    pd.testing.assert_frame_equal(res, again)


def test_taxo_ses_supports_mntd_abundances_and_custom_pools(capsys):
    d = community_d()
    comm = pd.DataFrame(
        [[3, 1, 0, 0], [0, 0, 2, 5], [1, 0, 0, 0]],
        index=["s1", "s2", "s3"], columns=list("ABCD"),
    )
    res = taxodist.taxo_ses(
        comm, d, index="mntd", abundance_weighted=True,
        pool=["A", "B", "C", "D", "E"], n_iter=99, seed=5,
    )
    assert list(res.columns) == [
        "sample", "S", "observed", "null_mean", "null_sd", "z", "NTI",
        "p_lower", "p_upper", "n_iter",
    ]
    assert list(res["S"]) == [2, 2, 1]
    assert np.isnan(res["observed"][2])
    assert np.isnan(res["p_lower"][2]) and np.isnan(res["p_upper"][2])
    assert all((res["p_lower"][:2] > 0) & (res["p_lower"][:2] <= 1))
    assert res.attrs["pool_size"] == 5
    taxodist.print_taxodist_ses(res)
    out = capsys.readouterr().out
    assert "abundance-weighted" in out
    assert "Nearest taxon index (MNTD)" in out


def test_the_pool_null_model_assigns_observed_abundances_to_drawn_taxa():
    d = community_d()
    full = d.to_numpy()

    class FirstTaxa:
        """Stand-in generator drawing the first taxa of the pool."""

        def choice(self, n, size, replace):
            return np.arange(size)

    import unittest.mock as mock
    with mock.patch.object(community.np.random, "default_rng", lambda seed=None: FirstTaxa()):
        res = taxodist.taxo_ses(
            {"D": 1, "E": 3}, d, abundance_weighted=True, n_iter=2,
        )
    expected = community._community_statistic(
        np.array([1.0, 3.0]), full[np.ix_([0, 1], [0, 1])], "mpd", True
    )
    assert res["null_mean"][0] == pytest.approx(expected)
    assert res["null_sd"][0] == pytest.approx(0)
    assert np.isnan(res["z"][0])


def test_the_independent_swap_keeps_richness_and_frequencies():
    rng = np.random.default_rng(9)
    comm = rng.binomial(1, 0.4, size=(8, 10)).astype(float)
    comm[comm == 1] = rng.integers(1, 6, size=int(comm.sum()))
    random = community._independent_swap(comm, 200, rng)
    np.testing.assert_array_equal((random > 0).sum(axis=1), (comm > 0).sum(axis=1))
    np.testing.assert_array_equal((random > 0).sum(axis=0), (comm > 0).sum(axis=0))
    assert not np.array_equal(random > 0, comm > 0)
    # Abundances move with presences.
    np.testing.assert_array_equal(np.sort(random[random > 0]), np.sort(comm[comm > 0]))
    # The input is not modified.
    assert comm.shape == (8, 10)
    # A matrix without checkerboards cannot change (and stops after
    # 100 * swaps attempts).
    full = np.ones((2, 2))
    np.testing.assert_array_equal(community._independent_swap(full, 5, rng), full)
    one_column = np.ones((3, 1))
    np.testing.assert_array_equal(community._independent_swap(one_column, 5, rng), one_column)


def test_the_independent_swap_performs_both_checkerboard_swaps():
    class Draws:
        """Stand-in generator returning fixed draws."""

        def __init__(self, values):
            self.values = list(values)

        def integers(self, high, size):
            return np.full(size, self.values.pop(0))

    # Rows 0/1 and columns 0/1: [[2, 0], [0, 5]] -> [[0, 2], [5, 0]].
    m = np.array([[2.0, 0.0], [0.0, 5.0]])
    swapped = community._independent_swap(m, 1, Draws([0, 0, 0, 0]))
    np.testing.assert_array_equal(swapped, [[0.0, 2.0], [5.0, 0.0]])
    # And back with the other orientation.
    back = community._independent_swap(swapped, 1, Draws([0, 0, 0, 0]))
    np.testing.assert_array_equal(back, m)


def test_taxo_ses_runs_the_independent_swap_null_model():
    d = community_d()
    comm = pd.DataFrame(
        [[1, 1, 0, 0, 0, 1], [0, 1, 1, 1, 0, 0], [1, 0, 0, 1, 1, 0]],
        index=["s1", "s2", "s3"], columns=list("ABCDEF"),
    )
    res = taxodist.taxo_ses(comm, d, null_model="independent_swap", n_iter=49, swaps=50, seed=2)
    assert len(res) == 3
    assert np.all(np.isfinite(res["null_mean"]))
    assert res.attrs["null_model"] == "independent_swap"
    # Taxa outside the pool are allowed by the swap model.
    res2 = taxodist.taxo_ses(
        comm, d, null_model="independent_swap", pool=["A", "B"], n_iter=3, swaps=5, seed=1,
    )
    assert res2.attrs["pool_size"] == 2
    with pytest.raises(ValueError, match="at least two samples"):
        taxodist.taxo_ses(["A", "B"], d, null_model="independent_swap")


def test_taxo_ses_validates_its_arguments(capsys):
    d = community_d()
    with pytest.raises(ValueError, match="mpd"):
        taxodist.taxo_ses(["A", "B"], d, index="pd")
    with pytest.raises(ValueError, match="pool"):
        taxodist.taxo_ses(["A", "B"], d, null_model="frequency")
    for bad in (0, 1.5, True, float("nan"), "5"):
        with pytest.raises(ValueError, match="positive integer"):
            taxodist.taxo_ses(["A", "B"], d, n_iter=bad)
    with pytest.raises(ValueError, match="positive integer"):
        taxodist.taxo_ses(["A", "B"], d, swaps=1.5)
    with pytest.raises(ValueError, match="True or False"):
        taxodist.taxo_ses(["A", "B"], d, abundance_weighted=None)
    with pytest.raises(ValueError, match="at least two"):
        taxodist.taxo_ses(["A", "B"], d, pool="A")
    with pytest.raises(ValueError, match="at least two"):
        taxodist.taxo_ses(["A", "B"], d, pool={"A": 1, "B": 2})
    with pytest.raises(ValueError, match="at least two"):
        taxodist.taxo_ses(["A", "B"], d, pool=["A", None])
    with pytest.raises(ValueError, match="belong to the pool"):
        taxodist.taxo_ses(["A", "B"], d, pool=["C", "D"])
    with pytest.raises(TypeError, match="distance matrix"):
        taxodist.taxo_ses(["A", "B"], np.zeros((2, 2)))
    zero_sd = taxodist.taxo_ses(["A", "B"], d, pool=["A", "B"], n_iter=5)
    assert np.isnan(zero_sd["z"][0])
    one = taxodist.taxo_ses(["A", "B", "C"], d, n_iter=1, seed=1)
    assert np.isnan(one["null_sd"][0]) and np.isnan(one["z"][0])
    # A user-built matrix has no metric.
    plain = pd.DataFrame(d.to_numpy(), index=d.index, columns=d.columns)
    res = taxodist.taxo_ses(["A", "B", "C"], plain, n_iter=5, seed=1)
    assert res.attrs["metric"] is None
    taxodist.print_taxodist_ses(res)
    assert "metric" not in capsys.readouterr().out


def test_taxo_ses_p_values_are_missing_when_a_null_value_is_missing():
    lineages = {"X": ["Biota", "X"], "Y": ["Biota", "Y"], "Z": ["Other", "Z"]}
    d = taxodist.distance_matrix(taxodist.taxo_from_lineages(lineages), progress=False)
    res = taxodist.taxo_ses(["X", "Y"], d, n_iter=30, seed=1)
    assert np.isfinite(res["observed"][0])
    assert np.isnan(res["null_mean"][0])
    assert np.isnan(res["p_lower"][0]) and np.isnan(res["p_upper"][0])


def test_taxo_ses_computes_distances_when_d_is_none(monkeypatch):
    def fake_distance_matrix(taxa, verbose=False, progress=True, metric="reciprocal"):
        return taxodist.distance_matrix(
            taxodist.taxo_from_lineages({t: COMMUNITY_LINEAGES[t] for t in taxa}),
            progress=False, metric=metric,
        )

    monkeypatch.setattr(community, "distance_matrix", fake_distance_matrix)
    comm = pd.DataFrame([[1, 1, 1]], index=["s"], columns=["A", "B", "D"])
    res = taxodist.taxo_ses(comm, n_iter=9, seed=1)
    assert res.attrs["pool_size"] == 3
    assert res.attrs["metric"] == "reciprocal"
    assert res["sample"][0] == "s"


def test_community_indices_reproduce_the_shared_fixture():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "expected_community.json")
    with open(path, encoding="utf-8") as handle:
        fx = json.load(handle)
    res = taxodist.taxo_from_lineages(fx["lineages"])
    comm = pd.DataFrame(fx["comm"], index=fx["sample_names"])

    def as_num(values):
        return np.array([np.nan if v is None else v for v in values], dtype=float)

    for metric in ("reciprocal", "path"):
        d = taxodist.distance_matrix(res, metric=metric, progress=False)
        full = d.loc[comm.columns, comm.columns].to_numpy()
        exp = fx[metric]
        for index in ("gamma_plus", "theta_plus"):
            np.testing.assert_allclose(
                lower(taxodist.taxo_dissimilarity(comm, d, index)), as_num(exp[index]),
                rtol=1e-12, err_msg=f"{metric} {index}",
            )
        for index in ("mpd", "mntd"):
            for weighted in (False, True):
                key = index + ("_weighted" if weighted else "")
                got = [
                    community._community_statistic(row, full, index, weighted)
                    for row in comm.to_numpy(dtype=float)
                ]
                np.testing.assert_allclose(
                    got, as_num(exp[key]), rtol=1e-12, err_msg=f"{metric} {key}"
                )
