"""
Community analyses on taxonomic distances: the taxonomic dissimilarities of
Clarke, Somerfield and Chapman (2006) and the standardized effect sizes of
MPD and MNTD (NRI and NTI) of Webb et al. (2002).
"""

import math
import numbers
import warnings
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .distance import _match_metric, distance_matrix
from .distinctness import _distinctness_matrix, _distinctness_samples


def taxo_dissimilarity(comm, d=None, index="gamma_plus", metric="ranked",
                       verbose=False, progress=True):
    """
    Taxonomic dissimilarity between samples

    Computes the taxonomic dissimilarity Gamma+ or Theta+ of Clarke,
    Somerfield and Chapman (2006) between every pair of samples. Unlike
    Bray-Curtis or Jaccard dissimilarities, they credit "near misses": two
    samples without any taxon in common are still similar when their taxa
    are closely related.

    Parameters
    ----------
    comm : pandas.DataFrame, dict, pandas.Series, or sequence of str
        The samples: a DataFrame with one row per sample and one column per
        taxon (column names are taxon names). Values greater than zero mean
        presence; abundances are not used.
    d : pandas.DataFrame, optional
        Square distance matrix (e.g. from distance_matrix()) whose labels
        include every taxon of ``comm``. If None, it is computed with
        distance_matrix() and ``metric``.
    index : {"gamma_plus", "theta_plus"}
        The dissimilarity.
    metric : str
        Distance metric used when ``d`` is None; see taxo_distance().
    verbose, progress : bool
        Passed to distance_matrix() when ``d`` is None.

    Returns
    -------
    pandas.DataFrame
        Symmetric dissimilarity matrix between samples (zero diagonal), the
        same structure as distance_matrix(). ``attrs["method"]`` records the
        index and ``attrs["metric"]`` the distance metric of ``d`` (None if
        unknown). Pairs involving a sample without taxa, or whose
        nearest-relative distances are missing or infinite, are NaN.

    Notes
    -----
    With w_ij the distance between taxon i of sample 1 and taxon j of
    sample 2, and s1, s2 the numbers of taxa in each sample,

    - Gamma+ = [sum_i min_j w_ij + sum_j min_i w_ij] / (s1 + s2)
    - Theta+ = [sum_i min_j w_ij / s1 + sum_j min_i w_ij / s2] / 2

    Taxa shared by both samples contribute zero. On a flat classification
    the indices reduce to the presence/absence Bray-Curtis (Sorensen) and
    Kulczynski dissimilarities, times that distance. Clarke et al. scale
    distances so that the largest path length is 100; the indices are
    returned here in the units of ``d``.
    """
    if not isinstance(index, str) or index not in ("gamma_plus", "theta_plus"):
        raise ValueError("index must be 'gamma_plus' or 'theta_plus'.")
    samples = _community_samples(comm)
    full, metric_used = _community_distances(
        list(samples.columns), d, metric, verbose, progress
    )
    present = samples.to_numpy(dtype=float) > 0
    n = len(samples)

    def pair(k, l):
        p = present[k]
        q = present[l]
        if not p.any() or not q.any():
            return np.nan
        w = full[np.ix_(p, q)]
        to_q = w.min(axis=1)
        to_p = w.min(axis=0)
        if not (np.all(np.isfinite(to_q)) and np.all(np.isfinite(to_p))):
            return np.nan
        if index == "gamma_plus":
            return (to_q.sum() + to_p.sum()) / (len(to_q) + len(to_p))
        return (to_q.mean() + to_p.mean()) / 2

    values = np.zeros((n, n))
    for l in range(n):
        for k in range(l + 1, n):
            values[k, l] = values[l, k] = pair(k, l)
    labels = list(samples.index)
    result = pd.DataFrame(values, index=labels, columns=labels)
    result.attrs["method"] = index
    result.attrs["metric"] = metric_used
    return result


def taxo_ses(comm, d=None, index="mpd", null_model="pool", pool=None, n_iter=999,
             abundance_weighted=False, swaps=1000, metric="reciprocal",
             verbose=False, progress=True, seed=None):
    """
    Standardized effect sizes of community taxonomic structure

    Compares the mean pairwise distance (MPD) or mean nearest-taxon distance
    (MNTD) among the taxa of each sample with its distribution under a null
    model, giving the standardized effect sizes known as the net relatedness
    index (NRI) and the nearest taxon index (NTI) of Webb et al. (2002),
    computed on taxonomic distances.

    Parameters
    ----------
    comm : sequence of str, dict, pandas.Series, or pandas.DataFrame
        The samples: a sequence of taxa (one sample), a dict or Series of
        abundances (one sample), or a DataFrame with one row per sample and
        one column per taxon.
    d : pandas.DataFrame, optional
        Square distance matrix whose labels include every taxon of ``comm``
        and of ``pool``. If None, it is computed with distance_matrix().
    index : {"mpd", "mntd"}
        The statistic.
    null_model : {"pool", "independent_swap"}
        ``"pool"`` (default): each sample is replaced by the same number of
        taxa drawn at random, without replacement, from ``pool``; with
        ``abundance_weighted=True`` the observed abundances are assigned to
        the drawn taxa (the null model of Webb et al. 2002).
        ``"independent_swap"``: the presence matrix of all samples is
        randomized with ``swaps`` independent swaps (Gotelli 2000), which
        keep the number of taxa in every sample and the number of samples in
        which every taxon occurs. Requires at least two samples.
    pool : sequence of str, optional
        Taxa forming the regional pool for the ``"pool"`` null model.
        Defaults to every label of ``d`` (or, when ``d`` is None, every taxon
        of ``comm``).
    n_iter : int
        Number of random communities.
    abundance_weighted : bool
        If True, MPD weights each pair of taxa by the product of their
        abundances (including each taxon with itself, which makes it Rao's
        quadratic entropy) and MNTD weights each taxon by its abundance, as
        in the picante package. Default False (presence only).
    swaps : int
        Number of successful swaps per random community for
        ``"independent_swap"``.
    metric : str
        Distance metric used when ``d`` is None. The default
        ``"reciprocal"`` uses the full depth of the classification.
    verbose, progress : bool
        Passed to distance_matrix() when ``d`` is None.
    seed : None, int, or numpy.random.Generator
        Source of randomness, passed to ``numpy.random.default_rng()``
        (replaces ``set.seed()`` in R; the random draws are not identical to
        R's).

    Returns
    -------
    pandas.DataFrame
        One row per sample with columns ``sample``, ``S`` (number of taxa),
        ``observed``, ``null_mean``, ``null_sd``, ``z`` (``(observed -
        null_mean) / null_sd``), ``NRI`` or ``NTI`` (``-z``), ``p_lower`` and
        ``p_upper`` (one-sided permutation p-values for values as low or as
        high as observed, ``(1 + count) / (1 + n_iter)``), and ``n_iter``.
        Samples with fewer than two taxa have NaN statistics. ``attrs``
        records ``index``, ``null_model``, ``abundance_weighted``,
        ``pool_size``, and ``metric``. print_taxodist_ses() prints a summary.

    Notes
    -----
    Positive NRI and NTI (low MPD and MNTD) indicate taxonomic clustering;
    negative values indicate overdispersion. Both depend on the pool, which
    should represent the taxa that could have occurred in the samples.
    """
    if not isinstance(index, str) or index not in ("mpd", "mntd"):
        raise ValueError("index must be 'mpd' or 'mntd'.")
    if not isinstance(null_model, str) or null_model not in ("pool", "independent_swap"):
        raise ValueError("null_model must be 'pool' or 'independent_swap'.")
    n_iter = _check_count(n_iter, "n_iter")
    swaps = _check_count(swaps, "swaps")
    if not isinstance(abundance_weighted, (bool, np.bool_)):
        raise ValueError("abundance_weighted must be True or False.")
    abundance_weighted = bool(abundance_weighted)
    samples = _community_samples(comm)
    columns = list(samples.columns)
    if pool is None:
        if d is None:
            pool = columns
        else:
            _check_distance_frame(d)
            pool = [str(label) for label in d.index]
    if isinstance(pool, str):
        pool = [pool]
    if (
        isinstance(pool, (bytes, Mapping))
        or not isinstance(pool, Iterable)
    ):
        raise ValueError("pool must be a sequence of at least two taxa.")
    pool = list(pool)
    if len(pool) < 2 or any(not isinstance(taxon, str) for taxon in pool):
        raise ValueError("pool must be a sequence of at least two taxa.")
    pool = list(dict.fromkeys(pool))
    outside = [taxon for taxon in columns if taxon not in pool]
    if outside and null_model == "pool":
        raise ValueError(
            "Every taxon of comm must belong to the pool. "
            f"Not in the pool: {', '.join(map(repr, outside))}."
        )
    taxa = columns + [taxon for taxon in pool if taxon not in columns]
    full, metric_used = _community_distances(taxa, d, metric, verbose, progress)
    if null_model == "independent_swap" and len(samples) < 2:
        raise ValueError("The 'independent_swap' null model needs at least two samples.")

    position = {taxon: i for i, taxon in enumerate(taxa)}
    column_idx = np.array([position[taxon] for taxon in columns], dtype=int)
    pool_idx = np.array([position[taxon] for taxon in pool], dtype=int)
    values = samples.to_numpy(dtype=float)

    def stat(x, idx):
        return _community_statistic(x, full[np.ix_(idx, idx)], index, abundance_weighted)

    observed = np.array([stat(row, column_idx) for row in values], dtype=float)
    S = (values > 0).sum(axis=1)

    rng = np.random.default_rng(seed)
    n_samples = len(samples)
    simulated = np.full((n_samples, n_iter), np.nan)
    for it in range(n_iter):
        if null_model == "pool":
            for k in range(n_samples):
                present = values[k] > 0
                count = int(present.sum())
                if count < 2:
                    continue
                drawn = pool_idx[rng.choice(len(pool_idx), size=count, replace=False)]
                simulated[k, it] = stat(values[k, present], drawn)
        else:
            random = _independent_swap(values, swaps, rng)
            for k in range(n_samples):
                simulated[k, it] = stat(random[k], column_idx)

    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        null_mean = simulated.mean(axis=1)
        null_sd = (
            simulated.std(axis=1, ddof=1) if n_iter > 1 else np.full(n_samples, np.nan)
        )
        z = (observed - null_mean) / null_sd
    z[~np.isfinite(z)] = np.nan
    tol = math.sqrt(np.finfo(float).eps)

    def p_value(k, lower):
        if math.isnan(observed[k]) or np.isnan(simulated[k]).any():
            return np.nan
        if lower:
            count = (simulated[k] <= observed[k] + tol).sum()
        else:
            count = (simulated[k] >= observed[k] - tol).sum()
        return (1 + count) / (n_iter + 1)

    out = pd.DataFrame({
        "sample": pd.Series(list(samples.index), dtype="object"),
        "S": S.astype("int64"),
        "observed": observed,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "z": z,
    })
    out["NRI" if index == "mpd" else "NTI"] = -z
    out["p_lower"] = [p_value(k, True) for k in range(n_samples)]
    out["p_upper"] = [p_value(k, False) for k in range(n_samples)]
    out["p_lower"] = out["p_lower"].astype("float64")
    out["p_upper"] = out["p_upper"].astype("float64")
    out["n_iter"] = np.int64(n_iter)
    out.attrs.update({
        "index": index,
        "null_model": null_model,
        "abundance_weighted": abundance_weighted,
        "pool_size": len(pool),
        "metric": metric_used,
    })
    return out


def print_taxodist_ses(x):
    """Print a taxo_ses() result with its settings; returns ``x``."""
    index = x.attrs.get("index")
    print(
        "Net relatedness index (MPD)" if index == "mpd" else "Nearest taxon index (MNTD)"
    )
    line = f"Null model: {x.attrs.get('null_model')}, pool of {x.attrs.get('pool_size')} taxa"
    if x.attrs.get("abundance_weighted"):
        line += ", abundance-weighted"
    metric = x.attrs.get("metric")
    if metric is not None:
        line += f"; metric: {metric}"
    print(line)
    print(x.to_string(index=False, float_format=lambda value: f"{value:.4g}", na_rep="NA"))
    return x


# -- Internal helpers ------------------------------------------------------------

def _check_count(value, arg):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
        or value < 1
        or value != math.floor(value)
    ):
        raise ValueError(f"{arg} must be a single positive integer.")
    return int(value)


def _community_samples(comm):
    if isinstance(comm, (list, tuple)) and len(comm) == 0:
        raise ValueError("comm contains no taxa.")
    samples = _distinctness_samples(comm)
    if samples.shape[1] == 0:
        raise ValueError("comm contains no taxa.")
    return samples


def _check_distance_frame(d):
    if not isinstance(d, pd.DataFrame) or getattr(d, "_taxodist_resolution", False):
        raise TypeError("d must be a distance matrix DataFrame (from distance_matrix())")


def _community_distances(taxa, d, metric, verbose, progress):
    """
    Full distance matrix (NumPy array) over ``taxa``, computing it when ``d``
    is None, and the metric of ``d`` (None if unknown).
    """
    if d is None:
        metric = _match_metric(metric)
        d = distance_matrix(taxa, verbose=verbose, progress=progress, metric=metric)
    full = _distinctness_matrix(d, taxa)
    return full, d.attrs.get("metric")


def _community_statistic(x, full, index, abundance_weighted):
    """
    MPD or MNTD of one sample (abundance vector ``x`` aligned with the rows
    of ``full``), following picante's definitions. NaN with fewer than two
    taxa or non-finite distances.
    """
    x = np.asarray(x, dtype=float)
    full = np.asarray(full, dtype=float)
    present = x > 0
    if present.sum() < 2:
        return np.nan
    w = full[np.ix_(present, present)]
    a = x[present]
    with np.errstate(invalid="ignore"):
        if index == "mpd":
            if abundance_weighted:
                weights = np.outer(a, a)
                value = (w * weights).sum() / weights.sum()
            else:
                cols, rows = np.triu_indices(len(a), 1)
                value = w[rows, cols].mean()
        else:
            w = w.copy()
            np.fill_diagonal(w, np.inf)
            nearest = w.min(axis=1)
            value = (nearest * a).sum() / a.sum() if abundance_weighted else nearest.mean()
    return float(value) if math.isfinite(value) else np.nan


def _independent_swap(samples, swaps, rng):
    """
    Independent swap randomization (Gotelli 2000): repeatedly picks two
    samples and two taxa forming a checkerboard (each sample has one of the
    two taxa) and swaps them, keeping every sample's richness and every
    taxon's number of occurrences. Abundances move with the presences. Stops
    after ``swaps`` successful swaps or ``100 * swaps`` attempts.
    """
    m = np.array(samples, dtype=float)
    n_row, n_col = m.shape
    if n_row < 2 or n_col < 2:
        # No checkerboard is possible.
        return m
    rows = m.tolist()
    done = 0
    attempts = 0
    limit = 100 * swaps
    chunk = 4096
    while done < swaps and attempts < limit:
        r1 = rng.integers(n_row, size=chunk).tolist()
        r2 = rng.integers(n_row - 1, size=chunk).tolist()
        k1 = rng.integers(n_col, size=chunk).tolist()
        k2 = rng.integers(n_col - 1, size=chunk).tolist()
        for i in range(chunk):
            if done >= swaps or attempts >= limit:
                break
            attempts += 1
            # Two distinct rows and two distinct columns, uniformly.
            ra, rb = r1[i], r2[i] + (r2[i] >= r1[i])
            ka, kb = k1[i], k2[i] + (k2[i] >= k1[i])
            row_a, row_b = rows[ra], rows[rb]
            a = row_a[ka] > 0
            b = row_a[kb] > 0
            c = row_b[ka] > 0
            e = row_b[kb] > 0
            if a and e and not b and not c:
                row_a[kb], row_a[ka] = row_a[ka], 0.0
                row_b[ka], row_b[kb] = row_b[kb], 0.0
                done += 1
            elif not a and not e and b and c:
                row_a[ka], row_a[kb] = row_a[kb], 0.0
                row_b[kb], row_b[ka] = row_b[ka], 0.0
                done += 1
    return np.array(rows, dtype=float).reshape(n_row, n_col)
