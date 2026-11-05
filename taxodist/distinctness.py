"""Taxonomic diversity and distinctness indices of Warwick and Clarke."""

import math
import numbers
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .distance import _match_metric, distance_matrix

_INDEX_COLUMNS = ["S", "Delta", "Delta_star", "Delta_plus", "Lambda_plus", "SDelta_plus"]


def taxo_distinctness(comm, d=None, metric="ranked", verbose=False, progress=True):
    """
    Taxonomic diversity and distinctness indices

    Computes the taxonomic diversity and distinctness indices of Warwick and
    Clarke for one or more samples: taxonomic diversity (Delta), taxonomic
    distinctness (Delta*), average taxonomic distinctness (Delta+), and
    variation in taxonomic distinctness (Lambda+).

    Parameters
    ----------
    comm : sequence of str, dict, pandas.Series, or pandas.DataFrame
        The samples. One of:

        - a sequence of taxon names (one sample, presence only);
        - a dict or Series of abundances indexed by taxon (one sample);
        - a DataFrame of abundances or presences with one row per sample and
          one column per taxon (column names are taxon names).
    d : pandas.DataFrame, optional
        Square distance matrix (e.g. from distance_matrix()) whose labels
        include every taxon present in ``comm``. If None (default), distances
        are computed with distance_matrix() and ``metric``.
    metric : str
        Distance metric used when ``d`` is None; see taxo_distance(). The
        default ``"ranked"`` gives the step weights of Clarke and Warwick (1
        for congeneric taxa, 2 for confamilial, ...).
    verbose, progress : bool
        Passed to distance_matrix() when ``d`` is None.

    Returns
    -------
    pandas.DataFrame
        One row per sample with columns ``sample``, ``S`` (number of taxa
        present), ``Delta``, ``Delta_star``, ``Delta_plus``, ``Lambda_plus``,
        and ``SDelta_plus`` (S * Delta+). The indices are NaN for samples
        with fewer than two taxa, or when a distance between present taxa is
        missing or infinite. ``attrs["metric"]`` records the metric of ``d``
        (None if unknown).

    Notes
    -----
    With x_i the abundance of taxon i, n = sum(x_i), S the number of taxa
    present, and w_ij the distance between taxa i and j, sums running over
    pairs i < j:

    - Delta = sum(w_ij x_i x_j) / [n (n - 1) / 2]
    - Delta* = sum(w_ij x_i x_j) / sum(x_i x_j)
    - Delta+ = sum(w_ij) / [S (S - 1) / 2]
    - Lambda+ = sum((w_ij - Delta+)^2) / [S (S - 1) / 2]

    With presence-only data, Delta and Delta* equal Delta+. Results agree
    with ``vegan::taxondive()`` and the R function ``taxo_distinctness()``.

    References
    ----------
    Warwick, R.M. & Clarke, K.R. (1995). Marine Ecology Progress Series 129,
    301-305. Clarke, K.R. & Warwick, R.M. (1998). Journal of Applied Ecology
    35, 523-531. Clarke, K.R. & Warwick, R.M. (2001). Marine Ecology Progress
    Series 216, 265-278.
    """
    samples = _distinctness_samples(comm)
    taxa = list(samples.columns)
    if d is None:
        metric = _match_metric(metric)
        d = distance_matrix(taxa, verbose=verbose, progress=progress, metric=metric)
    full = _distinctness_matrix(d, taxa)
    rows = []
    for name, values in zip(samples.index, samples.to_numpy(dtype=float)):
        rows.append({"sample": name, **_distinctness_indices(values, full)})
    out = pd.DataFrame(rows, columns=["sample"] + _INDEX_COLUMNS)
    out["S"] = out["S"].astype("int64")
    for column in _INDEX_COLUMNS[1:]:
        out[column] = out[column].astype("float64")
    out.attrs["metric"] = d.attrs.get("metric")
    return out


class TaxodistDistinctnessTest(dict):
    """Result of taxo_distinctness_test(); ``print()`` shows a summary."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __str__(self):
        metric = self.get("metric")
        header = (
            f"{self['S']} taxa vs {self['n_iter']} random samples from a pool "
            f"of {self['pool_size']}"
        )
        if metric is not None:
            header += f" (metric: {metric})"
        return "\n".join([
            "Taxonomic distinctness test",
            header,
            self["statistics"].to_string(
                index=False, float_format=lambda value: f"{value:.4g}"
            ),
        ])


def taxo_distinctness_test(sample, d, n_iter=999, seed=None):
    """
    Test taxonomic distinctness against random samples from a species pool

    Compares the average taxonomic distinctness Delta+ and its variation
    Lambda+ of a sample with their distribution in random samples of the same
    number of taxa drawn without replacement from a master list (species
    pool), as proposed by Clarke and Warwick.

    Parameters
    ----------
    sample : sequence of str
        The taxa present in the sample.
    d : pandas.DataFrame
        Square distance matrix over the master list, for example from
        distance_matrix(). Every taxon in ``sample`` must be one of its labels.
    n_iter : int
        Number of random samples. Default 999.
    seed : None, int, or numpy.random.Generator
        Source of randomness, passed to ``numpy.random.default_rng()``. Pass
        an integer (or a Generator) for reproducible results; this replaces
        ``set.seed()`` in R. The random draws are not identical to R's.

    Returns
    -------
    TaxodistDistinctnessTest
        A dictionary (with attribute access) containing:

        - statistics: DataFrame with one row for ``Delta_plus`` and one for
          ``Lambda_plus`` and columns ``statistic``, ``observed``,
          ``expected``, ``lower``, ``upper`` (the 2.5% and 97.5% quantiles of
          the random samples, R's default quantile type 7), ``p_lower``, and
          ``p_upper`` (one-sided permutation p-values ``(1 + count) /
          (n_iter + 1)`` for values as low or as high as observed).
        - simulated: NumPy array of shape ``(n_iter, 2)``; column 0 holds
          the simulated Delta+ and column 1 the simulated Lambda+.
        - S, pool_size, n_iter, metric: test settings (``metric`` is
          ``d.attrs.get("metric")``).

    Notes
    -----
    The expected Delta+ is the mean distance over all pairs of the master
    list, which does not depend on the sample size. The expected Lambda+ is
    the mean of the simulated values. A Delta+ below the lower limit
    indicates that the sample is less taxonomically distinct than expected
    from the pool.
    """
    if not isinstance(d, pd.DataFrame) or getattr(d, "_taxodist_resolution", False):
        raise TypeError("d must be a distance matrix DataFrame over the master list")
    if isinstance(sample, str):
        sample = [sample]
    if (
        isinstance(sample, (bytes, Mapping))
        or not isinstance(sample, Iterable)
    ):
        raise TypeError("sample must be a sequence of taxon names (character strings)")
    sample = list(sample)
    if any(not isinstance(taxon, str) for taxon in sample):
        raise TypeError("sample must be a sequence of taxon names (character strings)")
    if (
        isinstance(n_iter, (bool, np.bool_))
        or not isinstance(n_iter, numbers.Real)
        or not math.isfinite(n_iter)
        or n_iter < 1
        or n_iter != math.floor(n_iter)
    ):
        raise ValueError("n_iter must be a single positive integer")
    n_iter = int(n_iter)
    sample = list(dict.fromkeys(sample))
    pool = [str(label) for label in d.index]
    full = d.to_numpy(dtype=float)
    position = {label: i for i, label in enumerate(pool)}
    missing = [taxon for taxon in sample if taxon not in position]
    if missing:
        raise ValueError(
            "Every sampled taxon must belong to the master list. "
            f"Not in d: {', '.join(map(repr, missing))}."
        )
    cols, rows = np.triu_indices(len(pool), 1)
    if not np.all(np.isfinite(full[rows, cols])):
        raise ValueError("The master list distances must all be finite.")
    S = len(sample)
    if S < 2 or S > len(pool):
        raise ValueError(
            f"The sample must contain between 2 and {len(pool)} taxa of the master list."
        )

    def stats_of(idx):
        sub = full[np.ix_(idx, idx)]
        c, r = np.triu_indices(len(idx), 1)
        omega = sub[r, c]
        delta_plus = omega.mean()
        return delta_plus, ((omega - delta_plus) ** 2).mean()

    rng = np.random.default_rng(seed)
    observed = np.array(stats_of([position[taxon] for taxon in sample]))
    simulated = np.array(
        [stats_of(rng.choice(len(pool), size=S, replace=False)) for _ in range(n_iter)],
        dtype=float,
    ).reshape(n_iter, 2)
    expected = np.array([full[rows, cols].mean(), simulated[:, 1].mean()])
    tol = math.sqrt(np.finfo(float).eps)
    statistics = pd.DataFrame({
        "statistic": ["Delta_plus", "Lambda_plus"],
        "observed": observed,
        "expected": expected,
        "lower": np.quantile(simulated, 0.025, axis=0, method="linear"),
        "upper": np.quantile(simulated, 0.975, axis=0, method="linear"),
        "p_lower": (1 + (simulated <= observed + tol).sum(axis=0)) / (n_iter + 1),
        "p_upper": (1 + (simulated >= observed - tol).sum(axis=0)) / (n_iter + 1),
    })
    return TaxodistDistinctnessTest(
        statistics=statistics,
        simulated=simulated,
        S=S,
        pool_size=len(pool),
        n_iter=n_iter,
        metric=d.attrs.get("metric"),
    )


def print_taxodist_distinctness(x):
    """Print a taxo_distinctness() table or a taxo_distinctness_test() result."""
    if x is None:
        return x
    if isinstance(x, TaxodistDistinctnessTest):
        print(str(x))
        return x
    print("Taxonomic distinctness")
    metric = x.attrs.get("metric")
    if metric is not None:
        print(f"Metric: {metric}")
    print(x.to_string(index=False))
    return x


# -- Internal helpers ------------------------------------------------------------

def _is_number(value):
    return isinstance(value, (numbers.Real, np.bool_)) and not isinstance(value, str)


def _distinctness_samples(comm):
    """
    Normalize the supported sample formats to a float DataFrame with one row
    per sample and one column per taxon.
    """
    if isinstance(comm, pd.DataFrame):
        if getattr(comm, "_taxodist_resolution", False):
            raise TypeError(_COMM_TYPE_MESSAGE)
        frame = comm.copy()
        if isinstance(frame.index, pd.RangeIndex) and frame.index.equals(
            pd.RangeIndex(len(frame))
        ):
            frame.index = (
                ["sample"] if len(frame) == 1
                else [f"sample{i + 1}" for i in range(len(frame))]
            )
        else:
            frame.index = [str(label) for label in frame.index]
    elif isinstance(comm, (Mapping, pd.Series)):
        items = dict(comm) if isinstance(comm, Mapping) else comm.to_dict()
        if any(not _is_number(value) for value in items.values()):
            raise TypeError(_COMM_TYPE_MESSAGE)
        frame = pd.DataFrame([list(items.values())], columns=list(items), index=["sample"])
    elif isinstance(comm, (str, bytes)) or isinstance(comm, Iterable):
        taxa = [comm] if isinstance(comm, str) else list(comm)
        if isinstance(comm, bytes) or not taxa:
            raise TypeError(_COMM_TYPE_MESSAGE)
        if all(_is_number(value) for value in taxa):
            raise ValueError("A numeric comm vector must be named by taxon (use a dict or Series).")
        if any(not isinstance(taxon, str) for taxon in taxa):
            raise TypeError(_COMM_TYPE_MESSAGE)
        if any(not taxon.strip() for taxon in taxa):
            raise ValueError("comm cannot contain missing or empty taxon names.")
        taxa = list(dict.fromkeys(taxa))
        frame = pd.DataFrame([[1.0] * len(taxa)], columns=taxa, index=["sample"])
    else:
        raise TypeError(_COMM_TYPE_MESSAGE)

    columns = list(frame.columns)
    if (
        any(not isinstance(column, str) for column in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError("The columns of comm must be named by unique taxa.")
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(_COMM_TYPE_MESSAGE) from error
    if np.isnan(values).any() or (values < 0).any():
        raise ValueError("Abundances in comm must be non-negative and not missing.")
    return pd.DataFrame(values, index=list(frame.index), columns=columns)


_COMM_TYPE_MESSAGE = (
    "comm must be a sequence of taxa, a dict or Series of abundances named by "
    "taxon, or a DataFrame of abundances (rows = samples, columns = taxa)."
)


def _distinctness_matrix(d, taxa):
    """Full distance matrix restricted to and ordered by ``taxa``."""
    if not isinstance(d, pd.DataFrame) or getattr(d, "_taxodist_resolution", False):
        raise TypeError("d must be a distance matrix DataFrame (from distance_matrix())")
    missing = [taxon for taxon in taxa if taxon not in d.index or taxon not in d.columns]
    if missing:
        raise ValueError(
            f"d has no distances for some taxa. Missing: {', '.join(map(repr, missing))}."
        )
    return d.loc[taxa, taxa].to_numpy(dtype=float)


def _distinctness_indices(x, full):
    present = x > 0
    S = int(present.sum())
    empty = {
        "S": S, "Delta": np.nan, "Delta_star": np.nan, "Delta_plus": np.nan,
        "Lambda_plus": np.nan, "SDelta_plus": np.nan,
    }
    if S < 2:
        return empty
    idx = np.flatnonzero(present)
    w = full[np.ix_(idx, idx)]
    a = x[idx]
    # Pairs in R's lower.tri() order (column by column).
    cols, rows = np.triu_indices(S, 1)
    omega = w[rows, cols]
    if not np.all(np.isfinite(omega)):
        return empty
    xx = np.outer(a, a)[rows, cols]
    n = a.sum()
    delta_plus = omega.mean()
    return {
        "S": S,
        "Delta": (omega * xx).sum() / (n * (n - 1) / 2),
        "Delta_star": (omega * xx).sum() / xx.sum(),
        "Delta_plus": delta_plus,
        "Lambda_plus": ((omega - delta_plus) ** 2).mean(),
        "SDelta_plus": S * delta_plus,
    }
