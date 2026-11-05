import re
import warnings
from collections.abc import Mapping
from numbers import Real
import numpy as np
import pandas as pd

from .fetch import (
    get_lineage, get_taxonomicon_id, lineage_ranks, _lookup_taxonomicon_id,
)

#: Supported distance metrics (see taxo_distance()).
METRICS = ("reciprocal", "path", "wu_palmer", "ranked")

_DEFAULT_RANK_LEVELS = (
    "kingdom", "phylum", "class", "order", "family", "genus", "species",
)

#: Rank levels used by the ``"ranked"`` metric, matched case-insensitively
#: against the last word of each node's rank (so "Anamorph genus" counts as a
#: genus). Equivalent of the R option ``taxodist.rank_levels``; for example
#: ``taxodist.distance.RANK_LEVELS = ["kingdom", "phylum", "class", "order"]``.
RANK_LEVELS = list(_DEFAULT_RANK_LEVELS)


def _match_metric(metric):
    """Validate a metric name and return it."""
    if not isinstance(metric, str):
        raise TypeError(f"metric must be one of {', '.join(map(repr, METRICS))}")
    if metric not in METRICS:
        raise ValueError(
            f"Unknown distance metric {metric!r}. "
            f"Use one of {', '.join(map(repr, METRICS))}."
        )
    return metric


def _rank_levels():
    """Validated, lower-case copy of ``RANK_LEVELS``."""
    levels = RANK_LEVELS
    if (
        isinstance(levels, (str, bytes))
        or not isinstance(levels, (list, tuple))
        or not levels
        or any(not isinstance(level, str) for level in levels)
    ):
        raise ValueError(
            "taxodist.distance.RANK_LEVELS must be a non-empty list of strings"
        )
    return [level.strip().lower() for level in levels]


def _project_lineage(lineage, ranks, levels):
    """
    Reduce a lineage to its nodes whose rank is one of ``levels`` (compared
    on the lower-cased last word of the rank), always keeping the root node,
    so that taxa of different kingdoms remain connected, and the terminal
    node.
    """
    n = len(lineage)
    if n == 0:
        return list(lineage)
    levels = set(levels)
    keep = [
        rank is not None and re.sub(r"^.*\s", "", rank, flags=re.DOTALL).lower() in levels
        for rank in ranks
    ]
    keep[0] = True
    keep[-1] = True
    return [node for node, kept in zip(lineage, keep) if kept]


def _has_ranks(lineage, ranks):
    return (
        ranks is not None
        and len(ranks) == len(lineage)
        and any(rank is not None for rank in ranks)
    )


def _metric_lineages(lineages, ranks, labels, metric, levels=None):
    """
    Lineages on which ``metric`` is computed: unchanged for depth-based
    metrics; reduced to the standard rank levels for ``"ranked"``, which
    raises an error for lineages without rank information.
    """
    if metric != "ranked":
        return list(lineages)
    if levels is None:
        levels = _rank_levels()
    ranks = list(ranks) if ranks is not None else [None] * len(lineages)
    unranked = [
        label for label, lineage, node_ranks in zip(labels, lineages, ranks)
        if lineage is not None and not _has_ranks(lineage, node_ranks)
    ]
    if unranked:
        raise ValueError(
            "The 'ranked' metric requires the rank of every lineage node. "
            f"No ranks are available for {', '.join(map(repr, map(str, unranked)))}. "
            "Supply `ranks` to taxo_from_lineages() or use another metric."
        )
    return [
        None if lineage is None else _project_lineage(lineage, node_ranks, levels)
        for lineage, node_ranks in zip(lineages, ranks)
    ]


def _metric_values(h, da, db, metric):
    """
    Distance from MRCA depth ``h`` and lineage depths ``da``, ``db``
    (vectorized). For "ranked", depths refer to reduced lineages.
    """
    h = np.asarray(h, dtype=float)
    da = np.asarray(da, dtype=float)
    db = np.asarray(db, dtype=float)
    h, da, db = np.broadcast_arrays(h, da, db)
    with np.errstate(divide="ignore", invalid="ignore"):
        if metric == "reciprocal":
            values = 1.0 / h
        elif metric == "path":
            values = np.where(h == 0, np.inf, da + db - 2 * h)
        elif metric == "wu_palmer":
            values = 1.0 - 2.0 * h / (da + db)
        elif metric == "ranked":
            values = np.where(h == 0, np.inf, (da + db - 2 * h) / 2.0)
        else:  # pragma: no cover - validated by _match_metric()
            raise ValueError(f"Unknown distance metric {metric!r}")
    values = np.array(values, dtype=float)
    values[(h > 0) & (h == da) & (h == db)] = 0.0
    return values


def _as_taxon_list(taxa):
    """
    Materialise a collection of taxon names as a list.

    A single string is one taxon (as a length-one character vector in R),
    and iterables such as generators are consumed once.
    """
    if isinstance(taxa, (str, bytes)):
        return [taxa]
    if isinstance(taxa, Mapping):
        raise TypeError("taxa must be a sequence of taxon names, not a dictionary")
    return list(taxa)


def _is_distance_frame(taxa):
    """True for a plain distance-matrix DataFrame (not a resolution table)."""
    return isinstance(taxa, pd.DataFrame) and not getattr(
        taxa, "_taxodist_resolution", False
    )

def _common_prefix_depth(lin_a, lin_b):
    """Return the length of the continuous common prefix."""
    depth = 0
    for node_a, node_b in zip(lin_a, lin_b):
        if node_a != node_b:
            break
        depth += 1
    return depth


def _distance_value(lin_a, lin_b, metric="reciprocal"):
    """Numeric-only hierarchy distance (depth-based metrics only)."""
    mrca_depth = _common_prefix_depth(lin_a, lin_b)
    return float(_metric_values(mrca_depth, len(lin_a), len(lin_b), metric))


def _compute_distance(lin_a, lin_b, name_a="A", name_b="B",
                      metric="reciprocal", ranks_a=None, ranks_b=None):
    """
    Internal helper to compute distance between two parsed lineages.
    Equivalent to the hidden .compute_distance in R. The ``"ranked"`` metric
    needs the node ranks of both lineages (``ranks_a``, ``ranks_b``); the
    MRCA is always reported on the full lineages.
    """
    depth_a = len(lin_a)
    depth_b = len(lin_b)

    # Shared ancestry is the continuous common prefix from the root. A name
    # repeated after divergence is a homonym, not a shared hierarchy node.
    mrca_depth = _common_prefix_depth(lin_a, lin_b)

    if metric == "ranked":
        reduced = _metric_lineages(
            [lin_a, lin_b], [ranks_a, ranks_b], [name_a, name_b], metric
        )
        distance = float(_metric_values(
            _common_prefix_depth(reduced[0], reduced[1]),
            len(reduced[0]), len(reduced[1]), metric,
        ))
    else:
        distance = float(_metric_values(mrca_depth, depth_a, depth_b, metric))

    return {
        "distance": distance,
        "mrca": None if mrca_depth == 0 else lin_a[mrca_depth - 1],
        "mrca_depth": mrca_depth,
        "depth_a": depth_a,
        "depth_b": depth_b,
        "taxon_a": name_a,
        "taxon_b": name_b,
        "metric": metric,
    }


def _lineage_for_metric(taxon, metric, verbose=False):
    """
    ``(lineage, ranks)`` of a taxon: plain lineages from get_lineage() (ranks
    None) for depth-based metrics, lineages with ranks from lineage_ranks()
    for ``"ranked"``. ``(None, None)`` when the taxon cannot be found.
    """
    if metric == "ranked":
        ranked = lineage_ranks(taxon, verbose=verbose)
        if ranked is None:
            return None, None
        return list(ranked["node"]), list(ranked["rank"])
    return get_lineage(taxon, verbose=verbose), None


def taxo_distance(taxon_a, taxon_b, verbose=False, metric="reciprocal"):
    """
    Compute the taxonomic hierarchy distance between two taxa

    Given two taxon names, retrieves their lineages from The Taxonomicon and
    computes a taxonomic distance based on the depth of their most recent
    common ancestor (MRCA). With the default metric:

    d(A, B) = 1 / depth(MRCA(A, B))

    The deeper the shared ancestor, the smaller (closer to zero) the distance.
    This metric ensures that taxa diverging at the same node are always
    equidistant from any third taxon, regardless of lineage depth differences
    below the split.

    Metrics
    -------
    Let h be the MRCA depth and d_A, d_B the lineage depths (numbers of
    nodes, root included). Every metric is zero for identical lineages.

    - ``"reciprocal"`` (default): 1 / h. An ultrametric.
    - ``"path"``: d_A + d_B - 2h, the number of edges between the two nodes
      in the classification tree.
    - ``"wu_palmer"``: 1 - 2h / (d_A + d_B), the complement of the
      Wu-Palmer similarity, in [0, 1).
    - ``"ranked"``: half the path length after reducing both lineages to the
      root, the nodes of the standard Linnaean ranks (``RANK_LEVELS``), and
      the terminal node. For taxa of the same rank these are Clarke and
      Warwick's step weights (1 congeneric, 2 confamilial, ...). Requires
      lineage ranks (see lineage_ranks()).

    Lineages without a shared root have infinite distance, except under
    ``"wu_palmer"``, where their distance is 1.

    Parameters
    ----------
    taxon_a : str
        A character string giving the first taxon name.
    taxon_b : str
        A character string giving the second taxon name.
    verbose : bool
        Logical. If True, prints progress messages. Default False.
    metric : str
        One of ``"reciprocal"`` (default), ``"path"``, ``"wu_palmer"``, or
        ``"ranked"``.

    Returns
    -------
    dict or None
        A dictionary with the following elements:
        - distance: Numeric. The distance between the two taxa. Returns 0
          only when their complete lineages are identical. Distinct
          ancestor-descendant nodes have positive distance.
        - mrca: Character. The name of the most recent common ancestor.
        - mrca_depth: Integer. The depth of the MRCA node.
        - depth_a: Integer. The lineage depth of taxon A.
        - depth_b: Integer. The lineage depth of taxon B.
        - taxon_a: Character. Name of the first taxon.
        - taxon_b: Character. Name of the second taxon.
        - metric: Character. The metric used.
        Returns None if either taxon cannot be found.
    """
    metric = _match_metric(metric)
    lin_a, ranks_a = _lineage_for_metric(taxon_a, metric, verbose=verbose)
    lin_b, ranks_b = _lineage_for_metric(taxon_b, metric, verbose=verbose)

    if lin_a is None:
        print(f"Error: Could not retrieve lineage for {taxon_a}")
        return None
    if lin_b is None:
        print(f"Error: Could not retrieve lineage for {taxon_b}")
        return None

    return _compute_distance(
        lin_a, lin_b, taxon_a, taxon_b,
        metric=metric, ranks_a=ranks_a, ranks_b=ranks_b,
    )


def mrca(taxon_a, taxon_b, verbose=False):
    """
    Compute the most recent common ancestor of two taxa

    Retrieves the lineages of two taxa and returns the name of their most
    recent common ancestor (MRCA) — the deepest node shared by both lineages.

    Parameters
    ----------
    taxon_a : str
        A character string giving the first taxon name.
    taxon_b : str
        A character string giving the second taxon name.
    verbose : bool
        Logical. If True, prints progress messages. Default False.

    Returns
    -------
    str or None
        A character string giving the name of the MRCA, or None if
        either taxon cannot be found or no common ancestor exists.
    """
    result = taxo_distance(taxon_a, taxon_b, verbose=verbose)
    if result is None:
        return None
    return result["mrca"]

def distance_matrix(taxa, verbose=False, progress=True, metric="reciprocal"):
    """
    Compute pairwise taxonomic distances for a set of taxa

    Given a list of taxon names, computes all pairwise taxonomic hierarchy
    distances
    and returns a symmetric distance matrix. Lineages are cached after first
    retrieval to minimise redundant network requests.

    Parameters
    ----------
    taxa : sequence of str, TaxodistResolution, or TaxodistBundle
        Taxon names are retrieved normally. Resolution objects reuse their
        stored lineages (and ranks) and preserve their input labels. Bundles
        return their validated stored matrix without network access or
        recomputation.
    verbose : bool
        Logical. If True, prints progress for each pair. Default False.
    progress : bool
        Logical. If True, shows progress status. Default True.
    metric : str
        Distance metric; see taxo_distance(). Ignored for a bundle, whose
        stored matrix is returned.

    Returns
    -------
    pandas.DataFrame
        A symmetric numeric DataFrame containing pairwise distances.
        Row and column names are set to the input taxon names.
        Taxa that could not be found are included with NaN distances.
        ``DataFrame.attrs["metric"]`` records the metric (the equivalent of
        the ``"method"`` attribute of R's ``dist`` object).
    """
    if getattr(taxa, "_taxodist_bundle", False):
        from .bundle import validate_taxodist_bundle

        validate_taxodist_bundle(taxa, check_distances=False)
        # Return a copy so that changes do not corrupt the stored bundle.
        return taxa["matrix"].copy()

    metric = _match_metric(metric)
    return _distance_matrix_impl(
        taxa, verbose=verbose, progress=progress, metric=metric,
        levels=_rank_levels() if metric == "ranked" else None,
    )


def _distance_matrix_impl(taxa, verbose, progress, metric, levels):
    labels, lineages, ranks = _matrix_input_ranked(
        taxa, verbose=verbose, progress=progress, ranks=metric == "ranked"
    )
    lineages = _metric_lineages(lineages, ranks, labels, metric, levels)
    n = len(labels)

    if progress:
        print(f"Computing distances for {n * (n - 1) // 2} pairs...")
    mat = _distance_block(lineages, lineages, metric)
    np.fill_diagonal(mat, 0.0)
    if progress:
        print("Done.")

    result = pd.DataFrame(mat, index=labels, columns=labels)
    result.attrs["metric"] = metric
    return result


def cross_distances(x, y, verbose=False, progress=True, metric="reciprocal"):
    """
    Compute taxonomic distances between two sets of taxa

    Returns the rectangular matrix of distances from every taxon in ``x`` to
    every taxon in ``y``, for example from the species of a community to a
    set of reference taxa, without computing the full square matrix of the
    combined set.

    Parameters
    ----------
    x, y : sequence of str, TaxodistResolution, or TaxodistBundle
        Taxon names, or resolutions whose stored lineages are reused.
    verbose : bool
        If True, prints retrieval messages.
    progress : bool
        If True, prints progress messages.
    metric : str
        Distance metric; see taxo_distance().

    Returns
    -------
    pandas.DataFrame
        One row per element of ``x`` and one column per element of ``y``.
        Taxa that could not be found have NaN distances; taxa without a
        common ancestor have ``inf`` (1 under ``"wu_palmer"``).
        ``attrs["metric"]`` records the metric.
    """
    metric = _match_metric(metric)
    need_ranks = metric == "ranked"
    levels = _rank_levels() if need_ranks else None
    labels_x, lineages_x, ranks_x = _matrix_input_ranked(
        x, verbose=verbose, progress=progress, ranks=need_ranks
    )
    labels_y, lineages_y, ranks_y = _matrix_input_ranked(
        y, verbose=verbose, progress=progress, ranks=need_ranks
    )
    lineages_x = _metric_lineages(lineages_x, ranks_x, labels_x, metric, levels)
    lineages_y = _metric_lineages(lineages_y, ranks_y, labels_y, metric, levels)
    result = pd.DataFrame(
        _distance_block(lineages_x, lineages_y, metric),
        index=labels_x,
        columns=labels_y,
        dtype=float,
    )
    result.attrs["metric"] = metric
    return result


def _matrix_input(taxa, verbose=False, progress=True):
    """Return ``(labels, lineages)`` for names, resolutions, or bundles."""
    labels, lineages, _ = _matrix_input_ranked(
        taxa, verbose=verbose, progress=progress, ranks=False
    )
    return labels, lineages


def _cell_ranks(value):
    """A resolution ``ranks`` cell as a list, or None when absent."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return list(value)


def _matrix_input_ranked(taxa, verbose=False, progress=True, ranks=False):
    """
    Return ``(labels, lineages, ranks)`` for names, resolutions, or bundles.

    ``ranks`` holds one rank list (or None) per lineage. Resolutions provide
    their stored ranks when the ``ranks`` column exists and aligns with the
    lineage. Names are retrieved with lineage_ranks() when ``ranks=True``
    and with get_lineage() otherwise (their ranks are then None).
    """
    if getattr(taxa, "_taxodist_bundle", False):
        taxa = taxa["resolution"]
    if getattr(taxa, "_taxodist_resolution", False):
        required = {"input", "status", "lineage"}
        if not required.issubset(taxa.columns):
            raise ValueError("Invalid TaxodistResolution object")
        lineages = list(taxa["lineage"])
        rank_lists = [None] * len(lineages)
        if "ranks" in taxa.columns:
            for i, (lineage, value) in enumerate(zip(lineages, taxa["ranks"])):
                node_ranks = _cell_ranks(value)
                if (
                    lineage is not None
                    and node_ranks is not None
                    and len(node_ranks) == len(lineage)
                ):
                    rank_lists[i] = node_ranks
        return list(taxa["input"]), lineages, rank_lists
    if isinstance(taxa, (str, bytes)):
        raise TypeError("taxa must be a sequence of taxon names, not a single string")
    if isinstance(taxa, (Mapping, pd.DataFrame)):
        # R rejects lists here; a dict or table of names is ambiguous.
        raise TypeError(
            "taxa must be a sequence of taxon names, a TaxodistResolution, or "
            "a TaxodistBundle"
        )

    labels = list(taxa)
    if progress:
        print(f"Fetching {len(labels)} lineages...")
    lineages, rank_lists = [], []
    for t in labels:
        if t is None or (isinstance(t, float) and np.isnan(t)):
            lineages.append(None)
            rank_lists.append(None)
        elif ranks:
            lineage, node_ranks = _lineage_for_metric(t, "ranked", verbose=verbose)
            lineages.append(lineage)
            rank_lists.append(node_ranks)
        else:
            lineages.append(get_lineage(t, verbose=verbose))
            rank_lists.append(None)
    if progress:
        print("Lineages fetched.")
    return labels, lineages, rank_lists


def _prefix_node_ids(lineages):
    """
    Integer matrix identifying each lineage node by its full path from root.

    Two lineages share the node at depth ``d`` if and only if their
    identifiers at ``d`` are equal, and equality at ``d`` implies equality at
    every shallower depth, so the MRCA depth of two taxa is the number of
    depths at which their identifiers agree. Missing depths are ``-1``.
    """
    lengths = [0 if lineage is None else len(lineage) for lineage in lineages]
    max_depth = max(lengths, default=0)
    ids = np.full((len(lineages), max_depth), -1, dtype=np.int64)
    codes = {}
    for row, lineage in enumerate(lineages):
        parent = -1
        for depth, node in enumerate(lineage or ()):
            key = (depth, parent, node)
            parent = codes.setdefault(key, len(codes))
            ids[row, depth] = parent
    return ids, np.asarray(lengths, dtype=np.int64)


def _distance_block(lineages_a, lineages_b, metric="reciprocal"):
    """
    Vectorized distances between every lineage in ``a`` and ``b``. For the
    ``"ranked"`` metric the lineages must already be reduced with
    ``_metric_lineages()``.
    """
    ids, lengths = _prefix_node_ids(list(lineages_a) + list(lineages_b))
    n_a = len(lineages_a)
    ids_a, ids_b = ids[:n_a], ids[n_a:]
    len_a, len_b = lengths[:n_a], lengths[n_a:]

    out = np.full((n_a, len(lineages_b)), np.nan)
    for column in range(len(lineages_b)):
        shared = (ids_a == ids_b[column]) & (ids_a >= 0)
        mrca_depth = shared.sum(axis=1)
        out[:, column] = _metric_values(mrca_depth, len_a, len_b[column], metric)

    missing_a = np.array([lineage is None for lineage in lineages_a], dtype=bool)
    missing_b = np.array([lineage is None for lineage in lineages_b], dtype=bool)
    out[missing_a, :] = np.nan
    out[:, missing_b] = np.nan
    return out


def closest_relative(taxon, candidates, verbose=False, metric="reciprocal"):
    """
    Find the closest relative of a taxon among a set of candidates

    Given a query taxon and a list of candidate taxa, returns the candidate
    with the smallest phylogenetic distance to the query.

    Parameters
    ----------
    taxon : str
        A character string giving the query taxon name.
    candidates : list of str
        A list of candidate taxon names to compare against.
    verbose : bool
        Logical. If True, prints progress messages. Default False.
    metric : str
        Distance metric; see taxo_distance().

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame with columns 'taxon' (candidate name) and 'distance'
        (tree metric distance), sorted by distance ascending. Returns None if
        the query taxon cannot be found.
    """
    metric = _match_metric(metric)
    candidates = _as_taxon_list(candidates)
    query_lin, query_ranks = _lineage_for_metric(taxon, metric, verbose=verbose)
    if query_lin is None:
        print(f"Error: Could not retrieve lineage for {taxon}")
        return None

    if len(candidates) == 0:
        return pd.DataFrame({
            "taxon": pd.Series(dtype="object"),
            "distance": pd.Series(dtype="float64")
        })

    results =[]
    for cand in candidates:
        cand_lin, cand_ranks = _lineage_for_metric(cand, metric, verbose=verbose)
        if cand_lin is None:
            results.append({"taxon": cand, "distance": float('nan')})
        else:
            dist_result = _compute_distance(
                query_lin, cand_lin, taxon, cand,
                metric=metric, ranks_a=query_ranks, ranks_b=cand_ranks,
            )
            results.append({"taxon": cand, "distance": dist_result["distance"]})

    df = pd.DataFrame(results)
    return df.sort_values(by="distance", na_position='last').reset_index(drop=True)

def focal_distances(focal, community, verbose=False, progress=True,
                    metric="reciprocal"):
    """
    Compute distances from a focal taxon to a community of taxa

    Given a focal taxon and a list of community taxa, retrieves all lineages
    and computes the taxonomic distance from the focal taxon to each member
    of the community. Returns a sorted DataFrame that also reports the most
    recent common ancestor (MRCA) for each pair, making it easy to interpret
    why a taxon is close or distant.

    Parameters
    ----------
    focal : str
        A character string giving the focal taxon name.
    community : list of str
        A list of community taxon names to compare against. The focal taxon
        may be included — it will receive a distance of 0.
    verbose : bool
        If True, prints progress messages. Default False.
    progress : bool
        If True, prints progress status. Default True.
    metric : str
        Distance metric; see taxo_distance().

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame with columns:
        - taxon: str. Community taxon name.
        - distance: float. Taxonomic distance to the focal taxon.
        - mrca: str. Name of the most recent common ancestor.
        - mrca_depth: int. Depth of the MRCA node.
        Rows are sorted by distance ascending (closest relatives first).
        Returns None if the focal taxon cannot be found.
    """
    metric = _match_metric(metric)
    community = _as_taxon_list(community)
    focal_lin, focal_ranks = _lineage_for_metric(focal, metric, verbose=verbose)
    if focal_lin is None:
        print(f"Error: Could not retrieve lineage for {focal}")
        return None

    if len(community) == 0:
        result = pd.DataFrame({
            "taxon": pd.Series(dtype="object"),
            "distance": pd.Series(dtype="float64"),
            "mrca": pd.Series(dtype="object"),
            "mrca_depth": pd.Series(dtype="Int64")
        })
        result.attrs["focal"] = focal
        return result

    if progress:
        print(f"Computing focal distances for {len(community)} taxa...")

    results = []
    for i, taxon in enumerate(community):
        if progress:
            print(f"  [{i + 1}/{len(community)}] {taxon}")

        cand_lin, cand_ranks = _lineage_for_metric(taxon, metric, verbose=verbose)
        if cand_lin is None:
            results.append({
                "taxon":      taxon,
                "distance":   float("nan"),
                "mrca":       None,
                "mrca_depth": None
            })
        else:
            res = _compute_distance(
                focal_lin, cand_lin, focal, taxon,
                metric=metric, ranks_a=focal_ranks, ranks_b=cand_ranks,
            )
            results.append({
                "taxon":      taxon,
                "distance":   res["distance"],
                "mrca":       res["mrca"],
                "mrca_depth": res["mrca_depth"]
            })

    if progress:
        print("Done.")

    df = pd.DataFrame(results, columns=["taxon", "distance", "mrca", "mrca_depth"])
    df["distance"] = df["distance"].astype("float64")
    # Nullable integers, as for an empty community.
    df["mrca_depth"] = pd.array(df["mrca_depth"], dtype="Int64")
    df = df.sort_values(by="distance", na_position="last").reset_index(drop=True)
    df.attrs["focal"] = focal
    return df

def lineage_depth(taxon, verbose=False):
    """
    Get the lineage depth of a taxon

    Returns the number of nodes in the lineage of a taxon, from root to tip.
    This reflects how deeply nested the taxon is within the taxonomic hierarchy.

    Parameters
    ----------
    taxon : str
        A character string giving the taxon name.
    verbose : bool
        Logical. If True, prints progress messages. Default False.

    Returns
    -------
    int or None
        An integer giving the lineage depth, or None if the taxon cannot
        be found.
    """
    lin = get_lineage(taxon, verbose=verbose)
    if lin is None:
        return None
    return len(lin)

def check_coverage(taxa, verbose=False):
    """
    Check whether a taxon is covered by The Taxonomicon

    Queries The Taxonomicon for a taxon name and returns a boolean Series indicating
    whether the taxon was found. Useful for pre-screening a list of names
    before running distance computations.

    Parameters
    ----------
    taxa : str or iterable of str
        One taxon name, or an iterable of taxon names (a single string is
        treated as one name, not as a sequence of characters).
    verbose : bool
        Logical. If True, prints progress messages. Default False.

    Returns
    -------
    pandas.Series
        A nullable boolean Series indexed by taxon names. True indicates the
        taxon was found, False that it is absent from The Taxonomicon, and
        ``pd.NA`` that the answer is unknown because the server could not be
        reached.
    """
    taxa = _as_taxon_list(taxa)
    outcome = {"found": True, "not_found": False}
    values = [
        outcome.get(_lookup_taxonomicon_id(t, verbose=verbose)[1], pd.NA)
        for t in taxa
    ]
    return pd.Series(values, index=list(taxa), dtype="boolean")


def taxo_cluster(taxa, method="average", **kwargs):
    """
    Cluster taxa by taxonomic distance

    Computes pairwise taxonomic distances and performs hierarchical clustering.

    Parameters
    ----------
    taxa : list of str, TaxodistResolution, TaxodistBundle, or pandas.DataFrame
        Taxon names, a resolution or bundle (passed to distance_matrix()), or
        a distance matrix DataFrame from distance_matrix().
    method : str
        Clustering method passed to scipy.cluster.hierarchy.linkage().
        Default "average" (UPGMA), which works well with taxonomic distances.
    verbose : bool
        Logical. If True, prints progress messages. Default False.
    progress : bool
        Logical. If True, shows progress status. Default True.
    metric : str
        Distance metric passed to distance_matrix() (see taxo_distance());
        the matrix records it in ``attrs["metric"]``.

    Returns
    -------
    dict
        A dictionary with:
        - hclust: The hierarchical clustering linkage matrix (SciPy format).
        - dist: The underlying distance matrix DataFrame.
        - method: The linkage method used.
    """
    # A TaxodistResolution is a DataFrame subclass but not a distance matrix:
    # like names and bundles, it is passed through distance_matrix().
    if _is_distance_frame(taxa):
        d = taxa
    else:
        d = distance_matrix(taxa, **kwargs)

    if np.isnan(d.values).any():
        warnings.warn("Distance matrix contains NaN values (taxa not found or server offline). Clustering skipped.")
        return {"hclust": None, "dist": d, "method": method}

    if not np.isfinite(d.values).all():
        warnings.warn("Distance matrix contains infinite values (no shared ancestor). Clustering skipped.")
        return {"hclust": None, "dist": d, "method": method}

    if d.shape[0] < 2:
        warnings.warn("At least two taxa are required for clustering. Clustering skipped.")
        return {"hclust": None, "dist": d, "method": method}

    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    condensed_dist = squareform(d.values, checks=False)
    hc = linkage(condensed_dist, method=method)

    return {"hclust": hc, "dist": d, "method": method}

def taxo_ordinate(taxa, k=2, **kwargs):
    """
    Ordinate taxa in taxonomic distance space

    Computes pairwise taxonomic distances and applies classical multidimensional
    scaling (PCoA) to project taxa into a low-dimensional space.

    Parameters
    ----------
    taxa : list of str, TaxodistResolution, TaxodistBundle, or pandas.DataFrame
        Taxon names, a resolution or bundle (passed to distance_matrix()), or
        a distance matrix DataFrame from distance_matrix().
    k : int
        Number of dimensions. Default 2.
    verbose : bool
        Logical. If True, prints progress messages. Default False.
    progress : bool
        Logical. If True, shows progress status. Default True.
    metric : str
        Distance metric passed to distance_matrix() (see taxo_distance());
        the matrix records it in ``attrs["metric"]``.

    Returns
    -------
    dict
        A dictionary with:
        - points: A DataFrame of coordinates (taxa x k dimensions).
        - dist: The underlying distance matrix DataFrame.
        - GOF: Goodness-of-fit (matches R's cmdscale GOF).
        - eig: The eigenvalues computed during PCoA.
    """
    if _is_distance_frame(taxa):
        d = taxa
    else:
        d = distance_matrix(taxa, **kwargs)

    if np.isnan(d.values).any():
        warnings.warn("Distance matrix contains NaN values. Ordination skipped.")
        return {"points": None, "dist": d, "GOF": None, "eig": None}

    if not np.isfinite(d.values).all():
        warnings.warn("Distance matrix contains infinite values (no shared ancestor). Ordination skipped.")
        return {"points": None, "dist": d, "GOF": None, "eig": None}

    mat = d.values
    n = mat.shape[0]

    if n < 2:
        warnings.warn("At least two taxa are required for ordination. Ordination skipped.")
        return {"points": None, "dist": d, "GOF": None, "eig": None}

    valid_k = (
        not isinstance(k, (bool, np.bool_))
        and isinstance(k, Real)
        and np.isfinite(k)
        and k >= 1
        and float(k).is_integer()
    )
    if not valid_k:
        raise ValueError("k must be a single positive integer.")
    k = int(k)

    max_k = n - 1
    if k > max_k:
        warnings.warn(f"k reduced from {k} to {max_k}, the maximum for {n} taxa.")
        k = max_k

    A = -0.5 * (mat ** 2)
    J = np.eye(n) - np.ones((n, n)) / n
    B = J.dot(A).dot(J)

    eigvals, eigvecs = np.linalg.eigh(B)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    selected_eigvals = eigvals[:k]
    positive = selected_eigvals > 0
    if np.count_nonzero(positive) < k:
        warnings.warn(
            f"only {np.count_nonzero(positive)} of the first {k} eigenvalues are > 0"
        )

    retained_eigvals = selected_eigvals[positive]
    retained_vectors = eigvecs[:, :k][:, positive]
    points = retained_vectors * np.sqrt(retained_eigvals)

    numerator = np.sum(retained_eigvals)
    absolute_total = np.sum(np.abs(eigvals))
    positive_total = np.sum(np.maximum(eigvals, 0))
    # Match stats::cmdscale(): GOF[0] uses all absolute eigenvalues and
    # GOF[1] uses only the positive eigenvalue total.
    gof_1 = numerator / absolute_total if absolute_total > 0 else float("nan")
    gof_2 = numerator / positive_total if positive_total > 0 else float("nan")

    points_df = pd.DataFrame(
        points, 
        index=d.index, 
        columns=[f"PC{i+1}" for i in range(points.shape[1])]
    )

    return {
        "points": points_df,
        "dist": d,
        "GOF": [gof_1, gof_2],
        "eig": eigvals
    }