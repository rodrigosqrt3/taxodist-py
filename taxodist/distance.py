import warnings
from numbers import Real
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from .fetch import get_lineage, get_taxonomicon_id, _lookup_taxonomicon_id

def _common_prefix_depth(lin_a, lin_b):
    """Return the length of the continuous common prefix."""
    depth = 0
    for node_a, node_b in zip(lin_a, lin_b):
        if node_a != node_b:
            break
        depth += 1
    return depth


def _distance_value(lin_a, lin_b):
    """Numeric-only hierarchy distance used by matrix construction."""
    mrca_depth = _common_prefix_depth(lin_a, lin_b)
    if mrca_depth == 0:
        return float("inf")
    if len(lin_a) == len(lin_b) == mrca_depth:
        return 0.0
    return 1.0 / mrca_depth


def _compute_distance(lin_a, lin_b, name_a="A", name_b="B"):
    """
    Internal helper to compute distance between two parsed lineages.
    Equivalent to the hidden .compute_distance in R.
    """
    depth_a = len(lin_a)
    depth_b = len(lin_b)

    # Shared ancestry is the continuous common prefix from the root. A name
    # repeated after divergence is a homonym, not a shared hierarchy node.
    mrca_depth = _common_prefix_depth(lin_a, lin_b)

    if mrca_depth == 0:
        return {
            "distance": float('inf'),
            "mrca": None,
            "mrca_depth": 0,
            "depth_a": depth_a,
            "depth_b": depth_b,
            "taxon_a": name_a,
            "taxon_b": name_b
        }

    mrca_name = lin_a[mrca_depth - 1]
    same_lineage = depth_a == depth_b == mrca_depth
    distance = 0.0 if same_lineage else 1.0 / mrca_depth

    return {
        "distance": distance,
        "mrca": mrca_name,
        "mrca_depth": mrca_depth,
        "depth_a": depth_a,
        "depth_b": depth_b,
        "taxon_a": name_a,
        "taxon_b": name_b
    }


def taxo_distance(taxon_a, taxon_b, verbose=False):
    """
    Compute the taxonomic hierarchy distance between two taxa

    Given two taxon names, retrieves their lineages from The Taxonomicon and
    computes a taxonomic distance based on the depth of their most recent
    common ancestor (MRCA):

    d(A, B) = 1 / depth(MRCA(A, B))

    The deeper the shared ancestor, the smaller (closer to zero) the distance.
    This metric ensures that taxa diverging at the same node are always
    equidistant from any third taxon, regardless of lineage depth differences
    below the split.

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
        Returns None if either taxon cannot be found.
    """
    lin_a = get_lineage(taxon_a, verbose=verbose)
    lin_b = get_lineage(taxon_b, verbose=verbose)

    if lin_a is None:
        print(f"Error: Could not retrieve lineage for {taxon_a}")
        return None
    if lin_b is None:
        print(f"Error: Could not retrieve lineage for {taxon_b}")
        return None

    return _compute_distance(lin_a, lin_b, taxon_a, taxon_b)


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

def distance_matrix(taxa, verbose=False, progress=True):
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
        stored lineages and preserve their input labels. Bundles return their
        validated stored matrix without network access or recomputation.
    verbose : bool
        Logical. If True, prints progress for each pair. Default False.
    progress : bool
        Logical. If True, shows progress status. Default True.

    Returns
    -------
    pandas.DataFrame
        A symmetric numeric DataFrame containing pairwise distances.
        Row and column names are set to the input taxon names.
        Taxa that could not be found are included with NaN distances.
    """
    if getattr(taxa, "_taxodist_bundle", False):
        from .bundle import validate_taxodist_bundle

        validate_taxodist_bundle(taxa, check_distances=False)
        return taxa["matrix"]

    labels, lineages = _matrix_input(taxa, verbose=verbose, progress=progress)
    n = len(labels)

    if progress:
        print(f"Computing distances for {n * (n - 1) // 2} pairs...")
    mat = _distance_block(lineages, lineages)
    np.fill_diagonal(mat, 0.0)
    if progress:
        print("Done.")

    return pd.DataFrame(mat, index=labels, columns=labels)


def cross_distances(x, y, verbose=False, progress=True):
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

    Returns
    -------
    pandas.DataFrame
        One row per element of ``x`` and one column per element of ``y``.
        Taxa that could not be found have NaN distances; taxa without a
        common ancestor have ``inf``.
    """
    labels_x, lineages_x = _matrix_input(x, verbose=verbose, progress=progress)
    labels_y, lineages_y = _matrix_input(y, verbose=verbose, progress=progress)
    return pd.DataFrame(
        _distance_block(lineages_x, lineages_y),
        index=labels_x,
        columns=labels_y,
        dtype=float,
    )


def _matrix_input(taxa, verbose=False, progress=True):
    """Return ``(labels, lineages)`` for names, resolutions, or bundles."""
    if getattr(taxa, "_taxodist_bundle", False):
        taxa = taxa["resolution"]
    if getattr(taxa, "_taxodist_resolution", False):
        required = {"input", "status", "lineage"}
        if not required.issubset(taxa.columns):
            raise ValueError("Invalid TaxodistResolution object")
        return list(taxa["input"]), list(taxa["lineage"])
    if isinstance(taxa, (str, bytes)):
        raise TypeError("taxa must be a sequence of taxon names, not a single string")

    labels = list(taxa)
    if progress:
        print(f"Fetching {len(labels)} lineages...")
    lineages = [
        None if t is None or (isinstance(t, float) and np.isnan(t))
        else get_lineage(t, verbose=verbose)
        for t in labels
    ]
    if progress:
        print("Lineages fetched.")
    return labels, lineages


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


def _distance_block(lineages_a, lineages_b):
    """Vectorized distances between every lineage in ``a`` and ``b``."""
    ids, lengths = _prefix_node_ids(list(lineages_a) + list(lineages_b))
    n_a = len(lineages_a)
    ids_a, ids_b = ids[:n_a], ids[n_a:]
    len_a, len_b = lengths[:n_a], lengths[n_a:]

    out = np.full((n_a, len(lineages_b)), np.nan)
    for column in range(len(lineages_b)):
        shared = (ids_a == ids_b[column]) & (ids_a >= 0)
        mrca_depth = shared.sum(axis=1)
        with np.errstate(divide="ignore"):
            values = 1.0 / mrca_depth
        identical = (mrca_depth > 0) & (mrca_depth == len_b[column]) & (len_a == len_b[column])
        values[identical] = 0.0
        out[:, column] = values

    missing_a = np.array([lineage is None for lineage in lineages_a], dtype=bool)
    missing_b = np.array([lineage is None for lineage in lineages_b], dtype=bool)
    out[missing_a, :] = np.nan
    out[:, missing_b] = np.nan
    return out


def closest_relative(taxon, candidates, verbose=False):
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

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame with columns 'taxon' (candidate name) and 'distance'
        (tree metric distance), sorted by distance ascending. Returns None if
        the query taxon cannot be found.
    """
    query_lin = get_lineage(taxon, verbose=verbose)
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
        cand_lin = get_lineage(cand, verbose=verbose)
        if cand_lin is None:
            results.append({"taxon": cand, "distance": float('nan')})
        else:
            dist_result = _compute_distance(query_lin, cand_lin, taxon, cand)
            results.append({"taxon": cand, "distance": dist_result["distance"]})

    df = pd.DataFrame(results)
    return df.sort_values(by="distance", na_position='last').reset_index(drop=True)

def focal_distances(focal, community, verbose=False, progress=True):
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
    focal_lin = get_lineage(focal, verbose=verbose)
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

        cand_lin = get_lineage(taxon, verbose=verbose)
        if cand_lin is None:
            results.append({
                "taxon":      taxon,
                "distance":   float("nan"),
                "mrca":       None,
                "mrca_depth": None
            })
        else:
            res = _compute_distance(focal_lin, cand_lin, focal, taxon)
            results.append({
                "taxon":      taxon,
                "distance":   res["distance"],
                "mrca":       res["mrca"],
                "mrca_depth": res["mrca_depth"]
            })

    if progress:
        print("Done.")

    df = pd.DataFrame(results)
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
    taxa : list of str
        A list of one or more taxon names.
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
    taxa : list of str or pandas.DataFrame
        A list of taxon names, or a distance matrix DataFrame from distance_matrix().
    method : str
        Clustering method. Default "average" (UPGMA), which works well 
        with taxonomic distances.
    verbose : bool
        Logical. If True, prints progress messages. Default False.
    progress : bool
        Logical. If True, shows progress status. Default True.

    Returns
    -------
    dict
        A dictionary with:
        - hclust: The hierarchical clustering linkage matrix (SciPy format).
        - dist: The underlying distance matrix DataFrame.
    """
    if isinstance(taxa, pd.DataFrame):
        d = taxa
    else:
        d = distance_matrix(taxa, **kwargs)

    if np.isnan(d.values).any():
        warnings.warn("Distance matrix contains NaN values (taxa not found or server offline). Clustering skipped.")
        return {"hclust": None, "dist": d}

    if not np.isfinite(d.values).all():
        warnings.warn("Distance matrix contains infinite values (no shared ancestor). Clustering skipped.")
        return {"hclust": None, "dist": d}

    if d.shape[0] < 2:
        warnings.warn("At least two taxa are required for clustering. Clustering skipped.")
        return {"hclust": None, "dist": d}

    condensed_dist = squareform(d.values, checks=False)
    hc = linkage(condensed_dist, method=method)

    return {"hclust": hc, "dist": d}

def taxo_ordinate(taxa, k=2, **kwargs):
    """
    Ordinate taxa in taxonomic distance space

    Computes pairwise taxonomic distances and applies classical multidimensional
    scaling (PCoA) to project taxa into a low-dimensional space.

    Parameters
    ----------
    taxa : list of str or pandas.DataFrame
        A list of taxon names, or a distance matrix DataFrame from distance_matrix().
    k : int
        Number of dimensions. Default 2.
    verbose : bool
        Logical. If True, prints progress messages. Default False.
    progress : bool
        Logical. If True, shows progress status. Default True.

    Returns
    -------
    dict
        A dictionary with:
        - points: A DataFrame of coordinates (taxa x k dimensions).
        - dist: The underlying distance matrix DataFrame.
        - GOF: Goodness-of-fit (matches R's cmdscale GOF).
        - eig: The eigenvalues computed during PCoA.
    """
    if isinstance(taxa, pd.DataFrame):
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