import math
import warnings
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from .fetch import get_lineage, get_taxonomicon_id

def _compute_distance(lin_a, lin_b, name_a="A", name_b="B"):
    """
    Internal helper to compute distance between two parsed lineages.
    Equivalent to the hidden .compute_distance in R.
    """
    depth_a = len(lin_a)
    depth_b = len(lin_b)

    shared =[]
    set_b = set(lin_b)
    for x in lin_a:
        if x in set_b and x not in shared:
            shared.append(x)

    if not shared:
        return {
            "distance": float('inf'),
            "mrca": None,
            "mrca_depth": 0,
            "depth_a": depth_a,
            "depth_b": depth_b,
            "taxon_a": name_a,
            "taxon_b": name_b
        }

    positions_in_a =[lin_a.index(x) + 1 for x in shared]
    mrca_idx = positions_in_a.index(max(positions_in_a))
    mrca_depth = positions_in_a[mrca_idx]
    
    mrca_name = lin_a[mrca_depth - 1]
    
    is_ancestral = (mrca_name == lin_a[-1]) or (mrca_name == lin_b[-1]) or (name_a in lin_b) or (name_b in lin_a)
    distance = 0.0 if is_ancestral else 1.0 / mrca_depth

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
    Compute the phylogenetic distance between two taxa

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
          if one taxon is an ancestor of the other.
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

    Given a list of taxon names, computes all pairwise phylogenetic distances
    and returns a symmetric distance matrix. Lineages are cached after first
    retrieval to minimise redundant network requests.

    Parameters
    ----------
    taxa : list of str
        A list of taxon names.
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
    n = len(taxa)
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 0.0)

    if progress:
        print(f"Fetching {n} lineages...")
        
    lineages =[get_lineage(t, verbose=verbose) for t in taxa]
    
    if progress:
        print("Lineages fetched.")
        total_pairs = int(n * (n - 1) / 2)
        print(f"Computing distances for {total_pairs} pairs...")

    for i in range(n - 1):
        for j in range(i + 1, n):
            if lineages[i] is not None and lineages[j] is not None:
                result = _compute_distance(lineages[i], lineages[j], taxa[i], taxa[j])
                mat[i, j] = result["distance"]
                mat[j, i] = result["distance"]

    if progress:
        print("Done.")

    return pd.DataFrame(mat, index=taxa, columns=taxa)


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

    if progress:
        print(f"Computing focal distances for {len(community)} taxa...")

    results = []
    for i, taxon in enumerate(community):
        if progress:
            print(f"  [{i + 1}/{len(community)}] {taxon}")

        if taxon == focal:
            results.append({
                "taxon":      taxon,
                "distance":   0.0,
                "mrca":       focal,
                "mrca_depth": len(focal_lin)
            })
        else:
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
        A boolean Series indexed by taxon names. True indicates the taxon was found,
        False indicates it was not.
    """    
    result = {}
    for t in taxa:
        res = get_taxonomicon_id(t, verbose=verbose)
        result[t] = res is not None
        
    return pd.Series(result)


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

    mat = d.values
    n = mat.shape[0]

    A = -0.5 * (mat ** 2)
    J = np.eye(n) - np.ones((n, n)) / n
    B = J.dot(A).dot(J)

    eigvals, eigvecs = np.linalg.eigh(B)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvals_k = np.maximum(eigvals[:k], 0)
    points = eigvecs[:, :k] * np.sqrt(eigvals_k)
    
    eig_pos = np.maximum(eigvals, 0)
    gof_1 = np.sum(eigvals_k) / np.sum(eig_pos) if np.sum(eig_pos) > 0 else 0
    gof_2 = np.sum(eigvals_k) / np.sum(np.abs(eigvals)) if np.sum(np.abs(eigvals)) > 0 else 0

    points_df = pd.DataFrame(
        points, 
        index=d.index, 
        columns=[f"PC{i+1}" for i in range(k)]
    )

    return {
        "points": points_df,
        "dist": d,
        "GOF": [gof_1, gof_2],
        "eig": eigvals
    }