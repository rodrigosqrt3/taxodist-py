import pandas as pd
from .fetch import get_lineage
from .distance import _compute_distance
from .distance import distance_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.cluster.hierarchy import dendrogram

def compare_lineages(taxon_a, taxon_b, verbose=False):
    """
    Compare lineages of two taxa side by side

    Prints the lineages of two taxa aligned at their most recent common
    ancestor, making the point of divergence easy to identify.

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
        Returns a dictionary with elements 'lineage_a', 'lineage_b',
        and 'mrca_depth'.
    """
    lin_a = get_lineage(taxon_a, verbose=verbose)
    lin_b = get_lineage(taxon_b, verbose=verbose)

    if lin_a is None or lin_b is None:
        print("Error: Could not retrieve one or both lineages")
        return None

    result = _compute_distance(lin_a, lin_b, taxon_a, taxon_b)
    mrca_d = result["mrca_depth"]

    print("Lineage Comparison")
    print(f"MRCA: {result['mrca']} at depth {mrca_d}\n")

    if mrca_d > 0:
        print(f"Shared lineage ({mrca_d} nodes):")
        for node in lin_a[:mrca_d]:
            print(f"  {node}")

    print(f"\n{taxon_a} only ({len(lin_a) - mrca_d} nodes):")
    if len(lin_a) > mrca_d:
        for node in lin_a[mrca_d:]:
            print(f"> {node}")

    print(f"\n{taxon_b} only ({len(lin_b) - mrca_d} nodes):")
    if len(lin_b) > mrca_d:
        for node in lin_b[mrca_d:]:
            print(f"> {node}")

    return {
        "lineage_a": lin_a,
        "lineage_b": lin_b,
        "mrca_depth": mrca_d
    }


def shared_clades(taxon_a, taxon_b, verbose=False):
    """
    List all clades shared between two taxa

    Returns the list of clade names forming the shared trunk of two taxa's
    lineages, from root down to (and including) their MRCA.

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
    list or None
        A list of shared clade names ordered from root to MRCA,
        or None if either taxon cannot be found.
    """
    lin_a = get_lineage(taxon_a, verbose=verbose)
    lin_b = get_lineage(taxon_b, verbose=verbose)
    
    if lin_a is None or lin_b is None:
        return None

    result = _compute_distance(lin_a, lin_b, taxon_a, taxon_b)
    if result["mrca_depth"] == 0:
        return []
        
    return lin_a[:result["mrca_depth"]]


def is_member(taxon, clade, verbose=False):
    """
    Test whether one taxon is nested within another

    Returns True if taxon is a member of clade — i.e., if the clade name
    appears in the taxon's lineage.

    Parameters
    ----------
    taxon : str
        A character string giving the taxon name to test.
    clade : str
        A character string giving the clade name to test membership in.
    verbose : bool
        Logical. If True, prints progress messages. Default False.

    Returns
    -------
    bool or None
        A logical value, or None if the taxon cannot be found.
    """
    lin = get_lineage(taxon, verbose=verbose)
    if lin is None:
        return None

    clade_lower = clade.lower()
    for node in lin:
        if node.lower().startswith(clade_lower):
            return True
            
    return False


def filter_clade(taxa, clade, verbose=False):
    """
    Filter a vector of taxa to those belonging to a given clade

    Given a list of taxon names and a clade name, returns only those taxa
    whose lineage includes the specified clade.

    Parameters
    ----------
    taxa : list of str
        A list of taxon names.
    clade : str
        A character string giving the clade to filter by.
    verbose : bool
        Logical. If True, prints progress messages. Default False.

    Returns
    -------
    list
        A list of taxa that are members of the specified clade.
    """
    kept =[]
    for t in taxa:
        res = is_member(t, clade, verbose=verbose)
        if res is True:
            kept.append(t)
    return kept

def taxo_heatmap(taxa, **kwargs):
    """
    Plot a taxonomic heatmap

    Computes pairwise taxonomic distances and plots a heatmap with hierarchical
    clustering dendrograms on the margins. Darker/hotter colors typically
    represent smaller distances (closer relatives).

    Parameters
    ----------
    taxa : list of str or pandas.DataFrame
        A list of taxon names, or a distance matrix DataFrame.
    **kwargs : dict
        Additional arguments passed to seaborn.clustermap.

    Returns
    -------
    pandas.DataFrame
        The underlying distance matrix.
    """
    if isinstance(taxa, pd.DataFrame):
        d = taxa
    else:
        d = distance_matrix(taxa)

    if np.isnan(d.values).any():
        warnings.warn("Distance matrix contains NaN values. Heatmap skipped.")
        return d
    sns.clustermap(d, cmap="YlGnBu_r", annot=True, **kwargs)
    plt.show()
    
    return d

def taxo_path(taxon_a, taxon_b, verbose=False):
    """
    Get the taxonomic path between two taxa

    Returns the full node-by-node path from one taxon up to their most recent
    common ancestor (MRCA) and back down to the other taxon. The result is a
    DataFrame with one row per node, making it easy to inspect, filter, or
    pipe into other functions.

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
    pandas.DataFrame or None
        A DataFrame with columns:
        - node: Character. The clade or taxon name at this step.
        - depth: Integer. The depth of this node in the full lineage.
        - direction: Character. One of "a" (ascending), "mrca", or "b" (descending).
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

    result = _compute_distance(lin_a, lin_b, taxon_a, taxon_b)
    mrca_d = result["mrca_depth"]

    path_data =[]

    if mrca_d > 0:
        side_a_nodes = lin_a[:mrca_d][::-1]
        depths_a = list(range(mrca_d - 1, 0, -1))
        for i, node in enumerate(side_a_nodes[:-1]):
            path_data.append({"node": node, "depth": depths_a[i], "direction": "a"})

    path_data.append({
        "node": result["mrca"], 
        "depth": mrca_d, 
        "direction": "mrca"
    })

    side_b_nodes = lin_b[mrca_d:]
    depths_b = list(range(1, len(side_b_nodes) + 1))
    for i, node in enumerate(side_b_nodes):
        path_data.append({"node": node, "depth": depths_b[i], "direction": "b"})

    df = pd.DataFrame(path_data)
    df.attrs["taxon_a"] = taxon_a
    df.attrs["taxon_b"] = taxon_b

    return df

def print_taxodist_result(x):
    """Print method for taxodist distance results"""
    if x is None:
        return x
    print("Taxonomic Distance")
    print(f"* {x['taxon_a']} vs {x['taxon_b']}")
    print(f"  Distance : {x['distance']}")
    print(f"  MRCA     : {x['mrca']} (depth {x['mrca_depth']})")
    print(f"  Depth A  : {x['depth_a']}")
    print(f"  Depth B  : {x['depth_b']}")
    return x

def plot_taxodist_cluster(x, main="Taxonomic Clustering", xlab="", sub="Method: hierarchical clustering", **kwargs):
    """Plot method for taxodist_cluster objects"""
    if x is None or x.get("hclust") is None:
        return x
        
    plt.figure(figsize=(10, 6))
    dendrogram(x["hclust"], labels=x["dist"].index.tolist(), **kwargs)
    plt.title(main)
    plt.xlabel(xlab)
    plt.suptitle(sub, fontsize=10, y=0.92)
    plt.show()
    return x

def plot_taxodist_ord(x, main="Taxonomic Ordination (PCoA)", xlab="PC1", ylab="PC2", labels=None, **kwargs):
    """Plot method for taxodist_ord objects"""
    if x is None or x.get("points") is None:
        return x
        
    points = x["points"]
    gof = round(x["GOF"][0], 3)
    
    plt.figure(figsize=(8, 6))
    # Cria o plot invisível (type="n" no R) para definir os limites
    plt.scatter(points.iloc[:, 0], points.iloc[:, 1], alpha=0.0)
    
    if labels is None:
        labels = points.index
        
    for i, label in enumerate(labels):
        plt.text(points.iloc[i, 0], points.iloc[i, 1], label, ha='center', va='center', **kwargs)
        
    plt.title(f"{main}  (GOF = {gof})")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()
    return x

def summary_taxodist_ord(obj):
    """Summary method for taxodist_ord objects"""
    if obj is None or obj.get("points") is None:
        return None
        
    print("Taxonomic Ordination Summary (PCoA)")
    gof = round(obj["GOF"][0] * 100, 2)
    print(f"Goodness-of-Fit (GOF): {gof}%\n")
    
    eig = obj.get("eig")
    if eig is not None:
        eig_pos = eig[eig > 0]
        var_exp = (eig_pos / np.sum(eig_pos)) * 100
        k = obj["points"].shape[1]
        
        df = pd.DataFrame({
            "Axis":[f"PC{i+1}" for i in range(k)],
            "Eigenvalue": eig[:k],
            "Variance_Pct": var_exp[:k],
            "Cumulative_Pct": np.cumsum(var_exp)[:k]
        })
        print(df.to_string(index=False, float_format=lambda val: f"{val:.4f}"))
        return df
    else:
        print("Warning: Eigenvalues not found in the object.")
        return None


def print_taxodist_path(x):
    """Print method for taxodist_path objects"""
    if x is None:
        return x
        
    taxon_a = x.attrs.get("taxon_a", "A")
    taxon_b = x.attrs.get("taxon_b", "B")
    print(f"Taxonomic Path: {taxon_a} \u2192 {taxon_b}")
    
    for _, row in x.iterrows():
        node = row["node"]
        direction = row["direction"]
        dep = row["depth"]
        
        if direction == "a":
            print(f"  {node}  (depth {dep})  \u2191")
        elif direction == "mrca":
            print(f"* {node}  (MRCA, depth {dep})")
        else:
            print(f"  {node}  (depth {dep})  \u2193")
            
    return x

def print_focal_distances(x):
    """Print method for focal_distances results"""
    if x is None:
        return x
    focal = x.attrs.get("focal", "?")
    print(f"Focal distances from {focal}")
    print(f"{len(x)} taxa • sorted closest → most distant\n")
    print(x.to_string(index=False))
    return x