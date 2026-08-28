import json
import os

import pandas as pd


def _matrix_from_json(value, fallback_labels):
    """Convert either the legacy or 0.6.0 JSON matrix representation."""
    if isinstance(value, dict):
        labels = value.get("labels", fallback_labels)
        values = value.get("data", [])
    else:
        labels = fallback_labels
        values = value
    return pd.DataFrame(values, index=labels, columns=labels, dtype=float)

def load_taxobase():
    """
    Load the pre-computed taxobase offline reference dataset.

    A pre-computed dataset spanning major groups across the tree of life,
    generated from the same reference object distributed with taxodist for R.
    It provides an offline fallback containing lineages, distance matrices,
    search output, and provenance metadata.

    Returns:
        dict: A dictionary containing pre-computed components:
            - taxa (list)
            - found_taxa (list)
            - coverage (dict)
            - matrix (pandas.DataFrame)
            - pairwise (dict)
            - lineage_homo (list)
            - lineage_tyrannosaurus (list)
            - closest (dict)
            - filter (list)
            - search (pandas.DataFrame)
            - statistical_taxa (list)
            - statistical_matrix (pandas.DataFrame)
            - metadata (dict)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "data", "taxobase.json")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    taxa = data.get("taxa", [])
    found_taxa = data.get("found_taxa", taxa)

    coverage = data.get("coverage", {})
    if isinstance(coverage, list):
        coverage = dict(zip(taxa, coverage))
    data["coverage"] = pd.Series(coverage, dtype=bool).reindex(taxa)

    data["matrix"] = _matrix_from_json(data.get("matrix", []), found_taxa)

    closest = data.get("closest", [])
    data["closest"] = pd.DataFrame(
        closest, columns=["taxon", "distance"]
    )

    search = data.get("search", [])
    data["search"] = pd.DataFrame(search, columns=["id", "name"])

    statistical_taxa = data.get("statistical_taxa", [])
    data["statistical_taxa"] = statistical_taxa
    data["statistical_matrix"] = _matrix_from_json(
        data.get("statistical_matrix", []), statistical_taxa
    )

    data.setdefault("metadata", {
        "source": "The Taxonomicon",
        "source_url": "http://taxonomicon.taxonomy.nl",
        "generated_on": None,
        "package_version": "legacy",
        "distance_definition": "legacy packaged data; rebuild for taxodist 0.7.0"
    })
    return data