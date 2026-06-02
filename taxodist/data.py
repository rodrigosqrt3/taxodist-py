import json
import os

def load_taxobase():
    """
    Load the pre-computed taxobase offline reference dataset.

    A pre-computed dataset containing 50 taxonomic clades
    spanning across the tree of life, fetched from The Taxonomicon.
    This provides a offline fallback containing lineages,
    distance matrices, and search queries for demonstrating the package.

    Returns:
        dict: A dictionary containing pre-computed components:
            - taxa (list)
            - found_taxa (list)
            - coverage (dict)
            - matrix (list of lists)
            - pairwise (dict)
            - lineage_homo (list)
            - lineage_tyrannosaurus (list)
            - closest (dict)
            - filter (list)
            - search (dict)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "data", "taxobase.json")
    
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)