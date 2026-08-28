"""
taxodist: Taxonomic Hierarchy Distances and Lineage Analysis

taxodist computes distances between taxonomic hierarchy nodes using
hierarchical lineage data retrieved from The Taxonomicon
(taxonomy.nl), a comprehensive curated classification of all life
based on Systema Naturae 2000.

Core functions
--------------
- get_lineage() — retrieve the full lineage of any taxon
- taxo_distance() — compute the hierarchy distance between two taxa
- mrca() — find the most recent common ancestor
- distance_matrix() — compute all pairwise distances for a set of taxa
- closest_relative() — find the closest relative among candidates
- focal_distances() — compare one focal taxon with a community
- compare_lineages() — print a side-by-side lineage comparison
- shared_clades() — list clades shared between two taxa
- is_member() — test clade membership
- filter_clade() — filter taxa by clade membership
- check_coverage() — check Taxonomicon coverage for a list of taxa
- lineage_depth() — get the lineage depth of a taxon
- clear_cache() — clear the session lineage cache
- cache_info() — inspect the session lineage cache
- load_taxobase() — load the built-in reference dataset

Mathematical background
-----------------------
The distance metric is based on the depth of the most recent common
ancestor (MRCA):

    d(A, A) = 0
    d(A, B) = 1 / depth(MRCA(A, B)), for distinct nodes

The deeper the shared ancestor, the smaller the distance. Distinct
ancestor-descendant nodes therefore have positive distance. The measure is
an ultrametric within each connected hierarchy. It represents classification
depth, not evolutionary time or phylogenetic branch length.

Data source
-----------
All lineage data is sourced from The Taxonomicon (taxonomy.nl), based on
Systema Naturae 2000 by S.J. Brands (1989 onwards). Please cite this
resource when using taxodist in published work.

References
----------
Brands, S.J. (1989 onwards). Systema Naturae 2000. Amsterdam,
The Netherlands. Retrieved from The Taxonomicon,
http://taxonomicon.taxonomy.nl.
"""

__version__ = "0.7.0"

from .fetch import (
    clear_cache, save_cache, load_cache, cache_info,
    get_taxonomicon_id, get_lineage_by_id, get_lineage, taxo_search
)

from .distance import (
    taxo_distance, mrca, distance_matrix, closest_relative,
    focal_distances, lineage_depth, check_coverage, taxo_cluster, taxo_ordinate
)

from .utils import (
    compare_lineages, shared_clades, is_member, filter_clade,
    taxo_heatmap, taxo_path,
    print_taxodist_result, plot_taxodist_cluster, plot_taxodist_ord,
    summary_taxodist_ord, print_taxodist_path
)

from .data import load_taxobase