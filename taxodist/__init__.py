"""
taxodist: Taxonomic Hierarchy Distances and Lineage Analysis

taxodist computes distances between taxonomic hierarchy nodes from ordered
lineages retrieved from one of three taxonomic sources: The Taxonomicon
(taxonomy.nl, the default), the Open Tree of Life reference taxonomy, and
the NCBI Taxonomy. The source is chosen with ``taxo_resolve(source=...)``
(``"taxonomicon"``, ``"ott"``, or ``"ncbi"``); name-based functions such as
get_lineage() and taxo_distance() query The Taxonomicon, and resolutions
from any source can be passed to the matrix, tree, bundle, and community
functions. Curated lineages can be used offline with taxo_from_lineages().

Core functions
--------------
- get_lineage() — retrieve the full lineage of any taxon
- lineage_ranks() — the rank of every node of a lineage
- taxo_distance() — compute the hierarchy distance between two taxa
- mrca() — find the most recent common ancestor
- distance_matrix() — compute all pairwise distances for a set of taxa
- cross_distances() — compute distances between two sets of taxa
- taxo_tree() — build a classification tree with named clades (Newick export)
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
- taxo_resolve() — resolve names into an auditable batch table
- taxo_from_lineages() — create an offline resolution from curated lineages
- taxo_bundle() — combine resolution, distances, metric, and provenance
- write_taxodist_bundle() / read_taxodist_bundle() — exchange portable JSON
- compare_bundles() — report what changed between two analyses
- taxo_distinctness() / taxo_distinctness_test() — Warwick and Clarke's
  taxonomic diversity and distinctness indices, and their randomization test
- taxo_dissimilarity() — taxonomic dissimilarities Gamma+ and Theta+ between
  samples (Clarke, Somerfield and Chapman 2006)
- taxo_ses() — standardized effect sizes of MPD and MNTD (NRI and NTI)
- taxo_sources() — the supported taxonomic sources (The Taxonomicon, the Open
  Tree Taxonomy, and the NCBI Taxonomy; ``taxo_resolve(source=...)``)
- compare_sources() — compare distances across taxonomic sources
- taxo_search() — list the Taxonomicon entries matching a name
- taxo_path(), taxo_cluster(), taxo_ordinate(), taxo_heatmap() — paths,
  clustering, ordination, and heatmaps
- save_cache() / load_cache() — persist the lineage cache

Requests to The Taxonomicon are retried after transient failures; set
``taxodist.fetch.MAX_TRIES`` to change the number of attempts (default 3).
Consecutive requests are spaced by at least ``taxodist.fetch.MIN_INTERVAL``
seconds (default 0.25; set it to 0 to disable throttling).
``taxonomicon_available()`` checks quickly whether the server can be reached.

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

Other metrics are selected with ``metric=``: ``"path"`` (edges between the
nodes), ``"wu_palmer"`` (1 - 2 depth(MRCA) / (depth(A) + depth(B))), and
``"ranked"`` (half the path through the standard Linnaean ranks listed in
``taxodist.distance.RANK_LEVELS``; Clarke and Warwick's step weights).

Data sources
------------
Lineages come from The Taxonomicon (the default), the Open Tree Taxonomy
(``source="ott"``; identifiers ``"ott:<id>"``), or the NCBI Taxonomy
(``source="ncbi"``; identifiers ``"ncbi:<id>"``). NCBI requests are spaced
by at least 0.4 s (0.12 s with an API key set in the ``NCBI_API_KEY``
environment variable or ``taxodist.sources.NCBI_API_KEY``). Please cite the
source of the classification used, as well as taxodist, in published work;
taxo_sources() lists the citations.

References
----------
Brands, S.J. (ed.) 1989-present. The Taxonomicon. Universal Taxonomic
Services, Zwaag, The Netherlands. http://taxonomicon.taxonomy.nl.

Rees, J.A. & Cranston, K. (2017). Automated assembly of a reference
taxonomy for phylogenetic data synthesis. Biodiversity Data Journal 5:
e12581. https://doi.org/10.3897/BDJ.5.e12581

Schoch, C.L. et al. (2020). NCBI Taxonomy: a comprehensive update on
curation, resources and tools. Database 2020: baaa062.
https://doi.org/10.1093/database/baaa062
"""

from ._version import __version__

from .fetch import (
    clear_cache, save_cache, load_cache, cache_info,
    get_taxonomicon_id, get_lineage_by_id, get_lineage, lineage_ranks,
    taxo_search, TaxodistResolution, taxo_resolve, taxo_from_lineages,
    taxonomicon_available
)

from .distance import (
    METRICS,
    taxo_distance, mrca, distance_matrix, cross_distances, closest_relative,
    focal_distances, lineage_depth, check_coverage, taxo_cluster, taxo_ordinate
)

from .distinctness import (
    TaxodistDistinctnessTest,
    taxo_distinctness,
    taxo_distinctness_test,
    print_taxodist_distinctness,
)

from .community import taxo_dissimilarity, taxo_ses, print_taxodist_ses

from .sources import TaxodistSourceComparison, taxo_sources, compare_sources

from .tree import TaxodistTree, taxo_tree

from .utils import (
    compare_lineages, shared_clades, is_member, filter_clade,
    taxo_heatmap, taxo_path,
    print_taxodist_result, plot_taxodist_cluster, plot_taxodist_ord,
    summary_taxodist_ord, print_taxodist_path, print_focal_distances
)

from .data import load_taxobase

from .bundle import (
    TaxodistBundle,
    TaxodistBundleComparison,
    taxo_bundle,
    validate_taxodist_bundle,
    write_taxodist_bundle,
    read_taxodist_bundle,
    compare_bundles,
)

__all__ = [
    "__version__",
    # retrieval, resolution, and cache
    "clear_cache", "save_cache", "load_cache", "cache_info",
    "get_taxonomicon_id", "get_lineage_by_id", "get_lineage", "lineage_ranks",
    "taxo_search", "TaxodistResolution", "taxo_resolve", "taxo_from_lineages",
    "taxonomicon_available",
    # distances
    "METRICS",
    "taxo_distance", "mrca", "distance_matrix", "cross_distances",
    "closest_relative", "focal_distances", "lineage_depth", "check_coverage",
    "taxo_cluster", "taxo_ordinate",
    # distinctness and community analyses
    "TaxodistDistinctnessTest", "taxo_distinctness", "taxo_distinctness_test",
    "print_taxodist_distinctness",
    "taxo_dissimilarity", "taxo_ses", "print_taxodist_ses",
    # taxonomic sources
    "TaxodistSourceComparison", "taxo_sources", "compare_sources",
    # trees
    "TaxodistTree", "taxo_tree",
    # lineage utilities and plots
    "compare_lineages", "shared_clades", "is_member", "filter_clade",
    "taxo_heatmap", "taxo_path",
    "print_taxodist_result", "plot_taxodist_cluster", "plot_taxodist_ord",
    "summary_taxodist_ord", "print_taxodist_path", "print_focal_distances",
    # reference data
    "load_taxobase",
    # bundles
    "TaxodistBundle", "TaxodistBundleComparison", "taxo_bundle",
    "validate_taxodist_bundle", "write_taxodist_bundle", "read_taxodist_bundle",
    "compare_bundles",
]