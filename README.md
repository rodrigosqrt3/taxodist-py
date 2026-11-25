# taxodist <picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/rodrigosqrt3/taxodist-py/main/images/taxodist_dark.png"><source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/rodrigosqrt3/taxodist-py/main/images/taxodist_sepia.png"><img alt="taxodist logo" src="https://raw.githubusercontent.com/rodrigosqrt3/taxodist-py/main/images/taxodist_sepia.png" align="right" height="200"></picture>

[![PyPI version](https://img.shields.io/pypi/v/taxodist.svg?color=blue)](https://pypi.org/project/taxodist/) &nbsp; [![Python Tests](https://github.com/rodrigosqrt3/taxodist-py/actions/workflows/python-app.yml/badge.svg)](https://github.com/rodrigosqrt3/taxodist-py/actions/workflows/python-app.yml) &nbsp; [![codecov](https://codecov.io/gh/rodrigosqrt3/taxodist-py/branch/main/graph/badge.svg)](https://app.codecov.io/gh/rodrigosqrt3/taxodist-py)

**Taxonomic hierarchy distances derived from lineage classifications.**

`taxodist` retrieves ordered taxonomic lineages and computes distances from
the depth of the most recent common ancestor. Lineages come from one of three
taxonomic sources: [The Taxonomicon](http://taxonomicon.taxonomy.nl) (the
default), the [Open Tree Taxonomy](https://tree.opentreeoflife.org/about/taxonomy-version),
and the [NCBI Taxonomy](https://www.ncbi.nlm.nih.gov/taxonomy), or from
lineages you supply yourself. Besides pairwise distances and matrices, the
package provides classification trees, clustering and ordination, Clarke and
Warwick's taxonomic distinctness, taxonomic dissimilarities between samples,
and standardized effect sizes of community relatedness.

The same package is available for [R](https://github.com/rodrigosqrt3/taxodist)
and [Julia](https://github.com/rodrigosqrt3/Taxodist.jl); the three
implementations use the same distance definitions and lineage semantics, give
identical results, and exchange analyses through a common JSON bundle format.
Documentation for all three is at [taxodist.github.io](https://taxodist.github.io).

## Installation

```bash
pip install taxodist
```
```bash
pip install git+https://github.com/rodrigosqrt3/taxodist-py.git
```

The plotting helpers (`taxo_heatmap()`, `plot_taxodist_cluster()`,
`plot_taxodist_ord()`) need matplotlib and seaborn, which are installed with
the `plot` extra:

```bash
pip install "taxodist[plot]"
```

## Basic usage

Name-based functions query The Taxonomicon:

```python
from taxodist import (
    get_lineage, taxo_distance, mrca, distance_matrix,
    filter_clade, taxo_path, save_cache, load_cache,
)

# Get a full lineage
get_lineage("Tyrannosaurus")

# Distance between two taxa
taxo_distance("Tyrannosaurus", "Velociraptor")

# Most recent common ancestor
mrca("Tyrannosaurus", "Triceratops")   # "Dinosauria"
mrca("Tyrannosaurus", "Homo")          # "Amniota"

# Pairwise distance matrix
theropods = ["Tyrannosaurus", "Velociraptor", "Spinosaurus", "Allosaurus"]
distance_matrix(theropods)

# Filter taxa by clade
taxa = ["Tyrannosaurus", "Triceratops", "Homo", "Quercus"]
filter_clade(taxa, "Dinosauria")

# Get the path between two taxa
taxo_path("Tyrannosaurus", "Velociraptor")

# Save and restore the lineage cache across sessions
save_cache()            # persistent per-user file; or save_cache("my_cache.pkl")
load_cache()
```

## Choosing a taxonomic source

`taxo_resolve()` and `taxo_bundle()` take a `source` argument:

| `source` | classification | identifiers | root of every lineage |
|---|---|---|---|
| `"taxonomicon"` (default) | The Taxonomicon | numeric (`"50841"`) | `"Biota"` |
| `"ott"` | Open Tree of Life reference taxonomy | `"ott:770309"` | `"life"` |
| `"ncbi"` | NCBI Taxonomy | `"ncbi:9606"` | `"cellular organisms"` |

The resolution records its source (`resolved.source`, with the version of the
Open Tree Taxonomy, and `resolved.source_url`), and works with every function
that accepts one: `distance_matrix()`, `cross_distances()`, `taxo_tree()`,
`taxo_bundle()`, and, through the matrix, the distinctness and community
functions. When `source` is set, the prefix may be omitted from identifiers in
the input. Clade names differ between sources: animals are `"Animalia"` in The
Taxonomicon and `"Metazoa"` in the other two, which matters for `within`.

```python
from taxodist import taxo_resolve, taxo_sources, compare_sources

taxo_sources()     # the sources, their identifier prefixes, licences, and citations

resolved = taxo_resolve(["Homo", "Pan", "Mus"], source="ncbi")
distance_matrix(resolved)

# How much do the distances depend on the classification?
print(compare_sources(["Homo", "Pan", "Mus", "Gallus", "Danio"],
                      sources=["ott", "ncbi"], seed=1))
```

`compare_sources()` resolves the same taxa in several sources and reports the
lineage depth of every taxon in each and the agreement between the distance
matrices (Spearman correlation and a Mantel permutation test). Distances are
comparable within one source, not across sources.

Requests to each source are spaced independently: at least
`taxodist.fetch.MIN_INTERVAL` seconds (default 0.25) for The Taxonomicon and
the Open Tree of Life, and at least 0.4 s for NCBI, which allows three
requests per second without an API key. With an
[NCBI API key](https://www.ncbi.nlm.nih.gov/books/NBK25497/) (set it in
the `NCBI_API_KEY` environment variable or in `taxodist.sources.NCBI_API_KEY`)
the limit is ten per second and requests are spaced by 0.12 s. Refused
requests (HTTP 429) and server errors are retried after a pause.

## Trees and distances between sets

```python
from taxodist import taxo_tree, cross_distances

# Classification tree whose internal nodes are named clades
tree = taxo_tree(theropods)
print(tree.to_newick())        # for Bio.Phylo, ete3, dendropy, ...
tree.cophenetic()              # equals distance_matrix(theropods)

# Rectangular matrix: community (rows) x reference taxa (columns)
cross_distances(["Tyrannosaurus", "Triceratops"], ["Velociraptor", "Homo"])
```

## Auditable resolution and portable bundles

`taxo_resolve()` resolves a complete input list before an analysis, preserving
the candidates considered, selected ID, lineage, lineage depth, and one of the
statuses `resolved`, `ambiguous`, `unresolved`, or `retrieval_error`.
`within` restricts homonyms to a clade, and the resulting matrix is identical
to the one computed from the names:

```python
resolved = taxo_resolve(
    ["Tyrannosaurus", "Nereis", "50841"],
    ambiguity="warn",
    within="Animalia",
)
distances = distance_matrix(resolved)
```

Curated or unpublished classifications use the same downstream API without
network access:

```python
from taxodist import taxo_from_lineages

curated = taxo_from_lineages({
    "Alpha": ["Biota", "Animalia", "Alpha"],
    "Beta": ["Biota", "Animalia", "Beta"],
}, source="Curated study")
distance_matrix(curated)
```

A bundle combines the audit table, matrix, metric definition, software
metadata, and source provenance. Its schema is shared with the R and Julia
implementations; missing distances are JSON `null` and disconnected distances
are encoded as `"Infinity"` or `"-Infinity"`.

```python
from taxodist import (
    taxo_bundle, write_taxodist_bundle, read_taxodist_bundle, compare_bundles,
)

bundle = taxo_bundle(resolved)
write_taxodist_bundle(bundle, "analysis.taxodist.json")
restored = read_taxodist_bundle("analysis.taxodist.json")
distance_matrix(restored)

# Months later: what changed in the source classification?
print(compare_bundles("analysis.taxodist.json",
                      taxo_bundle(list(restored.resolution["input"]))))
```

## The distance metric

`taxodist` measures separation by asking how deep the most recent common
ancestor (MRCA) occurs in the continuous common prefix of two ordered
lineages:

$$
d(A,B) =
\begin{cases}
0, & A = B, \\
\dfrac{1}{\operatorname{depth}(\operatorname{MRCA}(A,B))}, & A \ne B.
\end{cases}
$$

A deeper shared ancestor produces a smaller distance. Zero is reserved for
identical hierarchy nodes. A taxon and one of its descendants therefore have
a positive distance even though they are connected by ancestry. Use
`is_member()` or `taxo_path()` when the question concerns containment rather
than distance between nodes.

Within each connected hierarchy this definition is an ultrametric. If two
lineages do not share a root, their distance is infinite. Missing lineages
produce `NaN` values in a distance matrix.

The values represent classification depth in the chosen source. They are not
evolutionary time, genetic distance, morphological divergence, or
phylogenetic branch lengths, and lineage depths differ between sources (The
Taxonomicon has many more unranked clades), so distances are comparable only
within one source.

### Other metrics and ranks

`taxo_distance()`, `distance_matrix()`, `cross_distances()`,
`closest_relative()`, `focal_distances()`, `taxo_heatmap()`, and
`taxo_bundle()` accept `metric=` (also passed through by `taxo_cluster()` and
`taxo_ordinate()`). With `h` the MRCA depth and `d_A`, `d_B` the lineage
depths:

| `metric` | distance | disconnected |
|---|---|---|
| `"reciprocal"` (default) | `1 / h` | `inf` |
| `"path"` | `d_A + d_B - 2h` (edges in the classification tree) | `inf` |
| `"wu_palmer"` | `1 - 2h / (d_A + d_B)` | `1` |
| `"ranked"` | half the path through the root, the standard Linnaean ranks, and the terminal node | `inf` |

Identical lineages always have distance 0. Matrices record the metric in
`DataFrame.attrs["metric"]`. The `"ranked"` metric reproduces Clarke and
Warwick's step weights (1 congeneric, 2 confamilial, ...) and needs the rank
of every node: `lineage_ranks("Passer")` returns each node with its rank,
`taxo_resolve()` stores them in a `ranks` column (for every source), and
`taxo_from_lineages()` accepts `ranks=`. The rank levels are configurable
through `taxodist.distance.RANK_LEVELS` (default kingdom, phylum, class,
order, family, genus, species).

```python
genus_ranks = [None, "Kingdom", "Genus"]
ranked = taxo_from_lineages(
    {"Alpha": ["Biota", "Animalia", "Alpha"], "Beta": ["Biota", "Animalia", "Beta"]},
    ranks={"Alpha": genus_ranks, "Beta": genus_ranks},
)
distance_matrix(ranked, metric="ranked")
```

Bundles (schema 1.1) record the metric, its rank levels, and the lineage
ranks; schema 1.0 bundles are still read.

## Community analyses

The functions below take a community as a list of taxa, a dict or Series of
abundances, or a DataFrame with one row per sample and one column per taxon,
together with a distance matrix whose labels include every taxon (from any
source). The example uses curated lineages, so it runs offline:

```python
import pandas as pd
from taxodist import (
    taxo_distinctness, taxo_distinctness_test, print_taxodist_distinctness,
    taxo_dissimilarity, taxo_ses, print_taxodist_ses,
)

ranks = [None, "Kingdom", "Phylum", "Class", "Order", "Family", "Genus"]
birds = ["Biota", "Animalia", "Chordata", "Aves"]
mammals = ["Biota", "Animalia", "Chordata", "Mammalia"]
lineages = {
    "Passer": birds + ["Passeriformes", "Passeridae", "Passer"],
    "Turdus": birds + ["Passeriformes", "Turdidae", "Turdus"],
    "Corvus": birds + ["Passeriformes", "Corvidae", "Corvus"],
    "Falco": birds + ["Falconiformes", "Falconidae", "Falco"],
    "Homo": mammals + ["Primates", "Hominidae", "Homo"],
    "Mus": mammals + ["Rodentia", "Muridae", "Mus"],
}
pool = distance_matrix(
    taxo_from_lineages(lineages, ranks={t: ranks for t in lineages}),
    metric="ranked", progress=False,
)
community = pd.DataFrame(
    {"Passer": [3, 1, 0], "Turdus": [1, 0, 0], "Corvus": [0, 2, 0],
     "Falco": [1, 0, 1], "Homo": [0, 0, 4], "Mus": [0, 1, 2]},
    index=["site1", "site2", "site3"],
)

# Warwick and Clarke's taxonomic diversity and distinctness indices
taxo_distinctness(community, pool)
print_taxodist_distinctness(
    taxo_distinctness_test(["Passer", "Turdus", "Falco"], pool, n_iter=999, seed=1)
)

# Taxonomic dissimilarity between samples (Gamma+ or Theta+)
taxo_dissimilarity(community, pool, index="gamma_plus")

# Standardized effect sizes of MPD (NRI) and MNTD (NTI)
print_taxodist_ses(taxo_ses(community, pool, index="mpd", n_iter=999, seed=1))
```

- `taxo_distinctness()` computes Warwick and Clarke's taxonomic diversity
  (`Delta`), distinctness (`Delta_star`), average taxonomic distinctness
  (`Delta_plus`), its variation (`Lambda_plus`), and `SDelta_plus`;
  `taxo_distinctness_test()` compares `Delta_plus` and `Lambda_plus` of a
  sample with random samples from a master list. Values match R's
  `taxo_distinctness()` and `vegan::taxondive()`.
- `taxo_dissimilarity()` computes the taxonomic dissimilarities Gamma+ and
  Theta+ of Clarke, Somerfield and Chapman (2006) between samples, returned as
  a square matrix for ordination, clustering, or PERMANOVA; samples without
  any taxon in common are still similar when their taxa are close relatives.
- `taxo_ses()` computes the standardized effect sizes of the mean pairwise
  distance (NRI) and mean nearest-taxon distance (NTI) of Webb et al. (2002)
  under a species-pool (`null_model="pool"`) or independent-swap
  (`null_model="independent_swap"`) null model, optionally weighted by abundance as in
  picante. Pass `seed=` for reproducible results.

The R, Python, and Julia implementations share the lineage parser semantics
(checked against the same test fixtures, including node ranks and real
responses of the Open Tree and NCBI APIs), the distance definitions, and the
bundle JSON schema (`taxodist/schema/taxodist-bundle-v1.schema.json`).

## Analysis helpers

The package also provides:

- `closest_relative()` and `focal_distances()`;
- `compare_lineages()` and `shared_clades()`;
- `is_member()` and `filter_clade()`;
- `check_coverage()` (`True`, `False`, or `pd.NA` when the server is
  unreachable) and `taxo_search()`;
- `taxo_cluster()` (identical to R's `hclust()` for all eight linkage
  methods), `taxo_ordinate()`, and `taxo_heatmap()`;
- `cache_info()`, `save_cache()`, and `load_cache()`.

Requests are retried automatically after transient network errors; set
`taxodist.fetch.MAX_TRIES` to change the number of attempts (default 3).
`taxonomicon_available()` checks quickly whether The Taxonomicon can be
reached.

Distance matrices are returned as symmetric `pandas.DataFrame` objects and can
be passed directly to the clustering, ordination, and plotting helpers.

## Reproducible reference data

`load_taxobase()` loads the packaged offline reference object. It is exported
from the same `taxobase` object distributed with the R package, preserving
taxon order, matrices, examples, and provenance metadata across both
implementations.

## Caching

Lineages and identifiers are cached automatically during a Python session.
Cache files use Python's pickle format:

```python
from taxodist import clear_cache

save_cache("taxodist-cache.pkl")
clear_cache()
load_cache("taxodist-cache.pkl")
```

Without a file argument, `save_cache()` and `load_cache()` use a persistent
per-user file (`%LOCALAPPDATA%\taxodist\Cache` on Windows,
`~/Library/Caches/taxodist` on macOS, `$XDG_CACHE_HOME/taxodist` elsewhere);
set `TAXODIST_CACHE_DIR` to choose another directory.

Cache files record the version of the lineage parser. Lineages saved by an
older parser are discarded with a warning when the file is loaded (taxon IDs
are kept) and are retrieved again when needed.

Only load cache files that you trust. As with any pickle file, loading an
untrusted file can execute arbitrary code.

## Data sources and citation

Published analyses should cite both the software and the classification the
lineages were taken from (`taxo_sources()` lists the citations, and every
resolution and bundle records its source):

- **The Taxonomicon** (`source="taxonomicon"`, the default):
  Brands, S.J. (ed.) 1989-present. *The Taxonomicon*. Universal Taxonomic
  Services, Zwaag, The Netherlands. http://taxonomicon.taxonomy.nl
- **Open Tree Taxonomy** (`source="ott"`), retrieved from the Open Tree of
  Life API (CC0; the version used is recorded in the resolution):
  Rees, J.A. & Cranston, K. (2017). Automated assembly of a reference
  taxonomy for phylogenetic data synthesis. *Biodiversity Data Journal* 5:
  e12581. https://doi.org/10.3897/BDJ.5.e12581
- **NCBI Taxonomy** (`source="ncbi"`), retrieved from the NCBI E-utilities:
  Schoch, C.L. et al. (2020). NCBI Taxonomy: a comprehensive update on
  curation, resources and tools. *Database* 2020: baaa062.
  https://doi.org/10.1093/database/baaa062

The NCBI E-utilities allow three requests per second without an API key and
ten with one; `taxodist` stays within these limits (see
[Choosing a taxonomic source](#choosing-a-taxonomic-source)). To cite
`taxodist` itself, use the metadata in `CITATION.cff` (GitHub's "Cite this
repository" button).

## Contributing

Found a taxon with an unexpected lineage or a difference between the Python
and R results? Please [open an issue](https://github.com/rodrigosqrt3/taxodist-py/issues)
with the queried names, the source, the identifiers when available, and the
retrieval date.
