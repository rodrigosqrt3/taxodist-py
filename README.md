# taxodist <picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/rodrigosqrt3/taxodist-py/main/images/taxodist_dark.png"><source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/rodrigosqrt3/taxodist-py/main/images/taxodist_sepia.png"><img alt="taxodist logo" src="https://raw.githubusercontent.com/rodrigosqrt3/taxodist-py/main/images/taxodist_sepia.png" align="right" height="200"></picture>

[![PyPI version](https://img.shields.io/pypi/v/taxodist.svg?color=blue)](https://pypi.org/project/taxodist/) &nbsp; [![Python Tests](https://github.com/rodrigosqrt3/taxodist-py/actions/workflows/python-app.yml/badge.svg)](https://github.com/rodrigosqrt3/taxodist-py/actions/workflows/python-app.yml) &nbsp; [![codecov](https://codecov.io/gh/rodrigosqrt3/taxodist-py/branch/main/graph/badge.svg)](https://app.codecov.io/gh/rodrigosqrt3/taxodist-py)

**Taxonomic hierarchy distances derived from lineage classifications.**

`taxodist` retrieves ordered taxonomic lineages from [The Taxonomicon](http://taxonomicon.taxonomy.nl) and computes distances from the depth of the most recent common ancestor. The Python and R implementations use the same distance definition and the same lineage semantics.

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

```python
from taxodist import (
    get_lineage, taxo_distance, mrca, distance_matrix,
    filter_clade, taxo_path, save_cache, load_cache,
    taxo_resolve, taxo_from_lineages, taxo_bundle,
    write_taxodist_bundle, read_taxodist_bundle
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
taxa =["Tyrannosaurus", "Triceratops", "Homo", "Quercus"]
filter_clade(taxa, "Dinosauria")

# Get the path between two taxa
taxo_path("Tyrannosaurus", "Velociraptor")

# Save and restore the lineage cache across sessions
save_cache()            # persistent per-user file; or save_cache("my_cache.pkl")
load_cache()
```

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
resolved = taxo_from_lineages({
    "Alpha": ["Biota", "Animalia", "Alpha"],
    "Beta": ["Biota", "Animalia", "Beta"],
}, source="Curated study")
```

A bundle combines the audit table, matrix, metric definition, software
metadata, and source provenance. Its schema is shared with the R and Julia
implementations; missing distances are JSON `null` and disconnected distances
are encoded as `"Infinity"` or `"-Infinity"`.

```python
bundle = taxo_bundle(resolved)
write_taxodist_bundle(bundle, "analysis.taxodist.json")
restored = read_taxodist_bundle("analysis.taxodist.json")
distance_matrix(restored)

# Months later: what changed in the source classification?
from taxodist import compare_bundles
print(compare_bundles("analysis.taxodist.json", taxo_bundle(list(restored.resolution["input"]))))
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

The values represent classification depth. They are not evolutionary time,
genetic distance, morphological divergence, or phylogenetic branch lengths.

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
`taxo_resolve()` stores them in a `ranks` column, and `taxo_from_lineages()`
accepts `ranks=`. The rank levels are configurable through
`taxodist.distance.RANK_LEVELS` (default kingdom, phylum, class, order,
family, genus, species).

```python
resolved = taxo_from_lineages(
    {"Alpha": ["Biota", "Animalia", "Alpha"], "Beta": ["Biota", "Animalia", "Beta"]},
    ranks={"Alpha": [None, "Kingdom", "Genus"], "Beta": [None, "Kingdom", "Genus"]},
)
distance_matrix(resolved, metric="ranked")
```

Bundles (schema 1.1) record the metric, its rank levels, and the lineage
ranks; schema 1.0 bundles are still read.

### Taxonomic distinctness

`taxo_distinctness()` computes Warwick and Clarke's taxonomic diversity
(`Delta`), distinctness (`Delta_star`), average taxonomic distinctness
(`Delta_plus`), its variation (`Lambda_plus`), and `SDelta_plus` for a list of
taxa, a dict or Series of abundances, or a DataFrame with one row per sample.
`taxo_distinctness_test()` compares `Delta_plus` and `Lambda_plus` of a sample
with random samples from a master list (pass `seed=` for reproducibility):

```python
pool = distance_matrix(resolved_master_list, metric="ranked")
taxo_distinctness(community_table, pool)
taxo_distinctness_test(["Alpha", "Beta", "Gamma"], pool, n_iter=999, seed=1)
```

The R, Python, and Julia implementations share the lineage parser semantics
(checked against the same TaxonTree test fixtures, including node ranks), the
distance definitions, and the bundle JSON schema
(`taxodist/schema/taxodist-bundle-v1.schema.json`). Distinctness values match
R's `taxo_distinctness()` and `vegan::taxondive()`.

## Analysis helpers

The package also provides:

- `closest_relative()` and `focal_distances()`;
- `compare_lineages()` and `shared_clades()`;
- `is_member()` and `filter_clade()`;
- `check_coverage()` (`True`, `False`, or `pd.NA` when the server is
  unreachable) and `taxo_search()`;
- `taxo_cluster()`, `taxo_ordinate()`, and `taxo_heatmap()`;
- `cache_info()`, `save_cache()`, and `load_cache()`.

Requests are retried automatically after transient network errors; set
`taxodist.fetch.MAX_TRIES` to change the number of attempts (default 3).
Consecutive requests are spaced by at least `taxodist.fetch.MIN_INTERVAL`
seconds (default 0.25) so that large batches do not overload The Taxonomicon.

Distance matrices are returned as symmetric `pandas.DataFrame` objects and can
be passed directly to the clustering, ordination, and plotting helpers.

## Reproducible reference data

`load_taxobase()` loads the packaged offline reference object. It is exported from the same `taxobase` object distributed with the R
package, preserving taxon order, matrices, examples, and provenance metadata
across both implementations.

## Caching

Lineages and Taxonomicon identifiers are cached automatically during a Python
session. Cache files use Python's pickle format:

```python
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

## Data source and citation

All retrieved lineage data originate from **The Taxonomicon**, based on
*Systema Naturae 2000*. Published analyses should cite both the software and
the underlying classification source:

> Brands, S.J. (1989 onwards). *Systema Naturae 2000*. Amsterdam, The
> Netherlands. Retrieved from The Taxonomicon,
> http://taxonomicon.taxonomy.nl.

## Contributing

Found a taxon with an unexpected lineage or a difference between the Python
and R results? Please [open an issue](https://github.com/rodrigosqrt3/taxodist-py/issues) with the queried names, numeric Taxonomicon IDs when available, and the retrieval date.