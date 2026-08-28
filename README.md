# taxodist <picture><source media="(prefers-color-scheme: dark)" srcset="images/taxodist_dark.png"><source media="(prefers-color-scheme: light)" srcset="images/taxodist_sepia.png"><img alt="taxodist logo" src="images/taxodist_sepia.png" align="right" height="200"></picture>

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

## Basic usage

```python
from taxodist import (
    get_lineage, taxo_distance, mrca, distance_matrix,
    filter_clade, taxo_path, save_cache, load_cache
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
save_cache("my_cache.pkl")
load_cache("my_cache.pkl")
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

See [Methodological notes](docs/methodology.md) for the formal properties,
assumptions, and limitations of the measure, and
[R/Python parity](docs/r-python-parity.md) for the compatibility contract.

## Analysis helpers

The package also provides:

- `closest_relative()` and `focal_distances()`;
- `compare_lineages()` and `shared_clades()`;
- `is_member()` and `filter_clade()`;
- `check_coverage()` and `taxo_search()`;
- `taxo_cluster()`, `taxo_ordinate()`, and `taxo_heatmap()`;
- `cache_info()`, `save_cache()`, and `load_cache()`.

Distance matrices are returned as symmetric `pandas.DataFrame` objects and can
be passed directly to the clustering, ordination, and plotting helpers.

## Reproducible reference data

`load_taxobase()` loads the packaged offline reference object. For release
0.7.0 it is exported from the same `taxobase` object distributed with the R
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