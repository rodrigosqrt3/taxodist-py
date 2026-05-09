# taxodist

[![PyPI version](https://badge.fury.io/py/taxodist-py.svg)](https://badge.fury.io/py/taxodist-py)
[![Python Tests](https://github.com/rodrigosqrt3/taxodist-py/actions/workflows/python-app.yml/badge.svg)](https://github.com/rodrigosqrt3/taxodist-py/actions/workflows/python-app.yml)
[![Coverage](https://codecov.io/gh/rodrigosqrt3/taxodist-py/branch/main/graph/badge.svg)](https://codecov.io/gh/rodrigosqrt3/taxodist-py)

**Taxonomic distance and phylogenetic lineage computation for any taxon on Earth.**

`taxodist` retrieves full hierarchical lineages from [The Taxonomicon](http://taxonomicon.taxonomy.nl) and computes a tree metric distance between any two taxa: a pair of dinosaurs, a dinosaur and a fungus, two species of fly, or an oak tree and a human.

## Installation

```bash
pip install taxodist
```
```bash
pip install git+https://github.com/rodrigosqrt3/taxodist.git
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
