# taxodist 0.1.0

* Initial release
* Compute taxonomic distances between any two taxa using The Taxonomicon
* Session-level caching for lineage data to minimize network requests
* Functions: `taxo_distance()`, `mrca()`, `distance_matrix()`, `closest_relative()`,
  `compare_lineages()`, `shared_clades()`, `is_member()`, `filter_clade()`,
  `lineage_depth()`, `check_coverage()`, `clear_cache()`

# taxodist 0.2.0

## New functions

* `taxo_cluster()`: hierarchical clustering of taxa by taxonomic distance.
* `taxo_ordinate()`: ordination (PCoA) of taxa in taxonomic distance space.
* `save_cache()`: serialises the session lineage cache to an `.rds` file.
* `load_cache()`: restores a previously saved cache, avoiding redundant network requests.
* `taxo_path()`: returns the full node-by-node path between two taxa as a
  tidy data frame, ascending from taxon A to their MRCA and descending to
  taxon B.

## Documentation

* Added a new vignette (`Statistical Applications of taxodist`) demonstrating 
  the integration of `taxodist` with `ape` (tree plotting) and `vegan` 
  (taxonomic distinctness, Mantel tests, and PERMANOVA).

## Minor improvements

* Added `vegan` to `Suggests` in `DESCRIPTION`.
* Documented the compatibility of `distance_matrix()` output with `vegan` 
  functions (`taxondive()`, `mantel()`, `adonis2()`).
* Documented the conversion of `taxo_cluster()` results into `phylo` objects 
  using `ape::as.phylo()`.

## Bug fixes

* Fixed incorrect MRCA computation caused by unnamed crown nodes (`[crown]`)
  being matched across different lineages as if they were the same ancestor.
* Fixed parsing of Taxonomicon rank prefixes (`Cohort`, `Subcohort`, 
  `Magnorder`, `Grandorder`, `Parvorder`, `Legion`) that were being retained 
  as bare rank names instead of the actual clade names (e.g., `Placentalia`, 
  `Boreoeutheria`, `Galloanserae`).
  
# taxodist 0.3.0

## New features
* `taxo_search()`: Added a new interactive search function that queries The Taxonomicon and         returns a tidy data frame of all available IDs, ranks, and authors for a given taxon name.
* Direct ID Support: `get_lineage()` and all distance functions (e.g.,                             `taxo_distance()`, `distance_matrix()`) now accept Taxonomicon numeric IDs. This                 provides a fallback for computing distances when homonyms or historical ranks cannot be          disambiguated by name alone.

## Network Handling
* Added a 30-second timeout to all HTTP requests to prevent the R session from hanging             indefinitely when The Taxonomicon servers are overloaded.
* Network failures, offline servers, or bad HTTP statuses now immediately emit a clear,            informative warning to the user rather than failing silently.

## Bug fixes
* Extended rank-prefix filter to remove additional bare rank tokens that were
  being retained as spurious lineage nodes: `Subgenus`, `Section`, `Division`,
  `Subdivision`, `Supercohort`, `Infracohort`, `Subsection`, `Candidatus`,
  `Parvphylum`, `Branch`, and `Go to` (a navigation artefact from The
  Taxonomicon page layout).
* Duplicate lineage nodes originating from data-quality issues in The
  Taxonomicon (e.g. `Uropygi` appearing twice in Thelyphonida, `Myxomycetes`
  appearing twice in Physarum) are now collapsed to a single occurrence.
  A warning is emitted when deduplication occurs so the user is aware of the
  upstream data issue.
* `get_taxonomicon_id()` now collects all biological matches for a taxon name
  before returning, and emits a warning when multiple valid biological entries
  are found in The Taxonomicon. This surfaces homonym ambiguity (e.g. `Nereis`
  matching both a polychaete worm and a butterfly genus) that was previously
  silent.
* `get_taxonomicon_id()` now properly follows taxonomic redirects and removes interface            noise (`N|T|P...`) from warnings.
* When multiple biological entries are found (e.g. `Bacteria`), the warning now lists all          available numeric IDs without duplicates, allowing users to make an informed choice.
* Made lineage parsing more robust by strictly truncating any philosophical or pre-basal           nodes (e.g. `organisms`) appearing before `Biota`.
* Added validation to `get_lineage_by_id()` to silently return `NULL` for non-numeric              strings, preventing upstream server fallback errors.
* Fixed a lineage parsing issue where the parent `Genus` was erroneously truncated from            species-level queries. Distances and Most Recent Common Ancestors (MRCAs) between congeneric     species are now computed accurately.
* Fixed an issue where taxonomic author names enclosed in double quotes (e.g., `"Redtenbacher,     1906"`) were evading the lineage cleaning pipeline and being erroneously retained as part of     the clade name.

# taxodist 0.4.0

## New features

* `cache_info()`: inspects the current session cache, reporting the number of
  cached lineages and IDs, total memory used, and the names of all taxa whose
  lineages are stored. Returns a named list invisibly for programmatic use.
* `focal_distances()`: computes taxonomic distances from a single focal taxon
  to all members of a community vector. Returns a sorted data frame with
  columns `taxon`, `distance`, `mrca`, and `mrca_depth`, making it easy to
  identify the closest and most distant relatives of a focal species in a
  community context.

## Bug fixes

* Fixed a lineage parsing bug where taxa with `Subphylum` and `Infraphylum`
  rank prefixes (e.g., Craniata, Vertebrata) were being incorrectly discarded.
  The dagger symbol (`†`) is now stripped before rank prefixes are removed,
  ensuring that prefixed names like `Subphylum † Craniata` are correctly
  parsed to `Craniata` rather than falling through as bare rank tokens.
  `Subphylum` and `Infraphylum` have been removed from `bare_ranks` accordingly.
* Fixed parsing of `auct.` author annotations (e.g., `Subphylum Craniata auct.`)
  that were preventing rank-prefixed names from being correctly cleaned.
* Fixed intermediate clades being silently dropped from lineages when their
  author strings contained ampersands, hyphens (e.g., Cavalier-Smith), or
  trailing commas without a year (e.g., `Romeriida Gauthier,`). The author
  removal pipeline now handles these patterns correctly.
* Fixed a bug where the dagger extinction marker (`†`) appearing between a
  rank prefix and a taxon name (e.g., `Family † Dromaeosauridae`) caused the
  taxon to be lost entirely. Dagger removal now precedes all other cleaning
  steps.
  
# taxodist 0.5.0

## New features

* `taxobase`: Added a built-in reference dataset containing 50 taxonomic clades. 
   The dataset includes pre-computed distance matrices, lineage paths, search 
   outputs, and coverage vectors. It acts as a complete offline fallback, allowing 
   package examples, vignettes, and unit tests to run instantly without requiring 
   a network connection to The Taxonomicon.

## Bug fixes

  * Fixed a bug where any taxon whose name contains the substring "planet" (e.g.               *Periplaneta*) was incorrectly discarded by the astronomical homonym filter in               `get_taxonomicon_id()` and `taxo_search()`. The filter now uses word boundaries              (`\bplanet\b`) so that genus names containing "planet" as an infix are no longer mistakenly   treated as non-biological entries.
  * Fixed a lineage parsing bug affecting genera whose Taxonomicon page does not link back to   the genus itself (e.g. *Periplaneta*, which only links to its constituent species). The      lineage is now correctly truncated before child species entries, and the genus name is       appended if absent.
  * Added `"Epifamily"` to the bare rank token filter so that unprefixed rank labels are not   retained as spurious lineage nodes.
  * Fixed a lineage parsing bug where the superscript type-marker `ᵀ` (U+1D40) immediately     following a taxon name prevented word-boundary matching in the cutoff search, causing the    lineage to overshoot the target node and incorrectly include descendant taxa as ancestors.   This silently affected any type genus or type species (i.e. the majority of genera), and     was particularly visible in deep clades such as *Dinosauria* where several child clades      were being parsed as ancestor nodes. Higher-rank taxa (clades, orders, families — anything   above genus level) were disproportionately affected.
  * Fixed a lineage parsing bug where navigation header text prepended to the                  `#divPageContent` block could contain the target taxon name (e.g. in a cited reference       title), causing the lineage cutoff to land in the junk header rather than the actual         taxonomic tree. Affected taxa received a garbage two- or three-line lineage and were         silently discarded by the `"Biota"` membership check in `get_taxonomicon_id()`.              *Drosophila* (the fly genus, ID 28940) was one such case.

# taxodist 0.6.0

## Breaking changes

* The distance definition is now a proper ultrametric. Distance is zero only
  when two complete lineages identify the same hierarchy node. Distinct
  ancestor-descendant pairs now receive `1 / depth(MRCA)` instead of zero.
  Use `is_member()` or `taxo_path()` when the question concerns clade
  membership or containment rather than distance between nodes.
* Package terminology now consistently describes the measure as a taxonomic
  hierarchy distance rather than a phylogenetic distance.

## Data and documentation

* `taxobase` is now distributed as a single documented list with provenance
  metadata, a compact reference distance matrix, and an offline matrix used to
  build the statistical applications vignette reproducibly.
* Added a methodological vignette defining the distance, demonstrating its
  ultrametric property, and documenting its assumptions and limitations.
* Revised the introduction and statistical applications vignettes to remove
  stale hand-written numerical output and distinguish taxonomic dendrograms
  from inferred phylogenetic trees.
* Added a package citation for taxodist and a separate citation for The
  Taxonomicon data source.

## Bug fixes and robustness

* MRCA detection now uses only the continuous common lineage prefix, preventing
  repeated names after divergence from being mistaken for shared ancestry.
* `is_member()` now performs exact, case-insensitive matching and treats regular
  expression characters literally.
* `taxo_path()` now correctly handles identical taxa, ancestor-descendant pairs,
  path direction, lineage depths, and pairs with no common ancestor.
* Empty and single-taxon inputs are handled safely by distance matrices,
  clustering, ordination, heatmaps, closest-relative queries, and focal
  distances. Ordination also reduces `k` when required by sample size.
* Clustering, ordination, and heatmaps now detect `NA` and infinite distances
  and return informative fallback objects instead of failing downstream.
* `load_cache()` validates cache structure, names, and value types before
  modifying the active cache.

# taxodist 0.7.0

## Performance

* Reused a persistent `requests.Session` across Taxonomicon searches and
  lineage retrievals. This enables HTTP connection pooling without changing
  resolution, parsing, caching, timeout, or public API behavior.
* Added a numeric-only pairwise path for `distance_matrix()`, avoiding one
  detailed result dictionary allocation per taxon pair.
* Cached final name-resolved lineages so warm matrix construction no longer
  repeats name filtering and regular-expression work.

## Compatibility and validation

* Preserved public return types and the continuous-prefix MRCA distance
  semantics shared with R and Julia.
* Added regression tests for valid and invalid resolved-lineage caches, quiet
  search paths, structured HTML edge cases, and optimized matrix construction.
* The test suite now reports complete statement and branch coverage.

# taxodist 0.8.0

## Auditable name resolution

* Added `taxo_resolve()` with explicit `resolved`, `ambiguous`, `unresolved`,
  and `retrieval_error` statuses, candidate tables, selected IDs, lineages,
  and lineage depths.
* Added explicit `warn`, `first`, and `error` ambiguity policies and direct
  numeric-ID resolution.
* Added `taxo_from_lineages()` for curated, frozen, unpublished, or fully
  offline classifications.
* `distance_matrix()` now accepts `TaxodistResolution` objects and reuses their
  stored lineages without network requests.

## Portable analysis bundles

* Added `TaxodistBundle`, `taxo_bundle()`, `write_taxodist_bundle()`, and
  `read_taxodist_bundle()` using the same version 1.0 JSON schema as R.
* Bundles preserve resolution records, candidate decisions, lineages,
  distances, metric definition, software metadata, and source provenance.
* Missing distances use JSON `null`; disconnected distances use the explicit
  strings `"Infinity"` and `"-Infinity"`.
* Bundle validation checks fields, statuses, candidate counts, lineage depths,
  selected candidates, matrix labels, symmetry, and agreement between stored
  matrices and stored lineages.
* `distance_matrix()` accepts a validated bundle and returns its stored matrix
  without retrieval or recomputation.


# taxodist 0.9.0

This release mirrors taxodist 0.9.0 for R. See `PORTING-0.9.0.md` for the
correspondence between the three implementations.

## Bug fixes

* `taxo_resolve()` now truncates each selected lineage at the queried taxon
  exactly as `get_lineage()` does. Previously it stored the raw page lineage,
  so for genera whose page does not list the genus itself (e.g.
  *Periplaneta*) the lineage stopped at the family and
  `distance_matrix(taxo_resolve(x))` could differ from `distance_matrix(x)`.
* Cleaned and uncleaned lineages are cached separately (`lin_<id>` and
  `rawlin_<id>`). Previously `get_lineage_by_id(id, clean=False)` poisoned the
  cache for later calls with `clean=True`. One request now fills both entries.
* Taxon names are normalized before lookup (surrounding whitespace removed,
  repeated spaces collapsed) and the queried name is located in the lineage
  case-insensitively. `get_lineage("tyrannosaurus")` no longer appends a
  spurious extra node.
* `taxo_heatmap()` accepts `cmap` and `annot` without raising "got multiple
  values for keyword argument", accepts resolutions and bundles, and has
  explicit `verbose` and `progress` arguments.
* `cache_info()` reports the names of cached taxa in `taxa` and their IDs in
  a new `ids` entry; it previously listed only the IDs.
* `get_lineage_by_id()` and `get_lineage()` raise `TypeError` for a
  collection of several IDs or names instead of silently returning `None`.
* `focal_distances()` reports the terminal lineage node as the MRCA of a
  community member identical to the focal taxon.
* The User-Agent header and the bundle `software.version` field now report
  the installed version instead of a hard-coded string.
* `distance_matrix()` raises `TypeError` for a single string instead of
  treating each character as a taxon name.

## Behavior changes

* `check_coverage()` returns a nullable `boolean` Series with `pd.NA` when
  The Taxonomicon could not be reached, instead of reporting such names as
  absent (`False`).
* `save_cache()` returns the path of the written file.
* Using a bundle in `distance_matrix()` validates its structure only; the
  full recomputation of every distance runs in `write_taxodist_bundle()` and
  `read_taxodist_bundle()` (`validate_taxodist_bundle(bundle,
  check_distances=True)`).

## Performance

* Distance matrices are computed with NumPy from integer prefix identifiers
  (each node identifies its full path from the root), replacing the
  pure-Python double loop over pairs.
* HTTP requests are retried with exponential backoff after network errors and
  server errors (5xx). Set `taxodist.fetch.MAX_TRIES` to change the number of
  attempts (default 3).
* Each candidate lineage is retrieved once while resolving a name (it was
  previously retrieved a second time during disambiguation).

## New features

* `taxo_tree()` builds a `TaxodistTree` directly from lineages: internal nodes
  are named clades, `to_newick()` exports the tree, and `cophenetic()`
  reproduces `distance_matrix()` exactly.
* `cross_distances(x, y)` computes the rectangular matrix between two sets of
  taxa.
* `compare_bundles(old, new)` reports which taxa changed status, identifier,
  or lineage between two bundles and which distances changed. JSON paths are
  accepted directly.
* `taxo_resolve()` and `taxo_bundle()` gain `within`, which restricts
  candidates to a clade (e.g. `within="Animalia"`).
* `save_cache()` and `load_cache()` default to a persistent per-user cache
  file (`TAXODIST_CACHE_DIR` overrides the directory).

## Tests

* New test modules cover resolution, vectorized matrices, trees, bundle
  comparison, and the persistent cache; coverage remains complete.

# taxodist 1.0.0

First stable release, matching taxodist 1.0.0 for R. The default distance
metric, the other metrics, and the bundle format (schema 1.1) are now stable:
later 1.x releases will keep their definitions and continue to read bundles
written by this version.

## New features (ported from the R package)

* **Ranks.** The rank of every lineage node (e.g. `"Kingdom"`, `"Clade"`,
  `"Superfamily"`, `"Genus"`) is read from The Taxonomicon and cached with
  the lineage (`ranks_<id>` / `rawranks_<id>`). The new `lineage_ranks()`
  returns a DataFrame with columns `node`, `rank` (None when unranked), and
  `depth`. `taxo_resolve()` results gain a `ranks` column, and
  `taxo_from_lineages()` accepts user-supplied ranks through `ranks=`.
* **Distance metrics.** `taxo_distance()`, `distance_matrix()`,
  `cross_distances()`, `closest_relative()`, `focal_distances()`,
  `taxo_heatmap()`, and `taxo_bundle()` gain a `metric` argument, also
  accepted by `taxo_cluster()` and `taxo_ordinate()` through `**kwargs`:
  `"reciprocal"` (default, unchanged), `"path"`, `"wu_palmer"`, and
  `"ranked"` (half the path through the standard Linnaean ranks, configurable
  with `taxodist.distance.RANK_LEVELS`; Clarke and Warwick's step weights).
  Matrices record the metric in `DataFrame.attrs["metric"]` (the R `dist`
  `"method"` attribute); `taxo_distance()` results gain a `metric` entry,
  which `print_taxodist_result()` shows.
* **Taxonomic distinctness.** `taxo_distinctness()` computes Δ, Δ*, Δ+, Λ+,
  and SΔ+ for one or more samples; `taxo_distinctness_test()` compares Δ+ and
  Λ+ of a sample with random draws from a master list (`seed=` for
  reproducibility) and returns a `TaxodistDistinctnessTest`;
  `print_taxodist_distinctness()` prints either result. Values equal R's
  `taxo_distinctness()` (and `vegan::taxondive()`).
* `taxonomicon_available()` reports whether The Taxonomicon can be reached,
  with a short timeout, as in R.
* **Bundle schema 1.1.** Bundles record the metric (and, for `"ranked"`, its
  `rank_levels`) and the `lineage_ranks` of every taxon (null for unranked
  nodes and missing lineages). Distances are verified with the stored
  metric. Schema 1.0 bundles are still read and must use the reciprocal
  metric. Bundles written by Python are read by R and vice versa for every
  metric.

## Breaking changes: lineage parsing

* Lineages are now read from the structure of The Taxonomicon page instead of
  its text. Each node's name is taken from the page's own markup for the
  scientific name, so author strings, rank labels, extinction and type
  markers, and vernacular names can no longer leak into, or remove, nodes.
  Previously many lineages contained malformed nodes such as
  `"Passeres (C."`, `"Delphinida de"`, or `"Alismatales R.Br. ex"`; because
  these contained spaces, name-based queries silently dropped them, so lineage
  depths, and therefore distances, differed from the classification and from
  ID-based queries. Some clades were also lost entirely (for example
  `Terrestrornithes` in passerine birds), taxa whose page row is labelled
  `[plesion]` or `Anamorph genus` were recorded under that label, and the `‡`
  marker was not removed. **Distances computed with this version can differ
  from earlier versions.** The parser is identical to the R implementation and
  is checked against the same trimmed TaxonTree fixtures
  (`tests/fixtures/expected_lineages.json`).
* Species are now recorded as nodes of their subspecies, and subgenera
  (`"Genus (Subgenus)"`) as nodes of their species, when they appear in the
  lineage of a name-based query.
* Pages without the structured hierarchy block are still parsed from text,
  now by extracting the scientific name at the start of each line rather than
  by deleting author strings. The `[crown]`/`[plesion]`/`[stem]` prefix and
  rank prefixes use exactly the same rank list as R (a rank must be followed
  by whitespace), and the cut-off at the page's own taxon is a
  case-sensitive whole-word match, as in R (it was case-insensitive).
* The `taxobase` reference dataset was rebuilt from the R package with the new
  parser.

## Bug fixes

* Taxon names containing regular-expression metacharacters, notably
  parentheses in subgenus names such as `"Microtus (Agricola)"`, are matched
  literally, and whole-word matching (ID lookup, `taxo_resolve()`, the text
  parser) uses Unicode-aware lookarounds instead of `\b`, so it no longer
  fails next to punctuation.
* `taxo_resolve()` reports a name whose candidates were all retrieved but are
  not biological (for example astronomical objects) as `"unresolved"` instead
  of `"retrieval_error"`; `"retrieval_error"` now means that a candidate
  lineage could not be retrieved.
* When a name has several biological entries, entries marked as invalid,
  rejected, nude, or illegitimate names (`nom. inval.`, `nom. rej.`,
  `nom. nud.`, `nom. illeg.`) are considered only after the others, in both
  `get_taxonomicon_id()` and `taxo_resolve()`.
* `load_cache()` discards lineages from cache files written by earlier
  versions (with a warning) and keeps their taxon IDs. `save_cache()` records
  the parser version under `"meta_parser_version"`; the key never enters the
  session cache.
* Search results are parsed exactly as in R: the row text is concatenated
  without separators, only links whose class is exactly `Valid` are used, the
  astronomical filter no longer includes the extra word `astronomy`, and the
  `N|T|P|R|B|L` prefix is removed only in its exact form. Candidate
  descriptions are therefore identical in both implementations.

## Python-specific fixes

* `taxo_cluster()` and `taxo_ordinate()` no longer treat a `TaxodistResolution`
  (a `DataFrame` subclass) as a distance matrix; resolutions and bundles are
  passed through `distance_matrix()`, as in R.
* `get_lineage()` and `get_lineage_by_id()` return copies of cached lineages
  and cache copies of fetched ones, so modifying a returned list no longer
  corrupts the cache. `distance_matrix(bundle)` returns a copy of the stored
  matrix.
* `check_coverage()`, `closest_relative()` (candidates), `focal_distances()`
  (community), and `filter_clade()` treat a single string as one taxon instead
  of iterating over its characters, and accept generators.
* `get_lineage()` and `get_lineage_by_id()` treat integral numbers, including
  `50841.0` and NumPy scalars, as Taxonomicon IDs (`"50841"`), as R does.
* `is_member()` and `filter_clade()` raise `TypeError` for a clade that is not
  a string; `is_member(taxon, None)` previously matched a node named `"None"`.
* `distance_matrix()`, `cross_distances()`, and `taxo_tree()` reject
  dictionaries, as R rejects lists.
* `focal_distances()` always returns `mrca_depth` as a nullable `Int64`
  column. `print_focal_distances()` is exported.
* `plot_taxodist_cluster()` shows the linkage method in its subtitle by
  default; `taxo_cluster()` records it under `"method"`.
* `pd.concat()` of resolutions keeps `source` and `source_url` when all inputs
  agree (and the earliest `retrieved_at`), and leaves them unset otherwise.
  `taxo_bundle()` assumes The Taxonomicon for a resolution without a source
  only when every ID is a Taxonomicon ID (the R default); otherwise it raises
  an error instead of mislabelling the provenance.

## Network

* Consecutive requests to The Taxonomicon are spaced by at least
  `taxodist.fetch.MIN_INTERVAL` seconds (default 0.25; the R option is
  `taxodist.min_interval`), so large batches do not overload the server. Set
  it to 0 to disable.

## Packaging

* matplotlib and seaborn are imported only by the plotting functions and moved
  to the optional `plot` extra (`pip install "taxodist[plot]"`). SciPy remains
  a core dependency because `taxo_cluster()` uses it; it is also imported
  lazily.
* The bundle JSON schema (`taxodist/schema/taxodist-bundle-v1.schema.json`) is
  now included in the package.
* The GPL-3 license text is shipped as `LICENSE` and declared with PEP 639
  metadata (`license = "GPL-3.0-or-later"`, `license-files`); this requires
  setuptools >= 77.
* The version is defined once, in `taxodist/_version.py`. Python 3.8 is no
  longer supported.
