import os
import re
import pickle
import urllib.parse
import warnings
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import requests
from bs4 import BeautifulSoup
import pandas as pd

_taxodist_cache = {}

# Keep one HTTP client for the lifetime of the Python process.  Top-level
# requests.get() creates a short-lived Session for every call, whereas a
# persistent Session reuses the connection pool for the many requests made by
# name resolution and lineage retrieval.
_http_session = requests.Session()
_http_session.headers.update({"User-Agent": "taxodist Python package/0.8.0"})

_RANK_PREFIXES = (
    "Clade|Kingdom|Phylum|Superphylum|Subphylum|Infraphylum|Class|Order|"
    "Suborder|Infraorder|Parvorder|Grandorder|Magnorder|Cohort|Subcohort|"
    "Legion|Family|Subfamily|Tribe|Subtribe|Genus|Species|Subkingdom|"
    "Infrakingdom|Superclass|Subclass|Infraclass|Superorder|Superfamily|"
    "Domain|Superkingdom|Grade|Subgrade|Supergrade"
)

_BARE_RANKS = {
    "Go to", "Superphylum", "Subfamily", "Suborder", "Epifamily",
    "Infraorder", "Superclass", "Subclass", "Superfamily", "Subgenus",
    "Section", "Division", "Candidatus", "Parvphylum", "Branch",
    "Supercohort", "Infracohort", "Subdivision", "Subsection", "Grade",
    "[unranked]", "(Supercluster)", "(Region)", "[crown]", ""
}


def _clean_lineage_label(raw_text):
    """Apply the same lineage-label cleaning sequence as taxodist for R."""
    text = re.sub(r"[\u2020\u1D40]", "", str(raw_text))
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(
        rf"^\[crown\]\s+({_RANK_PREFIXES})?\s*",
        "",
        text,
    )
    text = re.sub(rf"^({_RANK_PREFIXES}) ", "", text)
    text = re.sub(
        r"\s+[A-Z][a-záàâãéèêíïóôõöúüç].*$",
        "",
        text,
    )
    text = re.sub(r"\s+[A-Z]\.[A-Z]\..*$", "", text)
    text = re.sub(r"\s+auct\..*$", "", text)
    text = re.sub(r"\s+von.*$", "", text)
    text = re.sub(r"\s+\([A-Z][a-z].*$", "", text)
    text = re.sub(r"\s+\(\d{4}\).*$", "", text)
    text = re.sub(r"\s+\[.*$", "", text)
    text = re.sub(r"\s+[A-Z]\.$", "", text)
    text = re.sub(r"\s+\([a-z].*$", "", text)
    text = re.sub(r'\s+".*$', "", text)
    text = re.sub(r'^".*', "", text)
    return text.strip()


def clear_cache():
    """
    Clear the taxodist lineage cache

    Clears all cached lineages stored in the current R session. Useful when
    you suspect cached data is stale or want to force fresh retrieval.

    Returns
    -------
    None
        Invisibly returns None.
    """
    global _taxodist_cache
    _taxodist_cache.clear()
    return None


def save_cache(file):
    """
    Save the taxodist lineage cache to disk

    Serialises the current session cache to a pickle file so it can be
    restored in a future session with load_cache(). Useful for
    reproducibility and for avoiding repeated network requests.

    Parameters
    ----------
    file : str
        Path to the pickle file to write (the `.pkl` extension is recommended).

    Returns
    -------
    None
        Invisibly returns None.
    """
    with open(file, 'wb') as f:
        pickle.dump(dict(_taxodist_cache), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cache saved to '{file}' ({len(_taxodist_cache)} entries).")
    return None


def load_cache(file):
    """
    Load a previously saved taxodist cache from disk

    Restores lineage data saved with save_cache() into the current session
    cache, avoiding network requests for taxa already retrieved in a previous
    session.

    Parameters
    ----------
    file : str
        Path to a pickle file created by save_cache().

    Returns
    -------
    None
        Invisibly returns None.
    """
    global _taxodist_cache
    if not os.path.exists(file):
        raise FileNotFoundError(f"Cache file not found: '{file}'")
    with open(file, 'rb') as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError("Invalid cache file: expected a dictionary created by save_cache().")

    if any(not isinstance(key, str) or not key for key in data):
        raise ValueError("Invalid cache file: cache keys must be non-empty strings.")

    for key, value in data.items():
        if key.startswith("id_"):
            if not isinstance(value, str) or not value:
                raise ValueError(
                    "Invalid cache file: taxon ID entries must be non-empty strings."
                )
        elif key.startswith("lin_"):
            if not isinstance(value, (list, tuple)) or not all(
                isinstance(node, str) for node in value
            ):
                raise ValueError(
                    "Invalid cache file: lineage entries must be sequences of strings."
                )
        elif key.startswith("resolved_lineage_"):
            if not isinstance(value, (list, tuple)) or not all(
                isinstance(node, str) for node in value
            ):
                raise ValueError(
                    "Invalid cache file: resolved lineage entries must be sequences of strings."
                )

    # Modify the active cache only after the complete file has been validated.
    _taxodist_cache.update(data)
    print(f"Cache loaded from '{file}' ({len(data)} entries).")
    return None

def cache_info():
    """
    Inspect the current taxodist lineage cache

    Reports the number of cached entries, their total memory footprint, and
    the names of all taxa whose lineages are stored in the current session.
    Useful for understanding what has already been retrieved before running
    further computations.

    Returns
    -------
    dict
        A dictionary with:
        - n_lineages: int. Number of cached lineages.
        - n_ids: int. Number of cached taxon IDs.
        - taxa: list. Names of taxa with cached lineages (id stripped).
        - size_bytes: int. Total memory used by the cache.
    """
    import sys

    lin_keys = [k for k in _taxodist_cache if k.startswith("lin_")]
    id_keys  = [k for k in _taxodist_cache if k.startswith("id_")]
    taxa_names = [k[len("lin_"):] for k in lin_keys]
    size_bytes = sys.getsizeof(_taxodist_cache)

    print("taxodist Cache")
    print(f"* Lineages cached : {len(lin_keys)}")
    print(f"* IDs cached      : {len(id_keys)}")
    print(f"* Memory used     : {size_bytes} bytes")

    if taxa_names:
        print("\nCached taxa:")
        for name in taxa_names:
            print(f"  {name}")
    else:
        print("\nNo lineages cached yet.")

    return {
        "n_lineages": len(lin_keys),
        "n_ids":      len(id_keys),
        "taxa":       taxa_names,
        "size_bytes": size_bytes
    }

def get_taxonomicon_id(taxon, verbose=False):
    """
    Find the Taxonomicon ID for a taxon name

    Queries The Taxonomicon (taxonomy.nl) to retrieve the internal numeric
    identifier for a given taxon name. The search filters out non-biological
    entities such as astronomical objects that may share the same name.

    Parameters
    ----------
    taxon : str
        A character string giving the taxon name to search for.
        Typically a genus name (e.g., "Tyrannosaurus") but species and higher
        ranks are also supported.
    verbose : bool
        Logical. If True, prints status messages during retrieval.
        Default is False.

    Returns
    -------
    str or None
        A character string containing the Taxonomicon numeric ID, or None
        if the taxon is not found.

    Details
    -------
    The function queries the static search endpoint at
    taxonomicon.taxonomy.nl/TaxonList.aspx and parses the resulting HTML
    to extract the taxon ID from the hierarchy link. When multiple matches
    exist (e.g., a genus name shared with an astronomical object), biological
    entries are prioritised by filtering for entries annotated as dinosaur,
    reptile, archosaur, animal, plant, fungus, or bacterium.
    """
    cache_key = f"id_{taxon}"
    if cache_key in _taxodist_cache:
        if verbose:
            print(f"Using cached ID for {taxon}")
        return _taxodist_cache[cache_key]

    if verbose:
        print(f"Searching Taxonomicon for {taxon}...")

    safe_taxon = urllib.parse.quote(str(taxon))
    url = f"http://taxonomicon.taxonomy.nl/TaxonList.aspx?subject=Entity&by=ScientificName&search={safe_taxon}"
    try:
        res = _http_session.get(url, timeout=30)
        if res.status_code != 200:
            warnings.warn(
                "Cannot reach The Taxonomicon server.\n"
                "The website (taxonomy.nl) appears to be offline or unreachable.\n"
                "Please try again later."
            )
            return None
    except requests.exceptions.RequestException:
        warnings.warn(
            "Cannot reach The Taxonomicon server.\n"
            "The website (taxonomy.nl) appears to be offline or unreachable.\n"
            "Please try again later."
        )
        return None

    try:
        soup = BeautifulSoup(res.text, "lxml")
    except Exception:
        warnings.warn("Could not parse the response from The Taxonomicon.")
        return None
    rows = soup.find_all("tr")
    bio_ids =[]

    for row in rows:
        text = row.get_text(separator=" ", strip=True)
        if re.search(r"\bastronomical\b|\bplanet\b|\bMinor planet\b|\bcomet\b|\bastronomy\b|\basteroid\b", text, flags=re.IGNORECASE):
            continue

        links_nodes = row.find_all("a", href=re.compile(r"TaxonTree"))
        if not links_nodes:
            continue

        valid_links =[a for a in links_nodes if "Valid" in a.get("class", [])]
        if not valid_links:
            continue

        target_link = valid_links[0]
        href = target_link.get("href", "")
        
        match = re.search(r"id=([0-9]+)", href)
        if not match:
            continue
        
        id_val = match.group(1)

        text_entry = re.sub(r"\s+", " ", text).strip()
        text_entry = re.sub(r"^N\s*\|\s*T\s*\|\s*P\s*\|\s*R\s*\|\s*B\s*\|\s*L\s*", "", text_entry, count=1)

        candidate_lin = get_lineage_by_id(id_val, clean=True, verbose=False)
        if candidate_lin is None or "Biota" not in candidate_lin:
            continue

        bio_ids.append({"id": id_val, "text": text_entry})

    if len(bio_ids) > 1:
        matched = []
        for entry in bio_ids:
            lin = get_lineage_by_id(entry["id"], clean=True, verbose=False)
            if lin is not None:
                # Match the taxon name as a distinct word in the lineage list
                word_pattern = rf"\b{re.escape(str(taxon))}\b"
                if any(re.search(word_pattern, node, flags=re.IGNORECASE) for node in lin):
                    matched.append(entry)
        if matched:
            bio_ids = matched
            
    if not bio_ids:
        if verbose:
            print(f"{taxon} not found in Taxonomicon")
        return None

    unique_ids =[]
    seen = set()
    for b in bio_ids:
        if b["id"] not in seen:
            seen.add(b["id"])
            unique_ids.append(b["id"])

    unique_bio_ids = []
    for uid in unique_ids:
        matches =[b for b in bio_ids if b["id"] == uid]
        unique_bio_ids.append(matches[0])

    if len(unique_bio_ids) > 1:
        warn_msg =[
            f"Multiple valid biological entries found for '{taxon}'.",
            f"Using: {unique_bio_ids[0]['text']} (ID: {unique_bio_ids[0]['id']})",
            f"To use a different entry, pass its numeric ID directly, e.g. `get_lineage(\"{unique_bio_ids[1]['id']}\")`.",
            "Other available IDs:"
        ]
        for i in range(1, len(unique_bio_ids)):
            warn_msg.append(f"* ID {unique_bio_ids[i]['id']}: {unique_bio_ids[i]['text']}")
        warnings.warn("\n".join(warn_msg))

    final_id = unique_bio_ids[0]["id"]
    _taxodist_cache[cache_key] = final_id
    if verbose:
        print(f"Found {taxon} with ID {final_id}")
        
    return final_id


def get_lineage_by_id(taxon_id, clean=True, verbose=False):
    """
    Retrieve the full taxonomic lineage of a taxon

    Given a Taxonomicon numeric ID, retrieves and parses the complete
    hierarchical lineage from root (Natura) to the taxon itself. The lineage
    is returned as a character vector ordered from root to tip.

    Parameters
    ----------
    taxon_id : str or int
        A numeric or character string giving the Taxonomicon ID.
        Obtain this with get_taxonomicon_id().
    clean : bool
        Logical. If True (default), removes philosophical root nodes
        above Biota (i.e., Natura, actualia, Mundus, naturalia) and strips
        dagger and superscript markers from names.
    verbose : bool
        Logical. If True, prints status messages. Default False.

    Returns
    -------
    list or None
        A character vector of clade names from root to tip, or None if
        retrieval fails.

    Details
    -------
    Lineage data is sourced from The Taxonomicon, which is based on
    Systema Naturae 2000 (Brands, S.J., 1989 onwards). The depth of lineages
    in The Taxonomicon substantially exceeds that of other programmatic sources
    such as the Open Tree of Life, particularly for well-studied clades such
    as Dinosauria, where intermediate clades at the level of superfamilies,
    tribes, and named subclades are fully resolved.
    """
    if taxon_id is None or str(taxon_id).strip() == "" or not re.match(r"^[0-9]+$", str(taxon_id)):
        return None

    taxon_id = str(taxon_id)
    cache_key = f"lin_{taxon_id}"

    if cache_key in _taxodist_cache:
        if verbose:
            print(f"Using cached lineage for ID {taxon_id}")
        return _taxodist_cache[cache_key]

    url = f"http://taxonomicon.taxonomy.nl/TaxonTree.aspx?id={taxon_id}&src=0"
    try:
        res = _http_session.get(url, timeout=30)
        if res.status_code != 200:
            if verbose:
                print(f"Could not retrieve lineage for ID {taxon_id}")
            return None
    except requests.exceptions.RequestException:
        if verbose:
            print(f"Could not retrieve lineage for ID {taxon_id}")
        return None

    try:
        soup = BeautifulSoup(res.text, "lxml")
    except Exception:
        if verbose:
            print(f"Could not parse lineage for ID {taxon_id}")
        return None
    subject_node = soup.select_one("#ctl00_divSubject b")
    content_node = soup.select_one("#divPageContent")
    
    current_name = subject_node.get_text(strip=True) if subject_node else None
    
    use_text_parsing = False
    texts = []
    
    if current_name and content_node:
        tree_text = content_node.get_text()
        raw_lines = [line.strip() for line in tree_text.split("\n")]
        raw_lines = [line for line in raw_lines if line]
        
        # Locate where tree starts (Natura)
        tree_start_idx = None
        for idx, line in enumerate(raw_lines):
            if line.startswith("Natura"):
                tree_start_idx = idx
                break
        
        if tree_start_idx is not None:
            raw_lines = raw_lines[tree_start_idx:]
            
        # Clean daggers for indexing safety
        raw_lines_search = [re.sub(r"[\u2020\u1D40†]", "", line) for line in raw_lines]
        
        # Cut off raw lines once the current name is matched as a distinct word
        word_pattern = rf"\b{re.escape(current_name)}\b"
        cutoff_idx = None
        for idx, line in enumerate(raw_lines_search):
            if re.search(word_pattern, line, flags=re.IGNORECASE):
                cutoff_idx = idx
                break
                
        if cutoff_idx is not None:
            texts = raw_lines[:cutoff_idx + 1]
            use_text_parsing = True
        else:
            texts = raw_lines
            use_text_parsing = True
            
    if not use_text_parsing:
        # Fallback to your robust link-based extraction method
        links = soup.find_all("a", href=re.compile(r"TaxonTree"))
        hrefs_all = [a.get("href", "") for a in links]
        valid_links = [a for a, h in zip(links, hrefs_all) if re.search(r"id=[0-9]", h)]
        hrefs_all = [h for h in hrefs_all if re.search(r"id=[0-9]", h)]

        hrefs = [a.get("href", "") for a in valid_links]
        own_pattern = f"id={taxon_id}(&|$)"
        
        own_idx = [i for i, h in enumerate(hrefs) if re.search(own_pattern, h)]
        if own_idx:
            valid_links = valid_links[:max(own_idx) + 1]
            
        texts = [a.get_text(separator=" ", strip=True) for a in valid_links]

    lineage = [_clean_lineage_label(raw_text) for raw_text in texts]
    lineage = [x for x in lineage if x not in _BARE_RANKS]
    lineage =[x for x in lineage if x != "" and not re.match(r"^\s*$", x)]
    lineage = [x for x in lineage if not x.startswith('"')]
    lineage = [x for x in lineage if not x.startswith("Population")]
    
    lineage = list(dict.fromkeys(lineage))

    if clean:
        try:
            idx_biota = lineage.index("Biota")
            lineage = lineage[idx_biota:]
        except ValueError:
            pass

    if not lineage:
        return None

    _taxodist_cache[cache_key] = lineage
    return lineage

def get_lineage(taxon, clean=True, verbose=False):
    """
    Retrieve the full taxonomic lineage of a taxon by name

    A convenience wrapper that combines get_taxonomicon_id() and
    get_lineage_by_id() into a single call. Given a taxon name, returns
    its complete lineage from root to tip.

    Parameters
    ----------
    taxon : str
        A character string giving the taxon name.
    clean : bool
        Logical. If True (default), removes philosophical root nodes
        and cleans formatting markers.
    verbose : bool
        Logical. If True, prints progress messages. Default False.

    Returns
    -------
    list or None
        A character vector of clade names ordered from root to tip, or
        None if the taxon cannot be found.
    """
    taxon_str = str(taxon)
    is_id = bool(re.match(r"^[0-9]+$", taxon_str))

    resolved_key = f"resolved_lineage_{int(bool(clean))}_{taxon_str}"
    if not is_id and resolved_key in _taxodist_cache:
        if verbose:
            print(f"Using cached resolved lineage for {taxon_str}")
        return list(_taxodist_cache[resolved_key])

    if is_id:
        id_val = taxon_str
    else:
        id_val = get_taxonomicon_id(taxon_str, verbose=verbose)

    if id_val is None:
        return None

    lineage = get_lineage_by_id(id_val, clean=clean, verbose=verbose)
    if lineage is None:
        return None

    if not is_id:
        if not re.search(r"\s", taxon_str):
            lineage =[x for x in lineage if not re.search(r" ", x)]
            lineage =[x for x in lineage if not x.startswith("[")]
            matches = [i for i, x in enumerate(lineage) if x == taxon_str]
            if matches:
                target_idx = matches[-1]
                lineage = lineage[:target_idx + 1]
            else:
                lineage.append(taxon_str)
        else:
            lineage =[x for x in lineage if not re.search(r" ", x) or x == taxon_str]
            matches = [i for i, x in enumerate(lineage) if x == taxon_str]
            if matches:
                target_idx = matches[0]
                lineage = lineage[:target_idx + 1]
            else:
                lineage.append(taxon_str)

    if not lineage:
        return None

    if not is_id:
        _taxodist_cache[resolved_key] = list(lineage)

    return lineage


def _taxo_search_details(taxon, verbose=False):
    """Return a structured Taxonomicon search result for internal callers."""
    if verbose:
        print(f"Searching Taxonomicon for '{taxon}'...")

    safe_taxon = urllib.parse.quote(str(taxon))
    url = (
        "http://taxonomicon.taxonomy.nl/TaxonList.aspx"
        f"?subject=Entity&by=ScientificName&search={safe_taxon}"
    )
    try:
        res = _http_session.get(url, timeout=30)
        if res.status_code != 200:
            if verbose:
                print("Could not reach Taxonomicon")
            return {"status": "retrieval_error", "results": None}
    except requests.exceptions.RequestException:
        if verbose:
            print("Could not reach Taxonomicon")
        return {"status": "retrieval_error", "results": None}

    try:
        soup = BeautifulSoup(res.text, "lxml")
        rows = soup.find_all("tr")
    except Exception:
        if verbose:
            print("Could not parse the Taxonomicon response")
        return {"status": "retrieval_error", "results": None}

    results = []
    for row in rows:
        text = row.get_text(separator=" ", strip=True)
        if re.search(
            r"\bastronomical\b|\bplanet\b|\bMinor planet\b|\bcomet\b|"
            r"\bastronomy\b|\basteroid\b",
            text,
            flags=re.IGNORECASE,
        ):
            continue

        links = row.find_all("a", href=re.compile(r"TaxonTree"))
        valid_links = [a for a in links if "Valid" in a.get("class", [])]
        if not valid_links:
            continue

        match = re.search(r"id=([0-9]+)", valid_links[0].get("href", ""))
        if not match:
            continue

        text_entry = re.sub(r"\s+", " ", text).strip()
        text_entry = re.sub(
            r"^N\s*\|\s*T\s*\|\s*P\s*\|\s*R\s*\|\s*B\s*\|\s*L\s*",
            "",
            text_entry,
            count=1,
        )
        results.append({"id": match.group(1), "name": text_entry})

    if not results:
        if verbose:
            print("No matches found.")
        return {"status": "not_found", "results": None}

    frame = pd.DataFrame(results).drop_duplicates(subset=["id"]).reset_index(drop=True)
    if verbose:
        print(f"Found {len(frame)} entries.")
    return {"status": "ok", "results": frame}


def taxo_search(taxon, verbose=False):
    """
    Search The Taxonomicon for a taxon name

    Queries The Taxonomicon database and returns a data frame of all available
    biological entries matching the search string. This is particularly useful
    for exploring homonyms, historical ranks, or taxonomic synonyms before
    computing distances.

    Parameters
    ----------
    taxon : str
        A character string giving the taxon name to search for.
    verbose : bool
        Logical. If True, prints status messages. Default False.

    Returns
    -------
    pandas.DataFrame or None
        A data frame with columns:
        - id: Character. The numeric Taxonomicon ID.
        - name: Character. The full taxon description, including rank and author.
        Returns None if no matches are found.
    """
    return _taxo_search_details(taxon, verbose=verbose)["results"]


RESOLUTION_COLUMNS = [
    "input",
    "resolved_name",
    "id",
    "status",
    "n_candidates",
    "lineage_depth",
    "lineage",
    "candidates",
]


def _utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _empty_candidates():
    return pd.DataFrame(
        {"id": pd.Series(dtype="object"), "name": pd.Series(dtype="object")}
    )


class TaxodistResolution(pd.DataFrame):
    """A pandas resolution table with source provenance."""

    _metadata = ["source", "source_url", "retrieved_at"]
    _taxodist_resolution = True

    @property
    def _constructor(self):
        return TaxodistResolution

    def summary_counts(self):
        counts = self["status"].value_counts()
        return {
            status: int(counts.get(status, 0))
            for status in (
                "resolved",
                "ambiguous",
                "unresolved",
                "retrieval_error",
            )
        }


def _make_resolution(rows, source, source_url, retrieved_at=None):
    frame = TaxodistResolution(rows, columns=RESOLUTION_COLUMNS)
    frame["n_candidates"] = frame["n_candidates"].astype("int64")
    frame["lineage_depth"] = pd.array(frame["lineage_depth"], dtype="Int64")
    frame.source = source
    frame.source_url = source_url
    frame.retrieved_at = retrieved_at or _utc_timestamp()
    return frame


def taxo_resolve(taxa, ambiguity="warn", verbose=False, progress=True):
    """Resolve names or numeric Taxonomicon IDs into an auditable table."""
    if ambiguity not in {"warn", "first", "error"}:
        raise ValueError("ambiguity must be 'warn', 'first', or 'error'")
    if isinstance(taxa, (str, bytes, Mapping)) or not isinstance(taxa, Iterable):
        raise TypeError("taxa must be a sequence of strings")
    taxa = list(taxa)
    if any(not isinstance(taxon, str) for taxon in taxa):
        raise TypeError("taxa must be a sequence of strings")
    if any(not taxon.strip() for taxon in taxa):
        raise ValueError("taxa cannot contain missing or empty values")

    def resolve_one(taxon):
        if re.fullmatch(r"[0-9]+", taxon):
            lineage = get_lineage_by_id(taxon, clean=True, verbose=verbose)
            if lineage is None:
                return {
                    "input": taxon,
                    "resolved_name": None,
                    "id": None,
                    "status": "retrieval_error",
                    "n_candidates": 0,
                    "lineage_depth": None,
                    "lineage": None,
                    "candidates": _empty_candidates(),
                }
            name = lineage[-1]
            return {
                "input": taxon,
                "resolved_name": name,
                "id": taxon,
                "status": "resolved",
                "n_candidates": 1,
                "lineage_depth": len(lineage),
                "lineage": list(lineage),
                "candidates": pd.DataFrame([{"id": taxon, "name": name}]),
            }

        search = _taxo_search_details(taxon, verbose=verbose)
        if search["status"] == "retrieval_error":
            return {
                "input": taxon,
                "resolved_name": None,
                "id": None,
                "status": "retrieval_error",
                "n_candidates": 0,
                "lineage_depth": None,
                "lineage": None,
                "candidates": _empty_candidates(),
            }
        candidates = search["results"]
        if search["status"] == "not_found" or candidates is None or candidates.empty:
            return {
                "input": taxon,
                "resolved_name": None,
                "id": None,
                "status": "unresolved",
                "n_candidates": 0,
                "lineage_depth": None,
                "lineage": None,
                "candidates": _empty_candidates(),
            }

        searched_candidates = candidates.reset_index(drop=True).copy()
        candidate_lineages = [
            get_lineage_by_id(identifier, clean=True, verbose=verbose)
            for identifier in candidates["id"]
        ]
        valid = [
            lineage is not None and "Biota" in lineage
            for lineage in candidate_lineages
        ]
        candidates = candidates.loc[valid].reset_index(drop=True)
        candidate_lineages = [
            lineage for lineage, keep in zip(candidate_lineages, valid) if keep
        ]
        if candidates.empty:
            return {
                "input": taxon,
                "resolved_name": None,
                "id": None,
                "status": "retrieval_error",
                "n_candidates": len(searched_candidates),
                "lineage_depth": None,
                "lineage": None,
                "candidates": searched_candidates,
            }

        if len(candidates) > 1:
            pattern = re.compile(rf"\b{re.escape(taxon)}\b", re.IGNORECASE)
            exact = [
                any(pattern.search(node) for node in lineage)
                for lineage in candidate_lineages
            ]
            if any(exact):
                candidates = candidates.loc[exact].reset_index(drop=True)
                candidate_lineages = [
                    lineage
                    for lineage, keep in zip(candidate_lineages, exact)
                    if keep
                ]

        selected_lineage = list(candidate_lineages[0])
        return {
            "input": taxon,
            "resolved_name": selected_lineage[-1],
            "id": str(candidates.iloc[0]["id"]),
            "status": "ambiguous" if len(candidates) > 1 else "resolved",
            "n_candidates": len(candidates),
            "lineage_depth": len(selected_lineage),
            "lineage": selected_lineage,
            "candidates": candidates.copy(),
        }

    unique_taxa = list(dict.fromkeys(taxa))
    resolved_unique = {}
    for index, taxon in enumerate(unique_taxa, start=1):
        resolved_unique[taxon] = resolve_one(taxon)
        if progress:
            print(f"Resolving taxa [{index}/{len(unique_taxa)}]: {taxon}")

    result = _make_resolution(
        [resolved_unique[taxon].copy() for taxon in taxa],
        source="The Taxonomicon",
        source_url="http://taxonomicon.taxonomy.nl",
    )
    ambiguous = result.loc[result["status"] == "ambiguous"]
    if not ambiguous.empty:
        details = ", ".join(
            f"{row.input} ({row.n_candidates} candidates)"
            for row in ambiguous.itertuples()
        )
        message = (
            "Ambiguous taxon names were resolved using the first candidate: "
            f"{details}. Inspect taxo_search() or pass numeric IDs explicitly."
        )
        if ambiguity == "error":
            raise ValueError(message)
        if ambiguity == "warn":
            warnings.warn(message, UserWarning, stacklevel=2)
    return result


def taxo_from_lineages(lineages, ids=None, source="user-supplied"):
    """Create an auditable offline resolution from root-to-tip lineages."""
    if not isinstance(lineages, dict):
        raise TypeError("lineages must be a dictionary with unique, non-empty names")
    labels = list(lineages)
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("lineages must have unique, non-empty names")
    if any(
        not isinstance(lineage, (list, tuple))
        or not lineage
        or any(not isinstance(node, str) or not node.strip() for node in lineage)
        for lineage in lineages.values()
    ):
        raise ValueError(
            "Every lineage must be a non-empty sequence without missing or empty nodes"
        )
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be one non-empty string")

    if ids is None:
        identifiers = [f"custom:{label}" for label in labels]
    elif isinstance(ids, dict):
        if not all(label in ids for label in labels):
            raise ValueError("Named ids must contain every lineage name")
        identifiers = [ids[label] for label in labels]
    elif isinstance(ids, Iterable) and not isinstance(ids, (str, bytes)):
        identifiers = list(ids)
    else:
        raise TypeError("ids must be a sequence or dictionary of strings")
    if (
        len(identifiers) != len(labels)
        or any(
            not isinstance(identifier, str) or not identifier.strip()
            for identifier in identifiers
        )
        or len(set(identifiers)) != len(identifiers)
    ):
        raise ValueError(
            "ids must contain one unique, non-empty identifier per lineage"
        )

    rows = []
    for label, identifier in zip(labels, identifiers):
        lineage = list(lineages[label])
        resolved_name = lineage[-1]
        rows.append(
            {
                "input": label,
                "resolved_name": resolved_name,
                "id": identifier,
                "status": "resolved",
                "n_candidates": 1,
                "lineage_depth": len(lineage),
                "lineage": lineage,
                "candidates": pd.DataFrame(
                    [{"id": identifier, "name": resolved_name}]
                ),
            }
        )
    return _make_resolution(rows, source=source, source_url=None)
