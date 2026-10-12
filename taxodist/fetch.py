import math
import os
import re
import sys
import time
import pickle
import urllib.parse
import warnings
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import requests
from bs4 import BeautifulSoup
import pandas as pd

from ._version import __version__

_taxodist_cache = {}

# Keep one HTTP client for the lifetime of the Python process.  Top-level
# requests.get() creates a short-lived Session for every call, whereas a
# persistent Session reuses the connection pool for the many requests made by
# name resolution and lineage retrieval.
_http_session = requests.Session()
_http_session.headers.update({"User-Agent": f"taxodist Python package/{__version__}"})

#: Number of attempts made for each request to The Taxonomicon. Network errors
#: and server errors (5xx) are retried with exponential backoff; client errors
#: (4xx) are not. Set ``taxodist.fetch.MAX_TRIES = 1`` to disable retries.
MAX_TRIES = 3

_sleep = time.sleep

_ASTRONOMICAL_PATTERN = (
    r"\bastronomical\b|\bplanet\b|\bMinor planet\b|\bcomet\b|"
    r"\bastronomy\b|\basteroid\b"
)


def _max_tries():
    try:
        tries = int(MAX_TRIES)
    except (TypeError, ValueError):
        return 1
    return max(tries, 1)


def _http_get(url):
    """GET ``url`` with retries; raise the last RequestException on failure."""
    tries = _max_tries()
    for attempt in range(1, tries + 1):
        try:
            response = _http_session.get(url, timeout=30)
        except requests.exceptions.RequestException:
            if attempt == tries:
                raise
        else:
            if response.status_code < 500 or attempt == tries:
                return response
        _sleep(min(2 ** (attempt - 1), 5))


def _normalize_taxon(taxon):
    """Trim and collapse whitespace so equivalent spellings share a cache key."""
    return re.sub(r"\s+", " ", str(taxon)).strip()


def _search_url(taxon):
    safe_taxon = urllib.parse.quote(str(taxon), safe="")
    return (
        "http://taxonomicon.taxonomy.nl/TaxonList.aspx"
        f"?subject=Entity&by=ScientificName&search={safe_taxon}"
    )


def _parse_search_rows(soup):
    """Return unique ``{"id", "name"}`` candidates from a TaxonList page."""
    results = []
    seen = set()
    for row in soup.find_all("tr"):
        text = row.get_text(separator=" ", strip=True)
        if re.search(_ASTRONOMICAL_PATTERN, text, flags=re.IGNORECASE):
            continue

        links = row.find_all("a", href=re.compile(r"TaxonTree"))
        valid_links = [a for a in links if "Valid" in a.get("class", [])]
        if not valid_links:
            continue

        match = re.search(r"id=([0-9]+)", valid_links[0].get("href", ""))
        if not match or match.group(1) in seen:
            continue

        text_entry = re.sub(r"\s+", " ", text).strip()
        text_entry = re.sub(
            r"^N\s*\|\s*T\s*\|\s*P\s*\|\s*R\s*\|\s*B\s*\|\s*L\s*",
            "",
            text_entry,
            count=1,
        )
        seen.add(match.group(1))
        results.append({"id": match.group(1), "name": text_entry})
    return results


def _finalize_lineage(lineage, taxon):
    """
    Truncate a name-resolved lineage at the queried taxon.

    Genus-level and higher queries drop nodes containing spaces (species and
    bracketed rank labels); binomial queries keep only the queried binomial.
    The taxon is matched exactly first and case-insensitively as a fallback,
    and appended only when it is absent from the page.
    """
    folded = taxon.casefold()

    def target_indices(nodes):
        exact = [i for i, node in enumerate(nodes) if node == taxon]
        if exact:
            return exact
        return [i for i, node in enumerate(nodes) if node.casefold() == folded]

    if " " not in taxon:
        lineage = [x for x in lineage if " " not in x and not x.startswith("[")]
        matches = target_indices(lineage)
        if matches:
            return lineage[:matches[-1] + 1]
    else:
        lineage = [x for x in lineage if " " not in x or x.casefold() == folded]
        matches = target_indices(lineage)
        if matches:
            return lineage[:matches[0] + 1]
    return lineage + [taxon]

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


def _default_cache_file():
    """
    Location of the persistent per-user cache file.

    ``TAXODIST_CACHE_DIR`` overrides the directory. Otherwise the platform's
    user cache directory is used: ``%LOCALAPPDATA%`` on Windows,
    ``~/Library/Caches`` on macOS, and ``$XDG_CACHE_HOME`` (default
    ``~/.cache``) elsewhere.
    """
    directory = os.environ.get("TAXODIST_CACHE_DIR")
    if not directory:
        if sys.platform.startswith("win"):
            base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
            directory = os.path.join(base, "taxodist", "Cache")
        elif sys.platform == "darwin":
            directory = os.path.expanduser("~/Library/Caches/taxodist")
        else:
            base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
            directory = os.path.join(base, "taxodist")
    return os.path.join(directory, "lineage-cache.pkl")


def save_cache(file=None):
    """
    Save the taxodist lineage cache to disk

    Serialises the current session cache to a pickle file so it can be
    restored in a future session with load_cache(). Useful for
    reproducibility and for avoiding repeated network requests.

    Parameters
    ----------
    file : str or None
        Path to the pickle file to write (the `.pkl` extension is
        recommended). If None (default), a persistent per-user cache file is
        used and its directory created if necessary (see
        ``TAXODIST_CACHE_DIR``).

    Returns
    -------
    str
        The path of the written file.
    """
    if file is None:
        file = _default_cache_file()
        os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'wb') as f:
        pickle.dump(dict(_taxodist_cache), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cache saved to '{file}' ({len(_taxodist_cache)} entries).")
    return file


def load_cache(file=None):
    """
    Load a previously saved taxodist cache from disk

    Restores lineage data saved with save_cache() into the current session
    cache, avoiding network requests for taxa already retrieved in a previous
    session.

    Parameters
    ----------
    file : str or None
        Path to a pickle file created by save_cache(). If None (default), the
        persistent per-user cache written by ``save_cache()`` is used.

    Returns
    -------
    None
        Invisibly returns None.
    """
    global _taxodist_cache
    if file is None:
        file = _default_cache_file()
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
        elif key.startswith(("lin_", "rawlin_")):
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
        - taxa: list. Names of taxa with cached lineages (the terminal node
          of each lineage; None for an empty lineage).
        - ids: list. Taxonomicon IDs of those lineages, in the same order.
        - size_bytes: int. Total memory used by the cache.
    """
    lin_keys = sorted(k for k in _taxodist_cache if k.startswith("lin_"))
    id_keys  = [k for k in _taxodist_cache if k.startswith("id_")]
    ids = [k[len("lin_"):] for k in lin_keys]
    taxa_names = [
        _taxodist_cache[k][-1] if _taxodist_cache[k] else None for k in lin_keys
    ]
    size_bytes = sys.getsizeof(_taxodist_cache)

    print("taxodist Cache")
    print(f"* Lineages cached : {len(lin_keys)}")
    print(f"* IDs cached      : {len(id_keys)}")
    print(f"* Memory used     : {size_bytes} bytes")

    if taxa_names:
        print("\nCached taxa:")
        for name, taxon_id in zip(taxa_names, ids):
            print(f"  {name} (ID {taxon_id})")
    else:
        print("\nNo lineages cached yet.")

    return {
        "n_lineages": len(lin_keys),
        "n_ids":      len(id_keys),
        "taxa":       taxa_names,
        "ids":        ids,
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
    to extract the taxon ID from the hierarchy link. Only candidates whose
    lineage descends from Biota are retained, which excludes astronomical
    and other non-biological homonyms. When several biological candidates
    remain, those whose lineage contains the queried name are preferred and a
    warning lists the alternatives. Leading, trailing, and repeated internal
    whitespace in ``taxon`` is ignored. Transient network failures are
    retried (see ``MAX_TRIES``).
    """
    return _lookup_taxonomicon_id(taxon, verbose=verbose)[0]


def _lookup_taxonomicon_id(taxon, verbose=False):
    """
    Workhorse for get_taxonomicon_id().

    Returns ``(id, status)`` where status is ``"found"``, ``"not_found"``,
    or ``"retrieval_error"``, so that callers such as check_coverage() can
    distinguish an absent name from an unreachable server.
    """
    taxon = _normalize_taxon(taxon)
    cache_key = f"id_{taxon}"
    if cache_key in _taxodist_cache:
        if verbose:
            print(f"Using cached ID for {taxon}")
        return _taxodist_cache[cache_key], "found"

    if verbose:
        print(f"Searching Taxonomicon for {taxon}...")

    unreachable = (
        "Cannot reach The Taxonomicon server.\n"
        "The website (taxonomy.nl) appears to be offline or unreachable.\n"
        "Please try again later."
    )
    try:
        res = _http_get(_search_url(taxon))
    except requests.exceptions.RequestException:
        warnings.warn(unreachable)
        return None, "retrieval_error"
    if res.status_code != 200:
        warnings.warn(unreachable)
        return None, "retrieval_error"

    try:
        soup = BeautifulSoup(res.text, "lxml")
    except Exception:
        warnings.warn("Could not parse the response from The Taxonomicon.")
        return None, "retrieval_error"

    bio_ids = []
    lineage_failed = False
    for candidate in _parse_search_rows(soup):
        candidate_lin = get_lineage_by_id(candidate["id"], clean=True, verbose=False)
        if candidate_lin is None:
            lineage_failed = True
            continue
        if "Biota" not in candidate_lin:
            continue
        bio_ids.append({**candidate, "lineage": candidate_lin})

    if len(bio_ids) > 1:
        # Prefer candidates whose lineage contains the name as a distinct word.
        word_pattern = re.compile(rf"\b{re.escape(taxon)}\b", re.IGNORECASE)
        matched = [
            entry for entry in bio_ids
            if any(word_pattern.search(node) for node in entry["lineage"])
        ]
        if matched:
            bio_ids = matched

    if not bio_ids:
        if lineage_failed:
            return None, "retrieval_error"
        if verbose:
            print(f"{taxon} not found in Taxonomicon")
        return None, "not_found"

    if len(bio_ids) > 1:
        warn_msg = [
            f"Multiple valid biological entries found for '{taxon}'.",
            f"Using: {bio_ids[0]['name']} (ID: {bio_ids[0]['id']})",
            f"To use a different entry, pass its numeric ID directly, e.g. `get_lineage(\"{bio_ids[1]['id']}\")`.",
            "Other available IDs:"
        ]
        for entry in bio_ids[1:]:
            warn_msg.append(f"* ID {entry['id']}: {entry['name']}")
        warnings.warn("\n".join(warn_msg))

    final_id = bio_ids[0]["id"]
    _taxodist_cache[cache_key] = final_id
    if verbose:
        print(f"Found {taxon} with ID {final_id}")

    return final_id, "found"


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

    Cleaned and uncleaned lineages are cached separately, so the value of
    ``clean`` used in an earlier call never affects later calls.
    """
    if isinstance(taxon_id, (list, tuple, set, frozenset)):
        raise TypeError("taxon_id must be a single ID, not a collection of IDs")
    if taxon_id is None:
        return None
    taxon_id = str(taxon_id).strip()
    if not re.fullmatch(r"[0-9]+", taxon_id):
        return None

    clean_key = f"lin_{taxon_id}"
    raw_key = f"rawlin_{taxon_id}"
    cache_key = clean_key if clean else raw_key

    if cache_key in _taxodist_cache:
        if verbose:
            print(f"Using cached lineage for ID {taxon_id}")
        return _taxodist_cache[cache_key]

    url = f"http://taxonomicon.taxonomy.nl/TaxonTree.aspx?id={taxon_id}&src=0"
    try:
        res = _http_get(url)
    except requests.exceptions.RequestException:
        if verbose:
            print(f"Could not retrieve lineage for ID {taxon_id}")
        return None
    if res.status_code != 200:
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
    if not lineage:
        return None

    # One request serves both variants, so cache both.
    cleaned = lineage[lineage.index("Biota"):] if "Biota" in lineage else list(lineage)
    _taxodist_cache[raw_key] = lineage
    _taxodist_cache[clean_key] = cleaned
    return cleaned if clean else lineage

def get_lineage(taxon, clean=True, verbose=False):
    """
    Retrieve the full taxonomic lineage of a taxon by name

    A convenience wrapper that combines get_taxonomicon_id() and
    get_lineage_by_id() into a single call. Given a taxon name, returns
    its complete lineage from root to tip.

    Parameters
    ----------
    taxon : str or int
        A single taxon name or numeric Taxonomicon ID. Leading, trailing, and
        repeated internal whitespace is ignored, and the queried name is
        matched case-insensitively when truncating the lineage.
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
    if isinstance(taxon, (list, tuple, set, frozenset)):
        raise TypeError("taxon must be a single taxon name or ID")
    if taxon is None or (isinstance(taxon, float) and math.isnan(taxon)):
        return None
    taxon_str = _normalize_taxon(taxon)
    if not taxon_str:
        return None
    is_id = bool(re.fullmatch(r"[0-9]+", taxon_str))

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

    if is_id:
        return lineage or None

    lineage = _finalize_lineage(lineage, taxon_str)
    _taxodist_cache[resolved_key] = list(lineage)
    return lineage


def _taxo_search_details(taxon, verbose=False):
    """Return a structured Taxonomicon search result for internal callers."""
    taxon = _normalize_taxon(taxon)
    if verbose:
        print(f"Searching Taxonomicon for '{taxon}'...")

    try:
        res = _http_get(_search_url(taxon))
    except requests.exceptions.RequestException:
        if verbose:
            print("Could not reach Taxonomicon")
        return {"status": "retrieval_error", "results": None}
    if res.status_code != 200:
        if verbose:
            print("Could not reach Taxonomicon")
        return {"status": "retrieval_error", "results": None}

    try:
        soup = BeautifulSoup(res.text, "lxml")
    except Exception:
        if verbose:
            print("Could not parse the Taxonomicon response")
        return {"status": "retrieval_error", "results": None}

    results = _parse_search_rows(soup)
    if not results:
        if verbose:
            print("No matches found.")
        return {"status": "not_found", "results": None}

    frame = pd.DataFrame(results)
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
    for column in ("input", "resolved_name", "id", "status"):
        frame[column] = frame[column].astype("object")
        frame.loc[frame[column].isna(), column] = None
    frame["n_candidates"] = frame["n_candidates"].astype("int64")
    frame["lineage_depth"] = pd.array(frame["lineage_depth"], dtype="Int64")
    frame.source = source
    frame.source_url = source_url
    frame.retrieved_at = retrieved_at or _utc_timestamp()
    return frame


def taxo_resolve(taxa, ambiguity="warn", verbose=False, progress=True, within=None):
    """
    Resolve names or numeric Taxonomicon IDs into an auditable table.

    Parameters
    ----------
    taxa : sequence of str
        Taxon names or numeric Taxonomicon IDs.
    ambiguity : {"warn", "first", "error"}
        How to handle names with several valid biological candidates.
    verbose, progress : bool
        Print retrieval messages / progress.
    within : str or None
        Optional clade name (e.g. ``"Animalia"``). When given, only
        candidates whose lineage contains this clade are considered, which
        resolves many cross-kingdom homonyms automatically. Matching is exact
        and case-insensitive. Names without a candidate inside the clade are
        reported as ``"unresolved"``. Numeric IDs are not filtered.

    Notes
    -----
    Names are matched after trimming and collapsing whitespace, and the stored
    lineage is truncated at the queried taxon exactly as in ``get_lineage()``,
    so ``distance_matrix(taxo_resolve(x))`` and ``distance_matrix(x)`` agree.
    """
    if ambiguity not in {"warn", "first", "error"}:
        raise ValueError("ambiguity must be 'warn', 'first', or 'error'")
    if isinstance(taxa, (str, bytes, Mapping)) or not isinstance(taxa, Iterable):
        raise TypeError("taxa must be a sequence of strings")
    taxa = list(taxa)
    if any(not isinstance(taxon, str) for taxon in taxa):
        raise TypeError("taxa must be a sequence of strings")
    if any(not taxon.strip() for taxon in taxa):
        raise ValueError("taxa cannot contain missing or empty values")
    if within is not None and (not isinstance(within, str) or not within.strip()):
        raise ValueError("within must be None or one non-empty clade name")
    within_clade = None if within is None else _normalize_taxon(within).casefold()

    def failed(taxon, status, candidates=None):
        candidates = _empty_candidates() if candidates is None else candidates
        return {
            "input": taxon,
            "resolved_name": None,
            "id": None,
            "status": status,
            "n_candidates": len(candidates),
            "lineage_depth": None,
            "lineage": None,
            "candidates": candidates,
        }

    def resolve_one(taxon):
        if re.fullmatch(r"[0-9]+", taxon):
            lineage = get_lineage_by_id(taxon, clean=True, verbose=verbose)
            if lineage is None:
                return failed(taxon, "retrieval_error")
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
            return failed(taxon, "retrieval_error")
        candidates = search["results"]
        if search["status"] == "not_found" or candidates is None or candidates.empty:
            return failed(taxon, "unresolved")

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
            return failed(taxon, "retrieval_error", searched_candidates)

        if within_clade is not None:
            inside = [
                within_clade in {node.casefold() for node in lineage}
                for lineage in candidate_lineages
            ]
            if not any(inside):
                return failed(taxon, "unresolved")
            candidates = candidates.loc[inside].reset_index(drop=True)
            candidate_lineages = [
                lineage for lineage, keep in zip(candidate_lineages, inside) if keep
            ]

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

        # Apply the same truncation as get_lineage() so that name-based and
        # resolution-based matrices are identical.
        selected_lineage = _finalize_lineage(list(candidate_lineages[0]), taxon)
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

    keys = [_normalize_taxon(taxon) for taxon in taxa]
    unique_keys = list(dict.fromkeys(keys))
    resolved_unique = {}
    for index, key in enumerate(unique_keys, start=1):
        resolved_unique[key] = resolve_one(key)
        if progress:
            print(f"Resolving taxa [{index}/{len(unique_keys)}]: {key}")

    rows = []
    for taxon, key in zip(taxa, keys):
        row = resolved_unique[key].copy()
        row["input"] = taxon
        rows.append(row)
    result = _make_resolution(
        rows,
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
            f"{details}. Inspect taxo_search(), use `within`, or pass numeric "
            "IDs explicitly."
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
