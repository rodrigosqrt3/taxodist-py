import math
import numbers
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
from bs4.element import NavigableString, Tag, PreformattedString
import numpy as np
import pandas as pd

from ._version import __version__

_taxodist_cache = {}

#: Version of the lineage parser. Saved caches record it so that lineages
#: produced by an older parser are not reused.
_PARSER_VERSION = 2
_PARSER_VERSION_KEY = "meta_parser_version"

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

#: Minimum interval, in seconds, between consecutive requests to The
#: Taxonomicon, so that large batches do not overload the server. Set
#: ``taxodist.fetch.MIN_INTERVAL = 0`` to disable throttling (equivalent of
#: the R option ``taxodist.min_interval``).
MIN_INTERVAL = 0.25

_sleep = time.sleep
_throttle_sleep = time.sleep
_last_request = None

_ASTRONOMICAL_PATTERN = (
    r"\bastronomical\b|\bplanet\b|\bMinor planet\b|\bcomet\b|\basteroid\b"
)


def _max_tries():
    try:
        tries = int(MAX_TRIES)
    except (TypeError, ValueError):
        return 1
    return max(tries, 1)


def taxonomicon_available(timeout=3):
    """
    Check whether The Taxonomicon can be reached.

    Sends a single quick request to The Taxonomicon and reports whether it
    answered. Useful to skip code that needs the server when there is no
    internet connection or the server is down; lineage retrieval itself
    already handles failed requests.

    Parameters
    ----------
    timeout : float
        Maximum time to wait for an answer, in seconds.

    Returns
    -------
    bool
        ``True`` if the server answered with a successful response, ``False``
        otherwise, including when there is no internet connection.

    Examples
    --------
    >>> taxonomicon_available()  # doctest: +SKIP
    True
    """
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, numbers.Real)
        or math.isnan(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a single positive number of seconds.")
    try:
        response = _http_session.get(
            "http://taxonomicon.taxonomy.nl/", timeout=timeout
        )
    except requests.exceptions.RequestException:
        return False
    return response.status_code == 200


def _throttle():
    """Wait until ``MIN_INTERVAL`` seconds have passed since the last request."""
    global _last_request
    interval = MIN_INTERVAL
    if (
        isinstance(interval, bool)
        or not isinstance(interval, numbers.Real)
        or math.isnan(interval)
        or interval <= 0
    ):
        return None
    if _last_request is not None:
        wait = interval - (time.monotonic() - _last_request)
        if wait > 0:
            _throttle_sleep(wait)
    _last_request = time.monotonic()
    return None


def _regex_escape(x):
    """Escape every regular-expression metacharacter in ``x``."""
    return re.escape(x)


def _word_pattern(x):
    """
    Pattern matching ``x`` as a whole word.

    Unicode-aware lookarounds are used instead of ``\\b``, which fails next to
    non-word characters (e.g. a name ending in ")").
    """
    return r"(?<!\w)" + _regex_escape(x) + r"(?!\w)"


def _contains_word(text, word, ignore_case=False):
    flags = re.IGNORECASE if ignore_case else 0
    return re.search(_word_pattern(word), text, flags) is not None


def _candidate_order(descriptions):
    """
    Order of homonymous candidates (0-based indices).

    Entries whose search description marks the name as invalid or rejected
    (e.g. "nom. inval.", "nom. rej.") are moved after the others, keeping the
    search order within each group.
    """
    pattern = re.compile(r"\bnom\.\s*(inval|rej|nud|illeg)\.", re.IGNORECASE)
    invalid = [bool(pattern.search(str(text))) for text in descriptions]
    return sorted(range(len(invalid)), key=lambda i: invalid[i])


def _http_get(url):
    """GET ``url`` with retries; raise the last RequestException on failure."""
    _throttle()
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


def _trimws(text):
    """Trim like R's ``trimws()``: only spaces, tabs, and line breaks."""
    return re.sub(r"^[ \t\r\n]+|[ \t\r\n]+$", "", text)


def _class_attr(tag):
    """The ``class`` attribute of ``tag`` as a single string, or None."""
    value = tag.get("class")
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return " ".join(value)
    return str(value)


def _parse_search_rows(soup):
    """
    Return unique ``{"id", "name"}`` candidates from a TaxonList page.

    Mirrors R's ``.parse_search_rows()``: astronomical homonyms, rows without
    a TaxonTree link whose class is exactly ``Valid``, and links without a
    numeric ID are skipped. The row text is concatenated without separators,
    whitespace is collapsed, and the ``N|T|P|R|B|L`` prefix is removed.
    """
    results = []
    seen = set()
    for row in soup.find_all("tr"):
        text = row.get_text().strip()
        if re.search(_ASTRONOMICAL_PATTERN, text, flags=re.IGNORECASE):
            continue

        links = row.find_all("a", href=re.compile(r"TaxonTree"))
        if not links:
            continue
        valid_links = [a for a in links if _class_attr(a) == "Valid"]
        if not valid_links:
            continue

        match = re.search(r"id=([0-9]+)", valid_links[0].get("href", ""))
        if not match:
            continue

        text_entry = _trimws(re.sub(r"\s+", " ", text))
        text_entry = re.sub(r"^N\|T\|P\|R\|B\|L\s*", "", text_entry, count=1)
        if match.group(1) in seen:
            continue
        seen.add(match.group(1))
        results.append({"id": match.group(1), "name": text_entry})
    return results


def _finalize_lineage(lineage, taxon):
    """
    Truncate a name-resolved lineage at the queried taxon.

    Bracketed labels are dropped, and nodes are filtered with
    ``_keep_lineage_node()``. The taxon is matched exactly first and
    case-insensitively as a fallback; the lineage is truncated at the last
    match, and the taxon is appended only when it is absent from the page
    (e.g. genera whose page links only to their species). Always returns a
    new list.
    """
    keep, append = _finalize_plan(lineage, taxon)
    out = [lineage[i] for i in keep]
    return out + [taxon] if append else out


def _finalize_ranks(ranks, lineage, taxon):
    """
    Apply the truncation of ``_finalize_lineage()`` to a rank list aligned
    with ``lineage``. An appended taxon receives the rank implied by the form
    of its name.
    """
    keep, append = _finalize_plan(lineage, taxon)
    out = [ranks[i] for i in keep]
    return out + [_rank_from_name(taxon)] if append else out


def _finalize_plan(lineage, taxon):
    """
    Indices (0-based) of ``lineage`` retained by ``_finalize_lineage()``, and
    whether the queried taxon must be appended.
    """
    candidates = [i for i, node in enumerate(lineage) if not node.startswith("[")]
    keep = _keep_lineage_node([lineage[i] for i in candidates], taxon)
    candidates = [i for i, kept in zip(candidates, keep) if kept]
    kept_nodes = [lineage[i] for i in candidates]
    matches = [k for k, node in enumerate(kept_nodes) if node == taxon]
    if not matches:
        folded = taxon.lower()
        matches = [k for k, node in enumerate(kept_nodes) if node.lower() == folded]
    if matches:
        return candidates[:matches[-1] + 1], False
    return candidates, True


def _keep_lineage_node(lineage, taxon):
    """
    Nodes without spaces are always kept. A node with spaces (a species,
    subspecies, or subgenus name) is kept only when it is the queried taxon
    or one of its name-ancestors: the species of a queried subspecies, or the
    subgenus ``Genus (Subgenus)`` of a queried species. Other multi-word nodes
    are descendants of the queried taxon or residual text and are dropped.
    """
    plain = [" " not in node for node in lineage]
    if " " not in taxon:
        return plain
    target = taxon.lower()
    genus = re.sub(r" .*$", "", target, flags=re.DOTALL)
    keep = []
    for node, is_plain in zip(lineage, plain):
        lower = node.lower()
        is_target = lower == target
        species_of_target = target.startswith(lower + " ")
        subgenus_of_target = (
            re.fullmatch(r"\S+ \([^()]+\)", node) is not None
            and re.sub(r" .*$", "", lower, flags=re.DOTALL) == genus
        )
        keep.append(is_plain or is_target or species_of_target or subgenus_of_target)
    return keep


def _parse_hierarchy(soup):
    """
    Parse the classification tree of a TaxonTree page.

    Each row of ``#ctl00_cphTabContent_divHierarchy`` is an <a> (an ancestor)
    or a <span> (the page's own taxon), followed by a <br>. The taxon name is
    the text of the row's <b> element. Rows after the page's own taxon are
    its children and are excluded.

    Returns ``(names, ranks, self_missing)``, or None when the page has no
    such hierarchy block (the caller then falls back to text parsing). The
    rank of each row is the text preceding its name (see
    ``_rank_before_name()``). When the page omits the row of its own taxon,
    the leading run of strictly increasing indentation is kept and
    ``self_missing`` is True.
    """
    container = soup.select_one("#ctl00_cphTabContent_divHierarchy")
    if container is None:
        return None

    names = []
    ranks = []
    indents = []
    is_self = []
    indent = 0

    def visit(node):
        nonlocal indent
        for child in node.children:
            if isinstance(child, NavigableString):
                # Comments, CDATA, doctypes, etc. are not text() nodes.
                if isinstance(child, PreformattedString):
                    continue
                indent += str(child).count(" ")
            elif isinstance(child, Tag):
                if child.name == "br":
                    indent = 0
                elif child.name in ("a", "span"):
                    row(child)
                else:
                    visit(child)

    def row(tag):
        if _class_attr(tag) in ("Ordinal", "Small"):
            return
        bold = tag.find("b")
        if bold is None:
            return
        name = re.sub(r"\s+", " ", _trimws(bold.get_text()))
        if not name:
            return
        names.append(name)
        ranks.append(_rank_before_name(tag.get_text(), name))
        indents.append(indent)
        is_self.append(tag.name == "span")

    visit(container)
    if not names:
        return None

    if any(is_self):
        keep = is_self.index(True) + 1
        return names[:keep], ranks[:keep], False

    # Some pages (e.g. unverified taxa) omit the row of the page's own taxon.
    # Keep the leading run of strictly increasing indentation, i.e. the path
    # from the root, and let the caller append the subject name.
    keep = 1
    while keep < len(indents) and indents[keep] > indents[keep - 1]:
        keep += 1
    return names[:keep], ranks[:keep], True


# -- Ranks ---------------------------------------------------------------------

def _rank_before_name(row_text, name):
    """
    Rank of a tree row: the text that precedes the taxon name (e.g.
    "[crown] Clade", "Superfamily ‡?", "Anamorph genus"), without crown,
    plesion, or stem qualifiers, extinction and type markers, quotes, or a
    "Taxon:" caption. Rows without such text are species, subspecies, or
    subgenera, whose rank is implied by the form of the name.
    """
    row_text = re.sub(r"\s+", " ", row_text)
    pos = row_text.find(name)
    prefix = row_text[:pos] if pos > 0 else ""
    return _normalize_rank(prefix, name)


def _normalize_rank(prefix, name):
    prefix = re.sub(r"^.*:", "", prefix, count=1, flags=re.DOTALL)
    prefix = re.sub(r"\[(crown|plesion|stem)\]", "", prefix)
    prefix = re.sub('[†‡ᵀ?"]', "", prefix)
    prefix = _trimws(re.sub(r"\s+", " ", prefix))
    return prefix if prefix else _rank_from_name(name)


def _rank_from_name(name):
    """
    Rank implied by the form of a scientific name: "Genus (Subgenus)" is a
    subgenus, a binomial a species, and a trinomial a subspecies. Uninomials
    carry no rank information (None).
    """
    if re.fullmatch(r"\S+ \([^()]+\)", name):
        return "Subgenus"
    words = len(_trimws(name).split())
    if words == 2:
        return "Species"
    if words == 3:
        return "Subspecies"
    return None


def _lineage_from_page(soup, taxon_id):
    """
    Extract the raw lineage (before trimming at Biota) of a TaxonTree page.

    Uses the structured hierarchy block when present and falls back to text
    parsing otherwise. Repeated names are collapsed to their first occurrence.
    """
    return _lineage_and_ranks_from_page(soup, taxon_id)[0]


def _lineage_and_ranks_from_page(soup, taxon_id):
    """
    Like ``_lineage_from_page()``, but returns ``(lineage, ranks)`` where
    ``ranks`` holds the rank of every node (None when it has none). This is
    the equivalent of the "ranks" attribute of R's ``.lineage_from_page()``.
    """
    subject = soup.select_one("#ctl00_divSubject b")
    current_name = None
    if subject is not None:
        current_name = _trimws(re.sub("[†‡ᵀ?]", "", subject.get_text().strip()))

    structured = _parse_hierarchy(soup)
    if structured is not None:
        lineage, ranks, self_missing = structured
        if self_missing and current_name and lineage[-1] != current_name:
            subject_text = soup.select_one("#ctl00_divSubject").get_text()
            lineage = lineage + [current_name]
            ranks = ranks + [_rank_before_name(subject_text, current_name)]
    else:
        lineage, ranks = _parse_lineage_text(soup, current_name, taxon_id)
    seen = set()
    out_lineage, out_ranks = [], []
    for node, rank in zip(lineage, ranks):
        if node in seen:
            continue
        seen.add(node)
        out_lineage.append(node)
        out_ranks.append(rank)
    return out_lineage, out_ranks


def _parse_lineage_text(soup, current_name, taxon_id):
    """
    Fallback parser for pages without the structured hierarchy block.

    The tree is read as plain text, one node per line, and rank prefixes,
    markers, and author strings are removed with regular expressions.
    Returns ``(lineage, ranks)``.
    """
    content = soup.select_one("#divPageContent")
    if current_name is not None and content is not None:
        tree_text = content.get_text().strip()
        raw_lines = [_trimws(line) for line in tree_text.split("\n")]
        raw_lines = [line for line in raw_lines if line]
        tree_start = next(
            (i for i, line in enumerate(raw_lines) if line.startswith("Natura")),
            None,
        )
        if tree_start is not None:
            raw_lines = raw_lines[tree_start:]
        pattern = re.compile(_word_pattern(current_name))
        cutoff = next(
            (
                i for i, line in enumerate(raw_lines)
                if pattern.search(re.sub("[†ᵀ]", "", line))
            ),
            None,
        )
        texts = raw_lines[:cutoff + 1] if cutoff is not None else raw_lines
    else:
        links = soup.find_all("a", href=re.compile(r"TaxonTree"))
        links = [a for a in links if re.search(r"id=[0-9]", a.get("href", ""))]
        own_pattern = re.compile(f"id={re.escape(str(taxon_id))}($|&)")
        own_idx = [
            i for i, a in enumerate(links) if own_pattern.search(a.get("href", ""))
        ]
        if own_idx:
            links = links[:max(own_idx) + 1]
        texts = [a.get_text().strip() for a in links]

    lineage, ranks = [], []
    for text in texts:
        node = _clean_lineage_label(text)
        if (
            node is None
            or node in _BARE_RANKS
            or node == ""
            or re.match(r"^\s*$", node)
            or node.startswith('"')
            or node.startswith("Population")
        ):
            continue
        rank = _text_rank(text)
        lineage.append(node)
        ranks.append(rank if rank is not None else _rank_from_name(node))
    return lineage, ranks


def _text_rank(raw_text):
    """Rank label at the start of a plain-text tree line, or None."""
    marked = re.sub("[†‡ᵀ]", "", str(raw_text))
    marked = re.sub(r"^\?\s*", "", marked, count=1)
    match = _TEXT_RANK_PATTERN.match(marked)
    return match.group(1) if match else None


_RANK_ALTERNATIVES = "|".join([
    "Superdomain", "Domain", "Superkingdom", "Kingdom", "Subkingdom",
    "Infrakingdom", "Superphylum", "Phylum", "Subphylum", "Infraphylum",
    "Superclass", "Class", "Subclass", "Infraclass", "Supercohort", "Cohort",
    "Subcohort", "Magnorder", "Grandorder", "Superorder", "Order", "Suborder",
    "Infraorder", "Parvorder", "Legion", "Superfamily", "Family", "Subfamily",
    "Tribe", "Subtribe", "Genus", "Species", "Subspecies", "Clade",
    "Supergrade", "Grade", "Subgrade",
])

_TEXT_RANK_PATTERN = re.compile(
    r"^(?:\[(?:crown|plesion|stem)\]\s+)?"
    r"(Anamorph\s+\w+|" + _RANK_ALTERNATIVES + r")\s+"
)

# Lower-case words that start an author string rather than continue a
# scientific name (e.g. "de Blainville", "van der Hoeven", "R.Br. ex").
_AUTHOR_PARTICLES = [
    "de", "del", "della", "des", "di", "do", "dos", "da", "das", "du", "d",
    "van", "von", "der", "den", "ter", "zu", "zur", "la", "le", "les", "ex",
    "in", "et", "al", "auct", "sensu", "nom", "non", "emend", "corrig", "ms",
    "pro", "sp", "spp", "var", "nec", "fide",
]

_SCIENTIFIC_NAME_PATTERN = re.compile(
    r"^[^\s\",\[(]+"
    r"(?:\s(?!(?:" + "|".join(_AUTHOR_PARTICLES) + r")(?![\w'’-]))"
    r"[a-zà-ÿ][\w-]*(?![\w.'’]))"
    r"{0,2}"
)

_BARE_RANKS = {
    "Go to", "Superphylum", "Subfamily", "Suborder", "Epifamily",
    "Infraorder", "Superclass", "Subclass", "Superfamily", "Subgenus",
    "Section", "Division", "Candidatus", "Parvphylum", "Branch",
    "Supercohort", "Infracohort", "Subdivision", "Subsection", "Grade",
    "[unranked]", "(Supercluster)", "(Region)", "[crown]", ""
}


def _extract_scientific_name(x):
    """
    Keep only the scientific name at the start of a cleaned tree line.

    The first word (a uninomial, possibly hyphenated) is kept, followed by up
    to two lower-case epithets for species and subspecies. Author names,
    years, vernacular names and annotations that follow are discarded.
    Quoted and bracketed lines are returned unchanged so the caller's filters
    can recognise them. ``None`` is returned unchanged; a list or tuple is
    processed element-wise.
    """
    if isinstance(x, (list, tuple)):
        return [_extract_scientific_name(item) for item in x]
    if x is None or re.match(r"^[\"\[(]", x):
        return x
    match = _SCIENTIFIC_NAME_PATTERN.match(x)
    return match.group(0) if match else x


def _clean_lineage_label(raw_text):
    """Apply the same lineage-label cleaning sequence as taxodist for R."""
    if raw_text is None:
        return None
    text = re.sub("[†‡ᵀ]", "", str(raw_text))
    text = re.sub(r"^\?\s*", "", text, count=1)
    text = re.sub(
        r"^\[(crown|plesion|stem)\]\s+(?:(?:" + _RANK_ALTERNATIVES + r")\s+)?",
        "",
        text,
        count=1,
    )
    text = re.sub(
        r"^(?:Anamorph\s+\w+|" + _RANK_ALTERNATIVES + r")\s+", "", text, count=1
    )
    text = re.sub("[†‡?]", "", text)
    text = text.strip()
    return _extract_scientific_name(text)


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
    data = dict(_taxodist_cache)
    data[_PARSER_VERSION_KEY] = _PARSER_VERSION
    with open(file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cache saved to '{file}' ({len(data)} entries).")
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

    # Lineages saved by an older parser may contain malformed nodes. Taxon IDs
    # remain valid, so only those are restored from such a cache.
    data = dict(data)
    parser_version = data.pop(_PARSER_VERSION_KEY, None)
    current = (
        isinstance(parser_version, numbers.Real)
        and not isinstance(parser_version, bool)
        and not math.isnan(parser_version)
        and parser_version >= _PARSER_VERSION
    )
    if not current:
        kept = {key: value for key, value in data.items() if key.startswith("id_")}
        n_dropped = len(data) - len(kept)
        data = kept
        if n_dropped > 0:
            entries = "entry was" if n_dropped == 1 else "entries were"
            warnings.warn(
                "The cache was written by an older version of taxodist. "
                f"{n_dropped} lineage {entries} discarded and will be retrieved "
                "again; taxon IDs were kept.",
                UserWarning,
                stacklevel=2,
            )

    # Store lineages as fresh lists so the loaded object is not shared.
    data = {
        key: list(value) if isinstance(value, (list, tuple)) else value
        for key, value in data.items()
    }
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
        matched = [
            entry for entry in bio_ids
            if any(
                _contains_word(node, taxon, ignore_case=True)
                for node in entry["lineage"]
            )
        ]
        if matched:
            bio_ids = matched
        preferred = _candidate_order([entry["name"] for entry in bio_ids])
        bio_ids = [bio_ids[i] for i in preferred]

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
    taxon_id = _id_string(taxon_id, "taxon_id must be a single ID, not a collection of IDs")
    if taxon_id is None:
        return None
    taxon_id = taxon_id.strip()
    if not re.fullmatch(r"[0-9]+", taxon_id):
        return None

    clean_key = f"lin_{taxon_id}"
    raw_key = f"rawlin_{taxon_id}"
    cache_key = clean_key if clean else raw_key

    if cache_key in _taxodist_cache:
        if verbose:
            print(f"Using cached lineage for ID {taxon_id}")
        # Return a copy so that callers cannot modify the cached lineage.
        return list(_taxodist_cache[cache_key])

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

    lineage, ranks = _lineage_and_ranks_from_page(soup, taxon_id)
    if not lineage:
        return None

    # One request serves both variants, so cache both (as private copies),
    # with their ranks.
    start = lineage.index("Biota") if "Biota" in lineage else 0
    cleaned = lineage[start:]
    _taxodist_cache[raw_key] = list(lineage)
    _taxodist_cache[clean_key] = list(cleaned)
    _taxodist_cache[f"rawranks_{taxon_id}"] = list(ranks)
    _taxodist_cache[f"ranks_{taxon_id}"] = list(ranks[start:])
    return cleaned if clean else lineage


def _format_taxon_id(taxon_id):
    """Taxon ID as the string used in cache keys."""
    value = _id_string(taxon_id, "taxon_id must be a single ID, not a collection of IDs")
    return "" if value is None else value.strip()


def _ranks_by_id(taxon_id, clean=True, verbose=False):
    """
    Ranks aligned with ``get_lineage_by_id(taxon_id, clean)``. Lineages cached
    without ranks (e.g. restored from a cache file of an older version) are
    retrieved again.
    """
    lineage = get_lineage_by_id(taxon_id, clean=clean, verbose=verbose)
    if lineage is None:
        return None
    taxon_id = _format_taxon_id(taxon_id)
    key = f"{'ranks_' if clean else 'rawranks_'}{taxon_id}"
    if key not in _taxodist_cache:
        _taxodist_cache.pop(f"lin_{taxon_id}", None)
        _taxodist_cache.pop(f"rawlin_{taxon_id}", None)
        lineage = get_lineage_by_id(taxon_id, clean=clean, verbose=verbose)
        if lineage is None or key not in _taxodist_cache:
            return None
    ranks = _taxodist_cache[key]
    if len(ranks) != len(lineage):
        return [None] * len(lineage)
    return list(ranks)


def _cached_ranks(taxon_id, lineage):
    """
    Cached ranks of a cleaned lineage, without any network request. Returns
    None ranks when none are cached or they do not align with ``lineage``.
    """
    ranks = _taxodist_cache.get(f"ranks_{_format_taxon_id(taxon_id)}")
    if ranks is not None and len(ranks) == len(lineage):
        return list(ranks)
    return [None] * len(lineage)


def _id_string(value, collection_message):
    """
    Convert a scalar taxon or ID to a string as R's ``as.character()`` does.

    Integral numbers (including integral floats and NumPy scalars) are
    formatted as integers, so ``50841.0`` becomes ``"50841"``. Missing values
    return None; collections raise TypeError.
    """
    if isinstance(value, (list, tuple, set, frozenset, dict, np.ndarray, pd.Series)):
        raise TypeError(collection_message)
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return str(value)
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        value = float(value)
        if math.isnan(value):
            return None
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return repr(value)
    return str(value)

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
        matched case-insensitively when truncating the lineage. Integral
        numbers (e.g. ``50841`` or ``50841.0``) are treated as IDs.
    clean : bool
        Logical. If True (default), removes philosophical root nodes
        and cleans formatting markers.
    verbose : bool
        Logical. If True, prints progress messages. Default False.

    Returns
    -------
    list or None
        A character vector of clade names ordered from root to tip, or
        None if the taxon cannot be found. The list is a new object; changing
        it does not affect the cache.
    """
    taxon = _id_string(taxon, "taxon must be a single taxon name or ID")
    if taxon is None:
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


def lineage_ranks(taxon, clean=True, verbose=False):
    """
    Retrieve the ranks of the nodes in a lineage

    Returns each node of the lineage of a taxon together with its rank in The
    Taxonomicon classification (for example "Kingdom", "Clade",
    "Superfamily", or "Genus"). The nodes are exactly those returned by
    get_lineage() for the same arguments.

    Parameters
    ----------
    taxon : str or int
        A single taxon name or numeric Taxonomicon ID (as in get_lineage()).
    clean : bool
        If True (default), removes philosophical root nodes above Biota.
    verbose : bool
        If True, prints progress messages. Default False.

    Returns
    -------
    pandas.DataFrame or None
        Columns ``node``, ``rank``, and ``depth`` (1-based), one row per
        lineage node from root to tip, or None if the taxon cannot be found.
        ``rank`` is None for nodes without a stated rank (such as the
        philosophical root nodes above Biota).

    Notes
    -----
    Ranks are read from the label that precedes each name in the
    classification, without the ``[crown]``, ``[plesion]``, and ``[stem]``
    qualifiers or extinction markers. Rows without a label are species,
    subspecies, or subgenera, whose rank is inferred from the form of the
    name. Many nodes are unranked clades ("Clade"). Ranks are used by the
    ``"ranked"`` distance metric; see taxo_distance().
    """
    taxon = _id_string(taxon, "taxon must be a single taxon name or ID")
    if taxon is None:
        return None
    taxon_str = _normalize_taxon(taxon)
    if not taxon_str:
        return None
    is_id = bool(re.fullmatch(r"[0-9]+", taxon_str))
    id_val = taxon_str if is_id else get_taxonomicon_id(taxon_str, verbose=verbose)
    if id_val is None:
        return None

    ranks = _ranks_by_id(id_val, clean=clean, verbose=verbose)
    lineage = get_lineage_by_id(id_val, clean=clean, verbose=verbose)
    if ranks is None or lineage is None:
        return None
    if not is_id:
        ranks = _finalize_ranks(ranks, lineage, taxon_str)
        lineage = _finalize_lineage(lineage, taxon_str)
    return pd.DataFrame(
        {
            "node": pd.Series(lineage, dtype="object"),
            "rank": pd.Series(ranks, dtype="object"),
            "depth": np.arange(1, len(lineage) + 1, dtype=np.int64),
        }
    )


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

    def __finalize__(self, other, method=None, **kwargs):
        result = super().__finalize__(other, method=method, **kwargs)
        if method == "concat":
            # pandas does not propagate _metadata through pd.concat(). Keep
            # the provenance only when every concatenated resolution agrees
            # on it; otherwise leave it unset so that taxo_bundle() cannot
            # silently mislabel the combined table.
            objs = list(getattr(other, "objs", None) or [])
            if objs and all(isinstance(obj, TaxodistResolution) for obj in objs):
                for name in ("source", "source_url"):
                    values = [getattr(obj, name, None) for obj in objs]
                    same = all(value == values[0] for value in values)
                    object.__setattr__(result, name, values[0] if same else None)
                times = [getattr(obj, "retrieved_at", None) for obj in objs]
                earliest = None if any(t is None for t in times) else min(times)
                object.__setattr__(result, "retrieved_at", earliest)
        return result

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


def _make_resolution(rows, source, source_url, retrieved_at=None, ranks=True):
    """
    Build a TaxodistResolution from row dictionaries. With ``ranks=True`` the
    optional ``ranks`` column (lineage ranks, aligned with ``lineage``) is
    included; rows without a ``ranks`` entry receive None.
    """
    columns = RESOLUTION_COLUMNS + (["ranks"] if ranks else [])
    rows = list(rows)
    frame = TaxodistResolution(rows, columns=columns)
    if ranks:
        # Lists must be assigned element-wise to keep one list per cell.
        frame["ranks"] = pd.Series(
            [
                None if row.get("ranks") is None else list(row["ranks"])
                for row in rows
            ],
            index=frame.index,
            dtype="object",
        )
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

    Returns
    -------
    TaxodistResolution
        One row per input with columns input, resolved_name, id, status,
        n_candidates, lineage_depth, lineage, candidates, and ranks (the rank
        of every lineage node, aligned with ``lineage``; None for unresolved
        inputs and for nodes without a rank).

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
            "ranks": None,
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
                "ranks": _cached_ranks(taxon, lineage),
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
        retrieved = [lineage is not None for lineage in candidate_lineages]
        valid = [
            lineage is not None and "Biota" in lineage
            for lineage in candidate_lineages
        ]
        candidates = candidates.loc[valid].reset_index(drop=True)
        candidate_lineages = [
            lineage for lineage, keep in zip(candidate_lineages, valid) if keep
        ]
        if candidates.empty:
            # Only a failed lineage retrieval is a retrieval error; candidates
            # that were retrieved but are not biological (e.g. astronomical
            # objects) mean the name is simply not a taxon.
            # As for names outside `within`, an unresolved record carries no
            # candidates (bundle validation requires this).
            if all(retrieved):
                return failed(taxon, "unresolved")
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
            exact = [
                any(_contains_word(node, taxon, ignore_case=True) for node in lineage)
                for lineage in candidate_lineages
            ]
            if any(exact):
                candidates = candidates.loc[exact].reset_index(drop=True)
                candidate_lineages = [
                    lineage
                    for lineage, keep in zip(candidate_lineages, exact)
                    if keep
                ]
            preferred = _candidate_order(list(candidates["name"]))
            candidates = candidates.iloc[preferred].reset_index(drop=True)
            candidate_lineages = [candidate_lineages[i] for i in preferred]

        # Apply the same truncation as get_lineage() so that name-based and
        # resolution-based matrices are identical.
        selected_lineage = _finalize_lineage(list(candidate_lineages[0]), taxon)
        selected_ranks = _finalize_ranks(
            _cached_ranks(candidates.iloc[0]["id"], candidate_lineages[0]),
            candidate_lineages[0],
            taxon,
        )
        return {
            "input": taxon,
            "resolved_name": selected_lineage[-1],
            "id": str(candidates.iloc[0]["id"]),
            "status": "ambiguous" if len(candidates) > 1 else "resolved",
            "n_candidates": len(candidates),
            "lineage_depth": len(selected_lineage),
            "lineage": selected_lineage,
            "candidates": candidates.copy(),
            "ranks": selected_ranks,
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


def taxo_from_lineages(lineages, ids=None, source="user-supplied", ranks=None):
    """
    Create an auditable offline resolution from root-to-tip lineages.

    Parameters
    ----------
    lineages : dict
        Mapping from input label to the lineage (a sequence of node names
        ordered from root to the focal taxon).
    ids : sequence or dict of str, optional
        One unique identifier per lineage (matched by label when a dict).
        Defaults to ``"custom:<label>"``.
    source : str
        Description of the lineage source.
    ranks : dict or sequence, optional
        The rank of every node of every lineage (None for unranked nodes), as
        a dict keyed by label or a sequence matched by position. Ranks are
        required only by the ``"ranked"`` distance metric. When omitted,
        every rank is None.
    """
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

    if ranks is None:
        rank_lists = [[None] * len(lineages[label]) for label in labels]
    else:
        if isinstance(ranks, Mapping):
            if not all(label in ranks for label in labels):
                raise ValueError("Named ranks must contain every lineage name")
            rank_lists = [ranks[label] for label in labels]
        elif isinstance(ranks, (list, tuple)):
            rank_lists = list(ranks)
        else:
            raise TypeError(
                "ranks must be a dictionary or list of rank sequences"
            )

        def valid(entry, lineage):
            return (
                isinstance(entry, (list, tuple))
                and len(entry) == len(lineage)
                and all(rank is None or isinstance(rank, str) for rank in entry)
            )

        if len(rank_lists) != len(labels) or not all(
            valid(entry, lineages[label]) for entry, label in zip(rank_lists, labels)
        ):
            raise ValueError(
                "ranks must give one rank (or None) per node of every lineage"
            )
        rank_lists = [list(entry) for entry in rank_lists]

    rows = []
    for label, identifier, node_ranks in zip(labels, identifiers, rank_lists):
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
                "ranks": node_ranks,
            }
        )
    return _make_resolution(rows, source=source, source_url=None)
