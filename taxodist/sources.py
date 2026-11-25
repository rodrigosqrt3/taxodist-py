"""
Taxonomic sources other than The Taxonomicon.

Every source resolves names to records with the same structure as
taxo_resolve() uses for The Taxonomicon, so the resulting
``TaxodistResolution`` works with every matrix, tree, bundle, and
distinctness function. Identifiers carry a source prefix (``"ott:770309"``,
``"ncbi:9606"``) so they are never confused with Taxonomicon IDs.
"""

import itertools
import json
import math
import numbers
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
import requests

from . import fetch as _fetch

_SOURCE_IDS = ("taxonomicon", "ott", "ncbi")

_OTT_API = "https://api.opentreeoflife.org/v3"
_NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_OTT_VERSION_URL = "https://tree.opentreeoflife.org/about/taxonomy-version"

#: NCBI E-utilities API key (equivalent of the R option
#: ``taxodist.ncbi_api_key``). When None, the ``NCBI_API_KEY`` environment
#: variable is used. With a key, NCBI allows ten requests per second instead
#: of three.
NCBI_API_KEY = None


def taxo_sources():
    """
    List the taxonomic sources supported by taxodist

    Describes the classifications from which taxo_resolve() can retrieve
    lineages, with their licence and preferred citation. Distances are only
    comparable within one source; use compare_sources() to compare sources.

    Returns
    -------
    pandas.DataFrame
        One row per source and columns ``id`` (the value accepted by the
        ``source`` argument of taxo_resolve()), ``name``, ``url``,
        ``id_prefix`` (prefix of the identifiers stored in resolutions),
        ``ranks`` (whether unranked clades are included), ``licence``, and
        ``citation``.

    Notes
    -----
    - ``"taxonomicon"``: The Taxonomicon (Brands 1989-present), a single
      curated classification with very deep lineages, many unranked clades,
      and extensive fossil coverage. Retrieved from its web pages.
    - ``"ott"``: the Open Tree of Life reference taxonomy (Rees and Cranston
      2017), a synthesis of NCBI, GBIF, WoRMS, IRMNG, and other taxonomies,
      retrieved from the Open Tree of Life API. Its version is recorded in
      the resolution.
    - ``"ncbi"``: the NCBI Taxonomy database (Schoch et al. 2020), covering
      organisms with sequence data, retrieved from the NCBI E-utilities.
      Requests are spaced by at least 0.4 s (NCBI allows three per
      second), or 0.12 s (ten per second) with an API key set
      in the ``NCBI_API_KEY`` environment variable or
      ``taxodist.sources.NCBI_API_KEY``.

    Lineages are ordered from the root of each source (``"Biota"``,
    ``"life"``, and ``"cellular organisms"`` respectively) to the taxon.
    Clade names differ between sources: for example, animals are
    ``"Animalia"`` in The Taxonomicon but ``"Metazoa"`` in the other two,
    which matters for the ``within`` argument of taxo_resolve().
    """
    return pd.DataFrame({
        "id": list(_SOURCE_IDS),
        "name": ["The Taxonomicon", "Open Tree Taxonomy", "NCBI Taxonomy"],
        "url": [
            "http://taxonomicon.taxonomy.nl",
            _OTT_VERSION_URL,
            "https://www.ncbi.nlm.nih.gov/taxonomy",
        ],
        "id_prefix": ["", "ott:", "ncbi:"],
        "ranks": ["ranked and unranked clades"] * 3,
        "licence": [
            "All rights reserved; cite the source",
            "CC0 (Open Tree of Life); underlying sources keep their terms",
            "Public domain (US government work)",
        ],
        "citation": [
            "Brands, S.J. (ed.) 1989-present. The Taxonomicon. Universal "
            "Taxonomic Services, Zwaag, The Netherlands.",
            "Rees, J.A. & Cranston, K. (2017). Automated assembly of a "
            "reference taxonomy for phylogenetic data synthesis. Biodiversity "
            "Data Journal 5: e12581.",
            "Schoch, C.L. et al. (2020). NCBI Taxonomy: a comprehensive update "
            "on curation, resources and tools. Database 2020: baaa062.",
        ],
    })


def _source_choices():
    return f"{_SOURCE_IDS[0]!r}, {_SOURCE_IDS[1]!r}, or {_SOURCE_IDS[2]!r}"


def _match_source(source):
    """Validate a source identifier and return it."""
    if not isinstance(source, str):
        raise TypeError(f"source must be a single string, one of {_source_choices()}.")
    if source not in _SOURCE_IDS:
        raise ValueError(
            f"Unknown taxonomic source {source!r}. "
            f"Use one of {_source_choices()}; see taxo_sources()."
        )
    return source


# -- HTTP ----------------------------------------------------------------------

def _source_interval(source):
    """
    Minimum interval between requests to each source. NCBI allows at most
    three requests per second without an API key and ten with one.
    """
    interval = _fetch.MIN_INTERVAL
    if (
        isinstance(interval, bool)
        or not isinstance(interval, numbers.Real)
        or math.isnan(interval)
        or interval < 0
    ):
        interval = 0
    if source == "ncbi":
        # Slightly above NCBI's nominal limits, which it enforces strictly.
        floor = 0.12 if _ncbi_api_key() else 0.4
        interval = max(interval, floor)
    return interval


def _ncbi_api_key():
    key = NCBI_API_KEY
    if key is None:
        key = os.environ.get("NCBI_API_KEY", "")
    return key if isinstance(key, str) else ""


def _source_request(source, method, url, body=None):
    """
    Perform one request and return ``{"status": int, "text": str}``. The
    status is None (and the text None) when the request failed without a
    response. Network errors, server errors (5xx), and refused requests (HTTP
    429) are retried after a pause; other client errors (4xx) are not.
    """
    _fetch._throttle(source, _source_interval(source))
    kwargs = {"timeout": 30}
    if body is not None:
        kwargs["data"] = json.dumps(body).encode("utf-8")
        kwargs["headers"] = {"Content-Type": "application/json"}
    tries = _fetch._max_tries()
    response = None
    for attempt in range(1, tries + 1):
        try:
            response = _fetch._http_session.request(method, url, **kwargs)
        except requests.exceptions.RequestException:
            response = None
        else:
            # Refused requests (HTTP 429) are retried like server errors;
            # other client errors are final.
            if response.status_code < 500 and response.status_code != 429:
                break
        if attempt < tries:
            _fetch._sleep(min(2 ** (attempt - 1), 5))
    if response is None:
        return {"status": None, "text": None}
    return {
        "status": int(response.status_code),
        "text": response.content.decode("utf-8", errors="replace"),
    }


def _parse_json(text):
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _response_json(res):
    return _parse_json(res["text"]) if res["status"] == 200 else None


def _normalize_source_rank(rank):
    """
    Rank label as used by The Taxonomicon: capitalised, with unranked nodes
    (other than explicit clades) as None. Accepts one rank or a list.
    """
    if isinstance(rank, (list, tuple)):
        return [_normalize_source_rank(value) for value in rank]
    if rank is None:
        return None
    rank = _fetch._trimws(str(rank).lower())
    if rank in ("", "no rank", "no rank - terminal", "cellular root", "acellular root"):
        return None
    return rank[0].upper() + rank[1:]


def _cache():
    return _fetch._taxodist_cache


# -- Open Tree Taxonomy --------------------------------------------------------

def _ott_about():
    """
    Name and URL of the Open Tree Taxonomy served by the API, e.g.
    "Open Tree Taxonomy 3.7 (ott3.7draft3)".
    """
    cached = _cache().get("src_ott_about")
    if cached is not None:
        return dict(cached)
    res = _source_request("ott", "POST", f"{_OTT_API}/taxonomy/about", body={})
    data = _response_json(res)
    if not isinstance(data, dict):
        return {"name": "Open Tree Taxonomy", "url": _OTT_VERSION_URL}
    about = {
        "name": f"Open Tree Taxonomy {data.get('version')} ({data.get('source')})",
        "url": data.get("weburl") or _OTT_VERSION_URL,
    }
    _cache()["src_ott_about"] = dict(about)
    return about


def _ott_match_names(names, verbose=False):
    """
    Candidates for each name: a list of ``{"id", "name", "synonym"}``
    dictionaries, an empty list when the name is unknown, or None when the
    request failed. Names are sent in batches of 250.
    """
    out = {name: None for name in names}
    todo = []
    for name in names:
        cached = _cache().get(f"src_ott_match_{name.lower()}")
        if cached is None:
            todo.append(name)
        else:
            out[name] = [dict(candidate) for candidate in cached]
    for start in range(0, len(todo), 250):
        batch = todo[start:start + 250]
        if verbose:
            plural = "" if len(batch) == 1 else "s"
            print(f"Matching {len(batch)} name{plural} in the Open Tree Taxonomy...")
        res = _source_request(
            "ott", "POST", f"{_OTT_API}/tnrs/match_names",
            body={"names": batch, "do_approximate_matching": False},
        )
        data = _response_json(res)
        if not isinstance(data, dict):
            continue
        by_query = {}
        for result in data.get("results") or []:
            by_query[str(result.get("name")).lower()] = result.get("matches") or []
        for name in batch:
            candidates = [
                {
                    "id": f"ott:{match['taxon']['ott_id']}",
                    "name": match["taxon"]["name"],
                    "synonym": match.get("is_synonym") is True,
                }
                for match in by_query.get(name.lower(), [])
            ]
            _cache()[f"src_ott_match_{name.lower()}"] = [dict(c) for c in candidates]
            out[name] = candidates
    return out


def _ott_lineage(identifier, verbose=False):
    """
    Lineage of an OTT taxon from the root ("life") to the taxon, as
    ``{"lineage": [...], "ranks": [...]}``; None when it cannot be retrieved.
    """
    number = re.sub(r"^ott:", "", identifier)
    key = f"src_ott_lin_{number}"
    cached = _cache().get(key)
    if cached is not None:
        return _copy_record(cached)
    if verbose:
        print(f"Retrieving lineage of ott:{number}...")
    res = _source_request(
        "ott", "POST", f"{_OTT_API}/taxonomy/taxon_info",
        body={"ott_id": int(number), "include_lineage": True},
    )
    data = _response_json(res)
    if not isinstance(data, dict) or data.get("name") is None:
        return None
    ancestors = list(reversed(data.get("lineage") or []))
    record = {
        "lineage": [node["name"] for node in ancestors] + [data["name"]],
        "ranks": _normalize_source_rank(
            [node.get("rank") for node in ancestors] + [data.get("rank")]
        ),
    }
    _cache()[key] = _copy_record(record)
    return record


def _copy_record(record):
    return {"lineage": list(record["lineage"]), "ranks": list(record["ranks"])}


# -- NCBI Taxonomy -------------------------------------------------------------

def _ncbi_url(endpoint, query):
    """E-utilities URL with ``tool=taxodist`` and the API key, if any."""
    query = dict(query)
    query["tool"] = "taxodist"
    key = _ncbi_api_key()
    if key:
        query["api_key"] = key
    encoded = "&".join(
        f"{name}={urllib.parse.quote(str(value), safe='')}"
        for name, value in query.items()
    )
    return f"{_NCBI_EUTILS}/{endpoint}?{encoded}"


def _ncbi_search(name, verbose=False):
    """
    NCBI identifiers matching a name: by scientific name, then by any name
    (synonyms, common names). An empty list when unknown; None on failure.
    """
    key = f"src_ncbi_search_{name.lower()}"
    if key in _cache():
        return list(_cache()[key])
    if verbose:
        print(f"Searching NCBI Taxonomy for {name}...")
    for field in ("Scientific Name", "All Names"):
        term = '"' + name.replace('"', "") + f'"[{field}]'
        res = _source_request(
            "ncbi", "GET",
            _ncbi_url("esearch.fcgi", {
                "db": "taxonomy", "term": term, "retmode": "json", "retmax": "50",
            }),
        )
        data = _response_json(res)
        if not isinstance(data, dict) or data.get("esearchresult") is None:
            return None
        ids = list(data["esearchresult"].get("idlist") or [])
        if ids:
            ids = [f"ncbi:{identifier}" for identifier in ids]
            _cache()[key] = list(ids)
            return ids
    _cache()[key] = []
    return []


def _ncbi_fetch(ids, verbose=False):
    """Retrieve and cache the lineages of NCBI taxa, in batches of 200."""
    numbers_ = list(dict.fromkeys(re.sub(r"^ncbi:", "", i) for i in ids))
    todo = [n for n in numbers_ if f"src_ncbi_lin_{n}" not in _cache()]
    for start in range(0, len(todo), 200):
        batch = todo[start:start + 200]
        if verbose:
            plural = "" if len(batch) == 1 else "s"
            print(f"Retrieving {len(batch)} lineage{plural} from NCBI Taxonomy...")
        res = _source_request(
            "ncbi", "GET",
            _ncbi_url("efetch.fcgi", {
                "db": "taxonomy", "id": ",".join(batch), "retmode": "xml",
            }),
        )
        if res["status"] != 200:
            continue
        _ncbi_parse_taxa(res["text"])


def _xml_text(element, path):
    node = element.find(path)
    return None if node is None else (node.text or "")


def _ncbi_parse_taxa(text):
    """Parse an efetch TaxaSet and cache the lineage of every taxon in it."""
    try:
        root = ET.fromstring(text)
    except (ET.ParseError, TypeError):
        return None
    if root.tag != "TaxaSet":
        return None
    for taxon in root.findall("./Taxon"):
        identifier = _xml_text(taxon, "./TaxId")
        ancestors = taxon.findall("./LineageEx/Taxon")
        record = {
            "lineage": [_xml_text(node, "./ScientificName") for node in ancestors]
            + [_xml_text(taxon, "./ScientificName")],
            "ranks": _normalize_source_rank(
                [_xml_text(node, "./Rank") for node in ancestors]
                + [_xml_text(taxon, "./Rank")]
            ),
        }
        _cache()[f"src_ncbi_lin_{identifier}"] = _copy_record(record)
        # A merged identifier answers with the taxon it was merged into.
        for old in taxon.findall("./AkaTaxIds/TaxId"):
            _cache()[f"src_ncbi_lin_{old.text or ''}"] = _copy_record(record)
    return None


def _ncbi_lineage(identifier, verbose=False):
    number = re.sub(r"^ncbi:", "", identifier)
    key = f"src_ncbi_lin_{number}"
    if key not in _cache():
        _ncbi_fetch([number], verbose=verbose)
    cached = _cache().get(key)
    return None if cached is None else _copy_record(cached)


# -- Resolution ----------------------------------------------------------------

def _source_lineage(source, identifier, verbose=False):
    if source == "ott":
        return _ott_lineage(identifier, verbose=verbose)
    return _ncbi_lineage(identifier, verbose=verbose)


def _source_id_pattern(source):
    return re.compile(rf"({source}:)?[0-9]+")


def _resolve_external(names, source, within_clade, verbose, progress, failed):
    """
    Resolve unique, normalized names in an external source. Returns one
    record per name, with the fields used by taxo_resolve().
    """
    pattern = _source_id_pattern(source)
    is_id = [pattern.fullmatch(name) is not None for name in names]
    search_names = [name for name, flag in zip(names, is_id) if not flag]

    if source == "ott":
        candidates_by_name = _ott_match_names(search_names, verbose=verbose)
    else:
        found = [_ncbi_search(name, verbose=verbose) for name in search_names]
        ids = [identifier for entry in found if entry for identifier in entry]
        if ids:
            _ncbi_fetch(ids, verbose=verbose)
        candidates_by_name = {}
        for name, entry in zip(search_names, found):
            if entry is None:
                candidates_by_name[name] = None
                continue
            candidates = []
            for identifier in entry:
                cached = _cache().get(f"src_ncbi_lin_{re.sub(r'^ncbi:', '', identifier)}")
                candidates.append({
                    "id": identifier,
                    "name": None if cached is None else cached["lineage"][-1],
                    "synonym": False,
                })
            candidates_by_name[name] = candidates

    records = []
    for k, (name, flag) in enumerate(zip(names, is_id), start=1):
        if flag:
            record = _resolve_external_id(name, source, verbose, failed)
        else:
            record = _resolve_external_name(
                name, candidates_by_name.get(name), source, within_clade,
                verbose, failed,
            )
        records.append(record)
        if progress:
            print(f"Resolving taxa [{k}/{len(names)}]: {name}")
    return records


def _candidate_frame(candidates):
    return pd.DataFrame(
        {
            "id": pd.Series([c["id"] for c in candidates], dtype="object"),
            "name": pd.Series([c["name"] for c in candidates], dtype="object"),
        }
    )


def _resolve_external_id(value, source, verbose, failed):
    identifier = value if value.startswith(f"{source}:") else f"{source}:{value}"
    record = _source_lineage(source, identifier, verbose=verbose)
    if record is None:
        return failed(value, "retrieval_error")
    lineage = record["lineage"]
    name = lineage[-1]
    return {
        "input": value,
        "resolved_name": name,
        "id": identifier,
        "status": "resolved",
        "n_candidates": 1,
        "lineage_depth": len(lineage),
        "lineage": list(lineage),
        "candidates": _candidate_frame([{"id": identifier, "name": name}]),
        "ranks": list(record["ranks"]),
    }


def _resolve_external_name(value, candidates, source, within_clade, verbose, failed):
    if candidates is None:
        return failed(value, "retrieval_error")
    # NCBI candidates have no name when their lineage could not be retrieved.
    named = [c for c in candidates if c["name"] is not None]
    if candidates and not named:
        return failed(value, "retrieval_error")
    candidates = named
    if not candidates:
        return failed(value, "unresolved")

    # Exact matches of the queried name take precedence over matches through
    # synonyms or other names, as the lineage match does for The Taxonomicon.
    exact = [c for c in candidates if c["name"].lower() == value.lower()]
    if exact:
        candidates = exact
    # NCBI returns identifiers in no particular order; lower (older)
    # identifiers come first, which is deterministic and usually puts the
    # best-known taxon first.
    if source == "ncbi":
        candidates = sorted(candidates, key=lambda c: float(re.sub(r"^ncbi:", "", c["id"])))
    candidates = sorted(candidates, key=lambda c: c["synonym"])

    needs_all = within_clade is not None or len(candidates) > 1
    records = [
        _source_lineage(source, c["id"], verbose=verbose) if needs_all or i == 0 else None
        for i, c in enumerate(candidates)
    ]
    if len(candidates) > 1:
        keep = _fetch._outermost_candidates(
            [None if r is None else r["lineage"] for r in records]
        )
        candidates = [c for c, flag in zip(candidates, keep) if flag]
        records = [r for r, flag in zip(records, keep) if flag]
    if within_clade is not None:
        if all(r is None for r in records):
            return failed(value, "retrieval_error", _candidate_frame(candidates))
        inside = [
            r is not None and within_clade in {node.casefold() for node in r["lineage"]}
            for r in records
        ]
        if not any(inside):
            return failed(value, "unresolved")
        candidates = [c for c, flag in zip(candidates, inside) if flag]
        records = [r for r, flag in zip(records, inside) if flag]
    record = records[0]
    if record is None:
        return failed(value, "retrieval_error", _candidate_frame(candidates))
    lineage = record["lineage"]
    n = len(candidates)
    return {
        "input": value,
        "resolved_name": lineage[-1],
        "id": candidates[0]["id"],
        "status": "ambiguous" if n > 1 else "resolved",
        "n_candidates": n,
        "lineage_depth": len(lineage),
        "lineage": list(lineage),
        "candidates": _candidate_frame(candidates),
        "ranks": list(record["ranks"]),
    }


def _source_metadata(source):
    if source == "taxonomicon":
        return {"name": "The Taxonomicon", "url": "http://taxonomicon.taxonomy.nl"}
    if source == "ott":
        return _ott_about()
    return {"name": "NCBI Taxonomy", "url": "https://www.ncbi.nlm.nih.gov/taxonomy"}


# -- Comparison of sources -----------------------------------------------------

class TaxodistSourceComparison(dict):
    """Result of compare_sources(); ``print()`` shows a summary."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __str__(self):
        sources = list(self["resolutions"])
        medians = []
        for source in sources:
            depths = pd.to_numeric(self["taxa"][f"depth_{source}"], errors="coerce")
            depths = depths.dropna()
            value = float(np.median(depths)) if len(depths) else float("nan")
            medians.append(f"{source} {'NA' if math.isnan(value) else f'{value:g}'}")
        agreement = self["agreement"].copy()
        agreement["spearman"] = agreement["spearman"].round(3)
        return "\n".join([
            "Comparison of taxonomic sources",
            f"* Sources: {', '.join(sources)}",
            f"* Metric: {self['metric']}",
            f"* Taxa resolved in every source: {len(self['shared'])} of {len(self['taxa'])}",
            f"Median lineage depth: {', '.join(medians)}",
            "",
            agreement.to_string(index=False, na_rep="NA"),
        ])

    __repr__ = __str__


def compare_sources(taxa, sources=("taxonomicon", "ott", "ncbi"), metric="reciprocal",
                    ambiguity="first", within=None, permutations=999,
                    verbose=False, progress=True, seed=None):
    """
    Compare taxonomic distances across sources

    Resolves the same taxa in several taxonomic sources and compares the
    resulting classifications: the depth of every lineage in each source and
    the agreement between the distance matrices, measured by rank correlation
    and a Mantel permutation test. This shows how much an analysis depends
    on the classification it is based on.

    Parameters
    ----------
    taxa : sequence of str
        Taxon names (identifiers are specific to one source and are not
        accepted). At least three.
    sources : sequence of str
        At least two different source identifiers; see taxo_sources().
    metric : str
        Distance metric; see taxo_distance().
    ambiguity : {"first", "warn", "error"}
        How ambiguous names are handled in each source; see taxo_resolve().
        Defaults to ``"first"``, so ambiguity is reported in the result
        rather than as warnings.
    within : str, dict, or None
        Optional clade restricting the candidates: one name used for every
        source, or a dict with one name per source (clade names differ
        between sources, e.g. ``"Animalia"`` and ``"Metazoa"``).
    permutations : int
        Number of permutations of the Mantel test; 0 skips the test.
    verbose, progress : bool
        Passed to taxo_resolve().
    seed : None, int, or numpy.random.Generator
        Source of randomness for the permutations, passed to
        ``numpy.random.default_rng()`` (replaces ``set.seed()`` in R; the
        permutations are not identical to R's).

    Returns
    -------
    TaxodistSourceComparison
        A dictionary (with attribute access) containing ``resolutions`` (a
        dict of TaxodistResolution by source), ``matrices`` (distance
        matrices restricted to the taxa resolved in every source), ``shared``
        (those taxa), ``taxa`` (one row per input with, for every source, its
        ``status_<source>``, ``id_<source>``, and ``depth_<source>``),
        ``agreement`` (one row per pair of sources with ``source_a``,
        ``source_b``, ``n_taxa``, ``spearman``, and ``p_value``, NaN when not
        computed), ``metric``, and ``permutations``.

    Notes
    -----
    The Mantel test permutes the taxa of one matrix and reports the
    proportion of permutations whose correlation is at least the observed
    one, counting the observed arrangement, i.e. ``(1 + count) / (1 +
    permutations)``. Lineage depths are not comparable across sources in
    absolute terms; correlations of distances are.
    """
    from .distance import _match_metric, distance_matrix
    from .fetch import taxo_resolve

    metric = _match_metric(metric)
    if isinstance(sources, (str, bytes, Mapping)) or not isinstance(sources, Iterable):
        raise ValueError("sources must name at least two different sources.")
    sources = list(sources)
    if (
        len(sources) < 2
        or any(not isinstance(source, str) for source in sources)
        or len(set(sources)) != len(sources)
    ):
        raise ValueError("sources must name at least two different sources.")
    for source in sources:
        _match_source(source)
    if isinstance(taxa, (str, bytes, Mapping)) or not isinstance(taxa, Iterable):
        raise ValueError("taxa must be a sequence of at least three names.")
    taxa = list(taxa)
    if len(taxa) < 3 or any(not isinstance(taxon, str) for taxon in taxa):
        raise ValueError("taxa must be a sequence of at least three names.")
    if (
        isinstance(permutations, (bool, np.bool_))
        or not isinstance(permutations, numbers.Real)
        or math.isnan(permutations)
        or permutations < 0
    ):
        raise ValueError("permutations must be a single non-negative number.")
    permutations = int(permutations)

    if within is None or isinstance(within, Mapping):
        within_by_source = {
            source: (None if within is None else within.get(source)) for source in sources
        }
    else:
        if isinstance(within, (list, tuple)) and len(within) == 1:
            within = within[0]
        if not isinstance(within, str):
            raise ValueError("within must be one clade name or a dict keyed by source.")
        within_by_source = {source: within for source in sources}

    resolutions = {
        source: taxo_resolve(
            taxa, ambiguity=ambiguity, verbose=verbose, progress=progress,
            within=within_by_source[source], source=source,
        )
        for source in sources
    }

    usable = [
        resolution["status"].isin(["resolved", "ambiguous"]).to_numpy()
        for resolution in resolutions.values()
    ]
    shared_rows = np.logical_and.reduce(usable)
    # Identical inputs are compared once.
    keep = shared_rows & ~pd.Series(taxa).duplicated().to_numpy()
    shared = [taxon for taxon, flag in zip(taxa, keep) if flag]

    matrices = {
        source: distance_matrix(
            resolution.loc[keep].reset_index(drop=True), progress=False, metric=metric,
        )
        for source, resolution in resolutions.items()
    }

    table = pd.DataFrame({"input": pd.Series(taxa, dtype="object")})
    for source, resolution in resolutions.items():
        table[f"status_{source}"] = pd.Series(list(resolution["status"]), dtype="object")
        ids = pd.Series(list(resolution["id"]), dtype="object")
        table[f"id_{source}"] = ids.where(ids.notna(), None)
        table[f"depth_{source}"] = pd.array(resolution["lineage_depth"], dtype="Int64")

    rng = np.random.default_rng(seed)
    rows = []
    for a, b in itertools.combinations(sources, 2):
        r, p = _mantel(matrices[a], matrices[b], permutations, rng)
        rows.append({
            "source_a": a, "source_b": b, "n_taxa": len(shared),
            "spearman": r, "p_value": p,
        })
    agreement = pd.DataFrame(
        rows, columns=["source_a", "source_b", "n_taxa", "spearman", "p_value"]
    )
    agreement["n_taxa"] = agreement["n_taxa"].astype("int64")
    agreement["spearman"] = agreement["spearman"].astype("float64")
    agreement["p_value"] = agreement["p_value"].astype("float64")

    return TaxodistSourceComparison(
        resolutions=resolutions,
        matrices=matrices,
        shared=shared,
        taxa=table,
        agreement=agreement,
        metric=metric,
        permutations=permutations,
    )


def _lower_triangle(matrix):
    """Lower-triangle values in the column-major order of R's ``dist``."""
    values = np.asarray(matrix, dtype=float)
    cols, rows = np.triu_indices(values.shape[0], 1)
    return values[rows, cols]


def _spearman(x, y):
    """Spearman correlation as R's ``cor(method = "spearman")``; NaN when
    either variable is constant."""
    # SciPy is imported lazily to keep `import taxodist` light.
    from scipy.stats import rankdata

    if len(x) < 2:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    if np.all(rx == rx[0]) or np.all(ry == ry[0]):
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _mantel(a, b, permutations, rng):
    """
    Spearman Mantel statistic between two square distance matrices over the
    same taxa, with a one-sided permutation p-value. Pairs with a non-finite
    distance in either matrix are excluded. Returns ``(r, p)``.
    """
    x = _lower_triangle(a)
    y = _lower_triangle(b)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    r = _spearman(x[ok], y[ok])
    if math.isnan(r) or permutations == 0:
        return r, float("nan")
    mb = np.asarray(b, dtype=float)
    n = mb.shape[0]
    count = 0
    for _ in range(permutations):
        perm = rng.permutation(n)
        yp = _lower_triangle(mb[np.ix_(perm, perm)])
        okp = np.isfinite(x) & np.isfinite(yp)
        rp = _spearman(x[okp], yp[okp])
        if not math.isnan(rp) and rp >= r:
            count += 1
    return r, (1 + count) / (1 + permutations)
