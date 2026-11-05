"""Lineage parser, shared fixtures, and parser-related behaviour.

Mirrors tests/testthat/test-parser.R of the R package. Real TaxonTree and
TaxonList pages, trimmed to the parts the parser reads, are stored in
tests/fixtures; expected_lineages.json holds the raw lineage (before trimming
at Biota) expected from each tree page. The same files are used by the R and
Julia implementations to keep the three parsers identical.
"""

import json
import pickle
import time
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from bs4 import BeautifulSoup

import taxodist
import taxodist.fetch as fetch

FIXTURES = Path(__file__).parent / "fixtures"


def read_fixture(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def fixture_page(name):
    return BeautifulSoup(read_fixture(name), "lxml")


def lineage_of(taxon_id):
    return fetch._lineage_from_page(fixture_page(f"tree_{taxon_id}.html"), taxon_id)


@pytest.fixture(autouse=True)
def fresh_cache():
    fetch.clear_cache()
    yield
    fetch.clear_cache()


EXPECTED = json.loads(read_fixture("expected_lineages.json"))


# -- Structured parser ---------------------------------------------------------

def test_fixture_set_is_not_trivial():
    assert len(EXPECTED) > 5


@pytest.mark.parametrize("taxon_id", sorted(EXPECTED))
def test_structured_parser_reproduces_the_expected_lineage(taxon_id):
    assert lineage_of(taxon_id) == EXPECTED[taxon_id]


def test_structured_parser_keeps_clean_names_despite_authors_and_markers():
    passer = lineage_of("59107")
    assert {"Passeres", "Passerida", "Terrestrornithes"} <= set(passer)
    assert not any(
        any(char in node for char in " ,(") for node in passer if node != "Passer"
    )
    assert passer[-1] == "Passer"
    # Children of the page's taxon (species) are excluded.
    assert not any(node.startswith("Passer ") for node in passer)

    microtus = lineage_of("62669")
    assert "Supramyomorpha" in microtus
    assert microtus[-1] == "Microtus"

    lipotes = lineage_of("68670")
    assert lipotes[-3:] == ["Lipotoidea", "Lipotidae", "Lipotes"]

    anamorph = lineage_of("204388")
    assert anamorph[-1] == "Epidermophyton"


def test_structured_parser_appends_the_subject_when_its_own_row_is_missing():
    yi = lineage_of("7248674")
    assert yi[-2:] == ["Scansoriopterygidae", "Yi"]
    assert "[plesion]" not in yi


def test_structured_parser_handles_species_pages():
    assert lineage_of("66295")[-2:] == ["Homo", "Homo sapiens"]


def test_get_lineage_by_id_uses_the_structured_parser_and_caches_both_variants():
    response = Mock(status_code=200, text=read_fixture("tree_59107.html"))
    with patch.object(fetch._http_session, "get", return_value=response) as get:
        cleaned = fetch.get_lineage_by_id("59107")
        raw = fetch.get_lineage_by_id("59107", clean=False)
    assert get.call_count == 1
    assert cleaned[0] == "Biota"
    assert raw[0] == "Natura"
    assert raw[-len(cleaned):] == cleaned


def hierarchy_page(rows):
    return BeautifulSoup(
        f'<html><body><div id="ctl00_cphTabContent_divHierarchy">{rows}</div></body></html>',
        "lxml",
    )


def test_structured_parser_skips_comments_unnamed_and_unclassed_rows():
    nbsp = "\xa0"
    # Rows without a class attribute are read; a row whose <b> is blank is
    # skipped. The comment holds non-breaking spaces that would wrongly deepen
    # the indentation of the next row (as R's text() nodes exclude comments).
    page = hierarchy_page(
        '<a href="TaxonTree.aspx?id=1">Clade <b>Biota</b></a><br>'
        f'{nbsp}<a href="TaxonTree.aspx?id=2">Kingdom <b> </b></a><br>'
        f'{nbsp}{nbsp}<a href="TaxonTree.aspx?id=3">Kingdom <b>Animalia</b></a><br>'
        f'<!--{nbsp * 3}-->{nbsp}<a href="TaxonTree.aspx?id=4">Clade <b>Other</b></a><br>'
    )
    assert fetch._parse_hierarchy(page) == (["Biota", "Animalia"], ["Clade", "Kingdom"], True)


def test_structured_parser_without_named_rows_returns_none():
    page = hierarchy_page(
        '<a class="Ordinal" href="TaxonTree.aspx?id=1"><b>Order</b></a><br>'
        '<a href="TaxonTree.aspx?id=2">Kingdom <b>  </b></a><br>'
    )
    assert fetch._parse_hierarchy(page) is None


def test_class_attribute_is_read_as_one_string_with_any_tree_builder():
    html = '<a class="Valid Extra">x</a><a>y</a>'
    listed = BeautifulSoup(html, "lxml").find_all("a")
    plain = BeautifulSoup(html, "lxml", multi_valued_attributes=None).find_all("a")
    assert fetch._class_attr(listed[0]) == "Valid Extra"
    assert fetch._class_attr(plain[0]) == "Valid Extra"
    assert fetch._class_attr(listed[1]) is None
    assert fetch._class_attr(plain[1]) is None


# -- Name-based truncation -----------------------------------------------------

def test_finalize_lineage_keeps_name_ancestors_of_species_and_subspecies():
    finalize = fetch._finalize_lineage
    assert finalize(["Biota", "Homo", "Homo sapiens"], "Homo sapiens sapiens") == [
        "Biota", "Homo", "Homo sapiens", "Homo sapiens sapiens"
    ]
    assert finalize(
        ["Biota", "Microtus", "Microtus (Microtus)", "Microtus arvalis"],
        "Microtus arvalis",
    ) == ["Biota", "Microtus", "Microtus (Microtus)", "Microtus arvalis"]
    # A subgenus is a child of a queried genus and is dropped.
    assert finalize(["Biota", "Microtus", "Microtus (Microtus)"], "Microtus") == [
        "Biota", "Microtus"
    ]
    # Species of another genus are not name-ancestors.
    assert finalize(["Biota", "Homo", "Pan troglodytes"], "Homo sapiens") == [
        "Biota", "Homo", "Homo sapiens"
    ]


def test_finalize_lineage_returns_a_new_list():
    lineage = ["Biota", "Homo"]
    result = fetch._finalize_lineage(lineage, "Homo")
    assert result == lineage and result is not lineage


# -- Fallback text parser ------------------------------------------------------

def test_fallback_text_parser_extracts_names_regardless_of_author_format():
    raw = [
        "Ichthyosauria de Blainville, 1835", "Foraminifera d'Orbigny, 1826",
        "Xenus Özdikmen, 2009", "Xenus O'Brien, 1990",
        "Xidae van der Hoeven, 1850", "Homo sapiens Linnaeus, 1758",
        "Homo sapiens sapiens Linnaeus, 1758",
        "Alismatales R.Br. ex Bercht. & J.Presl", "Firmicutes corrig. Gibbons",
        "Scombriformes - mackerels", "Romeriida Gauthier,",
        "Accipitres (C. Linnaeus, 1758)", "neomura Cavalier-Smith, 1987",
        "Pan-Aves Gauthier & de Queiroz, 2001",
    ]
    assert fetch._extract_scientific_name(raw) == [
        "Ichthyosauria", "Foraminifera", "Xenus", "Xenus", "Xidae",
        "Homo sapiens", "Homo sapiens sapiens", "Alismatales", "Firmicutes",
        "Scombriformes", "Romeriida", "Accipitres", "neomura", "Pan-Aves",
    ]
    # Quoted and bracketed lines are left for the caller's filters.
    assert fetch._extract_scientific_name(['"Setaphyta"', "[unranked]"]) == [
        '"Setaphyta"', "[unranked]"
    ]
    assert fetch._extract_scientific_name(None) is None


def _text_page(subject, body):
    return BeautifulSoup(
        f'<html><body><div id="ctl00_divSubject"><b>{subject}</b></div>'
        f'<div id="divPageContent">{body}</div></body></html>',
        "lxml",
    )


def test_fallback_text_parser_strips_double_daggers_and_qualified_markers():
    page = _text_page(
        "Lipotes",
        "Natura - nature\nClade Biota Wagner 2004\n"
        "Superfamily ‡?Lipotoidea de Muizon, 1988\n"
        "[plesion] Family †Lipotidae Zhou, 1978\n"
        "Genus ‡?Lipotesᵀ Miller, 1918\n",
    )
    assert fetch._lineage_from_page(page, "1") == [
        "Natura", "Biota", "Lipotoidea", "Lipotidae", "Lipotes"
    ]


def test_fallback_text_parser_matches_r_on_prefixes_and_cutoff():
    # Expected values were produced by the R implementation.
    page = _text_page(
        "Homo",
        "Natura\nClade Biota\n[crown]Clade Mammalia\n[crown] Clade Eutheria Gill\n"
        "[stem] Theria Parker & Haswell\n[plesion] Genus Xus\nhomo lower line\n"
        "Order Primates Linnaeus, 1758\n? Family Hominidae Gray, 1825\n"
        "Anamorph Genus Epidermophyton Sabour.\nSubgenus\n[unranked]\n"
        '"Setaphyta"\nPopulation X\nGenus Homo Linnaeus, 1758\nHomo sapiens Linnaeus\n',
    )
    assert fetch._lineage_from_page(page, "1") == [
        "Natura", "Biota", "[crown]Clade Mammalia", "Eutheria", "Theria", "Xus",
        "homo lower line", "Primates", "Hominidae", "Epidermophyton", "Homo",
    ]
    page = _text_page(
        "Zed",
        "Natura\nBiota\nClassis Foo\nCladeBar\nClade  Baz  Qux\nTribe\tTrib Smith\n"
        "Infraorder Inf\nZed\n",
    )
    assert fetch._lineage_from_page(page, "1") == [
        "Natura", "Biota", "Classis", "CladeBar", "Baz", "Trib", "Inf", "Zed"
    ]


def test_fallback_text_parser_cutoff_is_case_sensitive_and_unicode_aware():
    page = _text_page(
        "Émys",
        "Natura\nBiota\nxÉmys Foo\nClass Reptilia Laurenti\nGenus Émys Duméril\n"
        "Émys orbicularis\n",
    )
    assert fetch._lineage_from_page(page, "1") == [
        "Natura", "Biota", "xÉmys", "Reptilia", "Émys"
    ]


def test_lineage_label_cleaning_requires_whitespace_after_crown_rank():
    assert fetch._clean_lineage_label("[crown] Clade Dinosauria") == "Dinosauria"
    assert fetch._clean_lineage_label("[stem] Subgrade Foo") == "Foo"
    assert fetch._clean_lineage_label("[crown] Cladex Foo") == "Cladex"
    # A missing label stays missing (NA propagates through R's pipeline).
    assert fetch._clean_lineage_label(None) is None


# -- Regular expressions ---------------------------------------------------------

def test_regex_escape_quotes_every_metacharacter():
    import re

    for text in ["Aus (Bus) cus", "a.b", "x|y", "a[b]", "a{2}", "a+b", "a^b$", "a\\b", "a*?"]:
        assert re.fullmatch(fetch._regex_escape(text), text), text
    assert fetch._contains_word("Microtus (Agricola)", "Microtus (Agricola)")
    assert fetch._contains_word("Aus (Bus) cus", "Aus (Bus)")
    assert not fetch._contains_word("Homopus", "Homo")
    assert fetch._contains_word("Émys", "Émys")
    assert not fetch._contains_word("xémys", "mys")
    assert not fetch._contains_word("Homo", "homo")
    assert fetch._contains_word("Homo", "homo", ignore_case=True)


# -- Search rows -----------------------------------------------------------------

# Produced by R's .parse_search_rows() on the same fixture files.
SEARCH_EXPECTED = {
    "search_Homo_sapiens.html": [
        {"id": "66295", "name": "Homo sapiens (Species) Linnaeus, 1758 - human (mammal)"},
    ],
    "search_Nereis.html": [
        {
            "id": "1109123",
            "name": "Nereis (Genus) Hübner, 1806, nom. illeg. see (Genus) "
                    "Mechanitis Fabricius, 1807 (butterfly)",
        },
        {
            "id": "15133",
            "name": "Nereis (Genus) Linnaeus, 1748, nom. inval. see (Genus) "
                    "Nereis Linnaeus, 1758 (animal)",
        },
    ],
    "search_Tyrannosaurus.html": [
        {"id": "50841", "name": "†Tyrannosaurus (Genus) Osborn, 1905 (dinosaur)"},
    ],
}


@pytest.mark.parametrize("name", sorted(SEARCH_EXPECTED))
def test_search_rows_match_r(name):
    assert fetch._parse_search_rows(fixture_page(name)) == SEARCH_EXPECTED[name]


def test_search_rows_require_class_exactly_valid():
    soup = BeautifulSoup(
        "<table>"
        "<tr><td>A</td><td><a class='Valid Extra' href='TaxonTree.aspx?id=1'>x</a></td></tr>"
        "<tr><td>B</td><td><a class='Valid' href='TaxonTree.aspx?id=2'>x</a></td></tr>"
        "<tr><td>astronomy C</td><td><a class='Valid' href='TaxonTree.aspx?id=3'>x</a></td></tr>"
        "</table>",
        "lxml",
    )
    # "astronomy" is not part of R's astronomical filter.
    assert fetch._parse_search_rows(soup) == [
        {"id": "2", "name": "Bx"}, {"id": "3", "name": "astronomy Cx"}
    ]


# -- Resolution ------------------------------------------------------------------

def test_taxo_resolve_reports_non_biological_candidates_as_unresolved():
    fetch._taxodist_cache["lin_1"] = ["Natura", "Mundus", "Venus"]
    with patch("taxodist.fetch._taxo_search_details", return_value={
        "status": "ok", "results": pd.DataFrame([{"id": "1", "name": "Venus - planet"}]),
    }):
        result = fetch.taxo_resolve(["Venus"], progress=False)
    assert result["status"].iloc[0] == "unresolved"
    # Unresolved records carry no candidates, as bundle validation requires.
    assert int(result["n_candidates"].iloc[0]) == 0


def test_taxo_resolve_keeps_retrieval_error_when_a_candidate_cannot_be_fetched():
    with patch("taxodist.fetch._taxo_search_details", return_value={
        "status": "ok", "results": pd.DataFrame([{"id": "abc", "name": "Broken"}]),
    }):
        result = fetch.taxo_resolve(["Broken"], progress=False)
    assert result["status"].iloc[0] == "retrieval_error"


def test_invalid_or_rejected_homonyms_are_ranked_after_valid_ones():
    descriptions = [
        "Amanita (Genus) Dill., nom. inval. see Amanita Pers.",
        "Amanita (Genus) Pers. (1797), nom. cons. (fungus)",
        "Amanita (Genus) Foo, nom. rej.",
    ]
    assert fetch._candidate_order(descriptions) == [1, 0, 2]
    assert fetch._candidate_order(["a", "b"]) == [0, 1]


AMANITA = pd.DataFrame([
    {"id": "1", "name": "Amanita (Genus) Dill., nom. inval. see Amanita Pers."},
    {"id": "2", "name": "Amanita (Genus) Pers. (1797), nom. cons. (fungus)"},
])


def test_candidate_order_is_applied_by_taxo_resolve_and_id_lookup():
    lineage = ["Biota", "Fungi", "Amanita"]
    with patch("taxodist.fetch._taxo_search_details", return_value={
        "status": "ok", "results": AMANITA,
    }), patch("taxodist.fetch.get_lineage_by_id", return_value=lineage):
        result = fetch.taxo_resolve(["Amanita"], ambiguity="first", progress=False)
    assert result["id"].iloc[0] == "2"
    assert list(result["candidates"].iloc[0]["id"]) == ["2", "1"]

    page = "<table>" + "".join(
        f"<tr><td>{row.name}</td><td><a class='Valid' href='TaxonTree.aspx?id={row.id}'></a></td></tr>"
        for row in AMANITA.itertuples()
    ) + "</table>"
    with patch.object(
        fetch._http_session, "get", return_value=Mock(status_code=200, text=page)
    ), patch("taxodist.fetch.get_lineage_by_id", return_value=lineage):
        with pytest.warns(UserWarning, match="Multiple valid biological entries"):
            assert fetch.get_taxonomicon_id("Amanita") == "2"


# -- Cache -----------------------------------------------------------------------

def test_cache_files_from_an_older_parser_keep_only_taxon_ids(tmp_path):
    path = tmp_path / "old.pkl"
    with path.open("wb") as stream:
        pickle.dump({"id_Passer": "59107", "lin_59107": ["Biota", "Passeres (C."]}, stream)
    with pytest.warns(UserWarning, match="older version"):
        fetch.load_cache(path)
    assert fetch._taxodist_cache == {"id_Passer": "59107"}

    fetch.clear_cache()
    with path.open("wb") as stream:
        pickle.dump({}, stream)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fetch.load_cache(path)

    fetch._taxodist_cache["lin_1"] = ["Biota", "A"]
    fetch.save_cache(path)
    with path.open("rb") as stream:
        assert pickle.load(stream)["meta_parser_version"] == 2
    fetch.clear_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fetch.load_cache(path)
    assert fetch._taxodist_cache == {"lin_1": ["Biota", "A"]}
    assert "meta_parser_version" not in fetch._taxodist_cache


# -- Throttling ------------------------------------------------------------------

def test_requests_are_throttled_by_min_interval(monkeypatch):
    monkeypatch.setattr(fetch, "MIN_INTERVAL", 0.2)
    fetch._throttle()
    start = time.monotonic()
    fetch._throttle()
    assert time.monotonic() - start >= 0.15

    monkeypatch.setattr(fetch, "MIN_INTERVAL", 0)
    start = time.monotonic()
    fetch._throttle()
    assert time.monotonic() - start < 0.1


def test_every_request_is_throttled(monkeypatch):
    calls = []
    monkeypatch.setattr(fetch, "_throttle", lambda: calls.append(1))
    with patch.object(fetch._http_session, "get", return_value=Mock(status_code=404)):
        fetch._http_get("http://example")
        fetch.get_lineage_by_id("1")
        fetch._taxo_search_details("Alpha")
    assert len(calls) == 3


def test_invalid_min_interval_disables_throttling(monkeypatch):
    sleeps = []
    monkeypatch.setattr(fetch, "_throttle_sleep", sleeps.append)
    for value in (None, "slow", float("nan"), -1, True):
        monkeypatch.setattr(fetch, "MIN_INTERVAL", value)
        fetch._throttle()
        fetch._throttle()
    assert sleeps == []
