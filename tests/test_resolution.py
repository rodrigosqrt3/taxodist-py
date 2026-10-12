"""Name normalization, lineage truncation, retries, and ID lookup (0.9.0)."""

import math
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

import taxodist
import taxodist.fetch as fetch


def html_response(text, status=200):
    response = Mock(status_code=status)
    response.text = text
    return response


SEARCH_ONE = """
<table>
  <tr><td>Alpha - animal</td>
  <td><a class="Valid" href="TaxonTree.aspx?id=101&src=0">tree</a></td></tr>
</table>
"""


@pytest.fixture(autouse=True)
def fresh_cache():
    fetch.clear_cache()
    yield
    fetch.clear_cache()


# ── Helpers ──────────────────────────────────────────────────────────────────

def test_normalize_and_search_url():
    assert fetch._normalize_taxon("  Homo   sapiens ") == "Homo sapiens"
    assert fetch._search_url("A&B C").endswith("search=A%26B%20C")


def test_max_tries_is_validated(monkeypatch):
    monkeypatch.setattr(fetch, "MAX_TRIES", 5)
    assert fetch._max_tries() == 5
    monkeypatch.setattr(fetch, "MAX_TRIES", "many")
    assert fetch._max_tries() == 1
    monkeypatch.setattr(fetch, "MAX_TRIES", 0)
    assert fetch._max_tries() == 1


def test_http_get_retries_network_and_server_errors(monkeypatch):
    pauses = []
    monkeypatch.setattr(fetch, "_sleep", pauses.append)
    ok = html_response("ok")
    with patch.object(
        fetch._http_session, "get",
        side_effect=[requests.exceptions.ConnectionError(), html_response("", 503), ok],
    ) as get:
        assert fetch._http_get("http://example") is ok
    assert get.call_count == 3
    assert pauses == [1, 2]


def test_http_get_returns_last_server_error_and_raises_last_exception():
    with patch.object(fetch._http_session, "get", return_value=html_response("", 500)) as get:
        assert fetch._http_get("http://example").status_code == 500
    assert get.call_count == fetch.MAX_TRIES

    with patch.object(
        fetch._http_session, "get", side_effect=requests.exceptions.Timeout()
    ):
        with pytest.raises(requests.exceptions.Timeout):
            fetch._http_get("http://example")


def test_http_get_does_not_retry_client_errors():
    with patch.object(fetch._http_session, "get", return_value=html_response("", 404)) as get:
        assert fetch._http_get("http://example").status_code == 404
    assert get.call_count == 1


def test_parse_search_rows_skips_duplicates_and_invalid_rows():
    soup = fetch.BeautifulSoup(
        """
        <table>
          <tr><td>Alpha</td><td><a class="Valid" href="TaxonTree.aspx?id=1">t</a></td></tr>
          <tr><td>Alpha again</td><td><a class="Valid" href="TaxonTree.aspx?id=1">t</a></td></tr>
          <tr><td>Invalid</td><td><a class="Invalid" href="TaxonTree.aspx?id=2">t</a></td></tr>
          <tr><td>Planet</td><td><a class="Valid" href="TaxonTree.aspx?id=3">t</a></td></tr>
        </table>
        """,
        "lxml",
    )
    assert fetch._parse_search_rows(soup) == [{"id": "1", "name": "Alpha t"}]


def test_finalize_lineage_genus_queries():
    lineage = ["Biota", "Dinosauria", "[unranked] x", "Tyrannosaurus", "Tyrannosaurus rex"]
    expected = ["Biota", "Dinosauria", "Tyrannosaurus"]
    assert fetch._finalize_lineage(lineage, "Tyrannosaurus") == expected
    assert fetch._finalize_lineage(lineage, "tyrannosaurus") == expected
    assert fetch._finalize_lineage(["Biota", "Blattidae"], "Periplaneta") == [
        "Biota", "Blattidae", "Periplaneta"
    ]


def test_finalize_lineage_binomial_queries():
    lineage = ["Biota", "Homo", "Homo sapiens", "Homo sapiens sapiens"]
    assert fetch._finalize_lineage(lineage, "homo sapiens") == [
        "Biota", "Homo", "Homo sapiens"
    ]
    assert fetch._finalize_lineage(["Biota", "Homo"], "Homo sapiens") == [
        "Biota", "Homo", "Homo sapiens"
    ]


# ── get_lineage / get_lineage_by_id ──────────────────────────────────────────

def test_get_lineage_normalizes_case_and_whitespace():
    fetch._taxodist_cache.update({
        "id_Tyrannosaurus": "1",
        "id_tyrannosaurus": "1",
        "lin_1": ["Biota", "Dinosauria", "Tyrannosaurus"],
    })
    expected = ["Biota", "Dinosauria", "Tyrannosaurus"]
    assert fetch.get_lineage("Tyrannosaurus") == expected
    assert fetch.get_lineage("  Tyrannosaurus ") == expected
    assert fetch.get_lineage("tyrannosaurus") == expected
    assert taxodist.taxo_distance("Tyrannosaurus", "tyrannosaurus")["distance"] == 0


def test_get_lineage_validates_input():
    with pytest.raises(TypeError, match="single taxon"):
        fetch.get_lineage(["A", "B"])
    assert fetch.get_lineage(None) is None
    assert fetch.get_lineage(math.nan) is None
    assert fetch.get_lineage("   ") is None


def test_get_lineage_by_id_validates_and_accepts_numbers():
    fetch._taxodist_cache["lin_50841"] = ["Biota", "Tyrannosaurus"]
    assert fetch.get_lineage_by_id(50841) == ["Biota", "Tyrannosaurus"]
    assert fetch.get_lineage_by_id(" 50841 ") == ["Biota", "Tyrannosaurus"]
    assert fetch.get_lineage(50841) == ["Biota", "Tyrannosaurus"]
    assert fetch.get_lineage_by_id(None) is None
    with pytest.raises(TypeError, match="single ID"):
        fetch.get_lineage_by_id(["1", "2"])


def test_cleaned_and_raw_lineages_are_cached_separately():
    page = (
        '<html><body><div id="ctl00_divSubject"><b>Foo</b></div>'
        '<div id="divPageContent">Natura\nactualia\nBiota\nAnimalia\nFoo\n</div>'
        "</body></html>"
    )
    with patch.object(fetch._http_session, "get", return_value=html_response(page)) as get:
        raw = fetch.get_lineage_by_id("999", clean=False)
        cleaned = fetch.get_lineage_by_id("999", clean=True)
    assert raw == ["Natura", "actualia", "Biota", "Animalia", "Foo"]
    assert cleaned == ["Biota", "Animalia", "Foo"]
    assert get.call_count == 1


# ── ID lookup statuses ───────────────────────────────────────────────────────

def test_lookup_reports_network_and_parse_failures():
    with patch.object(
        fetch._http_session, "get", side_effect=requests.exceptions.ConnectionError()
    ):
        with pytest.warns(UserWarning, match="Cannot reach"):
            assert fetch._lookup_taxonomicon_id("Alpha") == (None, "retrieval_error")

    with patch.object(fetch._http_session, "get", return_value=html_response("x", 404)):
        with pytest.warns(UserWarning, match="Cannot reach"):
            assert fetch._lookup_taxonomicon_id("Alpha") == (None, "retrieval_error")

    with patch.object(fetch._http_session, "get", return_value=html_response("x")), \
            patch("taxodist.fetch.BeautifulSoup", side_effect=ValueError("bad")):
        with pytest.warns(UserWarning, match="Could not parse"):
            assert fetch._lookup_taxonomicon_id("Alpha") == (None, "retrieval_error")


def test_lookup_separates_absent_names_from_failed_candidates(capsys):
    with patch.object(fetch._http_session, "get", return_value=html_response(SEARCH_ONE)):
        with patch("taxodist.fetch.get_lineage_by_id", return_value=None):
            assert fetch._lookup_taxonomicon_id("Alpha") == (None, "retrieval_error")
        with patch("taxodist.fetch.get_lineage_by_id", return_value=["Natura", "Rock"]):
            assert fetch._lookup_taxonomicon_id("Alpha", verbose=True) == (None, "not_found")
        with patch("taxodist.fetch.get_lineage_by_id", return_value=["Biota", "Alpha"]):
            assert fetch._lookup_taxonomicon_id("  Alpha ") == ("101", "found")
    assert "not found" in capsys.readouterr().out
    assert fetch._taxodist_cache["id_Alpha"] == "101"
    assert fetch._lookup_taxonomicon_id("Alpha") == ("101", "found")


def test_check_coverage_returns_na_when_unknown():
    statuses = {
        "Found": ("1", "found"),
        "Absent": (None, "not_found"),
        "Offline": (None, "retrieval_error"),
    }
    with patch(
        "taxodist.distance._lookup_taxonomicon_id",
        side_effect=lambda taxon, **kwargs: statuses[taxon],
    ):
        result = taxodist.check_coverage(["Found", "Absent", "Offline"])
    assert str(result.dtype) == "boolean"
    assert result["Found"] is True or bool(result["Found"])
    assert not bool(result["Absent"])
    assert result["Offline"] is pd.NA


# ── taxo_resolve ─────────────────────────────────────────────────────────────

def test_taxo_resolve_truncates_like_get_lineage():
    with patch("taxodist.fetch._taxo_search_details", return_value={
        "status": "ok",
        "results": pd.DataFrame([{"id": "2", "name": "Genus Periplaneta"}]),
    }), patch("taxodist.fetch.get_lineage_by_id", return_value=["Biota", "Insecta", "Blattidae"]):
        result = fetch.taxo_resolve(["Periplaneta"], progress=False)
    assert result["lineage"].iloc[0] == ["Biota", "Insecta", "Blattidae", "Periplaneta"]
    assert result["resolved_name"].iloc[0] == "Periplaneta"
    assert result["lineage_depth"].iloc[0] == 4


def test_taxo_resolve_resolves_normalized_duplicates_once():
    searches = []

    def search(taxon, **kwargs):
        searches.append(taxon)
        return {"status": "ok", "results": pd.DataFrame([{"id": "101", "name": taxon}])}

    with patch("taxodist.fetch._taxo_search_details", side_effect=search), \
            patch("taxodist.fetch.get_lineage_by_id", return_value=["Biota", "Alpha"]):
        result = fetch.taxo_resolve(["Alpha", " Alpha  "], progress=True)
    assert searches == ["Alpha"]
    assert list(result["input"]) == ["Alpha", " Alpha  "]
    assert list(result["id"]) == ["101", "101"]


def test_taxo_resolve_within_restricts_candidates():
    candidates = pd.DataFrame([
        {"id": "201", "name": "Nereis worm"},
        {"id": "202", "name": "Nereis plant"},
    ])
    lineages = {
        "201": ["Biota", "Animalia", "Annelida", "Nereis"],
        "202": ["Biota", "Plantae", "Nereis"],
    }
    with patch("taxodist.fetch._taxo_search_details", return_value={
        "status": "ok", "results": candidates,
    }), patch(
        "taxodist.fetch.get_lineage_by_id",
        side_effect=lambda identifier, **kwargs: lineages[identifier],
    ):
        animal = fetch.taxo_resolve(["Nereis"], within="annelida", progress=False)
        fungus = fetch.taxo_resolve(["Nereis"], within="Fungi", progress=False)

    assert animal["status"].iloc[0] == "resolved"
    assert animal["id"].iloc[0] == "201"
    assert animal["n_candidates"].iloc[0] == 1
    assert fungus["status"].iloc[0] == "unresolved"
    assert fungus["n_candidates"].iloc[0] == 0
    assert fungus["lineage"].iloc[0] is None

    with pytest.raises(ValueError, match="within"):
        fetch.taxo_resolve(["Nereis"], within="  ")


def test_distance_matrix_agrees_for_names_and_resolution():
    ids = {"Periplaneta": "1", "Blatta": "2"}
    fetch._taxodist_cache.update({
        "id_Periplaneta": "1",
        "id_Blatta": "2",
        "lin_1": ["Biota", "Insecta", "Blattidae"],
        "lin_2": ["Biota", "Insecta", "Blattidae", "Blatta"],
    })
    with patch(
        "taxodist.fetch._taxo_search_details",
        side_effect=lambda taxon, **kwargs: {
            "status": "ok",
            "results": pd.DataFrame([{"id": ids[taxon], "name": taxon}]),
        },
    ):
        resolution = fetch.taxo_resolve(list(ids), progress=False)
    by_name = taxodist.distance_matrix(list(ids), progress=False)
    by_resolution = taxodist.distance_matrix(resolution, progress=False)
    pd.testing.assert_frame_equal(by_name, by_resolution)
    assert by_name.iloc[0, 1] == pytest.approx(1 / 3)
