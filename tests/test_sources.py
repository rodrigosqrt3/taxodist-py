"""
Open Tree Taxonomy and NCBI Taxonomy sources, and compare_sources().

Requests are answered from real responses stored in fixtures/sources (shared
with the R and Julia implementations; see tools/record_source_fixtures.R in
the R package); a request without a stored response behaves as a failed
request.
"""

import json
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import requests

import taxodist
import taxodist.fetch as fetch
import taxodist.sources as sources

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "sources")


def fixture_path(name):
    return os.path.join(FIXTURES, name)


def read_fixture_response(name):
    path = fixture_path(name)
    if not os.path.exists(path):
        return {"status": None, "text": None}
    with open(path, encoding="utf-8") as handle:
        return {"status": 200, "text": handle.read()}


def fixture_request(source, method, url, body=None):
    if url.endswith("/taxonomy/about"):
        return read_fixture_response("ott_about.json")
    if url.endswith("/tnrs/match_names"):
        return read_fixture_response("ott_match_names.json")
    if url.endswith("/taxonomy/taxon_info"):
        return read_fixture_response(f"ott_taxon_info_{body['ott_id']}.json")
    if "esearch.fcgi" in url:
        term = urllib.parse.unquote(re.sub(r".*[?&]term=([^&]*).*", r"\1", url))
        name = re.sub(r'^"(.*)"\[.*$', r"\1", term).lower()
        field = "scientific" if "[Scientific Name]" in term else "all"
        return read_fixture_response(
            f"ncbi_esearch_{re.sub(r'[^a-z0-9]+', '_', name)}_{field}.json"
        )
    if "efetch.fcgi" in url:
        ids = urllib.parse.unquote(re.sub(r".*[?&]id=([^&]*).*", r"\1", url)).split(",")
        taxa = []
        for identifier in ids:
            path = fixture_path(f"ncbi_taxon_{identifier}.xml")
            if os.path.exists(path):
                taxon = ET.parse(path).getroot().find("./Taxon")
                taxa.append(ET.tostring(taxon, encoding="unicode"))
        return {"status": 200, "text": "<TaxaSet>" + "".join(taxa) + "</TaxaSet>"}
    return {"status": None, "text": None}


def failing_request(source, method, url, body=None):
    return {"status": None, "text": None}


@pytest.fixture
def local_sources(monkeypatch):
    """Answer source requests with ``request`` (default: the fixtures)."""
    def install(request=fixture_request):
        fetch.clear_cache()
        monkeypatch.setattr(sources, "_source_request", request)
    install()
    yield install
    fetch.clear_cache()


NAMES = [
    "Homo", "Pan", "Mus", "Drosophila", "Nereis", "Quercus",
    "Fakeosaurus", "human", "Homo sapiens",
]
IDS = {"ott": ["ott:770309", "417957"], "ncbi": ["ncbi:9606", "9598"]}


def test_resolutions_reproduce_the_recorded_expectations(local_sources):
    with open(fixture_path("expected_resolutions.json"), encoding="utf-8") as handle:
        expected = json.load(handle)
    assert set(expected) == {"ott", "ott_within_metazoa", "ncbi", "ncbi_within_metazoa"}
    for key, exp in expected.items():
        source = key.replace("_within_metazoa", "")
        within = "Metazoa" if "within" in key else None
        res = taxodist.taxo_resolve(
            NAMES + IDS[source], source=source, progress=False,
            ambiguity="first", within=within,
        )
        assert res.source == exp["source"], key
        assert res.source_url == exp["source_url"], key
        assert len(res) == len(exp["records"])
        for i, rec in enumerate(exp["records"]):
            info = f"{key} {rec['input']}"
            row = res.iloc[i]
            assert row["input"] == rec["input"], info
            assert row["resolved_name"] == rec["resolved_name"], info
            assert row["status"] == rec["status"], info
            assert row["id"] == rec["id"], info
            assert row["n_candidates"] == rec["n_candidates"], info
            depth = row["lineage_depth"]
            assert (None if pd.isna(depth) else int(depth)) == rec["lineage_depth"], info
            assert row["lineage"] == rec["lineage"], info
            assert row["ranks"] == rec["ranks"], info
            assert list(row["candidates"]["id"]) == [c["id"] for c in rec["candidates"]], info
            assert list(row["candidates"]["name"]) == [c["name"] for c in rec["candidates"]], info


def test_external_sources_resolve_the_expected_taxa(local_sources):
    ott = taxodist.taxo_resolve(["Homo", "Nereis", "Fakeosaurus"], source="ott", progress=False)
    assert list(ott["id"]) == ["ott:770309", "ott:305738", None]
    assert list(ott["status"]) == ["resolved", "resolved", "unresolved"]
    assert ott["lineage"][0][-3:] == ["Hominidae", "Homininae", "Homo"]
    assert ott["lineage"][0][0] == "life"
    assert ott["ranks"][0][-1] == "Genus"
    assert ott.source.startswith("Open Tree Taxonomy 3.7")

    with pytest.warns(UserWarning, match="Drosophila"):
        ncbi = taxodist.taxo_resolve(["Mus", "Drosophila", "human"], source="ncbi", progress=False)
    # The nominotypical subgenus Mus (Mus) is not counted as a homonym.
    assert ncbi["id"][0] == "ncbi:10088"
    assert ncbi["status"][0] == "resolved"
    # Drosophila is a true homonym (a fly and a fungus); the fly has the lower ID.
    assert ncbi["id"][1] == "ncbi:7215"
    assert ncbi["status"][1] == "ambiguous"
    # A common name is matched through "All Names".
    assert ncbi["resolved_name"][2] == "Homo sapiens"
    assert ncbi["lineage"][0][0] == "cellular organisms"
    assert ncbi.source == "NCBI Taxonomy"


def test_duplicated_inputs_and_progress(local_sources, capsys):
    res = taxodist.taxo_resolve([" Homo ", "Homo", "Pan"], source="ott")
    assert list(res["input"]) == [" Homo ", "Homo", "Pan"]
    assert list(res["id"]) == ["ott:770309", "ott:770309", "ott:417957"]
    out = capsys.readouterr().out
    assert "Resolving taxa [1/2]: Homo" in out
    assert "Resolving taxa [2/2]: Pan" in out


def test_resolutions_from_other_sources_feed_every_matrix_function(local_sources, tmp_path):
    res = taxodist.taxo_resolve(["Homo", "Pan", "Mus", "Quercus"], source="ott", progress=False)
    d = taxodist.distance_matrix(res, progress=False)
    assert d.loc["Homo", "Pan"] < d.loc["Homo", "Mus"] < d.loc["Homo", "Quercus"]
    assert isinstance(taxodist.taxo_tree(res, progress=False), taxodist.TaxodistTree)
    ranked = taxodist.distance_matrix(res, progress=False, metric="ranked")
    assert ranked.loc["Homo", "Pan"] < ranked.loc["Homo", "Quercus"]

    bundle = taxodist.taxo_bundle(res, progress=False)
    assert bundle["source"]["name"].startswith("Open Tree Taxonomy")
    path = tmp_path / "bundle.json"
    taxodist.write_taxodist_bundle(bundle, str(path))
    back = taxodist.read_taxodist_bundle(str(path))
    assert back["source"]["name"] == bundle["source"]["name"]
    assert list(back["resolution"]["id"]) == list(res["id"])
    np.testing.assert_array_equal(back["matrix"].to_numpy(), d.to_numpy())

    bundle2 = taxodist.taxo_bundle(["Homo", "Pan"], source="ncbi", progress=False)
    assert bundle2["source"]["name"] == "NCBI Taxonomy"
    assert bundle2["source"]["url"] == "https://www.ncbi.nlm.nih.gov/taxonomy"


def test_failed_requests_are_retrieval_errors(local_sources):
    local_sources(failing_request)
    ott = taxodist.taxo_resolve(["Homo", "770309"], source="ott", progress=False)
    assert list(ott["status"]) == ["retrieval_error", "retrieval_error"]
    assert ott.source == "Open Tree Taxonomy"
    assert ott.source_url == "https://tree.opentreeoflife.org/about/taxonomy-version"
    ncbi = taxodist.taxo_resolve(["Homo", "ncbi:9606"], source="ncbi", progress=False)
    assert list(ncbi["status"]) == ["retrieval_error", "retrieval_error"]


def test_a_candidate_whose_lineage_cannot_be_retrieved_is_a_retrieval_error(local_sources):
    def request(source, method, url, body=None):
        if "taxon_info" in url:
            return {"status": 500, "text": ""}
        return fixture_request(source, method, url, body)

    local_sources(request)
    res = taxodist.taxo_resolve(["Homo"], source="ott", progress=False)
    assert res["status"][0] == "retrieval_error"
    assert list(res["candidates"][0]["id"]) == ["ott:770309"]
    assert res["n_candidates"][0] == 1
    inside = taxodist.taxo_resolve(["Homo"], source="ott", progress=False, within="Metazoa")
    assert inside["status"][0] == "retrieval_error"


def test_within_restricts_candidates_of_external_sources(local_sources):
    plants = taxodist.taxo_resolve(
        ["Homo", "Quercus"], source="ott", progress=False, within="Chloroplastida"
    )
    assert list(plants["status"]) == ["unresolved", "resolved"]
    assert list(plants["n_candidates"]) == [0, 1]


def test_taxo_sources_describes_every_source():
    s = taxodist.taxo_sources()
    assert list(s["id"]) == ["taxonomicon", "ott", "ncbi"]
    assert list(s.columns) == ["id", "name", "url", "id_prefix", "ranks", "licence", "citation"]
    assert all(s["citation"].str.len() > 0)
    assert list(s["id_prefix"]) == ["", "ott:", "ncbi:"]


def test_sources_are_validated():
    with pytest.raises(ValueError, match="Unknown taxonomic source"):
        taxodist.taxo_resolve(["Homo"], source="gbif")
    with pytest.raises(TypeError, match="single string"):
        taxodist.taxo_resolve(["Homo"], source=["ott", "ncbi"])
    with pytest.raises(TypeError, match="single string"):
        taxodist.taxo_resolve(["Homo"], source=None)


def test_source_ranks_follow_the_taxonomicon_convention():
    assert sources._normalize_source_rank(
        ["genus", "no rank", "clade", "cellular root", "", None, "Superfamily",
         "no rank - terminal", " species group "]
    ) == ["Genus", None, "Clade", None, None, None, "Superfamily", None, "Species group"]


def test_nested_candidates_of_the_same_name_are_not_homonyms():
    outer = fetch._outermost_candidates
    assert outer([["a", "Mus"], ["a", "Mus", "Mus (Mus)"], ["b", "Mus"], None]) == [
        True, False, True, True,
    ]
    assert outer([["a", "x"]]) == [True]


def test_the_taxonomicon_ignores_nominotypical_subgenera_when_resolving(monkeypatch):
    fetch.clear_cache()
    monkeypatch.setattr(fetch, "_taxo_search_details", lambda *a, **k: {
        "status": "ok",
        "results": pd.DataFrame({"id": ["2", "1"], "name": ["Mus (Mus)", "Mus"]}),
    })

    def lineage(identifier, **kwargs):
        if identifier == "1":
            return ["Biota", "Rodentia", "Mus"]
        return ["Biota", "Rodentia", "Mus", "Mus (Mus)"]

    monkeypatch.setattr(fetch, "get_lineage_by_id", lineage)
    res = fetch.taxo_resolve(["Mus"], progress=False)
    assert res["status"][0] == "resolved"
    assert res["id"][0] == "1"
    assert res["n_candidates"][0] == 1
    assert res.source == "The Taxonomicon"


def test_the_taxonomicon_id_lookup_ignores_nominotypical_subgenera(monkeypatch):
    fetch.clear_cache()
    search = """
    <table>
      <tr><td>Mus (Mus)</td><td><a class="Valid" href="TaxonTree.aspx?id=2&src=0">t</a></td></tr>
      <tr><td>Mus</td><td><a class="Valid" href="TaxonTree.aspx?id=1&src=0">t</a></td></tr>
    </table>
    """
    response = Mock(status_code=200)
    response.text = search
    monkeypatch.setattr(fetch, "_http_get", lambda url: response)

    def lineage(identifier, **kwargs):
        if identifier == "1":
            return ["Biota", "Rodentia", "Mus"]
        return ["Biota", "Rodentia", "Mus", "Mus (Mus)"]

    monkeypatch.setattr(fetch, "get_lineage_by_id", lineage)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert fetch._lookup_taxonomicon_id("Mus") == ("1", "found")
    fetch.clear_cache()


def test_the_ncbi_request_rate_depends_on_the_api_key(monkeypatch):
    monkeypatch.setattr(fetch, "MIN_INTERVAL", 0)
    monkeypatch.setattr(sources, "NCBI_API_KEY", "")
    assert sources._source_interval("ncbi") == 0.4
    monkeypatch.setattr(sources, "NCBI_API_KEY", "abc")
    assert sources._source_interval("ncbi") == 0.12
    assert "api_key=abc" in sources._ncbi_url("esearch.fcgi", {"db": "taxonomy"})
    assert sources._source_interval("ott") == 0
    monkeypatch.setattr(fetch, "MIN_INTERVAL", "bad")
    assert sources._source_interval("ott") == 0
    monkeypatch.setattr(fetch, "MIN_INTERVAL", 0.5)
    assert sources._source_interval("ncbi") == 0.5
    monkeypatch.setattr(sources, "NCBI_API_KEY", 123)
    assert sources._ncbi_api_key() == ""
    monkeypatch.setattr(sources, "NCBI_API_KEY", None)
    monkeypatch.setenv("NCBI_API_KEY", "fromenv")
    assert sources._ncbi_api_key() == "fromenv"
    monkeypatch.delenv("NCBI_API_KEY")
    assert sources._ncbi_api_key() == ""
    url = sources._ncbi_url("esearch.fcgi", {"term": '"Homo sapiens"[All Names]'})
    assert url == (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        "?term=%22Homo%20sapiens%22%5BAll%20Names%5D&tool=taxodist"
    )


def test_ncbi_merged_identifiers_and_malformed_responses_are_handled():
    fetch.clear_cache()
    sources._ncbi_parse_taxa("not xml")
    sources._ncbi_parse_taxa("<Other><Taxon><TaxId>1</TaxId></Taxon></Other>")
    assert "src_ncbi_lin_1" not in fetch._taxodist_cache
    xml = (
        "<TaxaSet><Taxon><TaxId>2</TaxId><ScientificName>B</ScientificName>"
        "<Rank>genus</Rank><AkaTaxIds><TaxId>3</TaxId></AkaTaxIds>"
        "<LineageEx><Taxon><TaxId>1</TaxId><ScientificName>A</ScientificName>"
        "<Rank>no rank</Rank></Taxon></LineageEx></Taxon></TaxaSet>"
    )
    sources._ncbi_parse_taxa(xml)
    merged = fetch._taxodist_cache["src_ncbi_lin_3"]
    assert merged == {"lineage": ["A", "B"], "ranks": [None, "Genus"]}
    # A node without a rank is unranked.
    sources._ncbi_parse_taxa(
        "<TaxaSet><Taxon><TaxId>5</TaxId><ScientificName>C</ScientificName>"
        "</Taxon></TaxaSet>"
    )
    assert fetch._taxodist_cache["src_ncbi_lin_5"] == {"lineage": ["C"], "ranks": [None]}
    assert sources._parse_json(None) is None
    assert sources._parse_json("{not json") is None
    fetch.clear_cache()


def response(status, content=b""):
    mock = Mock(status_code=status)
    mock.content = content
    return mock


def test_external_requests_are_built_and_throttled_per_source(monkeypatch):
    monkeypatch.setattr(sources, "NCBI_API_KEY", "")
    seen = []
    monkeypatch.setattr(fetch, "_throttle", lambda source, interval: seen.append((source, interval)))
    with patch.object(
        fetch._http_session, "request", return_value=response(200, b'{"a":1}')
    ) as request:
        res = sources._source_request("ott", "POST", "http://example.org", body={"a": 1})
    assert res == {"status": 200, "text": '{"a":1}'}
    assert seen == [("ott", 0)]
    args, kwargs = request.call_args
    assert args == ("POST", "http://example.org")
    assert kwargs["data"] == b'{"a": 1}'
    assert kwargs["headers"] == {"Content-Type": "application/json"}

    with patch.object(
        fetch._http_session, "request", side_effect=requests.exceptions.ConnectionError()
    ) as request:
        assert sources._source_request("ncbi", "GET", "http://example.org") == {
            "status": None, "text": None,
        }
    assert request.call_count == fetch.MAX_TRIES
    assert seen[-1] == ("ncbi", 0.4)

    # Server errors and refused requests (429) are retried; the last response
    # is returned. Other client errors are not retried.
    pauses = []
    monkeypatch.setattr(fetch, "_sleep", pauses.append)
    with patch.object(
        fetch._http_session, "request",
        side_effect=[response(429), response(200, b"ok")],
    ) as request:
        assert sources._source_request("ncbi", "GET", "http://x") == {"status": 200, "text": "ok"}
    assert request.call_count == 2
    assert pauses == [1]
    with patch.object(fetch._http_session, "request", return_value=response(429)) as request:
        assert sources._source_request("ncbi", "GET", "http://x")["status"] == 429
    assert request.call_count == fetch.MAX_TRIES
    with patch.object(
        fetch._http_session, "request",
        side_effect=[response(503), response(200, "é".encode("utf-8"))],
    ) as request:
        assert sources._source_request("ott", "GET", "http://x") == {"status": 200, "text": "é"}
    assert request.call_count == 2
    with patch.object(fetch._http_session, "request", return_value=response(404)) as request:
        assert sources._source_request("ott", "GET", "http://x")["status"] == 404
    assert request.call_count == 1
    with patch.object(fetch._http_session, "request", return_value=response(500)) as request:
        assert sources._source_request("ott", "GET", "http://x")["status"] == 500
    assert request.call_count == fetch.MAX_TRIES


def test_throttling_is_per_source(monkeypatch):
    sleeps = []
    monkeypatch.setattr(fetch, "_throttle_sleep", sleeps.append)
    monkeypatch.setattr(fetch, "_last_request", {})
    fetch._throttle("ncbi", 10)
    fetch._throttle("ott", 10)
    assert sleeps == []
    fetch._throttle("ncbi", 10)
    assert len(sleeps) == 1 and 9 < sleeps[0] <= 10


def test_compare_sources_compares_distances_across_sources(local_sources, capsys):
    cmp = taxodist.compare_sources(
        ["Homo", "Pan", "Mus", "Quercus", "Fakeosaurus", "Homo"],
        sources=["ott", "ncbi"], permutations=99, seed=42, progress=False,
    )
    assert isinstance(cmp, taxodist.TaxodistSourceComparison)
    assert cmp.shared == ["Homo", "Pan", "Mus", "Quercus"]
    assert len(cmp.taxa) == 6
    assert {"status_ott", "id_ncbi", "depth_ncbi"} <= set(cmp.taxa.columns)
    assert list(cmp.taxa["status_ott"]) == [
        "resolved", "resolved", "resolved", "resolved", "unresolved", "resolved",
    ]
    assert cmp.taxa["depth_ott"].isna().tolist() == [False] * 4 + [True, False]
    assert list(cmp.agreement["n_taxa"]) == [4]
    assert cmp.agreement["spearman"][0] > 0.9
    assert 0 < cmp.agreement["p_value"][0] <= 1
    assert list(cmp.matrices["ott"].index) == cmp.shared
    assert cmp.metric == "reciprocal" and cmp.permutations == 99
    text = str(cmp)
    assert "Comparison of taxonomic sources" in text
    assert "Taxa resolved in every source: 4 of 6" in text
    assert "Median lineage depth: ott " in text
    assert repr(cmp) == text
    with pytest.raises(AttributeError):
        cmp.missing

    no_test = taxodist.compare_sources(
        ["Homo", "Pan", "Mus"], sources=["ott", "ncbi"], permutations=0,
        within={"ott": "Metazoa"}, progress=False,
    )
    assert np.isnan(no_test.agreement["p_value"][0])
    shared_within = taxodist.compare_sources(
        ["Homo", "Pan", "Mus"], sources=("ott", "ncbi"), permutations=0,
        within="Metazoa", progress=False,
    )
    assert shared_within.shared == ["Homo", "Pan", "Mus"]
    single = taxodist.compare_sources(
        ["Homo", "Pan", "Mus"], sources=["ott", "ncbi"], permutations=0,
        within=["Metazoa"], progress=False,
    )
    assert single.shared == ["Homo", "Pan", "Mus"]


def test_compare_sources_matches_r(local_sources):
    # Spearman correlations and the taxa table computed by R's
    # compare_sources() on the same fixtures (permutations = 0).
    taxa = ["Homo", "Pan", "Mus", "Quercus", "Drosophila", "Nereis", "Fakeosaurus", "Homo"]
    expected = {"reciprocal": 1.0, "path": 0.95487986864796792, "ranked": 0.94575073060740711}
    for metric, value in expected.items():
        cmp = taxodist.compare_sources(
            taxa, sources=["ott", "ncbi"], permutations=0, metric=metric, progress=False,
        )
        assert cmp.agreement["spearman"][0] == pytest.approx(value, abs=1e-14), metric
        assert np.isnan(cmp.agreement["p_value"][0])
    assert list(cmp.taxa["status_ncbi"]) == [
        "resolved", "resolved", "resolved", "resolved", "ambiguous", "resolved",
        "unresolved", "resolved",
    ]
    assert list(cmp.taxa["id_ott"]) == [
        "ott:770309", "ott:417957", "ott:1068778", "ott:791121", "ott:34907",
        "ott:305738", None, "ott:770309",
    ]
    assert cmp.taxa["depth_ott"].tolist() == [33, 33, 31, 20, 32, 17, pd.NA, 33]
    assert cmp.taxa["depth_ncbi"].tolist() == [30, 30, 29, 19, 35, 15, pd.NA, 30]
    assert cmp.shared == ["Homo", "Pan", "Mus", "Quercus", "Drosophila", "Nereis"]


def test_compare_sources_without_shared_taxa_prints_na(local_sources):
    local_sources(failing_request)
    cmp = taxodist.compare_sources(
        ["Homo", "Pan", "Mus"], sources=["ott", "ncbi"], progress=False
    )
    assert cmp.shared == []
    assert np.isnan(cmp.agreement["spearman"][0])
    assert "Median lineage depth: ott NA, ncbi NA" in str(cmp)


def test_compare_sources_validates_its_arguments(local_sources):
    with pytest.raises(ValueError, match="at least two"):
        taxodist.compare_sources(["a", "b", "c"], sources=["ott"])
    with pytest.raises(ValueError, match="at least two"):
        taxodist.compare_sources(["a", "b", "c"], sources="ott")
    with pytest.raises(ValueError, match="at least two"):
        taxodist.compare_sources(["a", "b", "c"], sources=["ott", "ott"])
    with pytest.raises(ValueError, match="Unknown"):
        taxodist.compare_sources(["a", "b", "c"], sources=["ott", "gbif"])
    with pytest.raises(ValueError, match="at least three"):
        taxodist.compare_sources(["a", "b"], sources=["ott", "ncbi"])
    with pytest.raises(ValueError, match="at least three"):
        taxodist.compare_sources("abc", sources=["ott", "ncbi"])
    with pytest.raises(ValueError, match="non-negative"):
        taxodist.compare_sources(["a", "b", "c"], sources=["ott", "ncbi"], permutations=-1)
    with pytest.raises(ValueError, match="non-negative"):
        taxodist.compare_sources(["a", "b", "c"], sources=["ott", "ncbi"], permutations=True)
    with pytest.raises(TypeError, match="metric"):
        taxodist.compare_sources(["a", "b", "c"], sources=["ott", "ncbi"], metric=None)
    with pytest.raises(ValueError, match="keyed by source"):
        taxodist.compare_sources(
            ["Homo", "Pan", "Mus"], sources=["ott", "ncbi"], within=["a", "b"],
            progress=False,
        )


def test_the_mantel_statistic_handles_degenerate_inputs():
    rng = np.random.default_rng(1)
    a = pd.DataFrame([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    b = pd.DataFrame([[0, np.inf, 2], [np.inf, 0, 3], [2, 3, 0]], dtype=float)
    r, p = sources._mantel(a, b, 10, rng)
    assert np.isnan(r) and np.isnan(p)
    flat = pd.DataFrame([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    assert np.isnan(sources._mantel(a, flat, 10, rng)[0])
    r, p = sources._mantel(a, a, 9, rng)
    assert r == pytest.approx(1)
    assert 0.1 <= p <= 1
    assert np.isnan(sources._spearman(np.array([1.0]), np.array([2.0])))
    # Ties get average ranks, as in R.
    x = np.array([1.0, 2.0, 2.0, 3.0])
    y = np.array([1.0, 3.0, 2.0, 4.0])
    assert sources._spearman(x, y) == pytest.approx(0.9486832980505138)


class FixedPermutations:
    """Stand-in random generator returning given permutations in turn."""

    def __init__(self, perms):
        self.perms = list(perms)

    def permutation(self, n):
        return np.array(self.perms.pop(0))


def test_mantel_p_value_is_one_plus_count_over_one_plus_permutations():
    m = pd.DataFrame(
        [[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]], dtype=float
    )
    # The identity reproduces the observed correlation (counted); the
    # reversal gives a lower one (not counted).
    rng = FixedPermutations([[0, 1, 2, 3], [3, 2, 1, 0], [0, 1, 2, 3]])
    r, p = sources._mantel(m, m, 3, rng)
    assert r == pytest.approx(1)
    assert p == pytest.approx((1 + 2) / (1 + 3))


def test_verbose_resolution_reports_each_request(local_sources, capsys):
    taxodist.taxo_resolve(["Homo", "Pan"], source="ott", progress=False, verbose=True)
    assert "Matching 2 names in the Open Tree Taxonomy" in capsys.readouterr().out
    taxodist.taxo_resolve(["Mus"], source="ott", progress=False, verbose=True)
    out = capsys.readouterr().out
    assert "Matching 1 name in the Open Tree Taxonomy" in out
    assert "Retrieving lineage of ott:1068778" in out
    taxodist.taxo_resolve(["Quercus"], source="ncbi", progress=False, verbose=True)
    assert "Searching NCBI Taxonomy for Quercus" in capsys.readouterr().out
    fetch.clear_cache()
    taxodist.taxo_resolve(["Quercus"], source="ncbi", progress=False, verbose=True)
    assert "Retrieving 1 lineage from NCBI Taxonomy" in capsys.readouterr().out
    fetch.clear_cache()
    taxodist.taxo_resolve(["Quercus", "Mus"], source="ncbi", progress=False, verbose=True)
    assert "Retrieving 3 lineages from NCBI Taxonomy" in capsys.readouterr().out


def test_results_are_cached(local_sources):
    calls = []

    def counting(source, method, url, body=None):
        calls.append(url)
        return fixture_request(source, method, url, body)

    local_sources(counting)
    first = taxodist.taxo_resolve(["Homo", "Mus"], source="ott", progress=False)
    n = len(calls)
    second = taxodist.taxo_resolve(["Homo", "Mus"], source="ott", progress=False)
    assert len(calls) == n
    assert list(first["id"]) == list(second["id"])
    # Cached lineages are private copies.
    second["lineage"][0].append("x")
    third = taxodist.taxo_resolve(["Homo", "ott:770309"], source="ott", progress=False)
    assert third["lineage"][0][-1] == "Homo"
    assert third["lineage"][1][-1] == "Homo"
    calls.clear()
    taxodist.taxo_resolve(["Quercus", "Fakeosaurus", "9606"], source="ncbi", progress=False)
    n = len(calls)
    again = taxodist.taxo_resolve(["Quercus", "Fakeosaurus", "9606"], source="ncbi", progress=False)
    assert len(calls) == n
    assert list(again["status"]) == ["resolved", "unresolved", "resolved"]


def test_missing_metadata_and_failed_lineage_batches_are_tolerated(local_sources):
    def request(source, method, url, body=None):
        if url.endswith("/taxonomy/about"):
            return {"status": 200, "text": '{"version": "9.9", "source": "ott9.9"}'}
        if "efetch.fcgi" in url:
            return {"status": 503, "text": ""}
        return fixture_request(source, method, url, body)

    local_sources(request)
    about = sources._ott_about()
    assert about == {
        "name": "Open Tree Taxonomy 9.9 (ott9.9)",
        "url": "https://tree.opentreeoflife.org/about/taxonomy-version",
    }
    # The name was found but its lineage could not be retrieved.
    res = taxodist.taxo_resolve(["Quercus"], source="ncbi", progress=False)
    assert res["status"][0] == "retrieval_error"
    assert res["n_candidates"][0] == 0


def test_malformed_answers_are_failures(local_sources):
    def request(source, method, url, body=None):
        if url.endswith("/tnrs/match_names"):
            return {"status": 200, "text": "[1, 2]"}
        if "esearch.fcgi" in url:
            return {"status": 200, "text": '{"header": {}}'}
        if url.endswith("/taxonomy/taxon_info"):
            return {"status": 200, "text": '{"lineage": []}'}
        return fixture_request(source, method, url, body)

    local_sources(request)
    ott = taxodist.taxo_resolve(["Homo", "770309"], source="ott", progress=False)
    assert list(ott["status"]) == ["retrieval_error", "retrieval_error"]
    ncbi = taxodist.taxo_resolve(["Homo"], source="ncbi", progress=False)
    assert ncbi["status"][0] == "retrieval_error"


def test_ott_synonyms_and_inexact_matches_come_last(local_sources):
    match = {
        "results": [{
            "name": "Alpha",
            "matches": [
                {"is_synonym": True, "taxon": {"ott_id": 1, "name": "Alpha"}},
                {"is_synonym": False, "taxon": {"ott_id": 2, "name": "Alpha"}},
                {"is_synonym": False, "taxon": {"ott_id": 3, "name": "Beta"}},
            ],
        }],
    }
    lineages = {
        1: {"name": "Alpha", "rank": "genus",
            "lineage": [{"name": "Z", "rank": "family"}, {"name": "life"}]},
        2: {"name": "Alpha", "rank": "genus", "lineage": [{"name": "life", "rank": "no rank"}]},
    }

    def request(source, method, url, body=None):
        if url.endswith("/tnrs/match_names"):
            return {"status": 200, "text": json.dumps(match)}
        if url.endswith("/taxonomy/taxon_info"):
            return {"status": 200, "text": json.dumps(lineages[body["ott_id"]])}
        return {"status": None, "text": None}

    local_sources(request)
    res = taxodist.taxo_resolve(["Alpha"], source="ott", progress=False, ambiguity="first")
    assert res["status"][0] == "ambiguous"
    assert list(res["candidates"][0]["id"]) == ["ott:2", "ott:1"]
    assert res["lineage"][0] == ["life", "Alpha"]
    assert res["ranks"][0] == [None, "Genus"]
    assert res.source == "Open Tree Taxonomy"
    with pytest.raises(ValueError, match="Ambiguous"):
        taxodist.taxo_resolve(["Alpha"], source="ott", progress=False, ambiguity="error")
