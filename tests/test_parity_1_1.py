"""Parity with the R development version (taxodist 1.1.0)."""

import json
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from bs4 import BeautifulSoup

import taxodist
import taxodist.fetch as fetch

FIXTURES = Path(__file__).parent / "fixtures"
RANK_LABEL = re.compile(r"^(Clade|Domain|Kingdom|Phylum|Class|Order|Family)( |$)")


# -- Stable ordering of ties (R's order() is stable) ---------------------------

TIED = {
    name: ["Biota", "Animalia", f"Clade{index % 3}", name]
    for index, name in enumerate(f"T{index:02d}" for index in range(60))
}
TIED["Query"] = ["Biota", "Animalia", "Clade0", "Query"]
TIED["Plant"] = ["Biota", "Plantae", "Plant"]


def _tied_lineage(taxon, **kwargs):
    lineage = TIED.get(taxon)
    return None if lineage is None else list(lineage)


def _stable_expected(candidates):
    distances = {
        name: 1 / 3 if TIED[name][2] == "Clade0" else
        1 / 1 if TIED[name][1] == "Plantae" else 1 / 2
        for name in candidates if name in TIED
    }
    known = sorted(distances, key=lambda name: distances[name])  # stable
    return known + [name for name in candidates if name not in TIED]


def test_closest_relative_keeps_input_order_for_ties():
    candidates = list(reversed([name for name in TIED if name != "Query"]))
    candidates.insert(7, "Missing")
    with patch("taxodist.distance.get_lineage", side_effect=_tied_lineage):
        result = taxodist.closest_relative("Query", candidates)
    assert list(result["taxon"]) == _stable_expected(candidates)
    assert result["taxon"].iloc[-1] == "Missing"


def test_focal_distances_keeps_input_order_for_ties():
    community = [name for name in TIED if name != "Query"][::2] + ["Missing", "Query"]
    with patch("taxodist.distance.get_lineage", side_effect=_tied_lineage):
        result = taxodist.focal_distances("Query", community, progress=False)
    expected = ["Query"] + [
        name for name in _stable_expected(community) if name != "Query"
    ]
    assert list(result["taxon"]) == expected


# -- compare_bundles() reports the metrics -------------------------------------

def _bundle(metric="reciprocal"):
    resolution = taxodist.taxo_from_lineages(
        {"A": ["Biota", "X", "A"], "B": ["Biota", "Y", "B"]},
        ranks={"A": ["domain", "phylum", "genus"], "B": ["domain", "phylum", "genus"]},
    )
    return taxodist.taxo_bundle(resolution, progress=False, metric=metric)


def test_compare_bundles_reports_the_metric_of_each_bundle():
    same = taxodist.compare_bundles(_bundle("path"), _bundle("path"))
    assert same.metric_old == "path" and same.metric_new == "path"
    assert same.metric_changed is False
    assert "!" not in str(same)

    changed = taxodist.compare_bundles(_bundle("reciprocal"), _bundle("path"))
    assert (changed.metric_old, changed.metric_new) == ("reciprocal", "path")
    assert changed.metric_changed is True
    text = str(changed)
    assert '! The bundles use different metrics: "reciprocal" and "path".' in text
    assert repr(changed) == text


def test_compare_bundles_detects_different_rank_levels(monkeypatch, capsys):
    old = _bundle("ranked")
    monkeypatch.setattr(taxodist.distance, "RANK_LEVELS", ["phylum", "genus"])
    new = _bundle("ranked")
    comparison = taxodist.compare_bundles(old, new)
    assert comparison.metric_old == comparison.metric_new == "ranked"
    assert comparison.metric_changed is True
    print(comparison)
    assert (
        '! The bundles use different rank levels for the "ranked" metric.'
        in capsys.readouterr().out
    )
    # Upper-case rank levels in a stored bundle are compared in lower case.
    new["metric"]["rank_levels"] = [level.upper() for level in old["metric"]["rank_levels"]]
    assert taxodist.compare_bundles(old, new).metric_changed is False


# -- Taxon names ------------------------------------------------------------------

def test_taxon_names_treat_unicode_spaces_as_whitespace():
    normalize = fetch._normalize_taxon
    assert normalize(" Homo  sapiens ") == "Homo sapiens"
    assert normalize("\tHomo  sapiens\n") == "Homo sapiens"
    assert normalize("Homo sapiens") == "Homo sapiens"
    assert normalize("　Homo \r\nsapiens ") == "Homo sapiens"
    # Not whitespace for R's [\h\v] either: kept unchanged.
    assert normalize("Homo​sapiens") == "Homo​sapiens"
    assert normalize("Homo\x1fsapiens") == "Homo\x1fsapiens"


# -- Text fallback parser (shared fixture) ---------------------------------------

def test_text_fallback_strips_no_break_space_indentation_shared_fixture():
    # expected_text_fallback.json is shared with the R and Julia
    # implementations: every fixture is parsed with its structured hierarchy
    # block removed, so the plain-text fallback parser is used.
    expected = json.loads(
        (FIXTURES / "expected_text_fallback.json").read_text(encoding="utf-8")
    )
    assert len(expected) > 5
    for taxon_id, record in expected.items():
        html = (FIXTURES / f"tree_{taxon_id}.html").read_text(encoding="utf-8")
        html = html.replace("ctl00_cphTabContent_divHierarchy", "xxHier")
        soup = BeautifulSoup(html, "lxml")
        assert fetch._parse_hierarchy(soup) is None, taxon_id
        lineage, ranks = fetch._lineage_and_ranks_from_page(soup, taxon_id)
        assert lineage == record["lineage"], taxon_id
        assert ranks == record["ranks"], taxon_id
        assert fetch._lineage_from_page(soup, taxon_id) == record["lineage"]
        # Rank labels must never be read as nodes.
        assert not any(RANK_LABEL.match(node) for node in lineage), taxon_id


# -- metric must be a single string --------------------------------------------

@pytest.mark.parametrize("metric", [None, ["path"], ("path",), np.array(["path"]), 1, b"path"])
def test_metric_must_be_a_single_string(metric):
    resolution = taxodist.taxo_from_lineages({"A": ["Biota", "A"], "B": ["Biota", "B"]})
    message = (
        r"metric must be a single string, one of 'reciprocal', 'path', "
        r"'wu_palmer', or 'ranked'"
    )
    for call in (
        lambda: taxodist.distance_matrix(resolution, metric=metric, progress=False),
        lambda: taxodist.cross_distances(resolution, resolution, metric=metric, progress=False),
        lambda: taxodist.taxo_bundle(resolution, metric=metric, progress=False),
        lambda: taxodist.taxo_distance("A", "B", metric=metric),
        lambda: taxodist.closest_relative("A", ["B"], metric=metric),
        lambda: taxodist.focal_distances("A", ["B"], metric=metric),
        lambda: taxodist.taxo_distinctness({"A": 1, "B": 1}, metric=metric),
    ):
        with pytest.raises(TypeError, match=message):
            call()


def test_unknown_metric_names_are_value_errors():
    resolution = taxodist.taxo_from_lineages({"A": ["Biota", "A"]})
    with pytest.raises(ValueError, match="Unknown distance metric 'paths'"):
        taxodist.distance_matrix(resolution, metric="paths", progress=False)
    # NumPy strings are strings.
    assert taxodist.distance_matrix(
        resolution, metric=np.str_("path"), progress=False
    ).attrs["metric"] == "path"


# -- User agent -------------------------------------------------------------------

def test_user_agent_follows_the_shared_format():
    expected = f"taxodist/{taxodist.__version__} (Python; +https://taxodist.github.io)"
    assert fetch.USER_AGENT == expected
    assert fetch._http_session.headers["User-Agent"] == expected
    assert expected == "taxodist/1.1.0 (Python; +https://taxodist.github.io)"


# -- taxo_bundle() source default ------------------------------------------------

def test_bundle_without_a_source_defaults_to_the_taxonomicon_like_r():
    resolution = taxodist.taxo_from_lineages(
        {"A": ["Biota", "A"]}, ids=["custom:A"], source="Curated"
    )
    resolution.source = None
    resolution.source_url = None
    bundle = taxodist.taxo_bundle(resolution, progress=False)
    assert bundle.source["name"] == "The Taxonomicon"
    assert bundle.source["url"] == "http://taxonomicon.taxonomy.nl"

    resolution.source = ""
    resolution.source_url = "https://example.org"
    assert taxodist.taxo_bundle(resolution, progress=False).source["url"] == "https://example.org"

    resolution.source = "Curated"
    resolution.source_url = None
    bundle = taxodist.taxo_bundle(resolution, progress=False)
    assert bundle.source["name"] == "Curated" and bundle.source["url"] is None


# -- Public API -------------------------------------------------------------


def test_all_lists_the_documented_public_api():
    import inspect
    import types

    public = {
        name for name in dir(taxodist)
        if not name.startswith("_")
        and not isinstance(getattr(taxodist, name), types.ModuleType)
    }
    assert set(taxodist.__all__) - {"__version__"} == public
    assert len(taxodist.__all__) == len(set(taxodist.__all__))
    for name in taxodist.__all__:
        obj = getattr(taxodist, name)
        if callable(obj):
            assert inspect.getdoc(obj), name
