"""Regression tests for Python-specific issues found in the 0.9.0 review."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import taxodist
import taxodist.fetch as fetch
from taxodist.distance import _matrix_input

ROOT = Path(__file__).resolve().parents[1]

LINEAGES = {
    "Alpha": ["Biota", "Animalia", "Chordata", "Alpha"],
    "Beta": ["Biota", "Animalia", "Chordata", "Beta"],
    "Gamma": ["Biota", "Animalia", "Arthropoda", "Gamma"],
}


@pytest.fixture(autouse=True)
def fresh_cache():
    fetch.clear_cache()
    yield
    fetch.clear_cache()


def fake_get_lineage(taxon, **kwargs):
    lineage = LINEAGES.get(taxon)
    return None if lineage is None else list(lineage)


# (1) Resolutions are routed through distance_matrix() --------------------------

def test_cluster_and_ordinate_accept_resolutions_and_bundles():
    resolution = taxodist.taxo_from_lineages(LINEAGES)
    expected = taxodist.distance_matrix(resolution, progress=False)

    clustered = taxodist.taxo_cluster(resolution, progress=False)
    pd.testing.assert_frame_equal(clustered["dist"], expected)
    assert clustered["hclust"].shape == (2, 4)

    ordination = taxodist.taxo_ordinate(resolution, progress=False)
    pd.testing.assert_frame_equal(ordination["dist"], expected)
    assert list(ordination["points"].index) == list(LINEAGES)

    bundle = taxodist.taxo_bundle(resolution, progress=False)
    pd.testing.assert_frame_equal(
        taxodist.taxo_cluster(bundle)["dist"], expected
    )


# (2) Cached lineages are not shared with callers --------------------------------

def test_mutating_returned_lineages_does_not_corrupt_the_cache():
    fetch._taxodist_cache.update({
        "id_Alpha": "1",
        "lin_1": ["Biota", "Animalia", "Alpha"],
        "rawlin_1": ["Natura", "Biota", "Animalia", "Alpha"],
    })
    by_id = fetch.get_lineage_by_id("1")
    by_id.append("Mutated")
    raw = fetch.get_lineage_by_id("1", clean=False)
    raw.clear()
    by_number = fetch.get_lineage("1")
    by_number[0] = "Mutated"
    by_name = fetch.get_lineage("Alpha")
    by_name.append("Mutated")

    assert fetch._taxodist_cache["lin_1"] == ["Biota", "Animalia", "Alpha"]
    assert fetch._taxodist_cache["rawlin_1"] == ["Natura", "Biota", "Animalia", "Alpha"]
    assert fetch.get_lineage("Alpha") == ["Biota", "Animalia", "Alpha"]


def test_fetched_lineages_are_cached_as_private_copies():
    page = (
        '<html><body><div id="ctl00_divSubject"><b>Foo</b></div>'
        '<div id="divPageContent">Natura\nBiota\nFoo\n</div></body></html>'
    )
    response = type("Response", (), {"status_code": 200, "text": page})()
    with patch.object(fetch._http_session, "get", return_value=response):
        lineage = fetch.get_lineage_by_id("5")
    lineage.append("Mutated")
    assert fetch._taxodist_cache["lin_5"] == ["Biota", "Foo"]


# (3) A single string is one taxon; iterables are materialised -------------------

def test_check_coverage_accepts_a_scalar_and_generators():
    with patch(
        "taxodist.distance._lookup_taxonomicon_id",
        side_effect=lambda taxon, **kwargs: ("1", "found"),
    ) as lookup:
        scalar = taxodist.check_coverage("Homo")
        generated = taxodist.check_coverage(name for name in ["Homo", "Pan"])
    assert list(scalar.index) == ["Homo"]
    assert list(generated.index) == ["Homo", "Pan"]
    assert generated.all()
    assert lookup.call_count == 3


@patch("taxodist.distance.get_lineage", side_effect=fake_get_lineage)
def test_scalar_candidates_and_community(mock_get_lineage):
    closest = taxodist.closest_relative("Alpha", "Beta")
    assert list(closest["taxon"]) == ["Beta"]
    focal = taxodist.focal_distances("Alpha", "Gamma", progress=False)
    assert list(focal["taxon"]) == ["Gamma"]
    from_generator = taxodist.closest_relative("Alpha", (t for t in ["Beta", "Gamma"]))
    assert list(from_generator["taxon"]) == ["Beta", "Gamma"]


@patch("taxodist.utils.get_lineage", side_effect=fake_get_lineage)
def test_filter_clade_accepts_a_scalar(mock_get_lineage):
    assert taxodist.filter_clade("Alpha", "Chordata") == ["Alpha"]
    assert taxodist.filter_clade(iter(["Alpha", "Gamma"]), "Chordata") == ["Alpha"]


# (4) The bundle schema is shipped ----------------------------------------------

def test_bundle_schema_is_packaged_and_describes_written_bundles(tmp_path):
    schema_path = Path(taxodist.__file__).parent / "schema" / "taxodist-bundle-v1.schema.json"
    assert schema_path.is_file()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    jsonschema = pytest.importorskip("jsonschema")
    bundle = taxodist.taxo_bundle(taxodist.taxo_from_lineages(LINEAGES), progress=False)
    path = tmp_path / "bundle.json"
    taxodist.write_taxodist_bundle(bundle, path)
    jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), schema)


# (5) License metadata ------------------------------------------------------------

def test_license_file_is_present_and_referenced():
    assert "GNU GENERAL PUBLIC LICENSE" in (ROOT / "LICENSE").read_text(
        encoding="utf-8"
    ).upper()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = "GPL-3.0-or-later"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject
    assert "License :: OSI Approved" not in pyproject


# (6) Plotting libraries are imported lazily -----------------------------------

def test_importing_taxodist_does_not_import_plotting_libraries():
    code = (
        "import sys, taxodist\n"
        "loaded = [m for m in ('matplotlib', 'seaborn', 'scipy') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=str(ROOT))


# (7) Provenance of concatenated resolutions ------------------------------------

def test_concat_keeps_shared_provenance_and_bundle_does_not_mislabel():
    first = taxodist.taxo_from_lineages({"Alpha": LINEAGES["Alpha"]}, source="Curated")
    second = taxodist.taxo_from_lineages({"Beta": LINEAGES["Beta"]}, source="Curated")
    combined = pd.concat([first, second], ignore_index=True)
    assert isinstance(combined, taxodist.TaxodistResolution)
    assert combined.source == "Curated"
    assert combined.retrieved_at == min(first.retrieved_at, second.retrieved_at)
    assert taxodist.taxo_bundle(combined, progress=False).source["name"] == "Curated"

    other = taxodist.taxo_from_lineages({"Gamma": LINEAGES["Gamma"]}, source="Other")
    mixed = pd.concat([first, other], ignore_index=True)
    assert mixed.source is None
    with pytest.raises(ValueError, match="no source provenance"):
        taxodist.taxo_bundle(mixed, progress=False)


# (11) Numeric IDs ------------------------------------------------------------------

def test_integral_numbers_are_taxon_ids():
    fetch._taxodist_cache["lin_50841"] = ["Biota", "Tyrannosaurus"]
    expected = ["Biota", "Tyrannosaurus"]
    with patch("taxodist.fetch.get_taxonomicon_id") as lookup:
        for value in (50841, 50841.0, np.int64(50841), np.float64(50841.0)):
            assert fetch.get_lineage_by_id(value) == expected
            assert fetch.get_lineage(value) == expected
    lookup.assert_not_called()
    assert fetch.get_lineage_by_id(50841.5) is None
    assert fetch.get_lineage_by_id(float("nan")) is None
    assert fetch.get_lineage_by_id(True) is None
    with pytest.raises(TypeError, match="single ID"):
        fetch.get_lineage_by_id(np.array([1, 2]))


# (12) is_member() with a missing clade --------------------------------------------

@patch("taxodist.utils.get_lineage", return_value=["Biota", "None"])
def test_is_member_rejects_a_missing_clade(mock_get_lineage):
    with pytest.raises(TypeError, match="clade"):
        taxodist.is_member("Alpha", None)
    with pytest.raises(TypeError, match="clade"):
        taxodist.filter_clade(["Alpha"], None)


# (13) distance_matrix(bundle) returns a copy ------------------------------------

def test_distance_matrix_of_a_bundle_is_a_copy():
    bundle = taxodist.taxo_bundle(taxodist.taxo_from_lineages(LINEAGES), progress=False)
    matrix = taxodist.distance_matrix(bundle)
    matrix.loc["Alpha", "Beta"] = 42.0
    assert bundle["matrix"].loc["Alpha", "Beta"] == pytest.approx(1 / 3)


# (14) Dictionaries are rejected like R lists ----------------------------------

def test_matrix_input_rejects_dictionaries():
    with pytest.raises(TypeError, match="sequence of taxon names"):
        _matrix_input({"Alpha": 1}, progress=False)
    with pytest.raises(TypeError):
        taxodist.distance_matrix({"Alpha": "Beta"}, progress=False)
    with pytest.raises(TypeError):
        taxodist.cross_distances({"Alpha": 1}, ["Beta"], progress=False)


# (15) focal_distances() dtypes and printer -------------------------------------

@patch("taxodist.distance.get_lineage", side_effect=fake_get_lineage)
def test_focal_distances_uses_nullable_integer_depths(mock_get_lineage, capsys):
    empty = taxodist.focal_distances("Alpha", [], progress=False)
    full = taxodist.focal_distances("Alpha", ["Beta", "Missing"], progress=False)
    complete = taxodist.focal_distances("Alpha", ["Beta"], progress=False)
    assert str(empty["mrca_depth"].dtype) == "Int64"
    assert str(full["mrca_depth"].dtype) == "Int64"
    assert str(complete["mrca_depth"].dtype) == "Int64"
    assert full["mrca_depth"].iloc[0] == 3
    assert full["mrca_depth"].iloc[1] is pd.NA

    assert taxodist.print_focal_distances(full) is full
    assert "Focal distances from Alpha" in capsys.readouterr().out


# (18) Dendrogram subtitle shows the linkage method ----------------------------

@patch("matplotlib.pyplot.show")
def test_cluster_plot_subtitle_names_the_linkage_method(mock_show):
    import matplotlib.pyplot as plt

    resolution = taxodist.taxo_from_lineages(LINEAGES)
    result = taxodist.taxo_cluster(resolution, method="complete", progress=False)
    assert result["method"] == "complete"
    taxodist.plot_taxodist_cluster(result)
    assert plt.gcf()._suptitle.get_text() == "Method: complete"
    plt.close("all")


# Version is defined in one place ------------------------------------------------

def test_version_is_single_sourced():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in pyproject
    assert 'version = {attr = "taxodist._version.__version__"}' in pyproject
    assert taxodist.__version__ == "1.0.0"
