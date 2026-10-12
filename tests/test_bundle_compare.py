"""Bundle validation cost and compare_bundles (0.9.0)."""

import numpy as np
import pytest

import taxodist
from taxodist.bundle import validate_taxodist_bundle


def make_bundle(lineages, **kwargs):
    return taxodist.taxo_bundle(
        taxodist.taxo_from_lineages(lineages, **kwargs), progress=False
    )


def test_using_a_bundle_skips_distance_recomputation(tmp_path):
    bundle = make_bundle({"Alpha": ["Biota", "Alpha"], "Beta": ["Biota", "Beta"]})
    bundle["matrix"].iloc[0, 1] = bundle["matrix"].iloc[1, 0] = 0.75
    assert taxodist.distance_matrix(bundle).iloc[0, 1] == 0.75
    assert validate_taxodist_bundle(bundle, check_distances=False) is bundle
    with pytest.raises(ValueError, match="stored distances"):
        taxodist.write_taxodist_bundle(bundle, tmp_path / "bundle.json")


def test_bundle_records_the_package_version():
    bundle = make_bundle({"Alpha": ["Biota", "Alpha"]})
    assert bundle["software"]["version"] == taxodist.__version__


def test_compare_identical_bundles():
    lineages = {"Alpha": ["Biota", "A", "Alpha"], "Beta": ["Biota", "B", "Beta"]}
    comparison = taxodist.compare_bundles(make_bundle(lineages), make_bundle(lineages))
    assert isinstance(comparison, taxodist.TaxodistBundleComparison)
    assert list(comparison.taxa["change"]) == ["unchanged", "unchanged"]
    assert comparison.distances.empty
    assert "No taxon changed" in str(comparison)
    with pytest.raises(AttributeError):
        comparison.missing_field


def test_compare_bundles_classifies_every_change():
    old = make_bundle(
        {
            "Same": ["Biota", "Animalia", "Same"],
            "Moved": ["Biota", "Animalia", "Moved"],
            "Renumbered": ["Biota", "Animalia", "Renumbered"],
            "Dropped": ["Biota", "Plantae", "Dropped"],
        },
        ids=["1", "2", "3", "4"],
    )
    new = make_bundle(
        {
            "Same": ["Biota", "Animalia", "Same"],
            "Moved": ["Biota", "Plantae", "Moved"],
            "Renumbered": ["Biota", "Animalia", "Renumbered"],
            "Added": ["Biota", "Fungi", "Added"],
        },
        ids=["1", "2", "33", "5"],
    )
    comparison = taxodist.compare_bundles(old, new)
    changes = dict(zip(comparison.taxa["input"], comparison.taxa["change"]))
    assert changes == {
        "Same": "unchanged", "Moved": "lineage_changed", "Renumbered": "id_changed",
        "Dropped": "removed", "Added": "added",
    }
    pairs = set(zip(comparison.distances["taxon_a"], comparison.distances["taxon_b"]))
    assert pairs == {("Same", "Moved"), ("Moved", "Renumbered")}
    assert (comparison.distances["distance_old"] == 0.5).all()
    assert (comparison.distances["distance_new"] == 1).all()
    text = str(comparison)
    assert "Moved: lineage_changed" in text and "Distances changed: 2" in text


def test_compare_bundles_detects_status_changes_and_reads_files(tmp_path):
    resolved = make_bundle({"Alpha": ["Biota", "Alpha"], "Beta": ["Biota", "Beta"]})
    failed = resolved["resolution"].copy()
    failed.at[1, "status"] = "retrieval_error"
    failed.at[1, "id"] = None
    failed.at[1, "resolved_name"] = None
    failed.at[1, "lineage"] = None
    failed["lineage_depth"] = failed["lineage_depth"].astype("Int64")
    failed.at[1, "lineage_depth"] = None
    failed_bundle = taxodist.taxo_bundle(failed, progress=False)
    path = tmp_path / "failed.json"
    taxodist.write_taxodist_bundle(failed_bundle, path)

    comparison = taxodist.compare_bundles(resolved, str(path))
    assert list(comparison.taxa["change"]) == ["unchanged", "status_changed"]
    assert comparison.distances["distance_old"].iloc[0] == 1
    assert np.isnan(comparison.distances["distance_new"].iloc[0])

    assert taxodist.compare_bundles(failed_bundle, path).distances.empty


def test_compare_bundles_validates_inputs():
    bundle = make_bundle({"Alpha": ["Biota", "Alpha"]})
    with pytest.raises(TypeError, match="old"):
        taxodist.compare_bundles({}, bundle)
    with pytest.raises(TypeError, match="new"):
        taxodist.compare_bundles(bundle, 42)
    assert taxodist.compare_bundles(bundle, bundle).distances.empty


def test_compare_bundles_handles_infinite_distances():
    lineages = {"A": ["Root A", "A"], "B": ["Root B", "B"]}
    comparison = taxodist.compare_bundles(make_bundle(lineages), make_bundle(lineages))
    assert comparison.distances.empty
