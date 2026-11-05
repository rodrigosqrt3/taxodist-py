"""Ranks, alternative distance metrics, schema 1.1 bundles, and taxonomic
distinctness.

Mirrors tests/testthat/test-metrics.R of the R package. Reference values
marked "from R" were generated with the R implementation
(``pkgload::load_all()`` + ``Rscript``) and must be reproduced exactly.
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from bs4 import BeautifulSoup

import taxodist
import taxodist.distance as distance
import taxodist.fetch as fetch
from taxodist.distance import _compute_distance

FIXTURES = Path(__file__).parent / "fixtures"
SCHEMA = Path(taxodist.__file__).parent / "schema" / "taxodist-bundle-v1.schema.json"
METRICS = ["reciprocal", "path", "wu_palmer", "ranked"]
LABELS = ["A", "B", "C", "D", "E"]


def fixture_page(name):
    return BeautifulSoup((FIXTURES / name).read_text(encoding="utf-8"), "lxml")


@pytest.fixture(autouse=True)
def fresh_cache():
    fetch.clear_cache()
    yield
    fetch.clear_cache()


def ranked_example():
    lineages = {
        "A": ["Biota", "Animalia", "Chordata", "Mammalia", "Carnivora", "Felidae", "Panthera", "A"],
        "B": ["Biota", "Animalia", "Chordata", "Mammalia", "Carnivora", "Felidae", "Panthera", "B"],
        "C": ["Biota", "Animalia", "Chordata", "Mammalia", "Carnivora", "Canidae", "Canis", "C"],
        "D": ["Biota", "Animalia", "Chordata", "Mammalia", "Primates", "Hominidae", "Homo", "D"],
        "E": ["Biota", "Animalia", "Arthropoda", "Insecta", "Diptera", "Muscidae", "Musca", "E"],
    }
    standard = [None, "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    return taxodist.taxo_from_lineages(
        lineages, ranks={label: list(standard) for label in lineages}
    )


def matrix(metric, resolution=None):
    resolution = ranked_example() if resolution is None else resolution
    return taxodist.distance_matrix(resolution, metric=metric, progress=False)


# -- Ranks ---------------------------------------------------------------------

EXPECTED_RANKS = json.loads((FIXTURES / "expected_ranks.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("taxon_id", sorted(EXPECTED_RANKS))
def test_structured_parser_reads_the_rank_of_every_node(taxon_id):
    # Shared with R and Julia: JSON null is a missing rank (None).
    page = fixture_page(f"tree_{taxon_id}.html")
    lineage, ranks = fetch._lineage_and_ranks_from_page(page, taxon_id)
    assert ranks == EXPECTED_RANKS[taxon_id]
    assert len(ranks) == len(lineage)


def test_passer_ranks():
    lineage, ranks = fetch._lineage_and_ranks_from_page(
        fixture_page("tree_59107.html"), "59107"
    )
    by_node = dict(zip(lineage, ranks))
    nodes = ["Animalia", "Chordata", "Reptilia", "Aves", "Passeriformes",
             "Passeridae", "Passer", "Neornithes", "Biota"]
    assert [by_node[n] for n in nodes] == [
        "Kingdom", "Phylum", "Class", "Subclass", "Order", "Family", "Genus",
        "Clade", "Clade",
    ]


def test_ranks_are_inferred_from_the_form_of_unlabelled_names():
    assert fetch._rank_from_name("Homo sapiens") == "Species"
    assert fetch._rank_from_name("Homo sapiens sapiens") == "Subspecies"
    assert fetch._rank_from_name("Microtus (Agricola)") == "Subgenus"
    assert fetch._rank_from_name("Natura") is None
    assert fetch._normalize_rank("[plesion] Family †", "X") == "Family"
    assert fetch._normalize_rank("Taxon: Genus †", "Yi") == "Genus"
    assert fetch._normalize_rank('Clade "', "X") == "Clade"


def test_the_text_fallback_parser_also_records_ranks():
    page = BeautifulSoup(
        '<html><body><div id="ctl00_divSubject"><b>Homo sapiens</b></div>'
        '<div id="divPageContent">Natura - nature\nClade Biota Wagner 2004\n'
        "Kingdom Animalia Linnaeus, 1758\n[crown] Clade Theria\n"
        "Genus Homo Linnaeus, 1758\nHomo sapiens Linnaeus, 1758\n</div></body></html>",
        "lxml",
    )
    lineage, ranks = fetch._lineage_and_ranks_from_page(page, "1")
    assert lineage == ["Natura", "Biota", "Animalia", "Theria", "Homo", "Homo sapiens"]
    assert ranks == [None, "Clade", "Kingdom", "Clade", "Genus", "Species"]


def test_get_lineage_by_id_caches_ranks_of_both_variants():
    from unittest.mock import Mock

    html = (FIXTURES / "tree_59107.html").read_text(encoding="utf-8")
    with patch.object(fetch._http_session, "get", return_value=Mock(status_code=200, text=html)):
        cleaned = fetch.get_lineage_by_id("59107")
    raw_ranks = fetch._taxodist_cache["rawranks_59107"]
    assert raw_ranks == EXPECTED_RANKS["59107"]
    assert fetch._taxodist_cache["ranks_59107"] == raw_ranks[-len(cleaned):]
    assert fetch._ranks_by_id("59107") == raw_ranks[-len(cleaned):]


def test_lineage_ranks_returns_nodes_aligned_with_get_lineage():
    cache = fetch._taxodist_cache
    cache["id_Passer"] = "59107"
    cache["lin_59107"] = ["Biota", "Animalia", "Passeridae", "Passer", "Passer domesticus"]
    cache["ranks_59107"] = ["Clade", "Kingdom", "Family", "Genus", "Species"]

    out = taxodist.lineage_ranks("Passer")
    assert list(out.columns) == ["node", "rank", "depth"]
    assert list(out["node"]) == taxodist.get_lineage("Passer")
    assert list(out["rank"]) == ["Clade", "Kingdom", "Family", "Genus"]
    assert list(out["depth"]) == [1, 2, 3, 4]

    # A queried name absent from the page is appended with its implied rank.
    cache["id_Passer montanus"] = "59107"
    out = taxodist.lineage_ranks("Passer montanus")
    assert out["node"].iloc[-1] == "Passer montanus"
    assert out["rank"].iloc[-1] == "Species"
    assert taxodist.lineage_ranks(None) is None


def test_ranks_missing_from_the_cache_are_retrieved_again():
    cache = fetch._taxodist_cache
    cache["lin_7"] = ["Biota", "Old"]
    with patch("taxodist.fetch.get_lineage_by_id", wraps=fetch.get_lineage_by_id) as get:
        with patch.object(fetch, "_http_get", side_effect=fetch.requests.exceptions.ConnectionError):
            assert fetch._ranks_by_id("7") is None
    assert get.call_count == 2
    assert "lin_7" not in cache
    assert fetch._cached_ranks("7", ["Biota", "Old"]) == [None, None]


def test_misaligned_cached_ranks_are_returned_as_missing():
    cache = fetch._taxodist_cache
    cache["lin_7"] = ["Biota", "Old"]
    cache["ranks_7"] = ["Clade"]
    with patch.object(fetch, "_http_get", side_effect=AssertionError("no request expected")):
        assert fetch._ranks_by_id("7") == [None, None]


def test_lineage_ranks_returns_none_for_blank_unknown_or_unavailable_taxa():
    assert taxodist.lineage_ranks("   ") is None
    with patch("taxodist.fetch.get_taxonomicon_id", return_value=None):
        assert taxodist.lineage_ranks("Nothing") is None
    with patch.object(fetch, "_http_get", side_effect=fetch.requests.exceptions.ConnectionError):
        assert fetch._ranks_by_id("7") is None
        assert taxodist.lineage_ranks("7") is None


def test_finalize_keeps_ranks_aligned_with_the_truncated_lineage():
    lineage = ["Biota", "[unranked] x", "Microtus", "Microtus (Microtus)", "Microtus arvalis"]
    ranks = ["Clade", None, "Genus", "Subgenus", "Species"]
    assert fetch._finalize_ranks(ranks, lineage, "Microtus") == ["Clade", "Genus"]
    assert fetch._finalize_ranks(ranks, lineage, "Microtus arvalis") == [
        "Clade", "Genus", "Subgenus", "Species"
    ]


def test_taxo_resolve_stores_cached_ranks():
    cache = fetch._taxodist_cache
    cache["lin_59107"] = ["Biota", "Animalia", "Passer", "Passer domesticus"]
    cache["ranks_59107"] = ["Clade", "Kingdom", "Genus", "Species"]
    with patch("taxodist.fetch._taxo_search_details", return_value={
        "status": "ok", "results": pd.DataFrame([{"id": "59107", "name": "Passer"}]),
    }):
        result = taxodist.taxo_resolve(["Passer", "59107"], progress=False)
    assert result["ranks"].iloc[0] == ["Clade", "Kingdom", "Genus"]
    assert result["ranks"].iloc[1] == ["Clade", "Kingdom", "Genus", "Species"]
    with patch("taxodist.fetch._taxo_search_details",
               return_value={"status": "not_found", "results": None}):
        missing = taxodist.taxo_resolve(["Nothing"], progress=False)
    assert missing["ranks"].iloc[0] is None


def test_taxo_from_lineages_validates_and_stores_ranks():
    res = ranked_example()
    assert "ranks" in res.columns
    assert res["ranks"].iloc[0][1] == "Kingdom"

    plain = taxodist.taxo_from_lineages({"A": ["Biota", "A"]})
    assert plain["ranks"].iloc[0] == [None, None]
    by_position = taxodist.taxo_from_lineages({"A": ["Biota", "A"]}, ranks=[[None, "Genus"]])
    assert by_position["ranks"].iloc[0] == [None, "Genus"]

    with pytest.raises(ValueError, match="one rank"):
        taxodist.taxo_from_lineages({"A": ["Biota", "A"]}, ranks={"A": ["Kingdom"]})
    with pytest.raises(ValueError, match="every lineage name"):
        taxodist.taxo_from_lineages({"A": ["Biota", "A"]}, ranks={"B": [None, "Genus"]})
    with pytest.raises(TypeError, match="list"):
        taxodist.taxo_from_lineages({"A": ["Biota", "A"]}, ranks="x")


# -- Metrics ---------------------------------------------------------------------

# From R: as.matrix(distance_matrix(ranked_example(), metric = m)).
R_MATRICES = {
    "reciprocal": [
        [0, 1 / 7, 0.2, 0.25, 0.5],
        [1 / 7, 0, 0.2, 0.25, 0.5],
        [0.2, 0.2, 0, 0.25, 0.5],
        [0.25, 0.25, 0.25, 0, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0],
    ],
    "path": [
        [0, 2, 6, 8, 12],
        [2, 0, 6, 8, 12],
        [6, 6, 0, 8, 12],
        [8, 8, 8, 0, 12],
        [12, 12, 12, 12, 0],
    ],
    "wu_palmer": [
        [0, 0.125, 0.375, 0.5, 0.75],
        [0.125, 0, 0.375, 0.5, 0.75],
        [0.375, 0.375, 0, 0.5, 0.75],
        [0.5, 0.5, 0.5, 0, 0.75],
        [0.75, 0.75, 0.75, 0.75, 0],
    ],
    "ranked": [
        [0, 1, 3, 4, 6],
        [1, 0, 3, 4, 6],
        [3, 3, 0, 4, 6],
        [4, 4, 4, 0, 6],
        [6, 6, 6, 6, 0],
    ],
}


@pytest.mark.parametrize("metric", METRICS)
def test_every_metric_matrix_equals_r(metric):
    mat = matrix(metric)
    assert list(mat.index) == LABELS and list(mat.columns) == LABELS
    np.testing.assert_array_equal(mat.to_numpy(), np.array(R_MATRICES[metric], dtype=float))
    assert mat.attrs["metric"] == metric


def test_reciprocal_is_ultrametric_and_path_satisfies_the_triangle_inequality():
    rec = matrix("reciprocal").to_numpy()
    pth = matrix("path").to_numpy()
    n = len(rec)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                assert rec[i, k] <= max(rec[i, j], rec[j, k]) + 1e-12
                assert pth[i, k] <= pth[i, j] + pth[j, k] + 1e-12


def test_metrics_treat_ancestors_identical_lineages_and_disconnected_roots():
    anc = ["Biota", "Animalia", "Chordata"]
    desc = ["Biota", "Animalia", "Chordata", "Mammalia", "Homo"]
    other = ["Root2", "X"]
    cd = _compute_distance
    assert cd(anc, desc, metric="path")["distance"] == 2
    assert cd(anc, desc, metric="wu_palmer")["distance"] == pytest.approx(1 - 6 / 8)
    assert cd(desc, desc, metric="wu_palmer")["distance"] == 0
    assert cd(desc, desc, metric="path")["distance"] == 0
    assert cd(anc, other, metric="path")["distance"] == np.inf
    assert cd(anc, other, metric="reciprocal")["distance"] == np.inf
    assert cd(anc, other, metric="wu_palmer")["distance"] == 1
    assert cd(anc, desc, metric="path")["metric"] == "path"
    ranks = [None, "Kingdom", "Phylum"]
    assert cd(anc, other, metric="ranked", ranks_a=ranks, ranks_b=["Clade", "Genus"])[
        "distance"
    ] == np.inf


def test_the_ranked_metric_uses_rank_levels(monkeypatch):
    monkeypatch.setattr(distance, "RANK_LEVELS", ["Kingdom", "phylum", "class", "order"])
    ranked = matrix("ranked")
    # Reduced lineages: root, Kingdom, Phylum, Class, Order, terminal.
    assert ranked.loc["A", "B"] == 1
    assert ranked.loc["A", "C"] == 1
    assert ranked.loc["A", "D"] == 2
    bundle = taxodist.taxo_bundle(ranked_example(), metric="ranked", progress=False)
    assert bundle.metric["rank_levels"] == ["kingdom", "phylum", "class", "order"]
    for bad in ([], "kingdom", [None]):
        monkeypatch.setattr(distance, "RANK_LEVELS", bad)
        with pytest.raises(ValueError, match="RANK_LEVELS"):
            matrix("ranked")


def test_the_ranked_metric_keeps_taxa_of_different_kingdoms_connected():
    res = taxodist.taxo_from_lineages(
        {
            "Homo": ["Biota", "Animalia", "Chordata", "Homo"],
            "Quercus": ["Biota", "Plantae", "Tracheophyta", "Quercus"],
        },
        ranks={
            "Homo": ["Clade", "Kingdom", "Phylum", "Genus"],
            "Quercus": ["Clade", "Kingdom", "Phylum", "Genus"],
        },
    )
    assert matrix("ranked", res).loc["Homo", "Quercus"] == 3


def test_anamorph_ranks_count_by_their_last_word():
    assert distance._project_lineage(
        ["Biota", "Fungi", "X", "Epidermophyton", "E. floccosum"],
        [None, "Kingdom", "Clade", "Anamorph genus", "Species"],
        distance._rank_levels(),
    ) == ["Biota", "Fungi", "Epidermophyton", "E. floccosum"]
    # As in R, an empty lineage is returned unchanged. (_metric_lineages()
    # rejects empty lineages as unranked before projecting them.)
    assert distance._project_lineage([], [], distance._rank_levels()) == []


def test_missing_ranks_cells_of_a_resolution_count_as_no_ranks():
    res = ranked_example()
    res.at[0, "ranks"] = np.nan
    with pytest.raises(ValueError, match="'A'"):
        matrix("ranked", res)
    np.testing.assert_array_equal(
        matrix("path", res).to_numpy(), matrix("path").to_numpy()
    )


def test_taxa_given_as_a_dictionary_are_rejected():
    with pytest.raises(TypeError, match="not a dictionary"):
        taxodist.closest_relative("A", {"B": 1})


def test_the_ranked_metric_requires_ranks():
    plain = taxodist.taxo_from_lineages({"A": ["Biota", "A"], "B": ["Biota", "B"]})
    with pytest.raises(ValueError, match="requires the rank"):
        matrix("ranked", plain)
    with pytest.raises(ValueError, match="Unknown distance metric"):
        matrix("nope", plain)
    with pytest.raises(TypeError):
        matrix(None, plain)


def test_metric_is_honoured_by_the_pairwise_and_analysis_functions():
    res = ranked_example()
    for metric in ("path", "ranked"):
        cross = taxodist.cross_distances(
            res.iloc[:2], res.iloc[2:], metric=metric, progress=False
        )
        full = matrix(metric)
        np.testing.assert_array_equal(cross.to_numpy(), full.iloc[:2, 2:].to_numpy())
        assert cross.attrs["metric"] == metric

    cl = taxodist.taxo_cluster(res, metric="path", progress=False)
    assert cl["dist"].attrs["metric"] == "path"
    ord_ = taxodist.taxo_ordinate(res, metric="wu_palmer", progress=False)
    assert ord_["dist"].attrs["metric"] == "wu_palmer"


RANKED_LINEAGES = {
    "Tyrannosaurus": (["Biota", "Dinosauria", "Theropoda", "Tyrannosaurus"],
                      ["Clade", "Clade", "Order", "Genus"]),
    "Velociraptor": (["Biota", "Dinosauria", "Theropoda", "Velociraptor"],
                     ["Clade", "Clade", "Order", "Genus"]),
    "Triceratops": (["Biota", "Dinosauria", "Ornithischia", "Triceratops"],
                    ["Clade", "Clade", "Order", "Genus"]),
}


def fake_lineage(taxon, verbose=False):
    return list(RANKED_LINEAGES[taxon][0])


def fake_lineage_ranks(taxon, verbose=False):
    nodes, ranks = RANKED_LINEAGES[taxon]
    return pd.DataFrame({"node": nodes, "rank": ranks, "depth": range(1, len(nodes) + 1)})


@patch("taxodist.distance.lineage_ranks", side_effect=fake_lineage_ranks)
@patch("taxodist.distance.get_lineage", side_effect=fake_lineage)
def test_taxo_distance_closest_relative_and_focal_distances_accept_a_metric(_get, _ranks):
    assert taxodist.taxo_distance("Tyrannosaurus", "Triceratops", metric="path")["distance"] == 4
    assert taxodist.taxo_distance("Tyrannosaurus", "Velociraptor", metric="ranked")["distance"] == 1
    result = taxodist.taxo_distance("Tyrannosaurus", "Triceratops", metric="ranked")
    assert result["distance"] == 2
    assert result["mrca"] == "Dinosauria" and result["metric"] == "ranked"

    cr = taxodist.closest_relative(
        "Tyrannosaurus", ["Triceratops", "Velociraptor"], metric="path"
    )
    assert list(cr["taxon"]) == ["Velociraptor", "Triceratops"]
    assert list(cr["distance"]) == [2, 4]

    fd = taxodist.focal_distances(
        "Tyrannosaurus", ["Triceratops", "Velociraptor"], metric="wu_palmer", progress=False
    )
    assert list(fd["distance"]) == pytest.approx([1 - 6 / 8, 1 - 4 / 8])

    names = taxodist.distance_matrix(
        ["Tyrannosaurus", "Velociraptor", "Triceratops"], metric="ranked", progress=False
    )
    assert names.loc["Tyrannosaurus", "Triceratops"] == 2
    with pytest.raises(ValueError, match="Unknown"):
        taxodist.taxo_distance("Tyrannosaurus", "Triceratops", metric="bad")


@patch("taxodist.distance.lineage_ranks", return_value=None)
def test_ranked_taxo_distance_returns_none_when_ranks_cannot_be_retrieved(_ranks, capsys):
    assert taxodist.taxo_distance("Nothing", "Tyrannosaurus", metric="ranked") is None
    assert "Could not retrieve lineage for Nothing" in capsys.readouterr().out


def test_print_taxodist_result_shows_the_metric(capsys):
    taxodist.print_taxodist_result(
        _compute_distance(["Biota", "A"], ["Biota", "B"], "A", "B", metric="path")
    )
    assert "Distance : 2.0 (path)" in capsys.readouterr().out


# -- Bundles (schema 1.1) -----------------------------------------------------------

@pytest.mark.parametrize("metric", METRICS)
def test_bundles_record_the_metric_and_ranks_and_survive_a_json_round_trip(tmp_path, metric):
    res = ranked_example()
    bundle = taxodist.taxo_bundle(res, metric=metric, progress=False)
    assert bundle.schema_version == "1.1"
    assert bundle.matrix.attrs["metric"] == metric
    path = tmp_path / f"{metric}.json"
    taxodist.write_taxodist_bundle(bundle, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(raw, json.loads(SCHEMA.read_text(encoding="utf-8")))
    back = taxodist.read_taxodist_bundle(path)
    assert back.metric["name"] == bundle.metric["name"]
    assert back.matrix.attrs["metric"] == metric
    np.testing.assert_array_equal(back.matrix.to_numpy(), bundle.matrix.to_numpy())
    assert back.resolution["ranks"].iloc[0] == res["ranks"].iloc[0]


def test_ranked_bundle_records_the_default_rank_levels():
    bundle = taxodist.taxo_bundle(ranked_example(), metric="ranked", progress=False)
    assert bundle.metric["name"] == "ranked_path_length"
    assert bundle.metric["rank_levels"] == list(distance._DEFAULT_RANK_LEVELS)
    assert list(bundle.metric) == [
        "name", "definition", "root_depth", "common_ancestor", "rank_levels"
    ]


def test_bundle_validation_recomputes_distances_with_the_stored_metric(tmp_path):
    bundle = taxodist.taxo_bundle(ranked_example(), metric="path", progress=False)
    path = tmp_path / "bundle.json"
    taxodist.write_taxodist_bundle(bundle, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["metric"]["name"] = "inverse_mrca_depth"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="stored distances"):
        taxodist.read_taxodist_bundle(path)
    raw["metric"]["name"] = "unknown_metric"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown distance metric"):
        taxodist.read_taxodist_bundle(path)
    raw["metric"] = {"name": "ranked_path_length"}
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="rank_levels"):
        taxodist.read_taxodist_bundle(path)


def test_a_single_rank_level_written_as_a_json_string_is_read(tmp_path, monkeypatch):
    default_levels = distance.RANK_LEVELS
    monkeypatch.setattr(distance, "RANK_LEVELS", ["kingdom"])
    bundle = taxodist.taxo_bundle(ranked_example(), metric="ranked", progress=False)
    monkeypatch.setattr(distance, "RANK_LEVELS", default_levels)
    path = tmp_path / "bundle.json"
    taxodist.write_taxodist_bundle(bundle, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    # R (jsonlite, auto_unbox) writes a length-one rank_levels as a string.
    raw["metric"]["rank_levels"] = "Kingdom"
    path.write_text(json.dumps(raw), encoding="utf-8")
    # Reading recomputes the distances with the stored level only; with the
    # default levels they would not match the stored matrix.
    back = taxodist.read_taxodist_bundle(path)
    np.testing.assert_array_equal(back.matrix.to_numpy(), bundle.matrix.to_numpy())
    assert not np.array_equal(back.matrix.to_numpy(), matrix("ranked").to_numpy())


def test_bundle_validation_checks_rank_alignment():
    bundle = taxodist.taxo_bundle(ranked_example(), progress=False)
    bundle.resolution.at[0, "ranks"] = ["Kingdom"]
    with pytest.raises(ValueError, match="lineage ranks"):
        taxodist.validate_taxodist_bundle(bundle, check_distances=False)


def test_schema_1_0_bundles_are_still_read_and_must_be_reciprocal(tmp_path):
    res = taxodist.taxo_from_lineages({"A": ["Biota", "X", "A"], "B": ["Biota", "Y", "B"]})
    path = tmp_path / "old.json"
    taxodist.write_taxodist_bundle(taxodist.taxo_bundle(res, progress=False), path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = "1.0"
    for record in raw["taxa"]:
        record.pop("lineage_ranks")
    path.write_text(json.dumps(raw), encoding="utf-8")
    old = taxodist.read_taxodist_bundle(path)
    assert old.schema_version == "1.0"
    assert "ranks" not in old.resolution.columns
    # Written again, a bundle without ranks has no lineage_ranks field.
    again = tmp_path / "again.json"
    taxodist.write_taxodist_bundle(old, again)
    assert "lineage_ranks" not in json.loads(again.read_text(encoding="utf-8"))["taxa"][0]

    raw["metric"]["name"] = "path_length"
    raw["matrix"]["values"] = [[0, 4], [4, 0]]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="only the reciprocal"):
        taxodist.read_taxodist_bundle(path)

    raw["schema_version"] = "2.0"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        taxodist.read_taxodist_bundle(path)


def test_bundle_json_keeps_none_ranks_as_null(tmp_path):
    path = tmp_path / "bundle.json"
    taxodist.write_taxodist_bundle(taxodist.taxo_bundle(ranked_example(), progress=False), path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["taxa"][0]["lineage_ranks"][0] is None
    assert raw["taxa"][0]["lineage_ranks"][1] == "Kingdom"
    assert list(raw["taxa"][0])[-1] == "lineage_ranks"


def test_missing_lineages_write_null_ranks(tmp_path):
    with patch("taxodist.fetch._taxo_search_details",
               return_value={"status": "not_found", "results": None}):
        missing = taxodist.taxo_resolve(["Nothing"], progress=False)
    resolution = pd.concat([ranked_example().iloc[:1], missing], ignore_index=True)
    resolution.source = "test"
    path = tmp_path / "bundle.json"
    taxodist.write_taxodist_bundle(taxodist.taxo_bundle(resolution, progress=False), path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["taxa"][1]["lineage_ranks"] is None
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(raw, json.loads(SCHEMA.read_text(encoding="utf-8")))


def test_bundle_written_by_r_is_read_and_reproduced():
    # tests/fixtures/bundle_ranked_r.json was written by R's
    # write_taxodist_bundle(taxo_bundle(ranked_example(), metric = "ranked")).
    bundle = taxodist.read_taxodist_bundle(FIXTURES / "bundle_ranked_r.json")
    assert bundle.software["language"] == "R"
    assert bundle.schema_version == "1.1"
    assert bundle.matrix.attrs["metric"] == "ranked"
    np.testing.assert_array_equal(
        bundle.matrix.to_numpy(), np.array(R_MATRICES["ranked"], dtype=float)
    )
    assert bundle.resolution["ranks"].iloc[0] == ranked_example()["ranks"].iloc[0]
    # Recomputed from the stored lineages and ranks.
    recomputed = taxodist.distance_matrix(bundle.resolution, metric="ranked", progress=False)
    np.testing.assert_array_equal(recomputed.to_numpy(), bundle.matrix.to_numpy())
    # And under the other metrics, the R reference matrices.
    for metric in ("reciprocal", "path", "wu_palmer"):
        other = taxodist.distance_matrix(bundle.resolution, metric=metric, progress=False)
        np.testing.assert_array_equal(other.to_numpy(), np.array(R_MATRICES[metric], dtype=float))


# -- Taxonomic distinctness --------------------------------------------------------

def as_matrix(values, labels):
    return pd.DataFrame(np.array(values, dtype=float), index=labels, columns=labels)


def test_taxo_distinctness_reproduces_the_hand_computed_indices(capsys):
    d = as_matrix([[0, 1, 3], [1, 0, 3], [3, 3, 0]], ["a", "b", "c"])
    out = taxodist.taxo_distinctness({"a": 2, "b": 1, "c": 1}, d)
    assert list(out.columns) == [
        "sample", "S", "Delta", "Delta_star", "Delta_plus", "Lambda_plus", "SDelta_plus"
    ]
    row = out.iloc[0]
    assert row["sample"] == "sample" and row["S"] == 3
    assert row["Delta"] == pytest.approx((1 * 2 + 3 * 2 + 3 * 1) / 6)
    assert row["Delta_star"] == pytest.approx((1 * 2 + 3 * 2 + 3 * 1) / 5)
    assert row["Delta_plus"] == pytest.approx(7 / 3)
    assert row["Lambda_plus"] == pytest.approx(np.mean((np.array([1, 3, 3]) - 7 / 3) ** 2))
    assert row["SDelta_plus"] == pytest.approx(7)

    presence = taxodist.taxo_distinctness(["a", "b", "c"], d)
    assert presence["Delta"].iloc[0] == pytest.approx(presence["Delta_plus"].iloc[0])
    assert presence["Delta_star"].iloc[0] == pytest.approx(presence["Delta_plus"].iloc[0])
    series = taxodist.taxo_distinctness(pd.Series({"a": 2, "b": 1, "c": 1}), d)
    pd.testing.assert_frame_equal(series, out)
    assert taxodist.print_taxodist_distinctness(presence) is presence
    assert "Taxonomic distinctness" in capsys.readouterr().out


# From R: taxo_distinctness(comm, distance_matrix(ranked_example(), metric =
# "ranked")), which equals vegan::taxondive().
R_DISTINCTNESS = {
    "S": [4, 3, 5],
    "Delta": [2.6727272727272728, 2.5714285714285716, 4.2999999999999998],
    "Delta_star": [3.5853658536585367, 3.8571428571428572, 4.2999999999999998],
    "Delta_plus": [4.166666666666667, 3.6666666666666665, 4.2999999999999998],
    "Lambda_plus": [3.8055555555555554, 0.22222222222222224, 2.6099999999999999],
    "SDelta_plus": [16.666666666666668, 11, 21.5],
}


def test_taxo_distinctness_equals_r_for_the_three_site_community():
    comm = pd.DataFrame(
        {"A": [5, 1, 1], "B": [3, 0, 1], "C": [1, 2, 1], "D": [0, 4, 1], "E": [2, 0, 1]},
        index=["s1", "s2", "s3"],
    )
    out = taxodist.taxo_distinctness(comm, matrix("ranked"))
    assert list(out["sample"]) == ["s1", "s2", "s3"]
    assert out.attrs["metric"] == "ranked"
    for column, values in R_DISTINCTNESS.items():
        np.testing.assert_allclose(out[column].to_numpy(dtype=float), values, rtol=1e-14, atol=0)


def test_taxo_distinctness_handles_small_samples_data_frames_and_bad_input():
    d = as_matrix([[0, 2], [2, 0]], ["a", "b"])
    one = taxodist.taxo_distinctness({"a": 3, "b": 0}, d)
    assert one["S"].iloc[0] == 1
    assert np.isnan(one["Delta_plus"].iloc[0])

    frame = pd.DataFrame({"a": [1, 0], "b": [1, 1]}, index=["x", "y"])
    out = taxodist.taxo_distinctness(frame, d)
    assert list(out["sample"]) == ["x", "y"]
    assert out["Delta_plus"].iloc[0] == 2 and np.isnan(out["Delta_plus"].iloc[1])
    unnamed = taxodist.taxo_distinctness(pd.DataFrame({"a": [1, 1], "b": [1, 0]}), d)
    assert list(unnamed["sample"]) == ["sample1", "sample2"]
    flags = taxodist.taxo_distinctness(pd.DataFrame({"a": [True], "b": [True]}), d)
    assert flags["sample"].iloc[0] == "sample" and flags["Delta_plus"].iloc[0] == 2

    inf = as_matrix([[0, np.inf], [np.inf, 0]], ["a", "b"])
    assert np.isnan(taxodist.taxo_distinctness(["a", "b"], inf)["Delta_plus"].iloc[0])

    with pytest.raises(ValueError, match="no distances"):
        taxodist.taxo_distinctness(["a", "z"], d)
    with pytest.raises(ValueError, match="named by taxon"):
        taxodist.taxo_distinctness([1, 2], d)
    with pytest.raises(ValueError, match="non-negative"):
        taxodist.taxo_distinctness({"a": -1, "b": 1}, d)
    with pytest.raises(TypeError, match="must be"):
        taxodist.taxo_distinctness([[1]], d)
    with pytest.raises(TypeError):
        taxodist.taxo_distinctness(["a", "b"], d.to_numpy())


@pytest.mark.parametrize(
    "comm, error, message",
    [
        (ranked_example(), TypeError, "comm must be"),
        ({"a": "x", "b": 1}, TypeError, "comm must be"),
        ([], TypeError, "comm must be"),
        (b"ab", TypeError, "comm must be"),
        (5, TypeError, "comm must be"),
        (["a", " "], ValueError, "missing or empty"),
        (pd.DataFrame([[1, 1]], columns=["a", "a"]), ValueError, "unique taxa"),
        (pd.DataFrame([[1, 1]], columns=[0, 1]), ValueError, "unique taxa"),
        (pd.DataFrame({"a": ["x"], "b": [1]}), TypeError, "comm must be"),
    ],
)
def test_taxo_distinctness_rejects_malformed_samples(comm, error, message):
    d = as_matrix([[0, 2], [2, 0]], ["a", "b"])
    with pytest.raises(error, match=message):
        taxodist.taxo_distinctness(comm, d)


def test_print_taxodist_distinctness_shows_the_metric_and_accepts_none(capsys):
    out = taxodist.taxo_distinctness(["A", "B", "C"], matrix("ranked"))
    assert taxodist.print_taxodist_distinctness(out) is out
    assert "Metric: ranked" in capsys.readouterr().out
    assert taxodist.print_taxodist_distinctness(None) is None
    assert capsys.readouterr().out == ""


def test_taxo_distinctness_computes_distances_when_d_is_not_supplied():
    full = matrix("ranked")

    def fake_matrix(taxa, verbose, progress, metric):
        assert metric == "ranked"
        result = full.loc[taxa, taxa].copy()
        result.attrs["metric"] = metric
        return result

    expected = taxodist.taxo_distinctness(["A", "B", "E"], full)
    with patch("taxodist.distinctness.distance_matrix", side_effect=fake_matrix):
        got = taxodist.taxo_distinctness(["A", "B", "E"])
    assert got["Delta_plus"].iloc[0] == expected["Delta_plus"].iloc[0]
    assert got.attrs["metric"] == "ranked"


def test_taxo_distinctness_test_compares_a_sample_with_random_draws(capsys):
    pool = matrix("ranked")
    test = taxodist.taxo_distinctness_test(["A", "B", "C"], pool, n_iter=499, seed=42)
    assert isinstance(test, taxodist.TaxodistDistinctnessTest)
    stats = test.statistics
    assert list(stats.columns) == [
        "statistic", "observed", "expected", "lower", "upper", "p_lower", "p_upper"
    ]
    assert list(stats["statistic"]) == ["Delta_plus", "Lambda_plus"]
    # From R (deterministic parts): observed values and the expected Delta+.
    assert stats["observed"].iloc[0] == pytest.approx(2.33333333333333348, rel=1e-15)
    assert stats["observed"].iloc[1] == pytest.approx(0.88888888888888884, rel=1e-15)
    assert stats["expected"].iloc[0] == pytest.approx(4.2999999999999998, rel=1e-15)
    assert (stats["lower"] <= stats["upper"]).all()
    assert ((stats["p_lower"] > 0) & (stats["p_lower"] <= 1)).all()
    assert ((stats["p_upper"] > 0) & (stats["p_upper"] <= 1)).all()
    assert test.simulated.shape == (499, 2)
    assert test.S == 3 and test.pool_size == 5 and test.n_iter == 499
    assert test.metric == "ranked"
    with pytest.raises(AttributeError, match="missing_field"):
        test.missing_field
    assert not hasattr(test, "missing_field")
    # The three closest taxa form the least distinct sample.
    assert stats["p_lower"].iloc[0] <= 0.2

    sim = test.simulated
    tol = np.sqrt(np.finfo(float).eps)
    assert stats["expected"].iloc[1] == pytest.approx(sim[:, 1].mean())
    assert list(stats["lower"]) == list(np.quantile(sim, 0.025, axis=0, method="linear"))
    assert stats["p_lower"].iloc[0] == (1 + (sim[:, 0] <= stats["observed"].iloc[0] + tol).sum()) / 500
    assert stats["p_upper"].iloc[1] == (1 + (sim[:, 1] >= stats["observed"].iloc[1] - tol).sum()) / 500

    a = taxodist.taxo_distinctness_test(["A", "E"], pool, n_iter=50, seed=1)
    b = taxodist.taxo_distinctness_test(["A", "E"], pool, n_iter=50, seed=np.random.default_rng(1))
    np.testing.assert_array_equal(a.simulated, b.simulated)
    pd.testing.assert_frame_equal(a.statistics, b.statistics)

    taxodist.print_taxodist_distinctness(test)
    out = capsys.readouterr().out
    assert "3 taxa vs 499 random samples from a pool of 5 (metric: ranked)" in out


def test_r_quantile_type_7_is_reproduced():
    # From R: quantile(c(1, 2, 3, 4, 10), c(0.025, 0.975)) = 1.1, 9.4
    values = np.array([1, 2, 3, 4, 10], dtype=float)
    assert np.quantile(values, [0.025, 0.975], method="linear") == pytest.approx([1.1, 9.4])


def test_taxo_distinctness_test_validates_its_input():
    pool = matrix("ranked")
    with pytest.raises(ValueError, match="master list"):
        taxodist.taxo_distinctness_test(["A", "Z"], pool)
    with pytest.raises(ValueError, match="between 2"):
        taxodist.taxo_distinctness_test("A", pool)
    with pytest.raises(TypeError, match="distance matrix"):
        taxodist.taxo_distinctness_test(["A", "B"], pool.to_numpy())
    with pytest.raises(ValueError, match="n_iter"):
        taxodist.taxo_distinctness_test(["A", "B"], pool, n_iter=0)
    with pytest.raises(ValueError, match="n_iter"):
        taxodist.taxo_distinctness_test(["A", "B"], pool, n_iter=True)
    with pytest.raises(TypeError, match="character"):
        taxodist.taxo_distinctness_test([1, 2], pool)
    with pytest.raises(TypeError, match="character"):
        taxodist.taxo_distinctness_test(5, pool)
    with pytest.raises(TypeError, match="character"):
        taxodist.taxo_distinctness_test(b"AB", pool)
    inf = as_matrix([[0, np.inf], [np.inf, 0]], ["a", "b"])
    with pytest.raises(ValueError, match="finite"):
        taxodist.taxo_distinctness_test(["a", "b"], inf)
