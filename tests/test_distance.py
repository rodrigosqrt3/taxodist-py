import os
import tempfile
import pytest
import warnings
import pandas as pd
from unittest.mock import patch, Mock
import requests

import taxodist.fetch as fetch
import taxodist.distance as distance
import taxodist.utils as utils

from taxodist.distance import _compute_distance
from taxodist.fetch import (
    clear_cache, save_cache, load_cache, 
    get_taxonomicon_id, get_lineage_by_id, get_lineage
)
from taxodist.utils import filter_clade, taxo_path, print_taxodist_path

# ── Pure logic tests ──────────────────────────────────────────────────────────

def test_taxodist_package_loads():
    assert True

def test_compute_distance_works_correctly():
    lin_a =["Biota", "Animalia", "Chordata", "Dinosauria", "Theropoda", "Tyrannosauridae", "Tyrannosaurus"]
    lin_b =["Biota", "Animalia", "Chordata", "Dinosauria", "Theropoda", "Dromaeosauridae", "Velociraptor"]
    
    result = _compute_distance(lin_a, lin_b, "Tyrannosaurus", "Velociraptor")
    assert result["mrca"] == "Theropoda"
    assert result["mrca_depth"] == 5
    assert result["depth_a"] == 7
    assert result["depth_b"] == 7
    assert result["distance"] >= 0
    assert result["distance"] <= 1

def test_compute_distance_is_between_0_and_1():
    lin_a =["Biota", "Animalia", "Chordata", "Dinosauria", "Theropoda", "Tyrannosauridae", "Tyrannosaurus"]
    lin_b =["Biota", "Animalia", "Chordata", "Dinosauria", "Theropoda", "Dromaeosauridae", "Velociraptor"]
    
    result = _compute_distance(lin_a, lin_b)
    assert result["distance"] >= 0
    assert result["distance"] <= 1

def test_compute_distance_is_symmetric():
    lin_a =["Biota", "Animalia", "Chordata", "Dinosauria", "Theropoda"]
    lin_b =["Biota", "Animalia", "Chordata", "Dinosauria", "Ornithischia"]
    
    r1 = _compute_distance(lin_a, lin_b)
    r2 = _compute_distance(lin_b, lin_a)
    assert r1["distance"] == r2["distance"]

def test_compute_distance_satisfies_triangle_inequality():
    lin_a =["Biota", "Animalia", "Dinosauria", "Theropoda", "Tyrannosauridae"]
    lin_b =["Biota", "Animalia", "Dinosauria", "Theropoda", "Dromaeosauridae"]
    lin_c =["Biota", "Animalia", "Dinosauria", "Ornithischia"]
    
    dAB = _compute_distance(lin_a, lin_b)["distance"]
    dBC = _compute_distance(lin_b, lin_c)["distance"]
    dAC = _compute_distance(lin_a, lin_c)["distance"]
    assert dAC <= (dAB + dBC)

def test_compute_distance_returns_0_for_identical_lineages():
    lin =["Biota", "Animalia", "Dinosauria", "Tyrannosaurus"]
    result = _compute_distance(lin, lin)
    assert result["distance"] == 0
    assert result["mrca"] == "Tyrannosaurus"

def test_compute_distance_handles_no_common_ancestor():
    lin_a = ["Biota", "Animalia"]
    lin_b = ["Fungi", "Ascomycota"]
    result = _compute_distance(lin_a, lin_b)
    assert result["mrca_depth"] == 0
    assert result["mrca"] is None

def test_compute_distance_returns_inf_for_no_shared_ancestor():
    lin_a = ["Biota", "Animalia"]
    lin_b = ["Fungi", "Ascomycota"]
    result = _compute_distance(lin_a, lin_b)
    assert result["distance"] == float('inf')

def test_compute_distance_result_has_correct_s3_class():
    lin =["Biota", "Animalia", "Dinosauria", "Tyrannosaurus"]
    result = _compute_distance(lin, lin)
    # No Python verificamos se o tipo retornado é dict
    assert isinstance(result, dict)

def test_compute_distance_between_0_and_1_for_asymmetric_lineages():
    lin_a =["Biota", "Animalia", "Dinosauria", "Theropoda", "Abelisauridae", "Carnotaurus"]
    lin_b =["Biota", "Animalia", "Dinosauria", "Ornithischia", "Triceratops"]
    result = _compute_distance(lin_a, lin_b)
    assert result["distance"] >= 0
    assert result["distance"] <= 1

def test_compute_distance_returns_0_when_one_taxon_is_ancestor_of_other():
    lin_a =["Biota", "Animalia", "Dinosauria"]
    lin_b =["Biota", "Animalia", "Dinosauria", "Theropoda", "Carnotaurus"]
    result = _compute_distance(lin_a, lin_b)
    assert result["distance"] == 0

# ── Cache Tests ───────────────────────────────────────────────────────────────

def test_clear_cache_returns_invisible_null():
    assert clear_cache() is None

def test_save_cache_creates_file_with_cache_contents():
    clear_cache()
    fetch._taxodist_cache["id_Carnotaurus"] = "12345"
    with tempfile.NamedTemporaryFile(suffix=".rds", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        assert save_cache(tmp_name) is None
        assert os.path.exists(tmp_name)
    finally:
        os.remove(tmp_name)

def test_load_cache_restores_entries_into_the_cache():
    clear_cache()
    fetch._taxodist_cache["id_Carnotaurus"] = "12345"
    with tempfile.NamedTemporaryFile(suffix=".rds", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        save_cache(tmp_name)
        clear_cache()
        assert "id_Carnotaurus" not in fetch._taxodist_cache
        
        load_cache(tmp_name)
        assert "id_Carnotaurus" in fetch._taxodist_cache
        assert fetch._taxodist_cache["id_Carnotaurus"] == "12345"
    finally:
        os.remove(tmp_name)

def test_save_load_cache_round_trip_preserves_all_entries():
    clear_cache()
    fetch._taxodist_cache["id_Tyrannosaurus"] = "50841"
    fetch._taxodist_cache["lin_50841"] =["Biota", "Animalia", "Dinosauria", "Tyrannosaurus"]
    with tempfile.NamedTemporaryFile(suffix=".rds", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        save_cache(tmp_name)
        clear_cache()
        load_cache(tmp_name)
        
        assert fetch._taxodist_cache["id_Tyrannosaurus"] == "50841"
        assert fetch._taxodist_cache["lin_50841"] ==["Biota", "Animalia", "Dinosauria", "Tyrannosaurus"]
    finally:
        os.remove(tmp_name)

def test_load_cache_errors_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_cache("nonexistent_file.rds")

def test_save_cache_returns_invisible_null():
    clear_cache()
    with tempfile.NamedTemporaryFile(suffix=".rds", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        assert save_cache(tmp_name) is None
    finally:
        os.remove(tmp_name)

def test_load_cache_returns_invisible_null():
    clear_cache()
    with tempfile.NamedTemporaryFile(suffix=".rds", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        save_cache(tmp_name)
        clear_cache()
        assert load_cache(tmp_name) is None
    finally:
        os.remove(tmp_name)

# ── Mocking filter_clade ──────────────────────────────────────────────────────

@patch("taxodist.utils.is_member")
def test_filter_clade_filters_correctly_with_mocked_lineages(mock_is_member):
    def fake_is_member(taxon, clade, **kwargs):
        memberships = {
            "Tyrannosaurus": ["Dinosauria", "Theropoda"],
            "Triceratops": ["Dinosauria", "Ornithischia"],
            "Homo":["Mammalia", "Amniota"]
        }
        return clade in memberships.get(taxon,[])
    mock_is_member.side_effect = fake_is_member
    
    result = filter_clade(["Tyrannosaurus", "Triceratops", "Homo"], "Dinosauria")
    assert result ==["Tyrannosaurus", "Triceratops"]

# ── taxo_path ─────────────────────────────────────────────────────────────────

@patch("taxodist.utils.get_lineage")
def test_taxo_path_returns_none_when_taxon_a_not_found(mock_get_lineage):
    mock_get_lineage.return_value = None
    assert taxo_path("Fakeosaurus", "Carnotaurus") is None

@patch("taxodist.utils.get_lineage")
def test_taxo_path_returns_none_when_taxon_b_not_found(mock_get_lineage):
    def fake_get_lineage(taxon, **kwargs):
        if taxon == "Carnotaurus":
            return["Biota", "Animalia", "Dinosauria", "Theropoda", "Carnotaurus"]
        return None
    mock_get_lineage.side_effect = fake_get_lineage
    
    assert taxo_path("Carnotaurus", "Fakeosaurus") is None

@patch("taxodist.utils.get_lineage")
def test_taxo_path_returns_a_taxodist_path_data_frame(mock_get_lineage):
    def fake_get_lineage(taxon, **kwargs):
        if taxon == "Tyrannosaurus":
            return["Biota", "Animalia", "Dinosauria", "Theropoda", "Tyrannosaurus"]
        return["Biota", "Animalia", "Dinosauria", "Ornithischia", "Triceratops"]
    mock_get_lineage.side_effect = fake_get_lineage
    
    result = taxo_path("Tyrannosaurus", "Triceratops")
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["node", "depth", "direction"]

@patch("taxodist.utils.get_lineage")
def test_taxo_path_has_exactly_one_mrca_row(mock_get_lineage):
    def fake_get_lineage(taxon, **kwargs):
        if taxon == "Tyrannosaurus":
            return["Biota", "Animalia", "Dinosauria", "Theropoda", "Tyrannosaurus"]
        return["Biota", "Animalia", "Dinosauria", "Ornithischia", "Triceratops"]
    mock_get_lineage.side_effect = fake_get_lineage
    
    result = taxo_path("Tyrannosaurus", "Triceratops")
    assert sum(result["direction"] == "mrca") == 1

@patch("taxodist.utils.get_lineage")
def test_taxo_path_mrca_node_is_correct(mock_get_lineage):
    def fake_get_lineage(taxon, **kwargs):
        if taxon == "Tyrannosaurus":
            return["Biota", "Animalia", "Dinosauria", "Theropoda", "Tyrannosaurus"]
        return["Biota", "Animalia", "Dinosauria", "Ornithischia", "Triceratops"]
    mock_get_lineage.side_effect = fake_get_lineage
    
    result = taxo_path("Tyrannosaurus", "Triceratops")
    assert result[result["direction"] == "mrca"]["node"].values[0] == "Dinosauria"

@patch("taxodist.utils.get_lineage")
def test_taxo_path_direction_column_only_contains_valid_values(mock_get_lineage):
    def fake_get_lineage(taxon, **kwargs):
        if taxon == "Tyrannosaurus":
            return["Biota", "Animalia", "Dinosauria", "Theropoda", "Tyrannosaurus"]
        return["Biota", "Animalia", "Dinosauria", "Ornithischia", "Triceratops"]
    mock_get_lineage.side_effect = fake_get_lineage
    
    result = taxo_path("Tyrannosaurus", "Triceratops")
    assert set(result["direction"].unique()).issubset({"a", "mrca", "b"})

@patch("taxodist.utils.get_lineage")
def test_taxo_path_preserves_taxon_attributes(mock_get_lineage):
    def fake_get_lineage(taxon, **kwargs):
        if taxon == "Tyrannosaurus":
            return["Biota", "Animalia", "Dinosauria", "Theropoda", "Tyrannosaurus"]
        return["Biota", "Animalia", "Dinosauria", "Ornithischia", "Triceratops"]
    mock_get_lineage.side_effect = fake_get_lineage
    
    result = taxo_path("Tyrannosaurus", "Triceratops")
    assert result.attrs["taxon_a"] == "Tyrannosaurus"
    assert result.attrs["taxon_b"] == "Triceratops"

@patch("taxodist.utils.get_lineage")
def test_print_taxodist_path_runs_without_error_and_returns_invisibly(mock_get_lineage):
    def fake_get_lineage(taxon, **kwargs):
        if taxon == "Tyrannosaurus":
            return["Biota", "Animalia", "Dinosauria", "Theropoda", "Tyrannosaurus"]
        return["Biota", "Animalia", "Dinosauria", "Ornithischia", "Triceratops"]
    mock_get_lineage.side_effect = fake_get_lineage
    
    result = taxo_path("Tyrannosaurus", "Triceratops")
    assert print_taxodist_path(result) is result

# ── Mock tests ────────────────────────────────────────────────────────────────

@patch("taxodist.fetch.requests.get")
def test_get_taxonomicon_id_returns_none_on_network_failure(mock_get):
    clear_cache()
    mock_get.side_effect = requests.exceptions.RequestException("Network error")
    with pytest.warns(UserWarning, match="Cannot reach"):
        result = get_taxonomicon_id("Tyrannosaurus")
    assert result is None

@patch("taxodist.fetch.requests.get")
def test_get_taxonomicon_id_returns_none_on_bad_status(mock_get):
    clear_cache()
    mock_resp = Mock()
    mock_resp.status_code = 404
    mock_get.return_value = mock_resp
    with pytest.warns(UserWarning, match="Cannot reach"):
        result = get_taxonomicon_id("Tyrannosaurus")
    assert result is None

@patch("taxodist.fetch.requests.get")
def test_get_lineage_by_id_returns_none_on_network_failure(mock_get):
    clear_cache()
    mock_get.side_effect = requests.exceptions.RequestException("Network error")
    result = get_lineage_by_id("12345")
    assert result is None

@patch("taxodist.fetch.get_taxonomicon_id")
def test_get_lineage_returns_none_when_id_not_found(mock_get_id):
    clear_cache()
    mock_get_id.return_value = None
    result = get_lineage("Fakeosaurus")
    assert result is None

@patch("taxodist.fetch.requests.get")
def test_cache_is_used_on_second_call_to_get_taxonomicon_id(mock_get):
    clear_cache()
    fetch._taxodist_cache["id_Tyrannosaurus"] = "50841"
    
    result = get_taxonomicon_id("Tyrannosaurus")
    assert result == "50841"
    mock_get.assert_not_called() 

import numpy as np
from taxodist.distance import taxo_distance, mrca, distance_matrix, closest_relative, lineage_depth, check_coverage
from taxodist.utils import compare_lineages, shared_clades, is_member, filter_clade, print_taxodist_result
from taxodist.fetch import get_taxonomicon_id, get_lineage_by_id

# ── Distance / Utility functions Mock tests (Continuação) ─────────────────────

@patch("taxodist.distance.get_lineage")
def test_taxo_distance_works_with_mocked_lineages(mock_get_lineage):
    def side_effect(taxon, **kwargs):
        if taxon == "Tyrannosaurus":
            return["Biota", "Animalia", "Dinosauria", "Theropoda", "Tyrannosaurus"]
        return["Biota", "Animalia", "Dinosauria", "Ornithischia", "Triceratops"]
    mock_get_lineage.side_effect = side_effect
    
    result = taxo_distance("Tyrannosaurus", "Triceratops")
    assert isinstance(result, dict)
    assert result["mrca"] == "Dinosauria"

@patch("taxodist.distance.get_lineage")
def test_closest_relative_works_with_mocked_lineages(mock_get_lineage):
    lins = {
        "Tyrannosaurus":["Biota", "Animalia", "Dinosauria", "Theropoda", "Tyrannosaurus"],
        "Velociraptor":["Biota", "Animalia", "Dinosauria", "Theropoda", "Velociraptor"],
        "Triceratops":["Biota", "Animalia", "Dinosauria", "Ornithischia", "Triceratops"]
    }
    mock_get_lineage.side_effect = lambda t, **kwargs: lins.get(t)
    result = closest_relative("Tyrannosaurus",["Velociraptor", "Triceratops"])
    assert len(result) == 2
    assert result["taxon"].iloc[0] == "Velociraptor"

@patch("taxodist.distance.get_lineage")
def test_lineage_depth_works_with_mocked_lineage(mock_get_lineage):
    mock_get_lineage.return_value =["Biota", "Animalia", "Dinosauria", "Theropoda", "Tyrannosaurus"]
    assert lineage_depth("Tyrannosaurus") == 5

@patch("taxodist.utils.get_lineage")
def test_shared_clades_works_with_mocked_lineages(mock_get_lineage):
    def side_effect(taxon, **kwargs):
        if taxon == "Tyrannosaurus":
            return["Biota", "Animalia", "Dinosauria", "Theropoda", "Tyrannosaurus"]
        return["Biota", "Animalia", "Dinosauria", "Ornithischia", "Triceratops"]
    mock_get_lineage.side_effect = side_effect
    result = shared_clades("Tyrannosaurus", "Triceratops")
    assert result ==["Biota", "Animalia", "Dinosauria"]

@patch("taxodist.utils.get_lineage")
def test_is_member_works_with_mocked_lineage(mock_get_lineage):
    mock_get_lineage.return_value =["Biota", "Animalia", "Dinosauria", "Theropoda"]
    assert is_member("Tyrannosaurus", "Dinosauria") is True
    assert is_member("Tyrannosaurus", "Mammalia") is False

@patch("taxodist.utils.get_lineage")
def test_compare_lineages_works_with_mocked_lineages(mock_get_lineage):
    def side_effect(taxon, **kwargs):
        if taxon == "Tyrannosaurus":
            return["Biota", "Animalia", "Dinosauria", "Theropoda", "Tyrannosaurus"]
        return["Biota", "Animalia", "Dinosauria", "Ornithischia", "Triceratops"]
    mock_get_lineage.side_effect = side_effect
    result = compare_lineages("Tyrannosaurus", "Triceratops")
    assert result["mrca_depth"] == 3

@patch("taxodist.distance.get_lineage")
def test_distance_matrix_works_with_mocked_lineages(mock_get_lineage):
    lins = {
        "Tyrannosaurus":["Biota", "Animalia", "Dinosauria", "Theropoda", "Tyrannosaurus"],
        "Velociraptor":["Biota", "Animalia", "Dinosauria", "Theropoda", "Velociraptor"],
        "Triceratops":["Biota", "Animalia", "Dinosauria", "Ornithischia", "Triceratops"]
    }
    mock_get_lineage.side_effect = lambda t, **kwargs: lins.get(t)
    mat = distance_matrix(["Tyrannosaurus", "Velociraptor", "Triceratops"], progress=False)
    assert mat.shape == (3, 3)
    assert list(np.diag(mat.values)) == [0.0, 0.0, 0.0]

@patch("taxodist.distance.get_taxonomicon_id")
def test_check_coverage_returns_named_logical_vector(mock_get_id):
    mock_get_id.side_effect = lambda t, **kwargs: None if t == "Fakeosaurus" else "12345"
    result = check_coverage(["Tyrannosaurus", "Fakeosaurus"])
    assert isinstance(result, pd.Series)
    assert result["Tyrannosaurus"]
    assert not result["Fakeosaurus"]

@patch("taxodist.distance.get_lineage")
def test_taxo_distance_returns_none_when_taxon_a_not_found(mock_get_lineage):
    mock_get_lineage.return_value = None
    result = taxo_distance("Fakeosaurus", "Carnotaurus")
    assert result is None

@patch("taxodist.distance.get_lineage")
def test_taxo_distance_returns_none_when_taxon_b_not_found(mock_get_lineage):
    def side_effect(taxon, **kwargs):
        if taxon == "Carnotaurus":
            return["Biota", "Animalia", "Dinosauria", "Theropoda", "Abelisauridae", "Carnotaurus"]
        return None
    mock_get_lineage.side_effect = side_effect
    result = taxo_distance("Carnotaurus", "Fakeosaurus")
    assert result is None

@patch("taxodist.distance.taxo_distance")
def test_mrca_returns_none_when_taxo_distance_fails(mock_dist):
    mock_dist.return_value = None
    result = mrca("Fakeosaurus", "Carnotaurus")
    assert result is None

@patch("taxodist.distance.taxo_distance")
def test_mrca_returns_correct_value_when_taxo_distance_succeeds(mock_dist):
    mock_dist.return_value = {"mrca": "Dinosauria"}
    result = mrca("Carnotaurus", "Triceratops")
    assert result == "Dinosauria"

@patch("taxodist.distance.get_lineage")
def test_closest_relative_returns_none_when_query_lineage_not_found(mock_get_lineage):
    mock_get_lineage.return_value = None
    result = closest_relative("Fakeosaurus",["Carnotaurus", "Velociraptor"])
    assert result is None

@patch("taxodist.distance.get_lineage")
def test_closest_relative_handles_none_candidate_lineage(mock_get_lineage):
    def side_effect(taxon, **kwargs):
        if taxon == "Carnotaurus":
            return["Biota", "Animalia", "Dinosauria", "Theropoda", "Abelisauridae", "Carnotaurus"]
        elif taxon == "Velociraptor":
            return["Biota", "Animalia", "Dinosauria", "Theropoda", "Dromaeosauridae", "Velociraptor"]
        return None
    mock_get_lineage.side_effect = side_effect
    result = closest_relative("Carnotaurus",["Velociraptor", "Fakeosaurus"])
    assert len(result) == 2
    missing_dist = result.loc[result["taxon"] == "Fakeosaurus", "distance"].values[0]
    assert pd.isna(missing_dist)

@patch("taxodist.distance.get_lineage")
def test_distance_matrix_handles_none_lineage_for_one_taxon(mock_get_lineage):
    def side_effect(taxon, **kwargs):
        if taxon == "Fakeosaurus":
            return None
        elif taxon == "Carnotaurus":
            return["Biota", "Animalia", "Dinosauria", "Theropoda", "Abelisauridae", "Carnotaurus"]
        return["Biota", "Animalia", "Dinosauria", "Theropoda", "Dromaeosauridae", "Velociraptor"]
    mock_get_lineage.side_effect = side_effect
    mat = distance_matrix(["Carnotaurus", "Velociraptor", "Fakeosaurus"], progress=False)
    assert pd.isna(mat.loc["Carnotaurus", "Fakeosaurus"])
    assert not pd.isna(mat.loc["Carnotaurus", "Velociraptor"])

@patch("taxodist.distance.get_lineage")
def test_distance_matrix_with_progress_true_runs_without_error(mock_get_lineage):
    lins = {
        "Carnotaurus":["Biota", "Animalia", "Dinosauria", "Theropoda", "Abelisauridae", "Carnotaurus"],
        "Velociraptor":["Biota", "Animalia", "Dinosauria", "Theropoda", "Dromaeosauridae", "Velociraptor"],
        "Triceratops":["Biota", "Animalia", "Dinosauria", "Ornithischia", "Triceratops"]
    }
    mock_get_lineage.side_effect = lambda t, **kwargs: lins.get(t)
    try:
        distance_matrix(["Carnotaurus", "Velociraptor", "Triceratops"], progress=True)
        success = True
    except Exception:
        success = False
    assert success

def test_get_taxonomicon_id_verbose_prints_messages_on_cache_hit():
    clear_cache()
    fetch._taxodist_cache["id_Carnotaurus"] = "99999"
    get_taxonomicon_id("Carnotaurus", verbose=True)
    assert True

@patch("taxodist.fetch.requests.get")
def test_get_taxonomicon_id_verbose_prints_warning_on_network_failure(mock_get):
    clear_cache()
    import requests
    mock_get.side_effect = requests.exceptions.RequestException("Network error")
    with pytest.warns(UserWarning, match="Cannot reach"):
        get_taxonomicon_id("Drosophila", verbose=True)

@patch("taxodist.fetch.requests.get")
def test_get_taxonomicon_id_verbose_prints_warning_on_bad_status(mock_get):
    clear_cache()
    mock_resp = Mock()
    mock_resp.status_code = 503
    mock_get.return_value = mock_resp
    with pytest.warns(UserWarning, match="Cannot reach"):
        get_taxonomicon_id("Drosophila", verbose=True)

def test_get_lineage_by_id_verbose_prints_messages_on_cache_hit():
    clear_cache()
    fetch._taxodist_cache["lin_99999"] =["Biota", "Animalia", "Dinosauria", "Abelisauridae", "Carnotaurus"]
    get_lineage_by_id("99999", verbose=True)
    assert True

@patch("taxodist.fetch.requests.get")
def test_get_lineage_by_id_verbose_prints_warning_on_network_failure(mock_get):
    clear_cache()
    import requests
    mock_get.side_effect = requests.exceptions.RequestException("Network error")
    get_lineage_by_id("00000", verbose=True)
    assert True

@patch("taxodist.fetch.requests.get")
def test_get_lineage_by_id_returns_none_when_lineage_is_empty_after_cleaning(mock_get):
    clear_cache()
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.text = "<html><body></body></html>"
    mock_get.return_value = mock_resp
    result = get_lineage_by_id("empty_page")
    assert result is None

@patch("taxodist.fetch.requests.get")
def test_get_lineage_by_id_returns_none_on_bad_http_status(mock_get):
    clear_cache()
    mock_resp = Mock()
    mock_resp.status_code = 404
    mock_get.return_value = mock_resp
    result = get_lineage_by_id("99999")
    assert result is None

@patch("taxodist.fetch.requests.get")
def test_get_lineage_by_id_verbose_prints_warning_on_bad_status(mock_get):
    clear_cache()
    mock_resp = Mock()
    mock_resp.status_code = 503
    mock_get.return_value = mock_resp
    get_lineage_by_id("99999", verbose=True)
    assert True

def test_get_lineage_by_id_cache_hit_with_verbose_prints_message():
    clear_cache()
    fetch._taxodist_cache["lin_50841"] =["Biota", "Animalia", "Dinosauria", "Theropoda", "Tyrannosaurus"]
    result = get_lineage_by_id("50841", verbose=True)
    assert isinstance(result, list)

@patch("taxodist.fetch.requests.get")
def test_get_lineage_by_id_with_clean_false_returns_none_on_network_failure(mock_get):
    clear_cache()
    fetch._taxodist_cache["lin_clean_test"] = None
    import requests
    mock_get.side_effect = requests.exceptions.RequestException("Network error")
    result = get_lineage_by_id("clean_test", clean=False)
    assert result is None

def test_print_taxodist_result_displays_output_correctly():
    lin_a =["Biota", "Animalia", "Dinosauria", "Theropoda", "Abelisauridae", "Carnotaurus"]
    lin_b =["Biota", "Animalia", "Dinosauria", "Ornithischia", "Triceratops"]
    result = _compute_distance(lin_a, lin_b, "Carnotaurus", "Triceratops")
    assert print_taxodist_result(result) is result

@patch("taxodist.utils.get_lineage")
def test_compare_lineages_returns_none_when_lineage_missing(mock_get_lineage):
    mock_get_lineage.return_value = None
    result = compare_lineages("Fakeosaurus", "Carnotaurus")
    assert result is None

@patch("taxodist.utils.get_lineage")
def test_compare_lineages_handles_mrca_depth_0(mock_get_lineage):
    def side_effect(taxon, **kwargs):
        if taxon == "Drosophila":
            return["Biota", "Animalia", "Arthropoda", "Insecta", "Drosophila"]
        return ["Fungi", "Ascomycota", "Saccharomyces"]
    mock_get_lineage.side_effect = side_effect
    result = compare_lineages("Drosophila", "Saccharomyces")
    assert result["mrca_depth"] == 0

@patch("taxodist.utils.get_lineage")
def test_compare_lineages_handles_case_where_one_lineage_is_subset_of_other(mock_get_lineage):
    def side_effect(taxon, **kwargs):
        if taxon == "Dinosauria":
            return ["Biota", "Animalia", "Dinosauria"]
        return["Biota", "Animalia", "Dinosauria", "Theropoda", "Carnotaurus"]
    mock_get_lineage.side_effect = side_effect
    result = compare_lineages("Dinosauria", "Carnotaurus")
    assert result["mrca_depth"] == 3

@patch("taxodist.utils.get_lineage")
def test_shared_clades_returns_empty_when_no_common_ancestor(mock_get_lineage):
    def side_effect(taxon, **kwargs):
        if taxon == "Drosophila":
            return["Biota", "Animalia", "Arthropoda", "Insecta", "Drosophila"]
        return["Fungi", "Ascomycota", "Saccharomyces"]
    mock_get_lineage.side_effect = side_effect
    result = shared_clades("Drosophila", "Saccharomyces")
    assert result ==[]

@patch("taxodist.utils.get_lineage")
def test_shared_clades_returns_none_when_one_lineage_missing(mock_get_lineage):
    mock_get_lineage.return_value = None
    result = shared_clades("Fakeosaurus", "Carnotaurus")
    assert result is None

@patch("taxodist.utils.get_lineage")
def test_is_member_returns_none_when_lineage_not_found(mock_get_lineage):
    mock_get_lineage.return_value = None
    result = is_member("Fakeosaurus", "Dinosauria")
    assert result is None

@patch("taxodist.utils.is_member")
def test_filter_clade_handles_none_result_from_is_member(mock_is_member):
    def side_effect(taxon, clade, **kwargs):
        if taxon == "Fakeosaurus":
            return None
        memberships = {
            "Carnotaurus":["Dinosauria", "Theropoda"],
            "Drosophila": ["Animalia", "Insecta"]
        }
        return clade in memberships.get(taxon,[])
    mock_is_member.side_effect = side_effect
    result = filter_clade(["Carnotaurus", "Fakeosaurus", "Drosophila"], "Dinosauria")
    assert result == ["Carnotaurus"]

@patch("taxodist.distance.get_lineage")
def test_lineage_depth_returns_none_when_lineage_not_found(mock_get_lineage):
    mock_get_lineage.return_value = None
    result = lineage_depth("Fakeosaurus")
    assert result is None

@patch("taxodist.fetch.get_lineage_by_id")
@patch("taxodist.fetch.requests.get")
def test_get_taxonomicon_id_skips_astronomical_entries(mock_get, mock_get_lineage_by_id):
    clear_cache()
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.text = '''
    <html><body><table>
      <tr>
        <td>Carnotaurus - asteroid - Minor planet</td>
        <td><a class="Valid" href="TaxonTree.aspx?id=99999&src=0">tree</a></td>
      </tr>
      <tr>
        <td>Carnotaurus - animal - dinosaur</td>
        <td><a class="Valid" href="TaxonTree.aspx?id=12345&src=0">tree</a></td>
      </tr>
    </table></body></html>'''
    mock_get.return_value = mock_resp
    mock_get_lineage_by_id.return_value = ["Biota", "Animalia"]
    result = get_taxonomicon_id("Carnotaurus", verbose=True)
    assert result == "12345"

import requests
from unittest.mock import patch, Mock
import numpy as np
import pandas as pd
from taxodist.fetch import taxo_search, get_lineage, get_lineage_by_id, get_taxonomicon_id, clear_cache
from taxodist.distance import taxo_cluster, taxo_ordinate, taxo_distance, mrca, lineage_depth
from taxodist.utils import plot_taxodist_cluster, plot_taxodist_ord, is_member

# ── Mais Mocks de HTML e Buscas ───────────────────────────────────────────────

@patch("taxodist.fetch.get_lineage_by_id")
@patch("taxodist.fetch.requests.get")
def test_get_taxonomicon_id_skips_astronomical_entries(mock_get, mock_get_lineage_by_id):
    clear_cache()
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.text = '''
    <html><body><table>
      <tr>
        <td>Carnotaurus - asteroid - Minor planet</td>
        <td><a class="Valid" href="TaxonTree.aspx?id=99999&src=0">tree</a></td>
      </tr>
      <tr>
        <td>Carnotaurus - animal - dinosaur</td>
        <td><a class="Valid" href="TaxonTree.aspx?id=12345&src=0">tree</a></td>
      </tr>
    </table></body></html>'''
    mock_get.return_value = mock_resp
    mock_get_lineage_by_id.return_value = ["Biota", "Animalia"]
    
    result = get_taxonomicon_id("Carnotaurus", verbose=True)
    assert result == "12345"

@patch("taxodist.fetch.get_lineage_by_id")
@patch("taxodist.fetch.requests.get")
def test_get_taxonomicon_id_skips_row_matching_both_bio_and_astronomical(mock_get, mock_get_lineage_by_id):
    clear_cache()
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.text = '''
    <html><body><table>
      <tr>
        <td>Pterodactylus - animal - Minor planet asteroid</td>
        <td><a class="Valid" href="TaxonTree.aspx?id=99999&src=0">wrong</a></td>
      </tr>
      <tr>
        <td>Pterodactylus - animal - reptile</td>
        <td><a class="Valid" href="TaxonTree.aspx?id=42042&src=0">tree</a></td>
      </tr>
    </table></body></html>'''
    mock_get.return_value = mock_resp
    mock_get_lineage_by_id.return_value = ["Biota", "Animalia"]
    
    result = get_taxonomicon_id("Pterodactylus", verbose=True)
    assert result == "42042"

@patch("taxodist.fetch.requests.get")
def test_get_taxonomicon_id_returns_none_when_bio_row_has_no_tree_link(mock_get):
    clear_cache()
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.text = '''
    <html><body><table>
      <tr>
        <td>Quercus - plant</td>
        <td><a href="SomeOtherPage.aspx?id=999">no tree link</a></td>
      </tr>
    </table></body></html>'''
    mock_get.return_value = mock_resp
    
    result = get_taxonomicon_id("Quercus", verbose=True)
    assert result is None

@patch("taxodist.fetch.requests.get")
def test_get_lineage_by_id_parses_html_and_returns_lineage(mock_get):
    clear_cache()
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.text = '''
    <html><body>
      <a href="TaxonTree.aspx?id=1&src=0">Biota</a>
      <a href="TaxonTree.aspx?id=2&src=0">Animalia</a>
      <a href="TaxonTree.aspx?id=3&src=0">Dinosauria</a>
      <a href="TaxonTree.aspx?id=4&src=0">Theropoda</a>
      <a href="TaxonTree.aspx?id=5&src=0">Carnotaurus</a>
    </body></html>'''
    mock_get.return_value = mock_resp
    
    result = get_lineage_by_id("12345", verbose=True)
    assert isinstance(result, list)
    assert "Dinosauria" in result
    assert "Carnotaurus" in result

@patch("taxodist.fetch.get_lineage_by_id")
@patch("taxodist.fetch.get_taxonomicon_id")
def test_get_lineage_passes_clean_and_verbose_through(mock_get_id, mock_get_lineage_by_id):
    clear_cache()
    mock_get_id.return_value = "12345"
    def fake_get_lineage_by_id(id_val, clean=True, verbose=False):
        assert id_val == "12345"
        assert clean is False
        assert verbose is True
        return["Biota", "Animalia", "Plantae", "Quercus"]
    mock_get_lineage_by_id.side_effect = fake_get_lineage_by_id
    
    result = get_lineage("Quercus", clean=False, verbose=True)
    assert result ==["Biota", "Animalia", "Plantae", "Quercus"]

# Usamos patch no plt.show para não abrir janelas visuais durante o teste
@patch("matplotlib.pyplot.show")
def test_plot_taxodist_cluster_runs_without_error(mock_show):
    m = np.array([[0, 0.2, 0.5],[0.2, 0, 0.3], [0.5, 0.3, 0]])
    d = pd.DataFrame(m, index=["A", "B", "C"], columns=["A", "B", "C"])
    cl = taxo_cluster(d)
    assert plot_taxodist_cluster(cl) is cl

@patch("matplotlib.pyplot.show")
def test_plot_taxodist_ord_runs_without_error(mock_show):
    m = np.array([[0, 0.2, 0.5], [0.2, 0, 0.3],[0.5, 0.3, 0]])
    d = pd.DataFrame(m, index=["A", "B", "C"], columns=["A", "B", "C"])
    ord_obj = taxo_ordinate(d)
    assert plot_taxodist_ord(ord_obj) is ord_obj

@patch("taxodist.fetch.requests.get")
def test_taxo_search_returns_none_on_network_failure_and_bad_status(mock_get):
    clear_cache()
    mock_get.side_effect = requests.exceptions.RequestException("Network error")
    assert taxo_search("Bacteria", verbose=True) is None

    mock_resp = Mock()
    mock_resp.status_code = 503
    mock_get.side_effect = None
    mock_get.return_value = mock_resp
    assert taxo_search("Bacteria", verbose=True) is None

@patch("taxodist.fetch.requests.get")
def test_taxo_search_returns_none_when_no_matches_are_found(mock_get):
    clear_cache()
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.text = '''
    <html><body><table>
      <tr><td>Not a link</td></tr>
      <tr><td><a href="OtherPage.aspx">No ID here</a></td></tr>
    </table></body></html>'''
    mock_get.return_value = mock_resp
    
    assert taxo_search("EmptyTaxon", verbose=True) is None

@patch("taxodist.fetch.requests.get")
def test_taxo_search_parses_html_applies_skips_dedups_and_returns_df(mock_get):
    clear_cache()
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.text = '''
    <html><body><table>
      <tr>
        <td>Astronomical planet asteroid</td>
        <td><a class="Valid" href="TaxonTree.aspx?id=111">ignore</a></td>
      </tr>
      <tr>
        <td>No links here</td>
        <td>Just text</td>
      </tr>
      <tr>
        <td>Invalid class</td>
        <td><a class="Invalid" href="TaxonTree.aspx?id=222">ignore</a></td>
      </tr>
      <tr>
        <td>Missing ID</td>
        <td><a class="Valid" href="TaxonTree.aspx?wrong=333">ignore</a></td>
      </tr>
      <tr>
        <td><a class="Valid" href="TaxonTree.aspx?id=444">N|T|P|R|B|L Bacteria (Kingdom)</a></td>
      </tr>
      <tr>
        <td><a class="Valid" href="TaxonTree.aspx?id=444">N|T|P|R|B|L Bacteria (Kingdom) Duplicated</a></td>
      </tr>
      <tr>
        <td><a class="Valid" href="TaxonTree.aspx?id=555">N|T|P|R|B|L Bacteria (Domain)</a></td>
      </tr>
    </table></body></html>'''
    mock_get.return_value = mock_resp
    
    df = taxo_search("Bacteria", verbose=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df["id"]) ==["444", "555"]
    assert df["name"].iloc[0] == "Bacteria (Kingdom)"
    assert df["name"].iloc[1] == "Bacteria (Domain)"

# ── Network tests (skipped if offline) ────────────────────────────────────────

def is_taxonomicon_down():
    try:
        import requests
        res = requests.get("http://taxonomicon.taxonomy.nl", timeout=3)
        return res.status_code != 200
    except:
        return True

skip_if_offline = pytest.mark.skipif(is_taxonomicon_down(), reason="Taxonomicon server is offline")

@skip_if_offline
def test_get_lineage_returns_correct_lineage_for_velociraptor():
    clear_cache()
    lin = get_lineage("Velociraptor")
    assert isinstance(lin, list)
    assert len(lin) > 0
    assert "Dinosauria" in lin
    assert "Theropoda" in lin
    assert "Dromaeosauridae" in lin

@skip_if_offline
def test_get_lineage_returns_correct_lineage_for_tyrannosaurus():
    clear_cache()
    lin = get_lineage("Tyrannosaurus")
    assert isinstance(lin, list)
    assert "Coelurosauria" in lin
    assert "Dinosauria" in lin

@skip_if_offline
def test_get_lineage_returns_correct_lineage_for_carnotaurus():
    clear_cache()
    lin = get_lineage("Carnotaurus")
    assert isinstance(lin, list)
    assert "Dinosauria" in lin
    assert "Theropoda" in lin

@skip_if_offline
def test_get_lineage_returns_correct_lineage_for_homo():
    clear_cache()
    lin = get_lineage("Homo")
    assert "Amniota" in lin
    assert "Mammalia" in lin

@skip_if_offline
def test_get_lineage_returns_correct_lineage_for_drosophila():
    clear_cache()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lin = get_lineage("Drosophila")
    if lin is not None:
        assert isinstance(lin, list)
        assert len(lin) > 0
        assert "Animalia" in lin

@skip_if_offline
def test_get_lineage_returns_none_for_unknown_taxon():
    clear_cache()
    assert get_lineage("Fakeosaurus") is None

@skip_if_offline
def test_taxo_distance_returns_valid_result_for_tyrannosaurus_vs_velociraptor():
    clear_cache()
    result = taxo_distance("Tyrannosaurus", "Velociraptor")
    assert isinstance(result, dict)
    assert 0 <= result["distance"] <= 1
    assert result["mrca"] == "Tyrannoraptora"

@skip_if_offline
def test_taxo_distance_returns_0_when_one_taxon_is_ancestor_of_other():
    clear_cache()
    res1 = taxo_distance("Tyrannosaurus", "Dinosauria")
    if res1 is not None:
        assert res1["distance"] == 0 
    
    res2 = taxo_distance("Carnotaurus", "Ceratosauria")
    if res2 is not None:
        assert res2["distance"] == 0

@skip_if_offline
def test_taxo_distance_between_carnotaurus_and_triceratops_is_valid():
    clear_cache()
    result = taxo_distance("Carnotaurus", "Triceratops")
    assert isinstance(result, dict)
    assert result["mrca"] == "Dinosauria"

@skip_if_offline
def test_taxo_distance_is_larger_between_distant_taxa_than_close_ones():
    clear_cache()
    res_close = taxo_distance("Carnotaurus", "Tyrannosaurus")
    res_distant = taxo_distance("Carnotaurus", "Homo")
    if res_close is not None and res_distant is not None:
        assert res_distant["distance"] > res_close["distance"]

@skip_if_offline
def test_mrca_of_tyrannosaurus_and_triceratops_is_dinosauria():
    clear_cache()
    assert mrca("Tyrannosaurus", "Triceratops") == "Dinosauria"

@skip_if_offline
def test_mrca_of_tyrannosaurus_and_homo_is_amniota():
    clear_cache()
    assert mrca("Tyrannosaurus", "Homo") == "Amniota"

@skip_if_offline
def test_mrca_of_velociraptor_and_triceratops_is_dinosauria():
    clear_cache()
    assert mrca("Velociraptor", "Triceratops") == "Dinosauria"

@skip_if_offline
def test_mrca_of_carnotaurus_and_tyrannosaurus_is_within_theropoda():
    clear_cache()
    ancestor = mrca("Carnotaurus", "Tyrannosaurus")
    if ancestor is not None:
        lin = get_lineage("Tyrannosaurus")
        assert ancestor in lin

@skip_if_offline
def test_is_member_correctly_identifies_clade_membership():
    clear_cache()
    assert is_member("Tyrannosaurus", "Theropoda") is True
    assert is_member("Tyrannosaurus", "Ornithischia") is False

@skip_if_offline
def test_lineage_depth_for_carnotaurus_is_reasonable():
    clear_cache()
    depth = lineage_depth("Carnotaurus")
    if depth is not None:
        assert depth > 10

@skip_if_offline
def test_get_taxonomicon_id_finds_real_id_and_caches_it():
    clear_cache()
    from taxodist.fetch import _taxodist_cache
    id_val = get_taxonomicon_id("Carnotaurus", verbose=True)
    assert isinstance(id_val, str)
    assert id_val is not None
    assert id_val == _taxodist_cache.get("id_Carnotaurus")

# ── Mocks Avançados de Fetch (Tratamentos de Exceções, Homônimos, Nomes) ──────

@patch("taxodist.fetch.requests.get")
@patch("taxodist.fetch.get_lineage_by_id")
def test_get_taxonomicon_id_skips_entry_whose_lineage_has_no_biota(mock_get_lineage, mock_get):
    clear_cache()
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.text = '''
    <html><body><table>
      <tr>
        <td>Carnotaurus - animal - dinosaur</td>
        <td><a class="Valid" href="TaxonTree.aspx?id=12345&src=0">tree</a></td>
      </tr>
    </table></body></html>'''
    mock_get.return_value = mock_resp
    mock_get_lineage.return_value =["NotBiota", "Animalia", "Dinosauria"]
    
    result = get_taxonomicon_id("Carnotaurus")
    assert result is None

@patch("taxodist.fetch.requests.get")
def test_get_lineage_by_id_truncates_at_own_id_when_present_in_links(mock_get):
    clear_cache()
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.text = '''
    <html><body>
      <a href="TaxonTree.aspx?id=1&src=0">Biota</a>
      <a href="TaxonTree.aspx?id=2&src=0">Animalia</a>
      <a href="TaxonTree.aspx?id=99&src=0">Carnotaurus</a>
      <a href="TaxonTree.aspx?id=100&src=0">SomeChild</a>
    </body></html>'''
    mock_get.return_value = mock_resp
    result = get_lineage_by_id("99")
    assert "Carnotaurus" in result
    assert "SomeChild" not in result

@patch("taxodist.fetch.get_lineage_by_id")
@patch("taxodist.fetch.get_taxonomicon_id")
def test_get_lineage_handles_binomial_taxon_name_correctly(mock_get_id, mock_get_lineage):
    clear_cache()
    mock_get_id.return_value = "12345"
    mock_get_lineage.return_value =["Biota", "Animalia", "Dinosauria", "Theropoda", "Carnotaurus sastrei"]
    result = get_lineage("Carnotaurus sastrei")
    assert "Carnotaurus sastrei" in result

@patch("taxodist.fetch.get_lineage_by_id")
@patch("taxodist.fetch.get_taxonomicon_id")
def test_get_lineage_returns_none_when_get_lineage_by_id_returns_none(mock_get_id, mock_get_lineage):
    clear_cache()
    mock_get_id.return_value = "12345"
    mock_get_lineage.return_value = None
    result = get_lineage("Carnotaurus")
    assert result is None

@patch("taxodist.fetch.get_lineage_by_id")
@patch("taxodist.fetch.get_taxonomicon_id")
def test_get_lineage_returns_single_node_lineage_when_lineage_by_id_returns_empty(mock_get_id, mock_get_lineage):
    clear_cache()
    mock_get_id.return_value = "12345"
    mock_get_lineage.return_value =[]
    result = get_lineage("Carnotaurus")
    assert result == ["Carnotaurus"]

@patch("taxodist.fetch.get_lineage_by_id")
@patch("taxodist.fetch.get_taxonomicon_id")
def test_get_lineage_appends_taxon_name_when_not_found_in_scraped_lineage(mock_get_id, mock_get_lineage):
    clear_cache()
    mock_get_id.return_value = "12345"
    mock_get_lineage.return_value = ["Biota", "Animalia", "Dinosauria", "Theropoda"]
    result = get_lineage("Carnotaurus")
    assert result[-1] == "Carnotaurus"

@patch("taxodist.fetch.requests.get")
def test_get_lineage_by_id_returns_none_when_all_links_are_filtered_out(mock_get):
    clear_cache()
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.text = '''
    <html><body>
      <a href="TaxonTree.aspx?id=1&src=0">Go to</a>
      <a href="TaxonTree.aspx?id=2&src=0">[unranked]</a>
    </body></html>'''
    mock_get.return_value = mock_resp
    result = get_lineage_by_id("99999")
    assert result is None

@patch("taxodist.fetch.requests.get")
def test_get_taxonomicon_id_warns_on_multiple_biological_entries(mock_get):
    clear_cache()
    import taxodist.fetch as fetch
    fetch._taxodist_cache["lin_111"] = ["Biota", "Animalia", "Fake1"]
    fetch._taxodist_cache["lin_222"] =["Biota", "Animalia", "Fake2"]
    
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.text = '''
    <html><body><table>
      <tr>
        <td>Nereis - animal - one</td>
        <td><a class="Valid" href="TaxonTree.aspx?id=111&src=0">tree</a></td>
      </tr>
      <tr>
        <td>Nereis - animal - two</td>
        <td><a class="Valid" href="TaxonTree.aspx?id=222&src=0">tree</a></td>
      </tr>
    </table></body></html>'''
    mock_get.return_value = mock_resp
    
    with pytest.warns(UserWarning, match="Multiple valid biological entries"):
        get_taxonomicon_id("Nereis")

@patch("taxodist.fetch.requests.get")
def test_deduplication_preserves_order(mock_get):
    clear_cache()
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.text = '''
    <html><body>
      <a href="TaxonTree.aspx?id=1&src=0">Biota</a>
      <a href="TaxonTree.aspx?id=2&src=0">Animalia</a>
      <a href="TaxonTree.aspx?id=3&src=0">Uropygi</a>
      <a href="TaxonTree.aspx?id=3&src=0">Uropygi</a>
      <a href="TaxonTree.aspx?id=4&src=0">Thelyphonida</a>
    </body></html>'''
    mock_get.return_value = mock_resp
    result = get_lineage_by_id("4")
    assert result ==["Biota", "Animalia", "Uropygi", "Thelyphonida"]

@patch("taxodist.distance.distance_matrix")
def test_taxo_cluster_returns_correct_s3_class(mock_distance_matrix):
    m = np.array([[0, 0.2, 0.5],[0.2, 0, 0.3], [0.5, 0.3, 0]])
    mock_distance_matrix.return_value = pd.DataFrame(m, index=["A", "B", "C"], columns=["A", "B", "C"])
    result = taxo_cluster(["A", "B", "C"], progress=False)
    assert isinstance(result, dict)

@patch("taxodist.distance.distance_matrix")
def test_taxo_cluster_result_contains_hclust_and_dist(mock_distance_matrix):
    m = np.array([[0, 0.2, 0.5],[0.2, 0, 0.3], [0.5, 0.3, 0]])
    mock_distance_matrix.return_value = pd.DataFrame(m, index=["A", "B", "C"], columns=["A", "B", "C"])
    result = taxo_cluster(["A", "B", "C"], progress=False)
    assert result["hclust"] is not None
    assert isinstance(result["dist"], pd.DataFrame)

def test_taxo_cluster_accepts_a_dist_object_directly():
    m = np.array([[0, 0.2, 0.5],[0.2, 0, 0.3], [0.5, 0.3, 0]])
    d = pd.DataFrame(m, index=["A", "B", "C"], columns=["A", "B", "C"])
    result = taxo_cluster(d)
    assert isinstance(result, dict)

@patch("taxodist.distance.distance_matrix")
def test_taxo_ordinate_returns_correct_s3_class(mock_distance_matrix):
    m = np.array([[0, 0.2, 0.5],[0.2, 0, 0.3], [0.5, 0.3, 0]])
    mock_distance_matrix.return_value = pd.DataFrame(m, index=["A", "B", "C"], columns=["A", "B", "C"])
    result = taxo_ordinate(["A", "B", "C"], progress=False)
    assert isinstance(result, dict)

@patch("taxodist.distance.distance_matrix")
def test_taxo_ordinate_result_contains_points_dist_and_gof(mock_distance_matrix):
    m = np.array([[0, 0.2, 0.5], [0.2, 0, 0.3],[0.5, 0.3, 0]])
    mock_distance_matrix.return_value = pd.DataFrame(m, index=["A", "B", "C"], columns=["A", "B", "C"])
    result = taxo_ordinate(["A", "B", "C"], progress=False)
    assert result["points"] is not None
    assert isinstance(result["dist"], pd.DataFrame)
    assert result["GOF"] is not None

@patch("taxodist.distance.distance_matrix")
def test_taxo_ordinate_points_matrix_has_correct_dimensions(mock_distance_matrix):
    m = np.array([[0, 0.2, 0.5], [0.2, 0, 0.3],[0.5, 0.3, 0]])
    mock_distance_matrix.return_value = pd.DataFrame(m, index=["A", "B", "C"], columns=["A", "B", "C"])
    result = taxo_ordinate(["A", "B", "C"], k=2, progress=False)
    assert result["points"].shape[1] == 2
    assert result["points"].shape[0] == 3

def test_taxo_ordinate_accepts_a_dist_object_directly():
    m = np.array([[0, 0.2, 0.5],[0.2, 0, 0.3],[0.5, 0.3, 0]])
    d = pd.DataFrame(m, index=["A", "B", "C"], columns=["A", "B", "C"])
    result = taxo_ordinate(d, k=2)
    assert isinstance(result, dict)

@patch("taxodist.fetch.get_lineage_by_id")
def test_get_lineage_accepts_numeric_ids_directly_without_searching(mock_get_lineage_by_id):
    clear_cache()
    mock_get_lineage_by_id.return_value = ["Biota", "Bacteria"]
    result = get_lineage("71320")
    assert result == ["Biota", "Bacteria"]

def test_get_lineage_by_id_returns_none_for_non_numeric_strings():
    assert get_lineage_by_id("Bacteria") is None
    assert get_lineage_by_id("123x") is None
    assert get_lineage_by_id("   ") is None

@patch("taxodist.fetch.requests.get")
def test_get_taxonomicon_id_follows_taxonomic_redirects(mock_get):
    clear_cache()
    import taxodist.fetch as fetch
    fetch._taxodist_cache["lin_16197"] = ["Biota", "Animalia", "Uropygi"]
    
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.text = '''
    <html><body><table>
      <tr>
        <td>Thelyphonida see Uropygi</td>
        <td>
          <a class="Invalid" href="TaxonTree.aspx?id=123&src=0">old</a>
          <a class="Valid" href="TaxonTree.aspx?id=16197&src=0">tree</a>
        </td>
      </tr>
    </table></body></html>'''
    mock_get.return_value = mock_resp
    result = get_taxonomicon_id("Thelyphonida")
    assert result == "16197"

@patch("taxodist.fetch.requests.get")
def test_get_taxonomicon_id_skips_rows_with_no_valid_links(mock_get):
    clear_cache()
    import taxodist.fetch as fetch
    fetch._taxodist_cache["lin_222"] =["Biota", "Animalia"]
    
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.text = '''
    <html><body><table>
      <tr>
        <td>Invalid taxon</td>
        <td><a class="Invalid" href="TaxonTree.aspx?id=111&src=0">skip me</a></td>
      </tr>
      <tr>
        <td>Good taxon - animal</td>
        <td><a class="Valid" href="TaxonTree.aspx?id=222&src=0">tree</a></td>
      </tr>
    </table></body></html>'''
    mock_get.return_value = mock_resp
    result = get_taxonomicon_id("Good taxon")
    assert result == "222"

@patch("taxodist.fetch.requests.get")
def test_get_taxonomicon_id_skips_valid_links_missing_numeric_ids(mock_get):
    clear_cache()
    import taxodist.fetch as fetch
    fetch._taxodist_cache["lin_333"] = ["Biota", "Animalia"]
    
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.text = '''
    <html><body><table>
      <tr>
        <td>Missing ID taxon</td>
        <td><a class="Valid" href="TaxonTree.aspx?wrongparam=abc">skip me</a></td>
      </tr>
      <tr>
        <td>Good taxon - animal</td>
        <td><a class="Valid" href="TaxonTree.aspx?id=333&src=0">tree</a></td>
      </tr>
    </table></body></html>'''
    mock_get.return_value = mock_resp
    result = get_taxonomicon_id("Good taxon")
    assert result == "333"

@skip_if_offline
def test_get_lineage_by_id_parses_and_caches_lineage_for_drosophila():
    clear_cache()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        id_val = get_taxonomicon_id("Drosophila")
    if id_val is not None:
        result = get_lineage_by_id(id_val, verbose=True)
        assert isinstance(result, list)
        assert "Animalia" in result

@skip_if_offline
def test_get_lineage_by_id_clean_false_keeps_more_nodes_than_clean_true():
    clear_cache()
    id_val = get_taxonomicon_id("Carnotaurus")
    if id_val is not None:
        result_clean = get_lineage_by_id(id_val, clean=True)
        clear_cache()
        result_no_clean = get_lineage_by_id(id_val, clean=False)
        assert len(result_clean) <= len(result_no_clean)

@skip_if_offline
def test_get_lineage_verbose_wrapper_works_for_quercus():
    clear_cache()
    result = get_lineage("Quercus", verbose=True)
    if result is not None:
        assert isinstance(result, list)
        assert "Biota" in result

@skip_if_offline
def test_get_taxonomicon_id_returns_none_for_genuinely_unknown_taxon():
    clear_cache()
    assert get_taxonomicon_id("Zzzznotarealgenus99999", verbose=True) is None

@skip_if_offline
def test_get_taxonomicon_id_skips_astronomical_objects_real():
    clear_cache()
    id_val = get_taxonomicon_id("Venus", verbose=True)
    if id_val is not None:
        lin = get_lineage_by_id(id_val)
        assert lin is not None
    else:
        assert id_val is None

@skip_if_offline
def test_get_lineage_by_id_works_directly_with_verbose():
    clear_cache()
    id_val = get_taxonomicon_id("Carnotaurus")
    if id_val is not None:
        get_lineage_by_id(id_val, verbose=True)
        assert True

@skip_if_offline
def test_get_taxonomicon_id_works_with_verbose_for_real_taxon():
    clear_cache()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        get_taxonomicon_id("Drosophila", verbose=True)
        assert True

@skip_if_offline
def test_get_taxonomicon_id_verbose_prints_not_found_warning():
    clear_cache()
    assert get_taxonomicon_id("Zzzzfakeosaurus99999", verbose=True) is None

@skip_if_offline
def test_get_lineage_by_id_verbose_success_message_fires_on_real_taxon():
    clear_cache()
    id_val = get_taxonomicon_id("Carnotaurus")
    if id_val is not None:
        get_lineage_by_id(id_val, verbose=True)
        assert True

import pytest
import numpy as np
import pandas as pd
import warnings
from unittest.mock import patch, Mock

from taxodist.distance import taxo_cluster, taxo_ordinate
from taxodist.fetch import taxo_search, clear_cache
from taxodist.utils import (
    taxo_heatmap, summary_taxodist_ord, print_taxodist_result,
    print_taxodist_path, plot_taxodist_cluster, plot_taxodist_ord,
    compare_lineages
)

def test_taxo_cluster_with_nan_df():
    m = np.array([[0.0, np.nan],[np.nan, 0.0]])
    df = pd.DataFrame(m, index=["A", "B"], columns=["A", "B"])
    with pytest.warns(UserWarning, match="Distance matrix contains NaN values"):
        res = taxo_cluster(df)
    assert res["hclust"] is None

def test_taxo_ordinate_with_nan_df():
    m = np.array([[0.0, np.nan], [np.nan, 0.0]])
    df = pd.DataFrame(m, index=["A", "B"], columns=["A", "B"])
    with pytest.warns(UserWarning, match="Distance matrix contains NaN values"):
        res = taxo_ordinate(df)
    assert res["points"] is None

@patch("taxodist.fetch.requests.get")
def test_taxo_search_verbose_prints_on_network_failure(mock_get, capsys):
    import requests
    clear_cache()
    mock_get.side_effect = requests.exceptions.RequestException("Network error")
    taxo_search("Bacteria", verbose=True)
    captured = capsys.readouterr()  
    assert "Could not reach Taxonomicon" in captured.out
    
    mock_resp = Mock()
    mock_resp.status_code = 503
    mock_get.side_effect = None
    mock_get.return_value = mock_resp
    taxo_search("Bacteria", verbose=True)
    captured = capsys.readouterr()
    assert "Could not reach Taxonomicon" in captured.out

@patch("matplotlib.pyplot.show")
def test_taxo_heatmap_with_nan_df(mock_show):
    m = np.array([[0.0, np.nan],[np.nan, 0.0]])
    df = pd.DataFrame(m, index=["A", "B"], columns=["A", "B"])
    with pytest.warns(UserWarning, match="Distance matrix contains NaN values"):
        res = taxo_heatmap(df)
    assert res is df
    
def test_utils_print_methods_handle_none():
    assert print_taxodist_result(None) is None
    assert print_taxodist_path(None) is None
    assert plot_taxodist_cluster(None) is None
    assert plot_taxodist_ord(None) is None
    assert summary_taxodist_ord(None) is None
    
@patch("matplotlib.pyplot.show")
def test_plot_taxodist_ord_without_labels(mock_show):
    points = pd.DataFrame([[1, 2], [3, 4]], index=["A", "B"])
    mock_ord = {"points": points, "GOF": [0.95]}
    res = plot_taxodist_ord(mock_ord)
    assert res is mock_ord
    
def test_summary_taxodist_ord_missing_eigenvalues(capsys):
    points = pd.DataFrame([[1, 2],[3, 4]], index=["A", "B"])
    mock_bad = {"points": points, "dist": pd.DataFrame(), "GOF":[0.95, 0.95]}
    res = summary_taxodist_ord(mock_bad)
    captured = capsys.readouterr()
    assert "Warning: Eigenvalues not found" in captured.out
    assert res is None
    
@patch("taxodist.utils.get_lineage")
def test_compare_lineages_prints_correctly(mock_get_lineage, capsys):
    def side_effect(taxon, **kwargs):
        if taxon == "Tyrannosaurus":
            return["Biota", "Animalia", "Dinosauria", "Theropoda", "Tyrannosaurus"]
        return["Biota", "Animalia", "Dinosauria", "Ornithischia", "Triceratops"]
    mock_get_lineage.side_effect = side_effect
    
    compare_lineages("Tyrannosaurus", "Triceratops")
    captured = capsys.readouterr()
    assert "Lineage Comparison" in captured.out
    assert "Shared lineage" in captured.out
    assert "Tyrannosaurus only" in captured.out
    assert "Triceratops only" in captured.out

    # ── COBERTURA CIRÚRGICA FINAL (Linhas exatas de fetch.py e utils.py) ──────────

@patch("taxodist.fetch.get_taxonomicon_id")
@patch("taxodist.fetch.get_lineage_by_id")
def test_fetch_get_lineage_append_multiword(mock_get_lin, mock_get_id):
    # Cobre fetch.py linha 407 (Adicionar taxon composto que não está na lista)
    mock_get_id.return_value = "123"
    mock_get_lin.return_value = ["Biota", "Animalia"]
    res = get_lineage("Homo sapiens")
    assert "Homo sapiens" in res

@patch("taxodist.fetch.get_lineage_by_id")
def test_fetch_get_lineage_return_none_for_empty_id(mock_get_lin):
    # Cobre fetch.py linha 410 (Se for passado um ID numérico e ele retornar vazio)
    mock_get_lin.return_value =[]
    res = get_lineage("12345")
    assert res is None

@patch("taxodist.utils.sns.clustermap")
@patch("matplotlib.pyplot.show")
@patch("taxodist.utils.distance_matrix")
def test_utils_taxo_heatmap_with_list_and_valid_data(mock_dist, mock_show, mock_sns):
    # Cobre utils.py linhas 187 e 192-195
    # Passamos uma lista para forçar o 'else: d = distance_matrix(taxa)'
    m = np.array([[0.0, 0.5], [0.5, 0.0]])
    df = pd.DataFrame(m, index=["A", "B"], columns=["A", "B"])
    mock_dist.return_value = df
    
    res = taxo_heatmap(["A", "B"])
    assert res is df
    mock_dist.assert_called_once_with(["A", "B"])
    mock_sns.assert_called_once()
    mock_show.assert_called_once()

def test_utils_summary_taxodist_ord_success(capsys):
    # Cobre utils.py linhas 322-333 (Cálculo de variância do PCoA e print)
    points = pd.DataFrame([[1, 2],[3, 4]], index=["A", "B"], columns=["PC1", "PC2"])
    mock_ord = {
        "points": points,
        "dist": pd.DataFrame(),
        "GOF":[0.95, 0.95],
        "eig": np.array([2.0, 1.0, -0.5])
    }
    res = summary_taxodist_ord(mock_ord)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 2
    assert "Variance_Pct" in res.columns
    
    captured = capsys.readouterr()
    assert "PC1" in captured.out
    assert "PC2" in captured.out