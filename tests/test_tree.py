"""taxo_tree and TaxodistTree (0.9.0)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import taxodist

DINOSAURS = {
    "Tyrannosaurus": ["Biota", "Dinosauria", "Theropoda", "Tyrannosaurus"],
    "Velociraptor": ["Biota", "Dinosauria", "Theropoda", "Velociraptor"],
    "Triceratops": ["Biota", "Dinosauria", "Ornithischia", "Triceratops"],
    "Homo": ["Biota", "Mammalia", "Homo"],
}


def test_taxo_tree_builds_named_clades():
    tree = taxodist.taxo_tree(taxodist.taxo_from_lineages(DINOSAURS), progress=False)
    assert isinstance(tree, taxodist.TaxodistTree)
    assert tree.n_tips == 4 and tree.n_node == 3
    assert tree.node_label == ["Biota", "Dinosauria", "Theropoda"]
    assert sorted(tree.tip_label) == sorted(DINOSAURS)
    assert tree.edge[0, 0] == 5
    assert (tree.edge_length >= 0).all()
    assert "TaxodistTree with 4 tips" in repr(tree)


def test_cophenetic_distances_reproduce_distance_matrix():
    lineages = dict(DINOSAURS)
    lineages.update({
        "Dinosauria": ["Biota", "Dinosauria"],
        "Rex1": DINOSAURS["Tyrannosaurus"],
        "Rex2": DINOSAURS["Tyrannosaurus"],
        "Theropoda": ["Biota", "Dinosauria", "Theropoda"],
        "Theropoda2": ["Biota", "Dinosauria", "Theropoda"],
    })
    resolution = taxodist.taxo_from_lineages(lineages)
    tree = taxodist.taxo_tree(resolution, progress=False)
    expected = taxodist.distance_matrix(resolution, progress=False)
    observed = tree.cophenetic().loc[expected.index, expected.columns]
    np.testing.assert_allclose(observed.to_numpy(), expected.to_numpy(), atol=1e-12)
    assert "" in tree.node_label
    assert "(Tyrannosaurus:0,Rex1:0,Rex2:0)Tyrannosaurus" in tree.to_newick()
    assert "(Theropoda:0,Theropoda2:0):" in tree.to_newick()


def test_identical_lineages_form_a_labeled_group():
    tree = taxodist.taxo_tree(taxodist.taxo_from_lineages({
        "A": ["Biota", "Alpha"], "B": ["Biota", "Alpha"], "C": ["Biota", "Beta"],
    }), progress=False)
    assert tree.node_label == ["Biota", "Alpha"]


def test_newick_export_quotes_labels():
    tree = taxodist.taxo_tree(taxodist.taxo_from_lineages({
        "Homo sapiens": ["Biota", "Homo", "Homo sapiens"],
        "O'Brien": ["Biota", "Homo", "O'Brien"],
        "Pan": ["Biota", "Pan"],
    }), progress=False)
    newick = tree.to_newick()
    assert newick.endswith(";")
    assert "'Homo sapiens'" in newick
    assert "'O''Brien'" in newick
    assert newick.endswith(")Biota;")


def test_taxo_tree_accepts_names_and_bundles():
    with patch(
        "taxodist.distance.get_lineage",
        side_effect=lambda taxon, **kwargs: DINOSAURS[taxon],
    ):
        tree = taxodist.taxo_tree(list(DINOSAURS), progress=False)
    assert tree.n_node == 3
    bundle = taxodist.taxo_bundle(taxodist.taxo_from_lineages(DINOSAURS), progress=False)
    assert taxodist.taxo_tree(bundle).n_node == 3


def test_taxo_tree_drops_missing_lineages_and_validates_roots():
    resolution = taxodist.taxo_from_lineages(DINOSAURS)
    resolution.at[2, "lineage"] = None
    with pytest.warns(UserWarning, match="Triceratops"):
        tree = taxodist.taxo_tree(resolution)
    assert "Triceratops" not in tree.tip_label

    two = taxodist.taxo_from_lineages({k: DINOSAURS[k] for k in list(DINOSAURS)[:2]})
    two.at[1, "lineage"] = None
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="At least two"):
        taxodist.taxo_tree(two)

    with pytest.raises(ValueError, match="same root"):
        taxodist.taxo_tree(taxodist.taxo_from_lineages({
            "A": ["Biota", "A"], "B": ["Natura", "B"],
        }))
