"""Classification trees built directly from lineages."""

import re
import warnings

import numpy as np
import pandas as pd

from .distance import _matrix_input


class TaxodistTree:
    """
    A rooted classification tree in the layout of an ape ``phylo`` object.

    Attributes
    ----------
    edge : numpy.ndarray
        ``(n_edges, 2)`` integer array of ``(parent, child)`` node numbers.
        Tips are numbered ``1..n_tips`` and internal nodes from
        ``n_tips + 1`` (the root), in preorder ("cladewise" order).
    edge_length : numpy.ndarray
        Branch length of each edge.
    tip_label : list of str
        Input labels of the tips, indexed by tip number minus one.
    node_label : list of str
        Clade names of the internal nodes, indexed by node number minus
        ``n_tips + 1``.
    n_node : int
        Number of internal nodes.
    """

    def __init__(self, edge, edge_length, tip_label, node_label):
        self.edge = np.asarray(edge, dtype=np.int64).reshape(-1, 2)
        self.edge_length = np.asarray(edge_length, dtype=float)
        self.tip_label = list(tip_label)
        self.node_label = list(node_label)
        self.n_node = len(self.node_label)

    @property
    def n_tips(self):
        return len(self.tip_label)

    def __repr__(self):
        return (
            f"TaxodistTree with {self.n_tips} tips and {self.n_node} internal "
            f"nodes (root: {self.node_label[0]!r})"
        )

    def _children(self):
        children = {}
        for (parent, child), length in zip(self.edge.tolist(), self.edge_length):
            children.setdefault(parent, []).append((child, length))
        return children

    def to_newick(self):
        """Return the tree in Newick format with node labels and branch lengths."""
        children = self._children()

        def quote(label):
            if label == "":
                return ""
            if re.search(r"[\s(),:;'\[\]]", label):
                return "'" + label.replace("'", "''") + "'"
            return label

        def render(node):
            if node <= self.n_tips:
                return quote(self.tip_label[node - 1])
            inner = ",".join(
                f"{render(child)}:{length:.15g}" for child, length in children[node]
            )
            return f"({inner}){quote(self.node_label[node - self.n_tips - 1])}"

        return render(self.n_tips + 1) + ";"

    def cophenetic(self):
        """Return the patristic distance matrix between tips as a DataFrame."""
        children = self._children()
        depth = {self.n_tips + 1: 0.0}
        parent_of = {}
        order = [self.n_tips + 1]
        for node in order:
            for child, length in children.get(node, []):
                depth[child] = depth[node] + length
                parent_of[child] = node
                order.append(child)

        def ancestors(node):
            path = [node]
            while node in parent_of:
                node = parent_of[node]
                path.append(node)
            return path

        paths = [ancestors(tip) for tip in range(1, self.n_tips + 1)]
        values = np.zeros((self.n_tips, self.n_tips))
        for i in range(self.n_tips):
            seen = set(paths[i])
            for j in range(i + 1, self.n_tips):
                mrca = next(node for node in paths[j] if node in seen)
                distance = depth[i + 1] + depth[j + 1] - 2 * depth[mrca]
                values[i, j] = values[j, i] = distance
        return pd.DataFrame(values, index=self.tip_label, columns=self.tip_label)


def taxo_tree(taxa, verbose=False, progress=True):
    """
    Build a taxonomic tree from lineages

    Builds a tree whose internal nodes are the named clades of the source
    classification, directly from the lineages of the supplied taxa. Unlike
    taxo_cluster(), which reconstructs a dendrogram from distances, the
    topology here is the classification itself.

    Parameters
    ----------
    taxa : sequence of str, TaxodistResolution, or TaxodistBundle
        Taxon names, or resolutions whose stored lineages are reused.
    verbose, progress : bool
        Passed to lineage retrieval for taxon names.

    Returns
    -------
    TaxodistTree
        Tip labels are the input labels and node labels are clade names.
        ``tree.to_newick()`` exports the tree for other software (e.g.
        Bio.Phylo, ete3, dendropy) and ``tree.cophenetic()`` returns its
        patristic distances. Taxa without a lineage are dropped with a
        warning.

    Notes
    -----
    Clades with a single descendant among the supplied taxa are collapsed, so
    every internal node is a point where the taxa diverge and is labeled with
    the deepest clade shared by its descendants.

    Branch lengths make the tree ultrametric with patristic distances equal
    to distance_matrix(): an internal node for a clade at depth ``h`` sits at
    height ``1 / (2h)`` and tips at height zero. A taxon that is an ancestor
    of another supplied taxon hangs from its clade node by a branch of length
    ``1 / (2h)``, and taxa with identical lineages are joined by zero-length
    branches. All lineages must share the same root node.
    """
    labels, lineages = _matrix_input(taxa, verbose=verbose, progress=progress)
    missing = [label for label, lineage in zip(labels, lineages) if lineage is None]
    if missing:
        warnings.warn(
            f"{len(missing)} taxa without a lineage were dropped from the tree: "
            + ", ".join(map(str, missing))
        )
        kept = [(label, lineage) for label, lineage in zip(labels, lineages)
                if lineage is not None]
        labels = [label for label, _ in kept]
        lineages = [lineage for _, lineage in kept]
    if len(lineages) < 2:
        raise ValueError("At least two taxa with lineages are required to build a tree.")
    lineages = [list(lineage) for lineage in lineages]
    if any(not lineage for lineage in lineages) or len({lin[0] for lin in lineages}) > 1:
        raise ValueError(
            "All lineages must share the same root node; taxa without a common "
            "ancestor have infinite distance and cannot be placed in one tree."
        )

    def build(indices, depth):
        # Taxa in `indices` share the lineage prefix of length `depth`.
        clade = lineages[indices[0]][depth - 1]
        ended = [i for i in indices if len(lineages[i]) == depth]
        continuing = [i for i in indices if len(lineages[i]) > depth]

        children = []
        if len(ended) == 1:
            children.append({"tip": ended[0]})
        elif len(ended) > 1:
            # Identical lineages: join at height zero so their distance is 0.
            children.append({
                "label": "" if continuing else clade,
                "height": 0.0,
                "children": [{"tip": i} for i in ended],
            })
        groups = {}
        for i in continuing:
            groups.setdefault(lineages[i][depth], []).append(i)
        children.extend(build(group, depth + 1) for group in groups.values())

        if len(children) == 1:
            return children[0]
        return {"label": clade, "height": 1.0 / (2 * depth), "children": children}

    root = build(list(range(len(lineages))), 1)

    n_tips = len(lineages)
    tip_label = [None] * n_tips
    node_label = []
    edges = []
    lengths = []
    counters = {"tip": 0, "node": n_tips}

    def visit(node, parent=None, parent_height=None):
        if "tip" in node:
            counters["tip"] += 1
            number = counters["tip"]
            tip_label[number - 1] = labels[node["tip"]]
            height = 0.0
        else:
            counters["node"] += 1
            number = counters["node"]
            node_label.append(node["label"])
            height = node["height"]
        if parent is not None:
            edges.append((parent, number))
            lengths.append(parent_height - height)
        for child in node.get("children", []):
            visit(child, number, height)

    visit(root)
    return TaxodistTree(edges, lengths, tip_label, node_label)
