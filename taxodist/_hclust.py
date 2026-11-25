"""
Hierarchical clustering identical to R's ``stats::hclust()``.

This is a port of the Fortran routines HCLUST and HCASS2 used by R
(F. Murtagh, 1986/1991; modifications for R by Ross Ihaka, Fritz Leisch, and
Martin Maechler, including the "ward.D2" method). Each agglomeration uses the
same nearest-neighbour list, the same comparisons (so ties are broken in the
same way), and the same floating-point operations in the same order, so the
merges, heights, and leaf order equal R's exactly. SciPy's ``linkage()``
differs from R in its tie breaking, in the median and centroid methods, and
does not provide "ward.D" or "mcquitty"; it is therefore not used.

The Lance-Williams updates are evaluated with NumPy on vectors of clusters,
which performs the same IEEE-754 operations as the Fortran loop.
"""

import sys

import numpy as np

#: Method names in the order of R's ``hclust()`` (Fortran ``iOpt`` = index + 1).
HCLUST_METHODS = (
    "ward.D", "single", "complete", "average", "mcquitty",
    "median", "centroid", "ward.D2",
)

# Fortran's DATA INF/1.D+300/.
_INF = 1e300


def match_hclust_method(method):
    """
    Full name of an hclust method, matched as R's ``hclust()`` does: ``"ward"``
    is renamed to ``"ward.D"`` with a message, then ``pmatch()`` is used (an
    exact match, or else a unique partial match).
    """
    if not isinstance(method, str):
        raise TypeError("method must be a single string naming a clustering method")
    if method == "ward":
        print(
            'The "ward" method has been renamed to "ward.D"; note new "ward.D2"',
            file=sys.stderr,
        )
        method = "ward.D"
    if method in HCLUST_METHODS:
        return method
    partial = [name for name in HCLUST_METHODS if method and name.startswith(method)]
    if len(partial) != 1:
        raise ValueError(f"invalid clustering method {method}")
    return partial[0]


def _first_min(values):
    """
    ``(index, value)`` of the first strict minimum below Fortran's INF, as in
    a loop ``IF (X(J) .LT. DMIN)`` started from ``DMIN = INF``; ``(None,
    INF)`` when no value is below INF.
    """
    if values.size == 0:
        return None, _INF
    position = int(np.argmin(values))
    value = values[position]
    if not value < _INF:
        return None, _INF
    return position, float(value)


def _hclust_fortran(diss, n, iopt, members):
    """
    HCLUST: agglomerations ``(ia, ib)`` (1-based, ``ia < ib``) and criterion
    values ``crit`` for ``n`` objects and the square dissimilarity matrix
    ``diss`` (modified in place).
    """
    flag = np.ones(n, dtype=bool)
    nn = np.zeros(n, dtype=np.int64)        # 0-based nearest neighbour
    disnn = np.zeros(n)
    membr = np.array(members, dtype=float)
    ia = np.zeros(n, dtype=np.int64)
    ib = np.zeros(n, dtype=np.int64)
    crit = np.zeros(n)
    im = jm = jj = 0
    is_ward = iopt in (1, 8)

    if iopt == 8:
        diss *= diss

    # Nearest neighbours to the right of each object.
    for i in range(n - 1):
        position, dmin = _first_min(diss[i, i + 1:])
        if position is not None:
            jm = i + 1 + position
        nn[i] = jm
        disnn[i] = dmin

    for step in range(n - 1):
        # Least dissimilarity in the list of nearest neighbours.
        candidates = np.where(flag[:n - 1], disnn[:n - 1], _INF)
        position, dmin = _first_min(candidates)
        if position is not None:
            im = position
            jm = int(nn[position])
        i2, j2 = min(im, jm), max(im, jm)
        ia[step] = i2 + 1
        ib[step] = j2 + 1
        if iopt == 8:
            dmin = float(np.sqrt(dmin))
        crit[step] = dmin
        flag[j2] = False

        # Update dissimilarities from the new cluster.
        ks = np.flatnonzero(flag)
        ks = ks[ks != i2]
        d1 = diss[i2, ks]
        d2 = diss[j2, ks]
        d12 = diss[i2, j2]
        m1, m2, mk = membr[i2], membr[j2], membr[ks]
        if is_ward:
            new = (m1 + mk) * d1 + (m2 + mk) * d2 - mk * d12
            new = new / (m1 + m2 + mk)
        elif iopt == 2:
            new = np.minimum(d1, d2)
        elif iopt == 3:
            new = np.maximum(d1, d2)
        elif iopt == 4:
            new = (m1 * d1 + m2 * d2) / (m1 + m2)
        elif iopt == 5:
            new = (d1 + d2) / 2
        elif iopt == 6:
            new = ((d1 + d2) - d12 / 2) / 2
        else:  # iopt == 7, centroid
            new = (m1 * d1 + m2 * d2 - m1 * m2 * d12 / (m1 + m2)) / (m1 + m2)
        diss[i2, ks] = new
        diss[ks, i2] = new

        right = ks > i2
        position, dmin = _first_min(new[right])
        if position is not None:
            jj = int(ks[right][position])
        left = ~right
        closer = new[left] < disnn[ks[left]]
        updated = ks[left][closer]
        disnn[updated] = new[left][closer]
        nn[updated] = i2

        membr[i2] = m1 + m2
        disnn[i2] = dmin
        nn[i2] = jj

        # Update the list of nearest neighbours insofar as required.
        for i in np.flatnonzero(flag[:n - 1] & ((nn[:n - 1] == i2) | (nn[:n - 1] == j2))):
            row = np.where(flag[i + 1:], diss[i, i + 1:], _INF)
            position, dmin = _first_min(row)
            if position is not None:
                jj = int(i + 1 + position)
            nn[i] = jj
            disnn[i] = dmin

    return ia[:n - 1], ib[:n - 1], crit[:n - 1]


def _hcass2(n, ia, ib):
    """HCASS2: R's ``merge`` matrix and ``order`` from the agglomerations."""
    ia = np.asarray(ia, dtype=np.int64)
    ib = np.asarray(ib, dtype=np.int64)
    iia = ia.copy()
    iib = ib.copy()
    for i in range(n - 2):
        k = min(ia[i], ib[i])
        later_a = np.flatnonzero(ia[i + 1:] == k) + i + 1
        later_b = np.flatnonzero(ib[i + 1:] == k) + i + 1
        iia[later_a] = -(i + 1)
        iib[later_b] = -(i + 1)
    iia = -iia
    iib = -iib
    for i in range(n - 1):
        if iia[i] > 0 and iib[i] < 0:
            iia[i], iib[i] = iib[i], iia[i]
        if iia[i] > 0 and iib[i] > 0:
            k1, k2 = min(iia[i], iib[i]), max(iia[i], iib[i])
            iia[i], iib[i] = k1, k2

    order = [int(iia[n - 2]), int(iib[n - 2])]
    for i in range(n - 2, 0, -1):
        j = order.index(i)
        order[j] = int(iia[i - 1])
        order.insert(j + 1, int(iib[i - 1]))
    merge = np.column_stack([iia, iib]).astype(np.int64)
    return merge, np.array([-value for value in order], dtype=np.int64)


def r_hclust(dist, method="complete"):
    """
    Hierarchical clustering with the algorithm of R's ``stats::hclust()``.

    Parameters
    ----------
    dist : array-like
        A square, symmetric dissimilarity matrix (only the lower triangle is
        used, as by R's ``as.dist()``).
    method : str
        One of ``HCLUST_METHODS``, matched as in R (``"ward"`` means
        ``"ward.D"``; unique abbreviations are accepted).

    Returns
    -------
    dict
        ``merge`` (an ``(n - 1, 2)`` integer array in R's convention:
        ``-j`` is observation ``j``, 1-based, and a positive ``k`` the
        cluster formed at step ``k``), ``height``, ``order`` (1-based leaf
        order), ``method`` (the full method name), and ``linkage`` (the same
        tree as a SciPy linkage matrix; see ``merge_to_linkage()``).
    """
    method = match_hclust_method(method)
    matrix = np.array(dist, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("dist must be a square matrix")
    n = matrix.shape[0]
    if n < 2:
        raise ValueError("must have n >= 2 objects to cluster")
    # R stores one value per pair: use the lower triangle for both halves.
    lower = np.tril(matrix, -1)
    matrix = lower + lower.T
    iopt = HCLUST_METHODS.index(method) + 1
    ia, ib, crit = _hclust_fortran(matrix, n, iopt, np.ones(n))
    merge, order = _hcass2(n, ia, ib)
    return {
        "merge": merge,
        "height": crit,
        "order": order,
        "method": method,
        "linkage": merge_to_linkage(merge, crit),
    }


def merge_to_linkage(merge, height):
    """
    Convert an R ``merge`` matrix and ``height`` vector to a SciPy linkage
    matrix: observation ``-j`` becomes leaf ``j - 1`` and the cluster of step
    ``k`` becomes ``n + k - 1``. The left/right order of each merge is kept,
    so ``scipy.cluster.hierarchy.dendrogram()`` and ``leaves_list()`` give
    R's ``order`` (minus one). Heights are kept as computed; the median and
    centroid methods can give non-monotone heights (inversions), which
    SciPy accepts for plotting.
    """
    merge = np.asarray(merge, dtype=np.int64)
    height = np.asarray(height, dtype=float)
    n = merge.shape[0] + 1
    linkage = np.zeros((n - 1, 4))
    sizes = np.zeros(n - 1, dtype=np.int64)
    for step, pair in enumerate(merge):
        size = 0
        for column, value in enumerate(pair):
            if value < 0:
                linkage[step, column] = -value - 1
                size += 1
            else:
                linkage[step, column] = n + value - 1
                size += sizes[value - 1]
        sizes[step] = size
        linkage[step, 2] = height[step]
        linkage[step, 3] = size
    return linkage
