"""Portable taxodist analysis bundles compatible with the R implementation."""

from datetime import datetime, timezone
import json
import math
import os

import numpy as np
import pandas as pd

from ._version import __version__
from .fetch import (
    RESOLUTION_COLUMNS,
    TaxodistResolution,
    _make_resolution,
    taxo_resolve,
)


class TaxodistBundle(dict):
    """Dictionary-like portable analysis bundle."""

    _taxodist_bundle = True

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def _utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


#: Schema version written by write_taxodist_bundle().
BUNDLE_SCHEMA_VERSION = "1.1"
_SUPPORTED_SCHEMA_VERSIONS = ("1.0", "1.1")

_METRIC_NAMES = {
    "inverse_mrca_depth": "reciprocal",
    "path_length": "path",
    "wu_palmer": "wu_palmer",
    "ranked_path_length": "ranked",
}


def _metric_spec(metric, levels=None):
    """Portable description of a metric, as stored in bundles."""
    from .distance import _rank_levels

    definitions = {
        "reciprocal": (
            "inverse_mrca_depth",
            "0 for identical lineages; otherwise 1 / depth(MRCA)",
        ),
        "path": (
            "path_length",
            "0 for identical lineages; otherwise depth(A) + depth(B) - "
            "2 depth(MRCA)",
        ),
        "wu_palmer": (
            "wu_palmer",
            "1 - 2 depth(MRCA) / (depth(A) + depth(B))",
        ),
        "ranked": (
            "ranked_path_length",
            "Half the path length between the lineages reduced to the root, "
            "the nodes whose rank is one of rank_levels, and the terminal node",
        ),
    }
    name, definition = definitions[metric]
    spec = {
        "name": name,
        "definition": definition,
        "root_depth": 1,
        "common_ancestor": "continuous common lineage prefix",
    }
    if metric == "ranked":
        spec["rank_levels"] = list(_rank_levels() if levels is None else levels)
    return spec


def _metric_from_spec(spec):
    """``(metric, rank_levels)`` from a stored metric description."""
    from .distance import _DEFAULT_RANK_LEVELS

    name = spec.get("name") if isinstance(spec, dict) else None
    if not isinstance(name, str) or name not in _METRIC_NAMES:
        raise ValueError("Invalid bundle: unknown distance metric.")
    metric = _METRIC_NAMES[name]
    if metric != "ranked":
        return metric, list(_DEFAULT_RANK_LEVELS)
    levels = spec.get("rank_levels")
    if isinstance(levels, str):
        # R writes a single rank level as a JSON string.
        levels = [levels]
    if (
        not isinstance(levels, (list, tuple))
        or not levels
        or any(not isinstance(level, str) for level in levels)
    ):
        raise ValueError("Invalid bundle: the ranked metric requires rank_levels.")
    return metric, [level.lower() for level in levels]


def taxo_bundle(taxa, ambiguity="warn", verbose=False, progress=True, within=None,
                metric="reciprocal"):
    """
    Combine resolution, distances, metric definition, and provenance.

    ``ambiguity``, ``verbose``, ``progress``, and ``within`` are passed to
    ``taxo_resolve()`` when ``taxa`` is a sequence of names. ``metric`` is the
    distance metric of the stored matrix (see taxo_distance()); the bundle
    records it (with the rank levels for ``"ranked"``), so the matrix can be
    verified from the stored lineages and ranks.
    """
    from .distance import _match_metric, _rank_levels, distance_matrix

    metric = _match_metric(metric)
    levels = _rank_levels() if metric == "ranked" else None
    resolution = (
        taxa
        if isinstance(taxa, TaxodistResolution)
        else taxo_resolve(
            taxa,
            ambiguity=ambiguity,
            verbose=verbose,
            progress=progress,
            within=within,
        )
    )
    source_name = getattr(resolution, "source", None)
    if source_name is None or pd.isna(source_name) or not str(source_name).strip():
        # As in R, a resolution without a recorded source is assumed to come
        # from The Taxonomicon. pd.concat() and other pandas operations can
        # drop the provenance, so this default is only applied when every
        # selected ID is a Taxonomicon ID; otherwise the source is unknown and
        # labelling it would be wrong.
        ids = [
            identifier for identifier in resolution["id"]
            if identifier is not None and not pd.isna(identifier)
        ]
        if not all(str(identifier).isdigit() for identifier in ids):
            raise ValueError(
                "The resolution has no source provenance (for example after "
                "pd.concat() of resolutions from different sources) and its "
                "IDs are not Taxonomicon IDs. Set `resolution.source` (and "
                "`resolution.source_url`, `resolution.retrieved_at`) before "
                "calling taxo_bundle()."
            )
        source_name = "The Taxonomicon"
    source_url = getattr(resolution, "source_url", None)
    if source_url is None and source_name == "The Taxonomicon":
        source_url = "http://taxonomicon.taxonomy.nl"
    metric_spec = _metric_spec(metric, levels)
    matrix = distance_matrix(resolution, progress=progress, metric=metric)
    return TaxodistBundle(
        schema_version=BUNDLE_SCHEMA_VERSION,
        created_at=_utc_timestamp(),
        source={
            "name": source_name,
            "url": source_url,
            "retrieved_at": getattr(resolution, "retrieved_at", None),
        },
        software={"name": "taxodist", "version": __version__, "language": "Python"},
        metric=metric_spec,
        resolution=resolution,
        matrix=matrix,
    )


def validate_taxodist_bundle(bundle, check_distances=True):
    """
    Validate the internal consistency of a TaxodistBundle.

    Structural checks are cheap and always run. With ``check_distances=True``
    (the default, used when writing and reading bundles) every stored
    distance is also recomputed from the stored lineages; using a bundle in
    ``distance_matrix()`` skips that step.
    """
    from .distance import _distance_matrix_impl

    if not isinstance(bundle, TaxodistBundle):
        raise TypeError("bundle must be a TaxodistBundle object")
    required = {
        "schema_version",
        "created_at",
        "source",
        "software",
        "metric",
        "resolution",
        "matrix",
    }
    if not required.issubset(bundle):
        raise ValueError("Invalid TaxodistBundle: required fields are missing")
    if bundle["schema_version"] not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported taxodist bundle schema: {bundle['schema_version']}"
        )
    metric, levels = _metric_from_spec(bundle["metric"])
    if bundle["schema_version"] == "1.0" and metric != "reciprocal":
        raise ValueError(
            "Invalid bundle: schema 1.0 supports only the reciprocal metric."
        )
    resolution = bundle["resolution"]
    if not isinstance(resolution, TaxodistResolution):
        raise ValueError("Invalid bundle: resolution has the wrong class")
    if not set(RESOLUTION_COLUMNS).issubset(resolution.columns):
        raise ValueError("Invalid bundle: resolution fields are missing")
    allowed = {"resolved", "ambiguous", "unresolved", "retrieval_error"}
    if not set(resolution["status"]).issubset(allowed):
        raise ValueError("Invalid bundle: unknown resolution status")

    has_ranks = "ranks" in resolution.columns
    for index, row in resolution.reset_index(drop=True).iterrows():
        number = index + 1
        candidates = row["candidates"]
        lineage = row["lineage"]
        count = row["n_candidates"]
        if pd.isna(count) or int(count) != count or count < 0:
            raise ValueError("Invalid bundle: candidate counts must be non-negative integers")
        if not isinstance(candidates, pd.DataFrame) or not {"id", "name"}.issubset(
            candidates.columns
        ):
            raise ValueError(f"Invalid bundle: malformed candidate table at row {number}")
        if len(candidates) != count:
            raise ValueError(f"Invalid bundle: candidate count mismatch at row {number}")
        depth = row["lineage_depth"]
        if lineage is None:
            if not pd.isna(depth):
                raise ValueError(f"Invalid bundle: lineage depth mismatch at row {number}")
        else:
            if (
                not isinstance(lineage, (list, tuple))
                or any(not isinstance(node, str) or not node.strip() for node in lineage)
            ):
                raise ValueError(f"Invalid bundle: malformed lineage at row {number}")
            if pd.isna(depth) or len(lineage) != int(depth):
                raise ValueError(f"Invalid bundle: lineage depth mismatch at row {number}")

        id_missing = pd.isna(row["id"])
        name_missing = pd.isna(row["resolved_name"])
        if row["status"] in {"resolved", "ambiguous"}:
            if id_missing or lineage is None or candidates.empty:
                raise ValueError(f"Invalid bundle: incomplete resolved record at row {number}")
            if str(candidates.iloc[0]["id"]) != row["id"]:
                raise ValueError(f"Invalid bundle: selected candidate mismatch at row {number}")
        else:
            if not id_missing or lineage is not None or not name_missing:
                raise ValueError(f"Invalid bundle: incomplete unresolved record at row {number}")
            if row["status"] == "unresolved" and not candidates.empty:
                raise ValueError(f"Invalid bundle: unresolved record has candidates at row {number}")
        # Ranks of a missing lineage are ignored (and written as null).
        if has_ranks and lineage is not None:
            ranks = _rank_cell(row["ranks"])
            if (
                not isinstance(ranks, (list, tuple))
                or len(ranks) != len(lineage)
                or any(rank is not None and not isinstance(rank, str) for rank in ranks)
            ):
                raise ValueError(
                    "Invalid bundle: lineage ranks do not match the lineage "
                    f"at row {number}"
                )

    matrix = bundle["matrix"]
    if not isinstance(matrix, pd.DataFrame):
        raise ValueError("Invalid bundle: matrix must be a pandas DataFrame")
    labels = list(resolution["input"])
    if list(matrix.index) != labels or list(matrix.columns) != labels:
        raise ValueError("Invalid bundle: matrix labels do not match the resolution inputs")
    if not check_distances:
        return bundle
    expected = _distance_matrix_impl(
        resolution, verbose=False, progress=False, metric=metric, levels=levels
    )
    if matrix.shape != expected.shape or not np.allclose(
        matrix.to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        rtol=0,
        atol=math.sqrt(np.finfo(float).eps),
        equal_nan=True,
    ):
        raise ValueError("Invalid bundle: stored distances do not match the stored lineages")
    return bundle


def _rank_cell(value):
    """A ``ranks`` cell (None, NaN after pandas operations, or a sequence)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def _portable_value(value):
    if pd.isna(value):
        return None
    if math.isinf(float(value)):
        return "Infinity" if value > 0 else "-Infinity"
    return float(value)


def write_taxodist_bundle(bundle, file, pretty=True):
    """
    Write a validated bundle as portable JSON (shared schema, version 1.1).

    The JSON records the metric (``metric.name`` is one of
    ``inverse_mrca_depth``, ``path_length``, ``wu_palmer``, or
    ``ranked_path_length``, the latter with ``rank_levels``) and, when the
    resolution has a ``ranks`` column, the ``lineage_ranks`` of every taxon
    (null for unranked nodes and for missing lineages).
    """
    validate_taxodist_bundle(bundle)
    resolution = bundle["resolution"]
    has_ranks = "ranks" in resolution.columns
    records = []
    for _, row in resolution.iterrows():
        record = {
            "input": row["input"],
            "resolved_name": (
                None if pd.isna(row["resolved_name"]) else row["resolved_name"]
            ),
            "id": None if pd.isna(row["id"]) else row["id"],
            "status": row["status"],
            "n_candidates": int(row["n_candidates"]),
            "lineage_depth": (
                None if pd.isna(row["lineage_depth"]) else int(row["lineage_depth"])
            ),
            "lineage": None if row["lineage"] is None else list(row["lineage"]),
            "candidates": row["candidates"][["id", "name"]].to_dict("records"),
        }
        if has_ranks:
            ranks = _rank_cell(row["ranks"])
            # null for a missing lineage; null entries for unranked nodes.
            record["lineage_ranks"] = (
                None if row["lineage"] is None or ranks is None
                else [None if rank is None else str(rank) for rank in ranks]
            )
        records.append(record)
    matrix = bundle["matrix"]
    portable = {
        "format": "taxodist_bundle",
        "schema_version": bundle["schema_version"],
        "created_at": bundle["created_at"],
        "source": bundle["source"],
        "software": bundle["software"],
        "metric": bundle["metric"],
        "taxa": records,
        "matrix": {
            "labels": list(matrix.index),
            "values": [
                [_portable_value(value) for value in row]
                for row in matrix.to_numpy(dtype=float)
            ],
        },
    }
    with open(file, "w", encoding="utf-8") as stream:
        json.dump(
            portable,
            stream,
            ensure_ascii=False,
            allow_nan=False,
            indent=2 if pretty else None,
        )
        stream.write("\n")
    return os.path.abspath(file)


def _parse_matrix(matrix_data):
    labels = [str(label) for label in matrix_data.get("labels", [])]
    rows = matrix_data.get("values", [])
    if len(rows) != len(labels):
        raise ValueError("Invalid bundle JSON: matrix row count does not match labels")
    values = []
    for row in rows:
        if len(row) != len(labels):
            raise ValueError("Invalid bundle JSON: matrix must be square")
        values.append(
            [
                np.nan
                if value is None
                else np.inf
                if value == "Infinity"
                else -np.inf
                if value == "-Infinity"
                else float(value)
                for value in row
            ]
        )
    matrix = pd.DataFrame(values, index=labels, columns=labels, dtype=float)
    if len(labels) and (matrix.values.diagonal() != 0).any():
        raise ValueError("Invalid bundle JSON: matrix diagonal must contain zeros")
    if not np.array_equal(matrix.values, matrix.values.T, equal_nan=True):
        raise ValueError("Invalid bundle JSON: distance matrix must be symmetric")
    return matrix


def read_taxodist_bundle(file):
    """
    Read and validate a portable bundle written by R, Python, or Julia.

    Schema versions 1.0 (reciprocal metric only) and 1.1 are supported. The
    stored distances are recomputed with the stored metric (and rank levels)
    and must match. The returned matrix records the metric in
    ``attrs["metric"]``.
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"Bundle file not found: {file}")
    try:
        with open(file, encoding="utf-8") as stream:
            raw = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not parse bundle JSON: {error}") from error
    if raw.get("format") != "taxodist_bundle":
        raise ValueError("Invalid bundle JSON: unrecognized format")
    if raw.get("schema_version") not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported taxodist bundle schema: {raw.get('schema_version')}"
        )
    required = {"created_at", "source", "software", "metric", "taxa", "matrix"}
    if not required.issubset(raw):
        raise ValueError("Invalid bundle JSON: required fields are missing")

    with_ranks = any("lineage_ranks" in record for record in raw["taxa"])
    rows = []
    for record in raw["taxa"]:
        candidate_frame = pd.DataFrame(
            record.get("candidates") or [], columns=["id", "name"]
        )
        row = {
            "input": record.get("input"),
            "resolved_name": record.get("resolved_name"),
            "id": record.get("id"),
            "status": record.get("status"),
            "n_candidates": record.get("n_candidates"),
            "lineage_depth": record.get("lineage_depth"),
            "lineage": record.get("lineage"),
            "candidates": candidate_frame,
        }
        if with_ranks:
            ranks = record.get("lineage_ranks")
            row["ranks"] = (
                None if ranks is None
                else [None if rank is None else str(rank) for rank in ranks]
            )
        rows.append(row)
    source = raw["source"]
    resolution = _make_resolution(
        rows,
        source=source.get("name"),
        source_url=source.get("url"),
        retrieved_at=source.get("retrieved_at"),
        ranks=with_ranks,
    )
    matrix = _parse_matrix(raw["matrix"])
    bundle = TaxodistBundle(
        schema_version=raw["schema_version"],
        created_at=raw["created_at"],
        source=source,
        software=raw["software"],
        metric=raw["metric"],
        resolution=resolution,
        matrix=matrix,
    )
    validate_taxodist_bundle(bundle)
    matrix.attrs["metric"] = _metric_from_spec(raw["metric"])[0]
    return bundle


class TaxodistBundleComparison(dict):
    """Result of compare_bundles(); ``print()`` shows a summary."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __str__(self):
        taxa = self["taxa"]
        changed = taxa.loc[taxa["change"] != "unchanged"]
        lines = [
            "taxodist Bundle Comparison",
            f"* Old bundle: {self['created_old']}",
            f"* New bundle: {self['created_new']}",
            f"* Taxa compared: {len(taxa)} ({len(changed)} changed)",
            f"* Distances changed: {len(self['distances'])}",
        ]
        if changed.empty:
            lines.append("No taxon changed between the two bundles.")
        else:
            lines.extend(f"> {row.input}: {row.change}" for row in changed.itertuples())
        return "\n".join(lines)


def compare_bundles(old, new, tolerance=math.sqrt(np.finfo(float).eps)):
    """
    Compare two taxodist bundles

    Reports how the resolution of each taxon and the pairwise distances
    changed between two bundles, for example when an analysis is repeated
    after The Taxonomicon updated its classification.

    Parameters
    ----------
    old, new : TaxodistBundle or str
        Bundles, or paths to JSON files written by write_taxodist_bundle().
    tolerance : float
        Distances closer than this are considered equal.

    Returns
    -------
    TaxodistBundleComparison
        A dictionary with:
        - taxa: DataFrame with one row per input label found in either bundle
          and columns input, change, status_old, status_new, id_old, id_new,
          depth_old, depth_new. ``change`` is one of "unchanged", "added",
          "removed", "status_changed", "id_changed", or "lineage_changed".
        - distances: DataFrame of pairs of inputs present in both bundles
          whose distance changed (taxon_a, taxon_b, distance_old,
          distance_new).
        - created_old, created_new: creation timestamps.
    """
    def as_bundle(value, name):
        if isinstance(value, (str, os.PathLike)):
            value = read_taxodist_bundle(value)
        if not isinstance(value, TaxodistBundle):
            raise TypeError(f"{name} must be a TaxodistBundle or the path to a bundle file")
        return validate_taxodist_bundle(value, check_distances=False)

    old = as_bundle(old, "old")
    new = as_bundle(new, "new")
    res_old = old["resolution"].reset_index(drop=True)
    res_new = new["resolution"].reset_index(drop=True)

    def first_positions(resolution):
        positions = {}
        for position, label in enumerate(resolution["input"]):
            positions.setdefault(label, position)
        return positions

    pos_old = first_positions(res_old)
    pos_new = first_positions(res_new)
    inputs = list(dict.fromkeys(list(res_old["input"]) + list(res_new["input"])))

    def value(resolution, positions, label, column):
        if label not in positions:
            return None
        item = resolution.iloc[positions[label]][column]
        return None if item is None or (not isinstance(item, list) and pd.isna(item)) else item

    rows = []
    for label in inputs:
        record = {"input": label}
        for suffix, resolution, positions in (
            ("old", res_old, pos_old),
            ("new", res_new, pos_new),
        ):
            record[f"status_{suffix}"] = value(resolution, positions, label, "status")
            record[f"id_{suffix}"] = value(resolution, positions, label, "id")
            record[f"depth_{suffix}"] = value(resolution, positions, label, "lineage_depth")
            record[f"lineage_{suffix}"] = value(resolution, positions, label, "lineage")
        if label not in pos_old:
            change = "added"
        elif label not in pos_new:
            change = "removed"
        elif record["status_old"] != record["status_new"]:
            change = "status_changed"
        elif record["id_old"] != record["id_new"]:
            change = "id_changed"
        elif record["lineage_old"] != record["lineage_new"]:
            change = "lineage_changed"
        else:
            change = "unchanged"
        record["change"] = change
        rows.append(record)

    taxa = pd.DataFrame(
        rows,
        columns=["input", "change", "status_old", "status_new", "id_old", "id_new",
                 "depth_old", "depth_new"],
    )
    taxa["depth_old"] = pd.array(taxa["depth_old"], dtype="Int64")
    taxa["depth_new"] = pd.array(taxa["depth_new"], dtype="Int64")

    shared = [label for label in inputs if label in pos_old and label in pos_new]
    changed_pairs = []
    if len(shared) > 1:
        matrix_old = old["matrix"].to_numpy(dtype=float)
        matrix_new = new["matrix"].to_numpy(dtype=float)
        for j in range(len(shared)):
            for i in range(j + 1, len(shared)):
                a, b = shared[j], shared[i]
                value_old = matrix_old[pos_old[b], pos_old[a]]
                value_new = matrix_new[pos_new[b], pos_new[a]]
                both_missing = np.isnan(value_old) and np.isnan(value_new)
                equal = both_missing or value_old == value_new or (
                    not np.isnan(value_old) and not np.isnan(value_new)
                    and abs(value_old - value_new) <= tolerance
                )
                if not equal:
                    changed_pairs.append({
                        "taxon_a": a,
                        "taxon_b": b,
                        "distance_old": value_old,
                        "distance_new": value_new,
                    })
    distances = pd.DataFrame(
        changed_pairs,
        columns=["taxon_a", "taxon_b", "distance_old", "distance_new"],
    )

    return TaxodistBundleComparison(
        taxa=taxa,
        distances=distances,
        created_old=old["created_at"],
        created_new=new["created_at"],
    )