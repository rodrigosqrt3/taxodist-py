"""Portable taxodist analysis bundles compatible with the R implementation."""

from datetime import datetime, timezone
import json
import math
import os

import numpy as np
import pandas as pd

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


def taxo_bundle(taxa, ambiguity="warn", verbose=False, progress=True):
    """Combine resolution, distances, metric definition, and provenance."""
    from .distance import distance_matrix

    resolution = (
        taxa
        if isinstance(taxa, TaxodistResolution)
        else taxo_resolve(
            taxa,
            ambiguity=ambiguity,
            verbose=verbose,
            progress=progress,
        )
    )
    source_name = getattr(resolution, "source", None)
    if source_name is None or pd.isna(source_name) or not str(source_name).strip():
        source_name = "The Taxonomicon"
    source_url = getattr(resolution, "source_url", None)
    if source_url is None and source_name == "The Taxonomicon":
        source_url = "http://taxonomicon.taxonomy.nl"
    return TaxodistBundle(
        schema_version="1.0",
        created_at=_utc_timestamp(),
        source={
            "name": source_name,
            "url": source_url,
            "retrieved_at": getattr(resolution, "retrieved_at", None),
        },
        software={"name": "taxodist", "version": "0.8.0", "language": "Python"},
        metric={
            "name": "inverse_mrca_depth",
            "definition": "0 for identical lineages; otherwise 1 / depth(MRCA)",
            "root_depth": 1,
            "common_ancestor": "continuous common lineage prefix",
        },
        resolution=resolution,
        matrix=distance_matrix(resolution, progress=progress),
    )


def validate_taxodist_bundle(bundle):
    """Validate the internal consistency of a TaxodistBundle."""
    from .distance import distance_matrix

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
    if bundle["schema_version"] != "1.0":
        raise ValueError(
            f"Unsupported taxodist bundle schema: {bundle['schema_version']}"
        )
    resolution = bundle["resolution"]
    if not isinstance(resolution, TaxodistResolution):
        raise ValueError("Invalid bundle: resolution has the wrong class")
    if not set(RESOLUTION_COLUMNS).issubset(resolution.columns):
        raise ValueError("Invalid bundle: resolution fields are missing")
    allowed = {"resolved", "ambiguous", "unresolved", "retrieval_error"}
    if not set(resolution["status"]).issubset(allowed):
        raise ValueError("Invalid bundle: unknown resolution status")

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

        if row["status"] in {"resolved", "ambiguous"}:
            if row["id"] is None or lineage is None or candidates.empty:
                raise ValueError(f"Invalid bundle: incomplete resolved record at row {number}")
            if str(candidates.iloc[0]["id"]) != row["id"]:
                raise ValueError(f"Invalid bundle: selected candidate mismatch at row {number}")
        else:
            if row["id"] is not None or lineage is not None or row["resolved_name"] is not None:
                raise ValueError(f"Invalid bundle: incomplete unresolved record at row {number}")
            if row["status"] == "unresolved" and not candidates.empty:
                raise ValueError(f"Invalid bundle: unresolved record has candidates at row {number}")

    matrix = bundle["matrix"]
    if not isinstance(matrix, pd.DataFrame):
        raise ValueError("Invalid bundle: matrix must be a pandas DataFrame")
    labels = list(resolution["input"])
    if list(matrix.index) != labels or list(matrix.columns) != labels:
        raise ValueError("Invalid bundle: matrix labels do not match the resolution inputs")
    expected = distance_matrix(resolution, progress=False)
    if matrix.shape != expected.shape or not np.allclose(
        matrix.to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        rtol=0,
        atol=math.sqrt(np.finfo(float).eps),
        equal_nan=True,
    ):
        raise ValueError("Invalid bundle: stored distances do not match the stored lineages")
    return bundle


def _portable_value(value):
    if pd.isna(value):
        return None
    if math.isinf(float(value)):
        return "Infinity" if value > 0 else "-Infinity"
    return float(value)


def write_taxodist_bundle(bundle, file, pretty=True):
    """Write a validated bundle using the shared JSON schema version 1.0."""
    validate_taxodist_bundle(bundle)
    records = []
    for row in bundle["resolution"].itertuples(index=False):
        records.append(
            {
                "input": row.input,
                "resolved_name": row.resolved_name,
                "id": row.id,
                "status": row.status,
                "n_candidates": int(row.n_candidates),
                "lineage_depth": (
                    None if pd.isna(row.lineage_depth) else int(row.lineage_depth)
                ),
                "lineage": None if row.lineage is None else list(row.lineage),
                "candidates": row.candidates[["id", "name"]].to_dict("records"),
            }
        )
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
    """Read and validate a portable bundle written by R, Python, or Julia."""
    if not os.path.exists(file):
        raise FileNotFoundError(f"Bundle file not found: {file}")
    try:
        with open(file, encoding="utf-8") as stream:
            raw = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not parse bundle JSON: {error}") from error
    if raw.get("format") != "taxodist_bundle":
        raise ValueError("Invalid bundle JSON: unrecognized format")
    if raw.get("schema_version") != "1.0":
        raise ValueError(
            f"Unsupported taxodist bundle schema: {raw.get('schema_version')}"
        )
    required = {"created_at", "source", "software", "metric", "taxa", "matrix"}
    if not required.issubset(raw):
        raise ValueError("Invalid bundle JSON: required fields are missing")

    rows = []
    for record in raw["taxa"]:
        candidate_frame = pd.DataFrame(
            record.get("candidates") or [], columns=["id", "name"]
        )
        rows.append(
            {
                "input": record.get("input"),
                "resolved_name": record.get("resolved_name"),
                "id": record.get("id"),
                "status": record.get("status"),
                "n_candidates": record.get("n_candidates"),
                "lineage_depth": record.get("lineage_depth"),
                "lineage": record.get("lineage"),
                "candidates": candidate_frame,
            }
        )
    source = raw["source"]
    resolution = _make_resolution(
        rows,
        source=source.get("name"),
        source_url=source.get("url"),
        retrieved_at=source.get("retrieved_at"),
    )
    bundle = TaxodistBundle(
        schema_version=raw["schema_version"],
        created_at=raw["created_at"],
        source=source,
        software=raw["software"],
        metric=raw["metric"],
        resolution=resolution,
        matrix=_parse_matrix(raw["matrix"]),
    )
    return validate_taxodist_bundle(bundle)