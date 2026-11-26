"""Validate locally supplied patient memberships for LabTOP checkpoint evaluation.

A saved model must be evaluated on its original held-out patients. This helper
reads an explicit local JSON split file without logging or rewriting patient
identifiers. It does not infer memberships from a new random seed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_GROUPS = ("train", "valid", "test")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys without exposing their contents."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Split JSON must not contain duplicate object keys")
        result[key] = value
    return result


def load_patient_splits(path: Path, patient_ids: Iterable[str]) -> dict[str, list[str]]:
    """Read and validate original training, validation and test memberships.

    Args:
        path: Local UTF-8 JSON file with exactly the keys ``train``, ``valid``
            and ``test``. Each value must be a nonempty list of string IDs.
        patient_ids: All patient IDs in the currently loaded dataset. Every
            ID must be a nonblank string, with no duplicate IDs.

    Returns:
        A new dictionary in train/valid/test order, with each membership list
        sorted lexicographically. IDs are preserved exactly; they are not
        converted, trimmed or logged.

    Raises:
        ValueError: If JSON or memberships are malformed, a patient is unknown,
            repeated or omitted, or the supplied dataset ID roster is invalid.
            Error messages describe counts and conditions, never patient IDs.
        OSError: If the local file cannot be read.
    """
    if isinstance(patient_ids, (str, bytes)):
        raise ValueError(  # noqa: TRY004 - uniform validation-error API
            "patient_ids must be an iterable of individual string IDs"
        )
    roster = list(patient_ids)
    if not roster or any(
        not isinstance(patient_id, str) or not patient_id.strip()
        for patient_id in roster
    ):
        raise ValueError("Dataset patient IDs must be nonempty, nonblank strings")
    known = set(roster)
    if len(known) != len(roster):
        raise ValueError("Dataset patient ID roster must not contain duplicates")

    try:
        content = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Split file must contain valid UTF-8 JSON") from error
    if not isinstance(content, dict) or set(content) != set(_GROUPS):
        raise ValueError("Split JSON must contain exactly train, valid and test")

    assigned: set[str] = set()
    result: dict[str, list[str]] = {}
    for group in _GROUPS:
        identifiers = content[group]
        if not isinstance(identifiers, list) or not identifiers:
            raise ValueError(f"The {group} split must be a nonempty list")
        if any(
            not isinstance(identifier, str) or not identifier.strip()
            for identifier in identifiers
        ):
            raise ValueError(f"The {group} split must contain nonblank string IDs")
        members = set(identifiers)
        if len(members) != len(identifiers):
            raise ValueError(f"The {group} split contains duplicate patient IDs")
        if members & assigned:
            raise ValueError("Patient memberships overlap across split groups")
        unknown_count = len(members - known)
        if unknown_count:
            raise ValueError(
                f"Split file contains {unknown_count} unknown patient ID(s)"
            )
        assigned.update(members)
        result[group] = sorted(members)

    missing_count = len(known - assigned)
    if missing_count:
        raise ValueError(
            f"Split file leaves {missing_count} loaded patient ID(s) unassigned"
        )
    return result
