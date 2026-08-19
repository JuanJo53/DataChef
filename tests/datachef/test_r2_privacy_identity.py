from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from datachef.contracts import UserIntent
from datachef.diagnostics import dataframe_fingerprint, diagnose_raw_dataframe
from datachef.privacy import (
    build_column_alias_map,
    build_planning_context,
    build_provider_planning_payload,
)


def test_provider_payload_uses_collision_free_aliases_and_no_content_hashes() -> None:
    source = pd.DataFrame(
        {
            "nombre_completo": ["Ana Example", "Beto Example"],
            "email": ["ana@example.test", "beto@example.test"],
            " Email ": ["third@example.test", "fourth@example.test"],
            "EMAIL": ["fifth@example.test", "sixth@example.test"],
            "__dc_private_001__": [1, 2],
            "internal_code": ["T-1", "T-2"],
            "measure": [10, 20],
        }
    )
    report = diagnose_raw_dataframe(source)
    assert next(
        item for item in report.column_profiles if item.name == "nombre_completo"
    ).possible_pii
    intent = UserIntent(
        intent_id="Alice",
        user_goal="Analyze Alice at 123 Main Street",
        protected_columns=("internal_code",),
    )

    alias_map = build_column_alias_map(report, intent)
    context = build_planning_context(report, intent, (), column_alias_map=alias_map)
    payload = build_provider_planning_payload(context)
    serialized = payload.model_dump_json()

    forbidden = (
        "nombre_completo",
        "email",
        " Email ",
        "EMAIL",
        "internal_code",
        "Alice",
        sha256(b"Alice").hexdigest()[:16],
        report.dataset_identity.fingerprint,
        report.dataset_identity.dataset_id,
        context.context_id,
    )
    assert all(value not in serialized for value in forbidden)
    names = tuple(column.name for column in payload.column_schema)
    assert len(names) == len(set(names)) == source.shape[1]
    assert any(name.startswith("__dc1_private_") for name in names)
    assert "__dc_private_001__" in names
    assert payload.context_reference

    second_context = build_planning_context(report, intent, ())
    assert second_context.provider_context_reference != payload.context_reference


def test_categorical_schema_semantics_change_local_fingerprint() -> None:
    baseline = pd.DataFrame(
        {"category": pd.Categorical(["A", "B"], categories=["A", "B"], ordered=True)}
    )
    reordered = pd.DataFrame(
        {"category": pd.Categorical(["A", "B"], categories=["B", "A"], ordered=True)}
    )
    unordered = pd.DataFrame(
        {"category": pd.Categorical(["A", "B"], categories=["A", "B"], ordered=False)}
    )
    numeric_categories = pd.DataFrame(
        {"category": pd.Categorical([1, 2], categories=[1, 2], ordered=True)}
    )

    fingerprints = {
        dataframe_fingerprint(frame)
        for frame in (baseline, reordered, unordered, numeric_categories)
    }
    assert len(fingerprints) == 4


def test_categorical_fingerprint_is_stable_across_processes() -> None:
    code = (
        "import pandas as pd; "
        "from datachef.diagnostics import dataframe_fingerprint; "
        "frame=pd.DataFrame({'category':pd.Categorical(['A','B'],"
        "categories=['B','A'],ordered=True)}); "
        "print(dataframe_fingerprint(frame))"
    )
    fingerprints = []
    for seed in ("11", "97"):
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            env={
                **os.environ,
                "PYTHONHASHSEED": seed,
                "PYTHONDONTWRITEBYTECODE": "1",
                "DATACHEF_OFFLINE": "true",
            },
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        fingerprints.append(completed.stdout.strip())

    assert fingerprints[0] == fingerprints[1]

# def test_null_counts_follow_provider_aliases() -> None:
#     source = pd.DataFrame(
#         {
#             "internal_code": [1, None, 3],
#             "measure": [10, 20, 30],
#         }
#     )
#     report = diagnose_raw_dataframe(source)
#     intent = UserIntent(
#         intent_id="null-count-alias",
#         protected_columns=("internal_code",),
#     )

#     context = build_planning_context(report, intent, ())
#     payload = build_provider_planning_payload(context)

#     alias = next(
#         name
#         for name in payload.null_counts
#         if name.startswith("__dc_private_")
#     )

#     assert payload.null_counts[alias] == 1
#     assert "internal_code" not in payload.null_counts

def test_provider_null_counts_follow_column_aliases() -> None:
    source = pd.DataFrame(
        {
            "internal_code": [1, None, 3],
            "measure": [10, 20, 30],
        }
    )
    report = diagnose_raw_dataframe(source)
    intent = UserIntent(
        intent_id="null-count-alias",
        protected_columns=("internal_code",),
    )

    context = build_planning_context(report, intent, ())
    payload = build_provider_planning_payload(context)

    aliases = tuple(
        name
        for name in payload.null_counts
        if name.startswith("__dc_private_")
    )

    assert len(aliases) == 1
    assert payload.null_counts[aliases[0]] == 1
    assert "internal_code" not in payload.null_counts