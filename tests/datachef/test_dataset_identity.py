from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from datachef.diagnostics import (
    DatasetShapeError,
    DatasetShapeFailure,
    dataframe_fingerprint,
    identify_dataset,
)


def test_fingerprint_is_stable_for_supported_deep_copies() -> None:
    source = pd.DataFrame(
        {
            "nullable": pd.Series([1, pd.NA], dtype="Int64"),
            "float_null": [np.nan, 2.5],
            "observed_at": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"]
            ),
            "category": pd.Series(["A", "B"], dtype="category"),
            "numpy_scalar": [np.int64(10), np.int64(20)],
        }
    )

    assert dataframe_fingerprint(source) == dataframe_fingerprint(source.copy(deep=True))
    assert identify_dataset(pd.DataFrame({"empty": pd.Series(dtype="string")})).row_count == 0


@pytest.mark.parametrize(
    ("frame", "failure"),
    [
        (pd.DataFrame([[1, 2]], columns=["duplicate", "duplicate"]), DatasetShapeFailure.DUPLICATE_COLUMN_LABEL),
        (pd.DataFrame([[1]], columns=[1]), DatasetShapeFailure.NON_STRING_COLUMN_LABEL),
        (pd.DataFrame({"object": [{"nondeterministic", "set"}]}), DatasetShapeFailure.UNSUPPORTED_OBJECT_VALUE),
    ],
)
def test_fingerprint_rejects_ambiguous_input_without_echoing_values(frame, failure) -> None:
    with pytest.raises(DatasetShapeError) as captured:
        identify_dataset(frame)

    assert captured.value.failure is failure
    assert "nondeterministic" not in str(captured.value)


def test_integer_and_string_labels_cannot_collide() -> None:
    string_identity = identify_dataset(pd.DataFrame([[1]], columns=["1"]))
    with pytest.raises(DatasetShapeError) as captured:
        identify_dataset(pd.DataFrame([[1]], columns=[1]))

    assert string_identity.fingerprint
    assert captured.value.failure is DatasetShapeFailure.NON_STRING_COLUMN_LABEL


def test_declared_identity_changes_are_observable() -> None:
    source = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}, index=[10, 20])
    baseline = dataframe_fingerprint(source)

    changed_value = source.copy(deep=True)
    changed_value.loc[10, "a"] = 9
    changed_dtype = source.astype({"a": "float64"})
    changed_index = source.copy(deep=True)
    changed_index.index = [11, 21]

    assert dataframe_fingerprint(changed_value) != baseline
    assert dataframe_fingerprint(changed_dtype) != baseline
    assert dataframe_fingerprint(source[["b", "a"]]) != baseline
    assert dataframe_fingerprint(source.iloc[::-1]) != baseline
    assert dataframe_fingerprint(changed_index) != baseline


def test_supported_fingerprint_is_stable_across_hash_seeds() -> None:
    code = (
        "import pandas as pd; "
        "from datachef.diagnostics import dataframe_fingerprint; "
        "print(dataframe_fingerprint(pd.DataFrame({'label':['A','B'],'value':[1,2]})))"
    )
    fingerprints = []
    for seed in ("1", "2"):
        environment = {
            **os.environ,
            "PYTHONHASHSEED": seed,
            "PYTHONDONTWRITEBYTECODE": "1",
            "DATACHEF_OFFLINE": "true",
        }
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        fingerprints.append(completed.stdout.strip())

    assert fingerprints[0] == fingerprints[1]
