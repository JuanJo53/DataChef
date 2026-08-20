from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


FIXTURES = Path(__file__).parent / "fixtures"


def test_pandas_parquet_round_trip_preserves_basic_contract(tmp_path: Path) -> None:
    original = pd.DataFrame(
        {
            "category": pd.Series(["alpha", "beta", None], dtype="string"),
            "quantity": pd.Series([1, None, 3], dtype="Int64"),
            "active": pd.Series([True, False, None], dtype="boolean"),
            "observed_at": pd.to_datetime(
                ["2026-08-01T12:00:00Z", None, "2026-08-03T12:00:00Z"],
                utc=True,
            ),
        }
    )
    parquet_path = tmp_path / "round-trip.parquet"

    original.to_parquet(parquet_path, engine="pyarrow", index=False)
    restored = pd.read_parquet(parquet_path, engine="pyarrow")

    assert list(restored.columns) == list(original.columns)
    assert restored.isna().sum().to_dict() == original.isna().sum().to_dict()
    assert_frame_equal(restored, original, check_dtype=True, check_like=False)


def test_tracked_csv_and_json_lines_fixtures_are_equivalent() -> None:
    csv_frame = pd.read_csv(
        FIXTURES / "quality_sample.csv",
        dtype=str,
        keep_default_na=False,
    )
    json_lines_frame = pd.read_json(
        FIXTURES / "quality_sample.jsonl",
        lines=True,
        dtype=False,
        convert_dates=False,
    ).astype(str)

    assert_frame_equal(csv_frame, json_lines_frame, check_dtype=False)
