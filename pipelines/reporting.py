import pandas as pd

from pipelines.ingestion import clean_raw_data, load_raw_data
from pipelines.transformation import transform_data


def build_dashboard() -> dict:
    raw_path = "data/raw/sample_data.csv"
    try:
        df = load_raw_data(raw_path)
    except FileNotFoundError:
        return {"summary": {"error": "Sample raw data not found."}}

    df = clean_raw_data(df)
    df = transform_data(df)

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
    }

    metrics = {
        "Total rows": len(df),
        "Numeric columns": len(df.select_dtypes(include=["number"]).columns),
    }

    history = pd.DataFrame(
        {
            "date": pd.date_range(start="2024-01-01", periods=min(10, len(df)), freq="D"),
            "value": df.select_dtypes(include=["number"]).sum(axis=1).tolist()[:10],
        }
    )

    return {"summary": summary, "metrics": metrics, "history": history}
