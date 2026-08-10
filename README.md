# DataChef ETL Demo

A simple demo ETL application that ingests raw data, transforms it, and presents a dashboard using Streamlit and Plotly.

## Structure

- `main.py` - application entrypoint
- `ui/` - Streamlit app and Plotly chart rendering
- `pipelines/` - ingestion, transformation, and reporting logic
- `utils/` - shared helpers for Spark, Gemini, and configuration
- `data/raw/` - raw input files
- `data/processed/` - processed output files
- `data/reports/` - generated reports

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Run the app

Start the Streamlit app with:

```bash
streamlit run main.py
```

Then open the browser URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Notes

- A virtual environment is recommended:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

- Set `GEMINI_API_KEY` in your environment if you use Gemini integration.
- Keep raw sample files in `data/raw/` and generated outputs in `data/processed/` or `data/reports/`.
