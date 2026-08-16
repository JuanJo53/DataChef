# DataChef ETL Demo

A simple demo ETL application that ingests raw data, transforms it, and presents a dashboard using Streamlit and Plotly.

## Current versus target behavior

The current repository contains the existing Streamlit UI, deterministic
ingestion and dashboard helpers, and a legacy transformation path that can ask
Gemini for Python and execute it. Phase 0B establishes only the offline
development foundation; it does not replace or secure that legacy path.

The approved Phase 1 target is a typed CrewAI workflow in which an
application-owned gateway returns declarative transformation plans and
deterministic allow-listed Python services execute them. That target is recorded
in `docs/adr/0001-llm-gateway-boundary.md` and is not implemented yet.

## Structure

- `main.py` - application entrypoint
- `ui/` - Streamlit app and Plotly chart rendering
- `pipelines/` - ingestion, transformation, and reporting logic
- `utils/` - shared helpers for Spark, Gemini, and configuration
- `data/raw/` - raw input files
- `data/processed/` - processed output files
- `data/reports/` - generated reports

## Requirements

The hackathon environment is proven with 64-bit Python 3.13.14. CrewAI 1.15.16
requires Python 3.10 or newer and earlier than 3.14.

Create the Windows virtual environment and install runtime dependencies using:

```powershell
python3.13-64 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install --prefer-binary -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` automatically applies `constraints-hackathon.txt`, which
reproduces the critical direct application stack proven during the hackathon.
It is intentionally not a full lock of every transitive package.

Contributors running tests should install the development entrypoint instead:

```powershell
.\.venv\Scripts\python.exe -m pip install --prefer-binary -r requirements-dev.txt
```

## Local configuration

Create the ignored local configuration from the tracked placeholder file:

```powershell
Copy-Item .env.example .env
```

New code uses `GOOGLE_API_KEY` as the canonical Gemini credential variable.
Enter its value manually in the local `.env`; never commit or paste it into a
chat or prompt. Leave `GEMINI_MODEL` empty until the separately approved live
compatibility test verifies an identifier. `GEMINI_API_KEY` remains only a
temporary alias in untouched legacy modules.

Inspect configuration without making a provider request:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_config
```

## Run the app

Start the Streamlit app with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run main.py
```

Then open the browser URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Safe offline tests

The normal pytest suite only collects tests under `tests/` and explicitly
ignores the external transformation smoke script:

```powershell
.\.venv\Scripts\python.exe -m pytest --collect-only -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe crew\ingestion_agent\test_ingestion.py
.\.venv\Scripts\python.exe crew\dashboard_agent\test_dashboard.py
```

Do not include `crew/transformation_agent/test_transformation.py` in offline
test commands. It can contact Gemini, execute generated Python, and write an
output pipeline.

## Test-data policy

- `tests/fixtures/`: small synthetic, sanitized, deterministic, tracked data.
- `data/local/`: private, unknown, large, or colleague-provided data; ignored.
- `data/baseline/`: local diagnostic evidence; ignored.

Place the colleague-provided Parquet file under `data/local/` until its content,
size, license, and privacy suitability for Git have been reviewed. Parquet test
files are generated in temporary directories by default.
