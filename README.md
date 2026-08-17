# DataChef ETL Demo

A simple demo ETL application that ingests raw data, transforms it, and presents a dashboard using Streamlit and Plotly.

## Current versus target behavior

`main.py` now starts the offline DataChef product: a staged Streamlit shell over
the reviewed application layer in `datachef/`. It runs entirely on this machine
and contacts no provider.

The legacy transformation path that asks Gemini for Python and executes it still
exists under `crew/transformation_agent/`, but the product no longer reaches it.
The approved Phase 1 target — an application-owned gateway returning declarative
plans that deterministic allow-listed services execute — is recorded in
`docs/adr/0001-llm-gateway-boundary.md`. The offline half of that target is
implemented; the provider-backed planner is deferred to Phase 2.

## What the product guarantees

The Streamlit layer renders evidence and collects input. It decides nothing.
Every one of the following is decided by `DataChefController` and merely
displayed:

- **Nothing runs without your approval.** A plan is prepared, reviewed, and shown
  in full before you approve it. Approval binds the dataset fingerprint, the plan
  ID and version, and the exact ordered operations.
- **Gold is earned, not asserted.** Execution replays independently and quality
  assurance is recomputed. Only a `PASS` verdict produces a gold table.
- **Downloads exist only for verified gold.** On warning, failure, rejection, or a
  raw-only session, no download control is rendered at all.
- **Every download is accounted for.** The bundle ships a cleaned CSV, a cleaned
  Parquet, the canonical transformation plan, the QA report, the execution change
  log, and a manifest recording a SHA-256 for each of the other five files.
- **Your file never touches disk.** Uploads are parsed in memory. Filenames and
  paths are never retained; artifact names are generated from dataset and plan
  identifiers.
- **A refresh cannot double-run anything.** Each action mints one command ID that
  is replayed verbatim, so a reload or double click performs no second effect.
- **Previews are local.** Preview rows are presentation only and never enter
  evidence, artifacts, the manifest, or any provider context.

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

Start the DataChef product with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run main.py
```

Then open the browser URL shown by Streamlit, usually:

```text
http://localhost:8501
```

**Acceptable row loss** starts at 0%, which refuses any plan that would remove a
row. If a plan is refused, the Plan screen shows the estimated removal per
operation next to your own setting, and a revise control lets you raise the
setting or drop a request without re-uploading the dataset.

The app walks six stages in order — Upload, Intent, Plan, Approval, Quality,
Results — and advances only when the application layer says the evidence
supports it. No credential is required: the offline product never contacts a
provider, and leaving `.env` unconfigured changes nothing about how it runs.

Use **Reset session** in the sidebar to clear the dataset, the diagnosis, the
intent, the workflow, and the command history, and to issue a fresh uploader.
**Show local data preview** is off by default and only affects what is drawn on
your screen.

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
