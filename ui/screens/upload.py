"""Upload and diagnosis stage: hand bytes to the controller, render evidence."""

from __future__ import annotations

from typing import Any

import streamlit as st

from datachef.application import (
    CsvParserOptions,
    JsonLinesParserOptions,
    JsonRecordsParserOptions,
    ParquetParserOptions,
    UploadFormat,
    UploadRequest,
)
from ui import state as ui_state
from ui.screens import render_findings, render_result

_ACCEPTED_SUFFIXES = ("csv", "json", "jsonl", "ndjson", "parquet")
_JSON_MODES = ("JSON records (a list of objects)", "JSON Lines (one object per line)")


def _suffix_of(name: str) -> str:
    return f".{name.rsplit('.', 1)[-1].lower()}" if "." in name else ""


def _request_for(upload: Any, json_mode: str) -> UploadRequest | None:
    """Build a typed request from in-memory bytes; no path is ever retained."""

    suffix = _suffix_of(upload.name)
    content = upload.getvalue()
    if suffix == ".csv":
        return UploadRequest(
            content=content,
            declared_suffix=suffix,
            format=UploadFormat.CSV,
            parser_options=CsvParserOptions(),
        )
    if suffix == ".parquet":
        return UploadRequest(
            content=content,
            declared_suffix=suffix,
            format=UploadFormat.PARQUET,
            parser_options=ParquetParserOptions(),
        )
    if suffix in {".jsonl", ".ndjson"}:
        return UploadRequest(
            content=content,
            declared_suffix=suffix,
            format=UploadFormat.JSON_LINES,
            parser_options=JsonLinesParserOptions(),
        )
    if suffix == ".json":
        if json_mode == _JSON_MODES[1]:
            return UploadRequest(
                content=content,
                declared_suffix=suffix,
                format=UploadFormat.JSON_LINES,
                parser_options=JsonLinesParserOptions(),
            )
        return UploadRequest(
            content=content,
            declared_suffix=suffix,
            format=UploadFormat.JSON_RECORDS,
            parser_options=JsonRecordsParserOptions(),
        )
    return None


def _render_source_summary(session: Any) -> None:
    identity = session.source.identity
    left, middle, right = st.columns(3)
    left.metric("Rows", identity.row_count)
    middle.metric("Columns", identity.column_count)
    right.metric("Dataset", identity.dataset_id)
    st.caption(f"Source fingerprint `{identity.fingerprint}`")
    if session.preview_enabled:
        st.markdown("#### Local preview")
        st.caption(
            "Preview rows are presentation only. They never enter evidence, "
            "artifacts, or the manifest."
        )
        st.dataframe(session.source.raw_copy().head(10), use_container_width=True)


def _render_diagnosis(session: Any) -> None:
    report = session.display_diagnostic_report
    if report is None:
        return
    st.markdown("### Deterministic diagnosis")
    evidence = report.legacy_evidence
    left, middle, right = st.columns(3)
    left.metric("Health score", evidence.health_score)
    middle.metric("Grade", evidence.health_grade)
    right.metric("Duplicate rows", report.duplicate_row_count)
    if not report.issues:
        st.success("No diagnostic issues were detected.")
        return
    st.markdown("#### Issues")
    for issue in report.issues:
        columns = ", ".join(issue.affected_columns) or "—"
        st.markdown(
            f"- `{issue.severity.value}` **{issue.title}** "
            f"({issue.kind.value}) — columns: {columns}"
        )


def render(controller: Any, state: Any) -> None:
    st.header("1 · Upload and diagnose")
    st.caption(
        "The file is parsed in memory. Nothing is written to disk and no "
        "provider is contacted."
    )
    session = controller.session

    json_mode = _JSON_MODES[0]
    upload = st.file_uploader(
        "Choose a CSV, JSON, JSON Lines, or Parquet file",
        type=list(_ACCEPTED_SUFFIXES),
        key=ui_state.uploader_key(session.uploader_generation),
    )
    if upload is not None and _suffix_of(upload.name) == ".json":
        json_mode = st.radio(
            "How should this .json file be read?",
            _JSON_MODES,
            key=ui_state.JSON_MODE_WIDGET,
            horizontal=True,
        )

    if upload is not None:
        request = _request_for(upload, json_mode)
        if request is None:
            st.error(
                "**UNSUPPORTED_FORMAT** — that file extension is not supported. "
                "Choose a .csv, .json, .jsonl, .ndjson, or .parquet file."
            )
        else:
            result = ui_state.remember_result(state, controller.load_upload(request))
            render_findings(result.findings)
            session = controller.session

    if session.source is None:
        render_result(ui_state.last_result(state))
        st.info("Upload a dataset to begin.")
        return

    _render_source_summary(session)

    if session.display_diagnostic_report is None:
        if st.button(
            "Run deterministic diagnosis",
            key=ui_state.DIAGNOSE_WIDGET,
            type="primary",
        ):
            result = ui_state.remember_result(state, controller.diagnose())
            render_findings(result.findings)
            st.rerun()
    _render_diagnosis(controller.session)
    render_result(ui_state.last_result(state))
