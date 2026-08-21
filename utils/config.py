"""Central configuration for DATAChef.

Model names used to be hardcoded in four different files, which meant two
people could not run the project with different models without editing (and
then conflicting over) each other's code. Everything is resolved here instead,
from environment variables, so each developer sets their own values in their
own .env and nobody touches the source.

Resolution order for a model, most specific first:

    1. the per-role variable   (DATACHEF_MODEL_TRANSFORM, ..._INSIGHTS, ..._CHAT)
    2. the shared variable     (DATACHEF_MODEL)
    3. the built-in default    (DEFAULT_MODELS below)

So one knob configures everything:

    DATACHEF_MODEL=gemini-3.6-flash

and a single stage can still be pinned separately when needed:

    DATACHEF_MODEL=gemini-3.6-flash
    DATACHEF_MODEL_CHAT=gemini-3.1-flash-lite

With no variables set, behaviour is exactly the built-in defaults, so an
existing checkout keeps working untouched.
"""

from __future__ import annotations

import os

try:  # Make .env visible without making python-dotenv a hard requirement.
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(), override=False)
except Exception:  # pragma: no cover - a missing/broken .env must never break import
    pass


# ---------------------------------------------------------------------
# Paths (pre-existing)
# ---------------------------------------------------------------------
DATA_RAW_PATH = os.getenv("DATA_RAW_PATH", "data/raw")
DATA_PROCESSED_PATH = os.getenv("DATA_PROCESSED_PATH", "data/processed")
REPORT_PATH = os.getenv("REPORT_PATH", "data/reports")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
SHARED_MODEL_ENV = "DATACHEF_MODEL"

# Roles, not agents: what the call is FOR. Defaults differ on purpose.
#
# "transform" is the critical path -- Stage 4 only unlocks once a
# transformation succeeds -- so it gets its own model, and therefore its own
# free-tier daily quota. "insights" and "chat" are optional niceties that both
# fall back to rules, but build_dashboard_spec fires on every dashboard render;
# on a shared model those renders would quietly eat the quota the transformation
# needs. Different default model = different quota pool.
#
# Both defaults are ALIASES rather than pinned versions: Google retires old
# versions (gemini-2.5-flash and gemini-2.5-flash-lite already return 404), and
# an alias always tracks the current model.
DEFAULT_MODELS: dict[str, str] = {
    "transform": "gemini-flash-lite-latest",
    "insights": "gemini-3.1-flash-lite",
    "chat": "gemini-3.1-flash-lite",
}

MODEL_ENV_BY_ROLE: dict[str, str] = {
    "transform": "DATACHEF_MODEL_TRANSFORM",
    "insights": "DATACHEF_MODEL_INSIGHTS",
    "chat": "DATACHEF_MODEL_CHAT",
}


def model_for(role: str) -> str:
    """Model name for a role: per-role env, then shared env, then default.

    Read at call time on purpose, so a .env loaded later (or an override set in
    a test) still applies, rather than being frozen at import.
    """
    if role not in DEFAULT_MODELS:
        raise ValueError(
            f"Unknown model role {role!r}. Expected one of: "
            f"{', '.join(sorted(DEFAULT_MODELS))}."
        )

    specific = os.getenv(MODEL_ENV_BY_ROLE[role], "").strip()
    if specific:
        return specific

    shared = os.getenv(SHARED_MODEL_ENV, "").strip()
    if shared:
        return shared

    return DEFAULT_MODELS[role]


def api_key() -> str | None:
    """Google API key. GOOGLE_API_KEY is canonical, GEMINI_API_KEY a legacy alias."""
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return key.strip() if key and key.strip() else None
