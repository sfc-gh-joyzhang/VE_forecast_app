"""
snow_connection.py
------------------
Utility helper for obtaining a Snowpark ``Session`` that works seamlessly when
running **inside Snowflake‑hosted Streamlit** *or* **locally during
development**.  Supports classic username/password **and** Okta/SAML SSO via the
``externalbrowser`` authenticator or an explicit Okta IdP URL.

Why you just saw *No default Session is found*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The previous runtime‑detection heuristic flagged *any* environment where the
``snowflake‑snowpark‑python`` package was importable as "inside Snowflake". When
run locally this led us to call ``get_active_session()``, which naturally blows
up because no session exists.  The detection logic is now stricter: we only
return "snowflake" if **calling** ``get_active_session()`` succeeds.

Quick‑start
~~~~~~~~~~~
>>> from snow_connection import get_session
>>> session = get_session()  # works everywhere

``.env`` (local only)
~~~~~~~~~~~~~~~~~~~~
Mandatory (pwd) | Mandatory (SSO) | Optional
----------------|-----------------|---------
SNOWFLAKE_ACCOUNT | SNOWFLAKE_ACCOUNT | SNOWFLAKE_ROLE
SNOWFLAKE_USER    | SNOWFLAKE_USER    | SNOWFLAKE_WAREHOUSE
SNOWFLAKE_PASSWORD| SNOWFLAKE_AUTHENTICATOR=externalbrowser *or* https://<okta>.okta.com | SNOWFLAKE_DATABASE / SCHEMA

If any required var is missing we raise a clear ``ValueError``.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Optional, TYPE_CHECKING

__all__ = ["get_session"]

# ---------------------------------------------------------------------------
# Runtime detection helpers
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from snowflake.snowpark import Session

def _active_snowflake_session() -> 'Session | None':
    """Return the active Snowflake session *or* ``None`` if not inside Snowflake."""
    try:
        from snowflake.snowpark.context import get_active_session  # type: ignore
        return get_active_session()
    except Exception:
        return None


def _detect_runtime() -> str:
    """Return ``"snowflake"`` if truly inside Snowflake; else ``"local"``."""
    # Official runtime flag set by Snowflake
    if os.getenv("SNOWPARK_RUNTIME") == "snowpark":
        return "snowflake"

    # Fall back to trying get_active_session()
    return "snowflake" if _active_snowflake_session() else "local"


# ---------------------------------------------------------------------------
# Local‑dev session builder
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _build_local_session() -> 'Session':
    """Create a *new* Snowpark Session (local dev only)."""
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(override=False)
    except ModuleNotFoundError:
        pass  # optional dependency

    try:
        from snowflake.snowpark import Session
    except ImportError as exc:
        raise ImportError(
            "snowflake-snowpark-python is required for local execution. "
            "Install it with `pip install snowflake-snowpark-python`."
        ) from exc

    authenticator = os.getenv("SNOWFLAKE_AUTHENTICATOR")

    cfg: Dict[str, Any] = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
    }

    if authenticator:  # SSO / Okta
        cfg["authenticator"] = authenticator
        if authenticator.lower() != "externalbrowser":
            cfg["password"] = os.getenv("SNOWFLAKE_PASSWORD")
        # Do NOT set password for externalbrowser
    else:  # classic pwd
        cfg["password"] = os.getenv("SNOWFLAKE_PASSWORD")

    # Optionals
    for key in ("role", "warehouse", "database", "schema"):
        env_key = f"SNOWFLAKE_{key.upper()}"
        if os.getenv(env_key):
            cfg[key] = os.getenv(env_key)

    # Validate
    required = ["account", "user"]
    # Only require password if:
    # - authenticator is not set, or
    # - authenticator is set and is not 'externalbrowser' and does not start with 'https://'
    require_password = (
        not authenticator or
        (
            authenticator.lower() != "externalbrowser"
            and not authenticator.lower().startswith("https://")
        )
    )
    if require_password:
        required.append("password")
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise ValueError(
            "Missing Snowflake connection parameters: " + ", ".join(missing)
        )

    return Session.builder.configs(cfg).create()


# ---------------------------------------------------------------------------
# In‑Snowflake session helper
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_snowflake_managed_session() -> 'Session':
    sess = _active_snowflake_session()
    if sess is None:
        raise RuntimeError(
            "No active Snowflake session found; did runtime detection fail?"
        )
    return sess


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_session(force_new: bool = False):
    """Return a Snowpark ``Session`` in any environment.

    * **Inside Snowflake** – reuse the already‑active session.
    * **Locally** – build one based on env vars / `.env`.
    """

    runtime = _detect_runtime()

    if runtime == "snowflake":
        return _get_snowflake_managed_session()

    if force_new:
        _build_local_session.cache_clear()

    return _build_local_session()
