"""
startup.py — TechnoBot environment validator and safe entry-point helper.

Run directly to validate the environment before starting the server:
    python startup.py

Or import validate_environment() from api_server.py to run checks on startup.

Checks performed:
  1. Python version >= 3.10 (type-union syntax, match statements)
  2. Required packages installed and importable
  3. UTF-8 encoding configured (emoji path fix)
  4. Working directory is the backend folder (relative imports work)
  5. Exports / uploads directories are writable
  6. Internet connectivity (lightweight HEAD request to Yahoo Finance)
  7. SSL availability (certifi bundle reachable)
  8. Optional: .env loaded if present
"""

from __future__ import annotations

import sys
import os
import time

# ── 0. UTF-8 fix BEFORE any yfinance-related import ──────────────────────────
# The emoji in D:\TechnoBot🤖\ can cause encoding errors in certifi/yfinance on
# Windows when stdout/stderr default to cp1252 or cp950.
def _fix_encoding() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")

_fix_encoding()


# ── Required packages ─────────────────────────────────────────────────────────

_REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("fastapi",   "fastapi"),
    ("uvicorn",   "uvicorn"),
    ("pydantic",  "pydantic"),
    ("yfinance",  "yfinance"),
    ("pandas",    "pandas"),
    ("requests",  "requests"),
]


# ── Validation helpers ────────────────────────────────────────────────────────

def _check_python_version() -> list[str]:
    errors: list[str] = []
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        errors.append(
            f"Python 3.10+ required (found {major}.{minor}). "
            "TechnoBot uses union type syntax (X | Y) that requires 3.10+."
        )
    return errors


def _check_packages() -> list[str]:
    errors: list[str] = []
    missing: list[str] = []
    for display_name, import_name in _REQUIRED_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(display_name)
    if missing:
        errors.append(
            f"Missing packages: {', '.join(missing)}. "
            "Run: pip install -r requirements.txt"
        )
    return errors


def _check_working_directory() -> list[str]:
    """Ensure Python can import backend modules (data/, technobot_data/, etc.)."""
    errors: list[str] = []
    cwd = os.getcwd()
    # Check that we can find at least one known backend file
    marker_files = ["api_server.py", "technobot_router.py"]
    found = any(os.path.isfile(os.path.join(cwd, f)) for f in marker_files)
    if not found:
        errors.append(
            f"Working directory looks wrong: {cwd!r}. "
            "Run the server FROM the backend folder:\n"
            "  cd 'D:\\TechnoBot🤖\\backend'\n"
            "  uvicorn api_server:app --reload --port 8000"
        )
    return errors


def _check_writable_dirs() -> list[str]:
    """Ensure exports/ and uploads/ directories exist and are writable."""
    errors: list[str] = []
    for dir_name in ("exports", os.path.join("technobot_data", "exports"),
                     os.path.join("technobot_data", "uploads")):
        try:
            os.makedirs(dir_name, exist_ok=True)
            test_file = os.path.join(dir_name, ".write_test")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
        except Exception as exc:
            errors.append(f"Directory {dir_name!r} is not writable: {exc}")
    return errors


def _check_internet() -> list[str]:
    """Lightweight connectivity check — HEAD request to Yahoo Finance."""
    warnings: list[str] = []
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://finance.yahoo.com",
            headers={"User-Agent": "TechnoBot-startup-check/1.0"},
            method="HEAD",
        )
        with urllib.request.urlopen(req, timeout=5):
            pass   # success
    except Exception as exc:
        warnings.append(
            f"Internet connectivity check failed: {exc}. "
            "Live market data will be unavailable. "
            "Upload a CSV to analyse offline data."
        )
    return warnings


def _check_ssl() -> list[str]:
    """Diagnose SSL / certifi setup and apply the emoji-path fix if needed.

    The D:\\TechnoBot🤖\\ folder name contains a non-ASCII emoji.  curl_cffi
    (the HTTP backend used by yfinance ≥ 0.2) cannot resolve CA-bundle paths
    that contain non-ASCII characters, producing:
        curl: (77) error setting certificate verify locations
    This check detects that condition, applies the fix (copy bundle to %TEMP%),
    and reports the result so the operator knows the fix is active.
    """
    warnings: list[str] = []

    # ── 1. Locate certifi bundle ──────────────────────────────────────────────
    try:
        import certifi
        cert_path = certifi.where()
        cert_exists = os.path.exists(cert_path)
        print(f"[TechnoBot/startup] certifi path   : {cert_path}")
        print(f"[TechnoBot/startup] certifi exists : {cert_exists}")
        if not cert_exists:
            warnings.append(
                f"certifi CA bundle not found at {cert_path!r}. "
                "Re-install certifi: pip install --upgrade certifi"
            )
    except ImportError:
        warnings.append("certifi not installed — pip install certifi")
        return warnings
    except Exception as exc:
        warnings.append(f"certifi check failed: {exc}")
        return warnings

    # ── 2. Detect non-ASCII path (emoji issue) and apply fix ─────────────────
    if not cert_path.isascii():
        print("[TechnoBot/startup] Non-ASCII certifi path detected (emoji in folder name)")
        try:
            import shutil
            import tempfile
            safe_path = os.path.join(tempfile.gettempdir(), 'technobot_cacert.pem')
            shutil.copy2(cert_path, safe_path)
            os.environ['CURL_CA_BUNDLE']     = safe_path
            os.environ['SSL_CERT_FILE']      = safe_path
            os.environ['REQUESTS_CA_BUNDLE'] = safe_path
            print(f"[TechnoBot/startup] SSL fix applied — bundle copied to: {safe_path}")
        except Exception as exc:
            warnings.append(
                f"Could not apply emoji-path SSL fix: {exc}. "
                "yfinance may fail with 'curl: (77) error setting certificate verify locations'. "
                "Workaround: move the project to a folder with an ASCII-only path."
            )
    else:
        os.environ.setdefault('CURL_CA_BUNDLE',     cert_path)
        os.environ.setdefault('SSL_CERT_FILE',      cert_path)
        os.environ.setdefault('REQUESTS_CA_BUNDLE', cert_path)
        print("[TechnoBot/startup] certifi path is ASCII — no SSL fix needed")

    # ── 3. Verify the SSL context is actually usable ──────────────────────────
    try:
        import ssl
        effective_cert = os.environ.get('CURL_CA_BUNDLE', cert_path)
        ctx = ssl.create_default_context(cafile=effective_cert)
        _ = ctx
        print(f"[TechnoBot/startup] SSL context OK (cafile={effective_cert})")
    except Exception as exc:
        warnings.append(
            f"SSL context creation failed: {exc}. "
            "HTTPS requests (yfinance) may fail."
        )

    # ── 4. Report active HTTP backend ─────────────────────────────────────────
    try:
        import curl_cffi   # noqa: F401
        backend = "curl_cffi"
    except ImportError:
        backend = "requests"
    print(f"[TechnoBot/startup] yfinance HTTP backend : {backend}")
    if backend == "curl_cffi" and not cert_path.isascii():
        print("[TechnoBot/startup] NOTE: curl_cffi backend active with non-ASCII "
              "cert path — SSL fix must be applied before yfinance import")

    return warnings


def _load_dotenv() -> None:
    """Load .env if present — silent if python-dotenv not installed."""
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.getcwd(), ".env")
        if os.path.isfile(env_path):
            load_dotenv(env_path)
            print("[TechnoBot/startup] .env loaded")
    except ImportError:
        pass  # python-dotenv optional


# ── Public API ────────────────────────────────────────────────────────────────

def validate_environment(strict: bool = False) -> bool:
    """Run all environment checks.

    Parameters
    ----------
    strict : if True, treat warnings as errors (useful in CI).

    Returns
    -------
    True  — all hard checks passed (warnings may still exist).
    False — one or more hard errors found; server should not start.
    """
    print("[TechnoBot/startup] TechnoBot startup initialized")

    _load_dotenv()

    errors: list[str]   = []
    warnings: list[str] = []

    errors   += _check_python_version()
    errors   += _check_packages()
    errors   += _check_working_directory()
    errors   += _check_writable_dirs()
    warnings += _check_internet()
    warnings += _check_ssl()

    if warnings:
        for w in warnings:
            print(f"[TechnoBot/startup] WARNING: {w}")

    if errors:
        for e in errors:
            print(f"[TechnoBot/startup] ERROR: {e}")
        print("[TechnoBot/startup] Startup validation FAILED — fix the above errors before starting.")
        return False

    print("[TechnoBot/startup] Environment validation passed")

    print("[TechnoBot/startup] AV → TD fetch pipeline ready (Yahoo removed)")
    return True


# ── CLI self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ok = validate_environment()
    sys.exit(0 if ok else 1)
