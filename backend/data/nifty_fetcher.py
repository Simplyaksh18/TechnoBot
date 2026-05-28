"""
data/nifty_fetcher.py — Fast, Yahoo-free market data fetcher for TechnoBot.

Fetch pipeline (stops at first success):
  1. In-memory cache (120 s TTL)       — zero network calls.
  2. Twelve Data    (1 attempt, 2 s timeout)  PRIMARY provider.
  3. Alpha Vantage  (1 attempt, 2 s timeout)  SECONDARY provider.
  4. Stale cache    — any age; used when both live providers fail.
  5. Return status="failed" so _cached_derive_market_context() activates
     the lightweight qualitative context (never shows "data unavailable").

Provider order: TD → AV (Fix 1).  TD has a free tier of 8 req/min;
the 120 s cache TTL (Fix 6) ensures at most 1 live TD call per ticker
per 2-minute window, well within the limit.

A global 3-second safety timer is checked before each live provider.
If the budget is exhausted, a [TechnoBot/perf] FAST FAIL log is emitted
and the pipeline moves directly to stale cache / lightweight.

Yahoo Finance removed
─────────────────────
  • No yf.download(), yf.Ticker(), fast_info calls.
  • No SSL / certifi / curl_cffi workaround.
  • No 429 cooldown, no rate-limit state.
  • No retry loops, no blocking sleeps.

Compatibility stubs (imported by technobot_router.py and startup.py)
──────────────────────────────────────────────────────────────────────
  reset_yahoo_call_count()  → no-op
  get_yahoo_call_count()    → always 0
  is_rate_limited()         → always False
  cooldown_remaining()      → always 0
  reset_rate_limit()        → no-op (logs a message)
  DEV_MODE                  → False (cooldown concept removed)
"""

from __future__ import annotations

import threading
import time
import sys
import os


# ── UTF-8 fix — retain for emoji folder path on Windows ──────────────────────
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

print("[TechnoBot/fetch] Yahoo Finance removed — using AV → TD pipeline")


# ── Supported interval configurations ─────────────────────────────────────────
# Kept so AV / TD wrappers can still reference period strings for logging.

SUPPORTED_INTERVALS: dict[str, dict] = {
    "1d":  {"interval": "1d",  "period": "6mo"},
    "1wk": {"interval": "1wk", "period": "2y"},
    "1mo": {"interval": "1mo", "period": "5y"},
}


# ── In-memory data cache ──────────────────────────────────────────────────────
# Key   : "{ticker}:{interval_key}"
# Value : result-dict + "_ts" (monotonic timestamp)
# Only successful (status="ok") results are stored.

_DATA_CACHE: dict[str, dict] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL  = 120  # seconds — raised to 120 s (Fix 6: both TD and AV free tiers)


def _cache_get(key: str) -> dict | None:
    with _CACHE_LOCK:
        entry = _DATA_CACHE.get(key)
        if entry and (time.monotonic() - entry["_ts"]) < _CACHE_TTL:
            return entry
    return None


def _cache_set(key: str, result: dict) -> None:
    with _CACHE_LOCK:
        _DATA_CACHE[key] = {**result, "_ts": time.monotonic()}


def cache_invalidate(ticker: str, interval_key: str | None = None) -> None:
    """Drop cache entries for a ticker.  Pass interval_key=None to drop all intervals."""
    with _CACHE_LOCK:
        if interval_key:
            _DATA_CACHE.pop(f"{ticker}:{interval_key}", None)
        else:
            for k in list(_DATA_CACHE.keys()):
                if k.startswith(f"{ticker}:"):
                    del _DATA_CACHE[k]


# ── Global fetch timeout ──────────────────────────────────────────────────────
# The entire AV → TD chain must finish within this many seconds.
# Each individual API call has its own 2 s timeout (set in the provider modules).
# If the global budget expires, FAST FAIL is triggered and the pipeline skips
# to stale cache / lightweight context.

_GLOBAL_TIMEOUT_S: float = 3.0


# ── Compatibility stubs ───────────────────────────────────────────────────────
# These functions are imported by technobot_router.py and startup.py.
# With Yahoo removed they are harmless no-ops that preserve the call interface.

DEV_MODE: bool = False   # Yahoo-era cooldown bypass — no longer applies


def reset_yahoo_call_count() -> None:
    """No-op — Yahoo removed."""


def get_yahoo_call_count() -> int:
    """Always 0 — no Yahoo calls are made."""
    return 0


def is_rate_limited() -> bool:
    """Always False — Yahoo 429 cooldown no longer exists."""
    return False


def cooldown_remaining() -> int:
    """Always 0 — no cooldown active."""
    return 0


def reset_rate_limit() -> None:
    """No-op — Yahoo cooldown concept removed."""
    print("[TechnoBot/fetch] reset_rate_limit: no-op (Yahoo removed)")


# ── Provider wrapper: Alpha Vantage ──────────────────────────────────────────

def _try_av_fallback(
    ticker:       str,
    interval_key: str,
    reason:       str,
) -> dict | None:
    """Attempt Alpha Vantage (primary live provider).

    Lazy-imports alphavantage_fetcher.  All errors are caught — returns None
    on any failure so the pipeline continues to Twelve Data.
    Timeout is enforced inside alphavantage_fetcher (_AV_TIMEOUT = 2 s).
    """
    try:
        from data.alphavantage_fetcher import fetch_from_alphavantage
        print(f"[TechnoBot/fetch] AV attempt  {ticker}/{interval_key} ({reason})")
        result = fetch_from_alphavantage(ticker, interval_key)
        if result is not None:
            print(f"[TechnoBot/fetch] AV SUCCESS  {ticker}/{interval_key}")
        return result
    except Exception as exc:
        print(f"[TechnoBot/fetch] AV error: {type(exc).__name__}: {exc}")
        return None


# ── Provider wrapper: Twelve Data ─────────────────────────────────────────────

def _try_td_fallback(
    ticker:       str,
    interval_key: str,
    reason:       str,
) -> dict | None:
    """Attempt Twelve Data (secondary live provider).

    Lazy-imports twelvedata_fallback.  Returns None when no API key is
    configured or when any error occurs (including rate-limit — the fallback
    now returns None for all failure modes, so no special status handling
    is needed here).
    Timeout is enforced inside twelvedata_fallback (_TD_TIMEOUT = 2 s).
    """
    try:
        from data.twelvedata_fallback import TD_AVAILABLE, fetch_twelvedata_fallback
        if not TD_AVAILABLE:
            return None
        print(f"[TechnoBot/fetch] TD attempt  {ticker}/{interval_key} ({reason})")
        result = fetch_twelvedata_fallback(ticker, interval_key, reason)
        if result is None:
            return None
        # Fix 5: Explicitly verify the result carries status="ok" and a DataFrame
        # before accepting — prevents silent pass-through of malformed dicts.
        if result.get("status") == "ok" and result.get("data") is not None:
            print(f"[TechnoBot/fetch] TD ACCEPTED by pipeline")
            print(f"[TechnoBot/fetch] TD USED  {ticker}/{interval_key}")
            return result
        # Fix 6: Any non-ok result (including obsolete rate_limited) falls through
        print(f"[TechnoBot/fetch] TD result rejected (status={result.get('status')}) "
              f"for {ticker}/{interval_key}")
        return None
    except Exception as exc:
        print(f"[TechnoBot/fetch] TD error: {type(exc).__name__}: {exc}")
        return None


# ── Stale cache fallback ──────────────────────────────────────────────────────

def _try_stale_cache(ticker: str, interval_key: str) -> dict | None:
    """Return the most recent cache entry for this ticker, ignoring TTL.

    Used only when both live providers have failed.  The age is logged so
    callers can see how stale the data is.  Tagged source="stale_cache"
    for internal tracking; never shown to users.
    """
    with _CACHE_LOCK:
        entry = _DATA_CACHE.get(f"{ticker}:{interval_key}")
        if entry and entry.get("data") is not None:
            age = int(time.monotonic() - entry["_ts"])
            result = {k: v for k, v in entry.items() if k != "_ts"}
            result["source"] = "stale_cache"
            print(f"[TechnoBot/fetch] STALE_CACHE {ticker}/{interval_key} "
                  f"age={age}s — serving cached data after provider failures")
            return result
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def safe_fetch(ticker: str, interval_key: str = "1d") -> dict:
    """Fetch OHLCV data via the TD → AV → stale-cache pipeline.

    Provider order (Fix 1 — TD is PRIMARY):
      1. In-memory cache (120 s TTL)     — zero network calls.
      2. Twelve Data    (1 attempt, 2 s timeout)  — primary live provider.
      3. Alpha Vantage  (1 attempt, 2 s timeout)  — secondary live provider.
      4. Stale cache    — any age; used when both live providers fail.
      5. Return status="failed" so _cached_derive_market_context() activates
         the lightweight qualitative context (never shows "data unavailable").

    Single-timeframe mode (Fix 2):
      Only the interval_key passed by the caller is fetched — no automatic
      weekly/monthly enrichment.  Multi-TF analysis is disabled at the
      generate_multitf_context() level (Fix 3).

    A global 3-second safety timer is checked before each live provider.
    If the budget is exhausted, a [TechnoBot/perf] FAST FAIL log is emitted
    and the pipeline moves directly to stale cache / lightweight.

    Returns
    -------
    dict with keys:
        data         : pd.DataFrame | None
        status       : "ok" | "failed"
        source       : "twelvedata" | "alphavantage" | "stale_cache" | None
        last_price   : float | None
        pct_change   : float | None
        rate_limited : bool           (always False — Yahoo removed)
        error_type   : str | None
        fail_reason  : str | None
    """
    _t0 = time.monotonic()

    _FAILED: dict = {
        "data": None, "status": "failed", "source": None,
        "last_price": None, "pct_change": None, "rate_limited": False,
        "error_type": "no_data", "fail_reason": None,
    }

    if not SUPPORTED_INTERVALS.get(interval_key):
        print(f"[TechnoBot/fetch] Unknown interval '{interval_key}' for {ticker}")
        return {**_FAILED, "fail_reason": f"Unsupported interval: {interval_key}"}

    # Fix 2: Single-TF mode — log exactly what is being fetched.
    # No secondary timeframe is fetched here; multi-TF is disabled in router.
    print(f"[TechnoBot/fetch] SINGLE TF MODE → {ticker}/{interval_key}")

    cache_key = f"{ticker}:{interval_key}"

    # ── 1. Cache hit — zero network calls (Fix 6) ─────────────────────────────
    cached = _cache_get(cache_key)
    if cached:
        rows = len(cached["data"]) if cached.get("data") is not None else "n/a"
        print(f"[TechnoBot/fetch] CACHE HIT  {ticker}/{interval_key} rows={rows}")
        return {k: v for k, v in cached.items() if k != "_ts"}

    # ── Timer helpers ─────────────────────────────────────────────────────────
    def _elapsed() -> float:
        return time.monotonic() - _t0

    def _timed_out(step: str) -> bool:
        elapsed = _elapsed()
        if elapsed >= _GLOBAL_TIMEOUT_S:
            print(f"[TechnoBot/perf] FAST FAIL triggered at '{step}' "
                  f"({elapsed:.2f}s elapsed, limit={_GLOBAL_TIMEOUT_S}s)")
            return True
        return False

    # ── 2. Twelve Data — PRIMARY (Fix 1) ─────────────────────────────────────
    if not _timed_out("pre-TD"):
        _td = _try_td_fallback(ticker, interval_key, "primary")
        if _td is not None:
            _rows_td = len(_td["data"]) if _td.get("data") is not None else 0
            # Fix 7: Source/rows log after successful API call
            print(f"[TechnoBot/fetch] SOURCE=twelvedata ticker={ticker} rows={_rows_td}")
            _cache_set(cache_key, _td)
            print(f"[TechnoBot/perf] fetch completed in {_elapsed():.2f}s "
                  f"(source=twelvedata)")
            return _td

    # ── 3. Alpha Vantage — SECONDARY (Fix 1) ─────────────────────────────────
    if not _timed_out("pre-AV"):
        _av = _try_av_fallback(ticker, interval_key, "td_failed")
        if _av is not None:
            _rows_av = len(_av["data"]) if _av.get("data") is not None else 0
            # Fix 7: Source/rows log after successful API call
            print(f"[TechnoBot/fetch] SOURCE=alphavantage ticker={ticker} rows={_rows_av}")
            _cache_set(cache_key, _av)
            print(f"[TechnoBot/perf] fetch completed in {_elapsed():.2f}s "
                  f"(source=alphavantage)")
            return _av

    # ── 4. Stale cache ────────────────────────────────────────────────────────
    _sc = _try_stale_cache(ticker, interval_key)
    if _sc is not None:
        _rows_sc = len(_sc["data"]) if _sc.get("data") is not None else 0
        print(f"[TechnoBot/fetch] SOURCE=stale_cache ticker={ticker} rows={_rows_sc}")
        print(f"[TechnoBot/perf] fetch completed in {_elapsed():.2f}s "
              f"(source=stale_cache)")
        return _sc

    # ── 5. All providers failed ───────────────────────────────────────────────
    # Return _FAILED — _cached_derive_market_context() will call
    # build_lightweight_context() so the user always gets a useful response.
    print(f"[TechnoBot/fetch] ALL FAILED  {ticker}/{interval_key} "
          f"elapsed={_elapsed():.2f}s — lightweight context will be used")
    return _FAILED


# ── Legacy compatibility ──────────────────────────────────────────────────────

def fetch_nifty(interval_key: str = "1d", ticker: str = "^NSEI") -> object:
    """Backward-compatible wrapper — returns the DataFrame or None."""
    return safe_fetch(ticker, interval_key).get("data")


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time as _t

    print(f"\n{'='*60}")
    print("Self-test — Yahoo-free AV → TD pipeline")
    print(f"Global timeout: {_GLOBAL_TIMEOUT_S}s per fetch")
    print(f"{'='*60}\n")

    _test_symbols = [
        ("RELIANCE.NS", "1d"),
        ("INFY.NS",     "1d"),
        ("TCS.NS",      "1d"),
        ("^NSEI",       "1d"),    # index — AV/TD may not support, expect lightweight
        ("INVALID.NS",  "1d"),    # invalid — expect lightweight
    ]

    for sym, iv in _test_symbols:
        t0 = _t.monotonic()
        r  = safe_fetch(sym, iv)
        elapsed = _t.monotonic() - t0
        rows = len(r["data"]) if r.get("data") is not None else 0
        print(
            f"{sym:<18} /{iv:<3}  "
            f"status={r['status']:<8}  "
            f"rows={rows:<5}  "
            f"src={str(r.get('source') or '-'):<14}  "
            f"t={elapsed:.2f}s"
        )
