"""
data/twelvedata_fallback.py — Twelve Data silent fallback provider for TechnoBot.

Called ONLY when Yahoo Finance (primary provider) has exhausted all retry
attempts.  The transition is completely invisible to the user:

  • The returned dict is schema-identical to safe_fetch() output.
  • Downstream code (derive_market_context, comparison, evidence) cannot
    distinguish Yahoo data from Twelve Data data.
  • No provider name appears in any user-facing response.
  • Provider switching is tracked internally via DEBUG_YF logs only.

Configuration  (set in .env or environment)
───────────────────────────────────────────
  TWELVEDATA_API_KEY   — required; free tier = 800 credits/day, 8 req/min
  TD_TIMEOUT_SEC       — optional; default 10 s per request
  TD_MAX_RETRIES       — optional; default 2 (total attempts including first)

Free tier limits
────────────────
  800 API credits/day.  Each time_series call = 1 credit.
  8 requests/minute.
  This fallback is only called after Yahoo fails, so daily volume
  should remain well within free-tier limits for normal usage.

Normalisation guarantee
────────────────────────
  Output DataFrame has:
    • DatetimeIndex (UTC-normalised, ascending)
    • Float64 columns: Open, High, Low, Close, Volume
    • No NaN rows in Close
  This is identical to what _flatten_and_clean() produces from yfinance.
"""

from __future__ import annotations

import os
import time
import threading
import requests
import pandas as pd
from collections import deque

from data.ticker_resolver import yahoo_to_twelvedata


# ── Configuration ─────────────────────────────────────────────────────────────

_TD_API_KEY:    str | None = os.environ.get("TWELVEDATA_API_KEY") or None
_TD_TIMEOUT:    int        = int(os.environ.get("TD_TIMEOUT_SEC",   "2"))   # hard cap
_TD_MAX_RETRY:  int        = int(os.environ.get("TD_MAX_RETRIES",   "1"))   # fail fast
_TD_BASE_URL:   str        = "https://api.twelvedata.com/time_series"

# Public flag — False when no API key is configured (safe_fetch checks this
# before attempting the fallback, avoiding an unnecessary import cost).
TD_AVAILABLE: bool = bool(_TD_API_KEY)

if TD_AVAILABLE:
    print(f"[TechnoBot/fallback] Twelve Data configured — fallback ENABLED "
          f"(timeout={_TD_TIMEOUT}s retries={_TD_MAX_RETRY})")
else:
    print("[TechnoBot/fallback] TWELVEDATA_API_KEY not set — fallback DISABLED. "
          "Set it in .env to enable automatic recovery from Yahoo failures.")


# ── Fix 1: Global rate limiter — 8 req/min free-tier enforcement ─────────────
# Tracks the wall-clock timestamp of each outgoing TD API call.
# Before every call, _td_throttle() checks whether the oldest of the last 8
# timestamps was less than 60 s ago; if so, it sleeps the remaining window.
#
# maxlen=8 means the deque automatically evicts the oldest entry when a 9th is
# appended — keeping only the 8 most recent timestamps.

_TD_CALL_TIMESTAMPS: deque = deque(maxlen=8)
_TD_THROTTLE_LOCK   = threading.Lock()


def _td_throttle() -> None:
    """Block until a TD API call is safe under the 8 req/min free-tier limit."""
    with _TD_THROTTLE_LOCK:
        now = time.time()
        if len(_TD_CALL_TIMESTAMPS) == 8:
            oldest  = _TD_CALL_TIMESTAMPS[0]
            elapsed = now - oldest
            if elapsed < 60:
                wait_time = 60 - elapsed
                print(f"[TechnoBot/TD] Throttling for {wait_time:.1f}s to avoid rate limit")
                time.sleep(wait_time)
        _TD_CALL_TIMESTAMPS.append(time.time())


# ── Rate-limit sentinel — returned by _fetch_td_raw when TD says "too many reqs"
# Using a module-level singleton lets the caller use identity check (`is`)
# to distinguish rate-limit from generic failure (None) without a type change.
_TD_RATE_LIMIT_SENTINEL: dict = {"_rate_limited": True}


# ── Interval / outputsize mapping ─────────────────────────────────────────────
# Twelve Data interval names differ from Yahoo Finance period strings.
# outputsize is chosen to match the number of rows Yahoo would normally return.

_TD_INTERVAL_MAP: dict[str, str] = {
    "1d":  "1day",
    "1wk": "1week",
    "1mo": "1month",
}

_TD_OUTPUTSIZE_MAP: dict[str, int] = {
    "1d":  130,   # ~6 months of trading days  (Yahoo period="6mo")
    "1wk": 110,   # ~2 years of weeks           (Yahoo period="2y")
    "1mo": 65,    # ~5 years of months          (Yahoo period="5y")
}


# ── Fallback usage tracking ───────────────────────────────────────────────────
# These counters are logged in debug mode only — never exposed to users.

_stats_lock          = threading.Lock()
_stats_td_success    = 0   # times Twelve Data returned usable data
_stats_td_failure    = 0   # times Twelve Data also failed
_stats_yahoo_failure = 0   # times Yahoo triggered a fallback attempt


def _record_td_success() -> None:
    global _stats_td_success
    with _stats_lock:
        _stats_td_success += 1


def _record_td_failure() -> None:
    global _stats_td_failure
    with _stats_lock:
        _stats_td_failure += 1


def _record_yahoo_failure() -> None:
    global _stats_yahoo_failure
    with _stats_lock:
        _stats_yahoo_failure += 1


def get_fallback_stats() -> dict:
    """Return a snapshot of fallback usage counters (for debug/health endpoints)."""
    with _stats_lock:
        return {
            "yahoo_failures":   _stats_yahoo_failure,
            "td_successes":     _stats_td_success,
            "td_failures":      _stats_td_failure,
            "td_available":     TD_AVAILABLE,
        }


# ── Twelve Data API call ───────────────────────────────────────────────────────

def _fetch_td_raw(
    td_symbol:   str,
    td_interval: str,
    outputsize:  int,
    attempt:     int = 1,
) -> list[dict] | dict | None:
    """Call the Twelve Data time_series endpoint.

    Returns:
      list[dict]               — 'values' list on success
      _TD_RATE_LIMIT_SENTINEL  — when TD responds with a rate-limit error
                                 (code 400/429 with "rate limit" in message)
      None                     — any other failure (timeout, network, API error)

    Does NOT raise — all errors are caught and logged.
    """
    params = {
        "symbol":     td_symbol,
        "interval":   td_interval,
        "outputsize": outputsize,
        "order":      "ASC",          # oldest first, matching yfinance default
        "apikey":     _TD_API_KEY,
    }
    try:
        resp = requests.get(
            _TD_BASE_URL,
            params=params,
            timeout=_TD_TIMEOUT,
            headers={"User-Agent": "TechnoBot/1.0"},
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.exceptions.Timeout:
        print(f"[TechnoBot/fallback] Twelve Data TIMEOUT "
              f"symbol={td_symbol} interval={td_interval} attempt={attempt}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[TechnoBot/fallback] Twelve Data REQUEST ERROR "
              f"symbol={td_symbol} attempt={attempt}: {e}")
        return None
    except Exception as e:
        print(f"[TechnoBot/fallback] Twelve Data UNEXPECTED ERROR "
              f"symbol={td_symbol} attempt={attempt}: {type(e).__name__}: {e}")
        return None

    status = payload.get("status", "error")
    if status != "ok":
        code    = payload.get("code", "?")
        message = payload.get("message", "unknown error")
        print(f"[TechnoBot/fallback] Twelve Data API error "
              f"symbol={td_symbol} status={status} code={code} msg={message!r}")
        # Fix 4: Detect rate-limit responses (TD uses code 429; some plans use 400)
        if "rate limit" in str(message).lower() or code in (400, 429):
            print(f"[TechnoBot/TD] Rate limit response detected "
                  f"symbol={td_symbol} code={code}")
            return _TD_RATE_LIMIT_SENTINEL
        return None

    values = payload.get("values")
    if not values:
        print(f"[TechnoBot/fallback] Twelve Data: empty 'values' list "
              f"symbol={td_symbol} interval={td_interval}")
        return None

    return values


# ── Normalisation ─────────────────────────────────────────────────────────────

def _normalise_td_response(
    values: list[dict],
    interval_key: str,
) -> pd.DataFrame | None:
    """Convert Twelve Data 'values' list to a yfinance-compatible DataFrame.

    The output DataFrame is schema-identical to what _flatten_and_clean()
    produces from a yfinance download:
      • DatetimeIndex (UTC, ascending)
      • Float64 columns: Open, High, Low, Close, Volume
      • No NaN rows in Close

    Twelve Data 'values' format (order="ASC"):
      [{"datetime": "2024-01-02", "open": "2500.00", "high": "2520.00",
        "low": "2490.00", "close": "2510.00", "volume": "1234567"}, ...]
    """
    if not values:
        return None

    rows: list[dict] = []
    for v in values:
        try:
            close_raw = v.get("close") or v.get("Close")
            if close_raw is None or close_raw in ("", "null", "None"):
                continue   # skip rows with no Close (same as yfinance dropna)

            rows.append({
                "Open":   float(v.get("open")   or v.get("Open")   or 0),
                "High":   float(v.get("high")   or v.get("High")   or 0),
                "Low":    float(v.get("low")    or v.get("Low")    or 0),
                "Close":  float(close_raw),
                "Volume": float(v.get("volume") or v.get("Volume") or 0),
                "_dt":    v.get("datetime") or v.get("Datetime", ""),
            })
        except (ValueError, TypeError):
            continue   # skip malformed rows silently

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]   # Twelve Data sometimes returns 0.0 for holidays

    # Build DatetimeIndex — normalise to UTC so it matches yfinance output
    try:
        df.index = pd.to_datetime(df["_dt"], utc=True)
        df.index.name = "Date"
        df = df.drop(columns=["_dt"])
        df = df.sort_index()
    except Exception as e:
        print(f"[TechnoBot/fallback] DateTime parsing failed: {e}")
        return None

    # Ensure correct dtypes
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"])

    if df.empty:
        return None

    return df


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_twelvedata_fallback(
    ticker:           str,
    interval_key:     str,
    yahoo_fail_reason: str = "unknown",
) -> dict | None:
    """Attempt to fetch OHLCV data from Twelve Data when Yahoo has failed.

    Parameters
    ----------
    ticker           : Yahoo Finance ticker (e.g. "RELIANCE.NS", "^NSEI")
    interval_key     : One of "1d", "1wk", "1mo"
    yahoo_fail_reason: Short string describing why Yahoo failed (for logging)

    Returns
    -------
    dict  — schema-identical to safe_fetch() output with status="ok"
            (error_type=None, fail_reason=None, source="twelvedata")
    None  — any TD failure (rate-limit, timeout, bad data); caller falls through
            to stale cache / lightweight context
    """
    if not TD_AVAILABLE:
        return None

    td_interval = _TD_INTERVAL_MAP.get(interval_key)
    outputsize  = _TD_OUTPUTSIZE_MAP.get(interval_key)
    if not td_interval or not outputsize:
        print(f"[TechnoBot/fallback] Unsupported interval '{interval_key}' "
              f"for Twelve Data fallback")
        return None

    # Convert Yahoo ticker → Twelve Data symbol
    td_symbol = yahoo_to_twelvedata(ticker)
    if not td_symbol:
        print(f"[TechnoBot/fallback] Cannot convert '{ticker}' to "
              f"Twelve Data format — skipping fallback")
        return None

    _record_yahoo_failure()
    print(f"[TechnoBot/fallback] Yahoo failed ({yahoo_fail_reason}) — "
          f"trying Twelve Data: {ticker} → {td_symbol} "
          f"interval={td_interval} outputsize={outputsize}")

    # Fix 2: Throttle before every TD API call (8 req/min free-tier guard).
    # Fix 3: Single attempt only — retries disabled to avoid double quota burn.
    _td_throttle()
    raw = _fetch_td_raw(td_symbol, td_interval, outputsize, attempt=1)

    # Fix 1: Rate-limit sentinel → None.  Caller already treats None as fallback
    # trigger — no new status formats needed or wanted.
    if raw is _TD_RATE_LIMIT_SENTINEL:
        _record_td_failure()
        print(f"[TechnoBot/TD] Rate limit confirmed — returning None "
              f"for {td_symbol}/{interval_key}")
        return None

    values: list[dict] | None = raw  # type: ignore[assignment]

    if values is None:
        _record_td_failure()
        print(f"[TechnoBot/fallback] Twelve Data also failed for "
              f"{td_symbol}/{interval_key} — returning no data")
        return None

    # ── Normalise to yfinance-identical DataFrame ─────────────────────────────
    df = _normalise_td_response(values, interval_key)
    if df is None or df.empty:
        _record_td_failure()
        print(f"[TechnoBot/fallback] Twelve Data normalisation failed for "
              f"{td_symbol}/{interval_key}")
        return None

    # ── Fix 3: Harden DataFrame before returning ──────────────────────────────
    # Downstream (nifty_context.py) expects uppercase column names matching the
    # yfinance / Alpha Vantage convention: Open, High, Low, Close, Volume.
    # Ensure every required column exists, force float64, drop bad rows, and
    # reject batches that are too small to compute meaningful indicators.
    df = df.copy()

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    df = df.astype({
        "Open":   "float64",
        "High":   "float64",
        "Low":    "float64",
        "Close":  "float64",
        "Volume": "float64",
    })

    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]   # remove holiday / zero-price rows

    # Fix 4: Relaxed row floor — accept >= 5 rows (matches nifty_context.py floor).
    # The previous < 10 threshold rejected valid partial data (new listings,
    # freshly-resumed tickers) and pushed the system into unnecessary lightweight mode.
    if df.empty or len(df) < 5:
        _record_td_failure()
        print(f"[TechnoBot/TD] Data rejected — insufficient rows ({len(df)}) "
              f"for {td_symbol}/{interval_key}")
        return None
    if len(df) < 10:
        print(f"[TechnoBot/context] LOW DATA but ACCEPTED rows={len(df)} "
              f"for {td_symbol}/{interval_key}")

    print(f"[TechnoBot/TD] FINAL DF rows={len(df)} cols={list(df.columns)}")

    _record_td_success()
    stats = get_fallback_stats()
    print(f"[TechnoBot/fetch] TD USED  {td_symbol}/{interval_key} rows={len(df)}")
    print(f"[TechnoBot/fallback] SUCCESS  {td_symbol}/{interval_key} "
          f"rows={len(df)} "
          f"[td_ok={stats['td_successes']} td_fail={stats['td_failures']}]")

    # Fix 2: Return schema-identical dict — exactly the same shape as AV output.
    # All keys that safe_fetch() consumers expect must be present.
    return {
        "data":         df,
        "status":       "ok",
        "source":       "twelvedata",   # internal tracking only; not shown to users
        "last_price":   None,
        "pct_change":   None,
        "rate_limited": False,
        "error_type":   None,
        "fail_reason":  None,
    }


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nTD_AVAILABLE : {TD_AVAILABLE}")
    print(f"API key set  : {'yes' if _TD_API_KEY else 'no'}")
    if not TD_AVAILABLE:
        print("Set TWELVEDATA_API_KEY in .env to run self-test.")
    else:
        for sym, iv in [("RELIANCE.NS", "1d"), ("^NSEI", "1d"), ("TCS.NS", "1wk")]:
            result = fetch_twelvedata_fallback(sym, iv, "self-test")
            if result:
                df = result["data"]
                print(f"  {sym}/{iv}: rows={len(df)} "
                      f"close_last={df['Close'].iloc[-1]:.2f}")
            else:
                print(f"  {sym}/{iv}: no data returned")
