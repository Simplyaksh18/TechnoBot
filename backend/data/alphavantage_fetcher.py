"""
data/alphavantage_fetcher.py — Alpha Vantage secondary fallback for TechnoBot.

Priority in safe_fetch():  Yahoo (primary) → Alpha Vantage (HERE) → TwelveData → stale cache

Called ONLY when Yahoo Finance has failed.  The returned dict is schema-identical
to safe_fetch() output so derive_market_context and the comparison engine cannot
distinguish the data source.

API limits (free tier)
───────────────────────
  25 requests / day
  5  requests / minute

Stability features
───────────────────
  • 120 s in-memory cache per ticker+interval — prevents duplicate calls during
    rapid or consecutive queries (cache is shared across threads).
  • Per-key in-flight deduplication — when two concurrent requests need the same
    ticker (e.g. a comparison query that fires both sides "at once"), the second
    request waits for the first to finish and then reads the cache, making exactly
    ONE API call total per ticker per 120 s window.
  • Sliding-window throttle — enforces ≤ 5 live calls per 60 s globally.
    NON-BLOCKING: if the rate window is full, returns None immediately so
    TwelveData can take over — no sleep, no queue.
  • Single-attempt, no backoff — fail fast to TwelveData within 4 s.
  • Per-provider cooldown — if AV returns an explicit rate-limit signal the
    module backs off for _AV_RL_COOLDOWN seconds and returns None immediately
    during that window, letting TwelveData take over.
  • Graceful degradation — every error path returns None (never raises), so
    safe_fetch() always continues to the next provider without interruption.

Configuration
───────────────
  ALPHAVANTAGE_API_KEY in .env — overrides the bundled default key.
"""

from __future__ import annotations

import os
import time
import threading
import collections
import requests
import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────

# Bundled free-tier key.  Override via ALPHAVANTAGE_API_KEY env var in .env.
_AV_DEFAULT_KEY = "N5EM5IA0L4UA7A8W"
_AV_API_KEY: str = os.environ.get("ALPHAVANTAGE_API_KEY") or _AV_DEFAULT_KEY

_AV_BASE_URL  = "https://www.alphavantage.co/query"
_AV_TIMEOUT   = int(os.environ.get("AV_TIMEOUT_SEC",  "2"))    # per-request timeout (hard cap)
_AV_MAX_RETRY = int(os.environ.get("AV_MAX_RETRIES",  "1"))    # 1 attempt — fail fast

# Public availability flag (always True — key is bundled)
AV_AVAILABLE: bool = True

print(f"[TechnoBot/alpha] Alpha Vantage configured — secondary fallback ENABLED "
      f"(timeout={_AV_TIMEOUT}s retries={_AV_MAX_RETRY})")


# ── Alpha Vantage endpoint + key maps ────────────────────────────────────────

_AV_FUNCTION_MAP: dict[str, str] = {
    "1d":  "TIME_SERIES_DAILY",
    "1wk": "TIME_SERIES_WEEKLY",
    "1mo": "TIME_SERIES_MONTHLY",
}

# The JSON key under which AV returns the time-series data
_AV_SERIES_KEY: dict[str, str] = {
    "TIME_SERIES_DAILY":   "Time Series (Daily)",
    "TIME_SERIES_WEEKLY":  "Weekly Time Series",
    "TIME_SERIES_MONTHLY": "Monthly Time Series",
}

# Approximate row count to keep (matches Yahoo Finance period lengths)
_AV_MAX_ROWS: dict[str, int] = {
    "1d":  130,   # ~6 months of trading days
    "1wk": 110,   # ~2 years of weeks
    "1mo": 65,    # ~5 years of months
}


# ── Ticker resolution: Yahoo format → Alpha Vantage BSE format ───────────────

# Index tickers (^NSEI etc.) are not available on AV free tier — return None
_AV_INDEX_PREFIXES = ("^", )

# Explicit overrides for tickers whose auto-conversion would produce wrong results
_AV_TICKER_OVERRIDE: dict[str, str | None] = {
    "^NSEI":      None,   # Nifty 50 — not on AV free tier
    "^BSESN":     None,   # Sensex — not on AV free tier
    "^NSEBANK":   None,
    "^CNXIT":     None,
    "^CNXPHARMA": None,
    "^CNXAUTO":   None,
    "^CNXFMCG":   None,
    "^CNXMETAL":  None,
    "^NSEMDCP50": None,
    "^INDIAVIX":  None,
    # Tickers with special characters need remapping
    "M&M.NS":     "M&M.BSE",       # AV supports & in BSE symbols
    "BAJAJ-AUTO.NS": "BAJAJ-AUTO.BSE",
    "L&TFH.NS":   "L&TFH.BSE",
}


def _to_av_symbol(yahoo_ticker: str) -> str | None:
    """Convert a Yahoo Finance ticker to Alpha Vantage BSE symbol.

    Conversion rules (applied in order):
      1. Explicit override map — handles indices, special characters.
      2. Tickers starting with "^" → None (indices not supported).
      3. Ticker ends with ".NS" → strip suffix, append ".BSE".
      4. Ticker ends with ".BSE" → use as-is.
      5. Ticker ends with ".BO" → strip ".BO", append ".BSE".
      6. No exchange suffix → append ".BSE" (assume Indian equity).

    Returns None when conversion is not possible or AV doesn't support
    the instrument (indices, US stocks, etc.).
    """
    # Explicit overrides (includes index blocking)
    if yahoo_ticker in _AV_TICKER_OVERRIDE:
        result = _AV_TICKER_OVERRIDE[yahoo_ticker]
        return result   # may be None — caller handles that

    # Reject index tickers not caught by the override map
    for prefix in _AV_INDEX_PREFIXES:
        if yahoo_ticker.startswith(prefix):
            return None

    ticker_upper = yahoo_ticker.upper()

    if ticker_upper.endswith(".NS"):
        return ticker_upper[:-3] + ".BSE"

    if ticker_upper.endswith(".BSE"):
        return ticker_upper   # already correct

    if ticker_upper.endswith(".BO"):
        return ticker_upper[:-3] + ".BSE"

    # No exchange suffix — assume Indian equity on BSE
    if "." not in ticker_upper:
        return ticker_upper + ".BSE"

    # Unknown suffix (e.g. ".L" London, ".T" Tokyo) — skip AV
    return None


# ── In-memory cache ───────────────────────────────────────────────────────────
# Separate from Yahoo/TD cache so AV results can have a different TTL.
# Cache key: "{yahoo_ticker}:{interval_key}"

_AV_CACHE: dict[str, dict] = {}
_AV_CACHE_LOCK = threading.Lock()
_AV_CACHE_TTL  = 120   # seconds — longer than Yahoo's 60 s to conserve 25/day limit


def _av_cache_get(key: str) -> dict | None:
    with _AV_CACHE_LOCK:
        entry = _AV_CACHE.get(key)
        if entry and (time.monotonic() - entry["_ts"]) < _AV_CACHE_TTL:
            return entry
    return None


def _av_cache_set(key: str, result: dict) -> None:
    with _AV_CACHE_LOCK:
        _AV_CACHE[key] = {**result, "_ts": time.monotonic()}


# ── In-flight deduplication ───────────────────────────────────────────────────
# When two threads need the same ticker simultaneously (e.g. a comparison query
# that resolves both instruments at once), one thread performs the fetch while
# the other waits and then reads the cache — exactly ONE API call is made.

_AV_INFLIGHT: dict[str, threading.Event] = {}
_AV_INFLIGHT_LOCK = threading.Lock()


def _inflight_wait(key: str, timeout: float = 0.5) -> bool:
    """If `key` is already being fetched, do a short non-blocking wait and return True.
    Returns False immediately if `key` is not in-flight (this thread should fetch it).

    Timeout deliberately short (0.5 s) — if the concurrent fetch hasn't finished
    in half a second we proceed independently rather than blocking the response.
    """
    with _AV_INFLIGHT_LOCK:
        event = _AV_INFLIGHT.get(key)
    if event is None:
        return False
    event.wait(timeout=timeout)
    return True


def _inflight_start(key: str) -> threading.Event:
    """Mark `key` as in-flight.  Returns the Event that the fetching thread
    must call .set() on when done."""
    event = threading.Event()
    with _AV_INFLIGHT_LOCK:
        _AV_INFLIGHT[key] = event
    return event


def _inflight_done(key: str, event: threading.Event) -> None:
    """Mark `key` as done, unblocking any waiting threads."""
    event.set()
    with _AV_INFLIGHT_LOCK:
        _AV_INFLIGHT.pop(key, None)


# ── Global throttle — ≤ 5 requests per 60 s ──────────────────────────────────
# Uses a sliding-window deque of the last 5 request timestamps.
# NON-BLOCKING: if the rate window is full, returns False immediately so the
# caller skips to TwelveData without any sleep or queue.

_AV_THROTTLE_LOCK      = threading.Lock()
_AV_REQUEST_TIMES: collections.deque = collections.deque(maxlen=5)
_AV_RATE_WINDOW_S      = 62    # slightly more than 60 s for safety margin


def _throttle_wait() -> bool:
    """Non-blocking rate-limit check.

    Returns True  — slot acquired, caller may proceed with the API call.
    Returns False — rate window full; caller must skip AV and try next provider.

    REMOVED: the previous blocking while-sleep loop (up to 70 s wait).
    Rationale: blocking here was the primary cause of 3–4 minute response times.
    AV is a fallback — if it's rate-limited, we move on immediately.
    """
    with _AV_THROTTLE_LOCK:
        now = time.monotonic()
        if len(_AV_REQUEST_TIMES) < 5:
            # Fewer than 5 recent requests — slot available
            _AV_REQUEST_TIMES.append(now)
            return True
        oldest = _AV_REQUEST_TIMES[0]
        if now - oldest >= _AV_RATE_WINDOW_S:
            # Oldest of the last 5 has aged out of the 60 s window — slot available
            _AV_REQUEST_TIMES.append(now)
            return True
        # Window full — skip immediately, no sleep
        remaining = int(_AV_RATE_WINDOW_S - (now - oldest))
        print(f"[TechnoBot/perf] SKIP AV (rate limit — no wait, "
              f"window resets in ~{remaining}s)")
        return False


# ── Per-provider rate-limit cooldown ─────────────────────────────────────────
# If AV explicitly returns a rate-limit response, back off for _AV_RL_COOLDOWN
# seconds and return None immediately during that window.

_AV_RL_LOCK     = threading.Lock()
_AV_RL_UNTIL    = 0.0
_AV_RL_COOLDOWN = 65   # seconds


def _av_in_cooldown() -> bool:
    with _AV_RL_LOCK:
        return time.monotonic() < _AV_RL_UNTIL


def _av_trigger_cooldown() -> None:
    global _AV_RL_UNTIL
    with _AV_RL_LOCK:
        now = time.monotonic()
        if now >= _AV_RL_UNTIL:
            _AV_RL_UNTIL = now + _AV_RL_COOLDOWN
            print(f"[TechnoBot/alpha] RATE LIMIT DETECTED — "
                  f"AV cooldown active for {_AV_RL_COOLDOWN}s")
        else:
            remaining = int(_AV_RL_UNTIL - now)
            print(f"[TechnoBot/alpha] AV cooldown already active "
                  f"({remaining}s remaining) — NOT extending")


_AV_RL_SIGNALS = (
    "thank you for using alpha vantage",   # AV's polite rate-limit message
    "our standard api call frequency",
    "5 calls per minute",
    "25 requests per day",
    "rate limit",
    "premium",
)


def _is_av_rate_limit(payload: dict) -> bool:
    """Return True if the AV JSON response is a rate-limit / quota message."""
    note = str(payload.get("Note") or payload.get("Information") or "").lower()
    return any(sig in note for sig in _AV_RL_SIGNALS)


# ── Response normalisation ────────────────────────────────────────────────────

def _normalise_av_response(
    time_series: dict,
    interval_key: str,
) -> pd.DataFrame | None:
    """Convert AV time-series dict to a yfinance-identical DataFrame.

    AV format (newest first by default, sorted ascending here):
      { "2024-01-15": {"1. open": "2500.00", "2. high": "2520.00",
                       "3. low": "2490.00",  "4. close": "2510.00",
                       "5. volume": "1234567"}, ... }

    Output DataFrame:
      DatetimeIndex (UTC, ascending)
      Float64 columns: Open, High, Low, Close, Volume
      No NaN rows in Close
    """
    max_rows = _AV_MAX_ROWS.get(interval_key, 130)
    rows: list[dict] = []

    for date_str, ohlcv in time_series.items():
        try:
            close_raw = (ohlcv.get("4. close") or ohlcv.get("4b. adjusted close")
                         or ohlcv.get("5. adjusted close"))
            if close_raw is None or close_raw in ("", "null", "None"):
                continue

            close_val = float(close_raw)
            if close_val <= 0:
                continue

            rows.append({
                "_dt":   date_str,
                "Open":  float(ohlcv.get("1. open",   0) or 0),
                "High":  float(ohlcv.get("2. high",   0) or 0),
                "Low":   float(ohlcv.get("3. low",    0) or 0),
                "Close": close_val,
                "Volume": float(ohlcv.get("5. volume", 0) or 0),
            })
        except (ValueError, TypeError, AttributeError):
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows)

    try:
        df.index = pd.to_datetime(df["_dt"], utc=True)
        df.index.name = "Date"
        df = df.drop(columns=["_dt"])
        df = df.sort_index()                  # ascending (oldest → newest)
        df = df.tail(max_rows)                # keep only as many rows as Yahoo would
    except Exception as e:
        print(f"[TechnoBot/alpha] DateTime parse error: {e}")
        return None

    # Final type enforcement
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])

    return df if not df.empty else None


# ── Usage stats (debug only) ──────────────────────────────────────────────────

_stats_lock         = threading.Lock()
_stats_av_success   = 0
_stats_av_failure   = 0
_stats_av_cache_hit = 0
_stats_av_throttled = 0


def get_av_stats() -> dict:
    """Return a snapshot of AV usage counters (for debug/health endpoints)."""
    with _stats_lock:
        return {
            "av_success":   _stats_av_success,
            "av_failure":   _stats_av_failure,
            "av_cache_hit": _stats_av_cache_hit,
            "av_throttled": _stats_av_throttled,
        }


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_from_alphavantage(
    ticker:       str,
    interval_key: str = "1d",
) -> dict | None:
    """Fetch OHLCV from Alpha Vantage.  Returns safe_fetch-compatible dict or None.

    Never raises — all error paths return None so safe_fetch() continues to
    the next provider (TwelveData) without interruption.

    Parameters
    ----------
    ticker       : Yahoo Finance ticker (e.g. "RELIANCE.NS", "INFY.NS")
    interval_key : One of "1d", "1wk", "1mo"
    """
    # ── Resolve ticker ────────────────────────────────────────────────────────
    av_symbol = _to_av_symbol(ticker)
    if av_symbol is None:
        # Index or unsupported ticker — skip silently
        return None

    function = _AV_FUNCTION_MAP.get(interval_key)
    if not function:
        print(f"[TechnoBot/alpha] Unsupported interval '{interval_key}' — skipping AV")
        return None

    cache_key = f"{ticker}:{interval_key}"

    # ── 1. Cache hit ──────────────────────────────────────────────────────────
    cached = _av_cache_get(cache_key)
    if cached:
        global _stats_av_cache_hit
        with _stats_lock:
            _stats_av_cache_hit += 1
        rows = len(cached["data"]) if cached.get("data") is not None else "n/a"
        print(f"[TechnoBot/alpha] CACHE HIT  {av_symbol}/{interval_key} rows={rows}")
        return {k: v for k, v in cached.items() if k != "_ts"}

    # ── 2. AV rate-limit cooldown check ──────────────────────────────────────
    if _av_in_cooldown():
        print(f"[TechnoBot/alpha] AV COOLDOWN — skipping {av_symbol}, "
              f"trying next provider")
        return None

    # ── 3. In-flight deduplication ───────────────────────────────────────────
    # If another thread is already fetching this key, wait and use its result.
    already_fetching = _inflight_wait(cache_key)
    if already_fetching:
        # The other thread finished — check cache
        cached = _av_cache_get(cache_key)
        if cached:
            rows = len(cached["data"]) if cached.get("data") is not None else "n/a"
            print(f"[TechnoBot/alpha] IN-FLIGHT HIT {av_symbol}/{interval_key} "
                  f"rows={rows} (waited for concurrent fetch)")
            return {k: v for k, v in cached.items() if k != "_ts"}
        # Concurrent fetch also failed — fall through to our own attempt
        # (This is rare; the concurrent thread may have hit a different error.)

    # ── 4. Throttle check ────────────────────────────────────────────────────
    slot_acquired = _throttle_wait()
    if not slot_acquired:
        global _stats_av_throttled
        with _stats_lock:
            _stats_av_throttled += 1
        return None

    # ── 5. Live fetch with retry ──────────────────────────────────────────────
    inflight_event = _inflight_start(cache_key)
    try:
        return _do_fetch(av_symbol, ticker, interval_key, function, cache_key)
    finally:
        _inflight_done(cache_key, inflight_event)


def _do_fetch(
    av_symbol:    str,
    yahoo_ticker: str,
    interval_key: str,
    function:     str,
    cache_key:    str,
) -> dict | None:
    """Perform the actual API call with exponential-backoff retry.
    Separated from fetch_from_alphavantage() so the in-flight event is always
    released (via the try/finally in the caller).
    """
    series_key = _AV_SERIES_KEY[function]
    params = {
        "function":   function,
        "symbol":     av_symbol,
        "outputsize": "compact",   # last 100 data points (~5 months daily)
        "datatype":   "json",
        "apikey":     _AV_API_KEY,
    }

    # No backoff delays — fail fast to TwelveData.
    # _AV_MAX_RETRY=1 means this loop runs exactly once under normal config.
    for attempt in range(1, _AV_MAX_RETRY + 1):
        if attempt > 1:
            print(f"[TechnoBot/alpha] RETRY {attempt}/{_AV_MAX_RETRY} "
                  f"{av_symbol}/{interval_key} (no backoff — fail fast)")

        print(f"[TechnoBot/alpha] FETCH  {av_symbol}/{interval_key} "
              f"function={function} attempt={attempt}")

        try:
            resp = requests.get(
                _AV_BASE_URL,
                params=params,
                timeout=_AV_TIMEOUT,
                headers={"User-Agent": "TechnoBot/1.0"},
            )
            resp.raise_for_status()
            payload = resp.json()

        except requests.exceptions.Timeout:
            print(f"[TechnoBot/perf] SKIP AV (timeout after {_AV_TIMEOUT}s — "
                  f"moving to next provider)")
            _record_av_failure()
            return None   # fail fast — no retry on timeout

        except requests.exceptions.HTTPError as e:
            status_code = getattr(e.response, "status_code", None)
            if status_code == 429:
                print(f"[TechnoBot/alpha] HTTP 429 {av_symbol} — activating AV cooldown")
                _av_trigger_cooldown()
                _record_av_failure()
                return None   # no point retrying during cooldown
            print(f"[TechnoBot/alpha] HTTP ERROR {status_code} {av_symbol}: {e}")
            continue

        except requests.exceptions.RequestException as e:
            print(f"[TechnoBot/alpha] REQUEST ERROR {av_symbol} attempt={attempt}: {e}")
            continue

        except Exception as e:
            print(f"[TechnoBot/alpha] UNEXPECTED {type(e).__name__} "
                  f"{av_symbol} attempt={attempt}: {e}")
            continue

        # ── Check for AV rate-limit / quota message in payload ───────────────
        if _is_av_rate_limit(payload):
            print(f"[TechnoBot/alpha] RATE LIMIT RESPONSE {av_symbol} — "
                  f"activating AV cooldown")
            _av_trigger_cooldown()
            _record_av_failure()
            return None   # no point retrying

        # ── Check for AV error message ───────────────────────────────────────
        if "Error Message" in payload:
            err = payload["Error Message"]
            print(f"[TechnoBot/alpha] API ERROR {av_symbol}: {err!r}")
            # Invalid symbol — do not retry, not a transient error
            _record_av_failure()
            return None

        # ── Extract time series ───────────────────────────────────────────────
        time_series = payload.get(series_key)
        if not time_series:
            print(f"[TechnoBot/alpha] EMPTY SERIES {av_symbol}/{interval_key} "
                  f"key={series_key!r} payload_keys={list(payload.keys())}")
            continue   # retry — may be a transient issue

        # ── Normalise ─────────────────────────────────────────────────────────
        df = _normalise_av_response(time_series, interval_key)
        if df is None or df.empty:
            print(f"[TechnoBot/alpha] NORMALISE FAILED {av_symbol}/{interval_key}")
            continue

        # ── Success ───────────────────────────────────────────────────────────
        _record_av_success()
        result = {
            "data":         df,
            "status":       "ok",
            "source":       "alphavantage",   # internal tracking; never user-facing
            "last_price":   None,
            "pct_change":   None,
            "rate_limited": False,
            "error_type":   None,
            "fail_reason":  None,
        }
        _av_cache_set(cache_key, result)
        print(f"[TechnoBot/alpha] FETCH SUCCESS  {av_symbol}/{interval_key} "
              f"rows={len(df)} "
              f"[ok={_stats_av_success} fail={_stats_av_failure} "
              f"cache={_stats_av_cache_hit}]")
        return result

    # All retry attempts exhausted
    print(f"[TechnoBot/alpha] ALL RETRIES FAILED  {av_symbol}/{interval_key} — "
          f"moving to TwelveData")
    _record_av_failure()
    return None


# ── Stats helpers (module-level so they persist across calls) ─────────────────

def _record_av_success() -> None:
    global _stats_av_success
    with _stats_lock:
        _stats_av_success += 1


def _record_av_failure() -> None:
    global _stats_av_failure
    with _stats_lock:
        _stats_av_failure += 1


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Alpha Vantage self-test")
    print(f"API key : {_AV_API_KEY[:8]}{'*' * (len(_AV_API_KEY) - 8)}")
    print(f"{'='*60}\n")

    _test_cases = [
        ("RELIANCE.NS", "1d"),
        ("INFY.NS",     "1d"),
        ("TCS.NS",      "1wk"),
        ("^NSEI",       "1d"),    # should return None (index)
        ("INVALID_XYZ", "1d"),    # should return None gracefully
    ]
    for sym, iv in _test_cases:
        result = fetch_from_alphavantage(sym, iv)
        if result and result.get("data") is not None:
            df = result["data"]
            print(f"  {sym:<18} /{iv:<3}  ✓  rows={len(df)} "
                  f"last_close={df['Close'].iloc[-1]:.2f}")
        elif result:
            print(f"  {sym:<18} /{iv:<3}  minimal  price={result.get('last_price')}")
        else:
            print(f"  {sym:<18} /{iv:<3}  ✗  (no data / not supported)")

    print(f"\nStats: {get_av_stats()}")
