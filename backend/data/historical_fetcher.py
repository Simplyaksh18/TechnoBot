"""
data/historical_fetcher.py — Historical OHLCV data for TechnoBot export / chart features.

Fetch strategy per request (no caching — historical data is intentionally
always fresh since users trigger these calls explicitly):
  1. yf.download() — primary.
  2. yf.download() retry after 0.3 s.
  3. yf.Ticker().history() — flat-column fallback.
  4. Weekly-resample fallback (1wk only): fetch daily then resample.
  5. Raise RuntimeError only when all four paths fail.
"""

import time
import os
import sys
from datetime import datetime

# FIX: Emoji in path (D:\TechnoBot🤖\) causes Unicode encoding errors in yfinance/certifi
# Set encoding to UTF-8 BEFORE importing yfinance
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

import pandas as pd
import yfinance as yf


# ── Timeframe configuration ───────────────────────────────────────────────────

TIMEFRAME_MAP: dict[str, dict] = {
    "1d":  {"period": "5y",  "interval": "1d",  "label": "~5y 1d"},
    "1wk": {"period": "10y", "interval": "1wk", "label": "~10y 1wk"},
    "1mo": {"period": "15y", "interval": "1mo", "label": "~15y 1mo"},
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex, deduplicate columns, drop rows missing Close."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    if "Close" in df.columns:
        df = df.dropna(subset=["Close"])
    else:
        df = df.dropna()
    return df


def _prepare_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a DateTimeIndex is available for resampling."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    prepared = df.copy()
    if "Date" in prepared.columns:
        prepared["Date"] = pd.to_datetime(prepared["Date"])
        prepared = prepared.set_index("Date")
    elif "index" in prepared.columns:
        prepared["index"] = pd.to_datetime(prepared["index"])
        prepared = prepared.set_index("index")
    else:
        prepared.index = pd.to_datetime(prepared.index)
    return prepared


def _resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame | None:
    """Resample OHLCV data to the requested frequency."""
    try:
        df = _prepare_datetime_index(df)
        agg: dict[str, str] = {}
        if "Open"   in df.columns: agg["Open"]   = "first"
        if "High"   in df.columns: agg["High"]   = "max"
        if "Low"    in df.columns: agg["Low"]    = "min"
        if "Close"  in df.columns: agg["Close"]  = "last"
        if "Volume" in df.columns: agg["Volume"] = "sum"
        if "Close" not in agg:
            return None
        resampled = df.resample(freq).agg(agg).dropna(subset=["Close"])
        return resampled.reset_index() if not resampled.empty else None
    except Exception:
        return None


def _resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame | None:
    """Resample daily OHLCV to weekly W-bars."""
    return _resample_ohlcv(df, "W")


def _resample_to_monthly(df: pd.DataFrame) -> pd.DataFrame | None:
    """Resample daily OHLCV to monthly M-bars."""
    return _resample_ohlcv(df, "ME")


# ── Core fetch ────────────────────────────────────────────────────────────────

def fetch_historical_data(ticker: str, timeframe: str = "1d"):
    """Fetch historical price data from Yahoo Finance.

    Applies a 4-stage resilient strategy: download → retry → history()
    → weekly-resample (1wk only).  Raises RuntimeError only when all fail.

    Returns
    -------
    df   : pd.DataFrame  (Date column + OHLCV; index is integer)
    meta : dict
    """
    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}. "
                         f"Valid values: {list(TIMEFRAME_MAP)}")

    cfg = TIMEFRAME_MAP[timeframe]

    def _download() -> pd.DataFrame | None:
        try:
            df = yf.download(
                ticker,
                period=cfg["period"],
                interval=cfg["interval"],
                auto_adjust=True,
                progress=False,
            )
            if df is None or df.empty:
                return None
            df = df.reset_index()
            df = _clean_df(df)
            return df if not df.empty else None
        except Exception:
            return None

    def _download_daily() -> pd.DataFrame | None:
        try:
            df = yf.download(
                ticker,
                period="10y",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if df is None or df.empty:
                return None
            df = df.reset_index()
            df = _clean_df(df)
            return df if not df.empty else None
        except Exception:
            return None

    def _history() -> pd.DataFrame | None:
        try:
            t  = yf.Ticker(ticker)
            df = t.history(period=cfg["period"], interval=cfg["interval"])
            if df is None or df.empty:
                return None
            df = df.reset_index()
            df = _clean_df(df)
            return df if not df.empty else None
        except Exception:
            return None

    def _history_daily() -> pd.DataFrame | None:
        try:
            t  = yf.Ticker(ticker)
            df = t.history(period="10y", interval="1d")
            if df is None or df.empty:
                return None
            df = df.reset_index()
            df = _clean_df(df)
            return df if not df.empty else None
        except Exception:
            return None

    # Stage 1: primary download
    df = _download()

    # Stage 2: retry after brief pause
    if df is None:
        print(f"[TechnoBot/hist] {ticker}/{timeframe}: primary empty — retry after 0.3 s")
        time.sleep(0.3)
        df = _download()

    # Stage 3: Ticker.history() fallback
    if df is None:
        print(f"[TechnoBot/hist] {ticker}/{timeframe}: using Ticker.history() fallback")
        df = _history()

    # Stage 4: weekly-resample from daily (1wk only)
    if df is None and timeframe == "1wk":
        print(f"[TechnoBot/hist] {ticker}/1wk: all weekly attempts failed "
              f"— trying daily→weekly resample")
        daily_raw = _download_daily()
        if daily_raw is None:
            daily_raw = _history_daily()
        if daily_raw is not None:
            df = _resample_to_weekly(daily_raw)
            if df is not None:
                print(f"[TechnoBot/hist] {ticker}/1wk: resampled from daily, "
                      f"rows={len(df)}")

    # Stage 5: monthly-resample from daily (1mo only)
    if df is None and timeframe == "1mo":
        print(f"[TechnoBot/hist] {ticker}/1mo: all monthly attempts failed "
              f"— trying daily→monthly resample")
        daily_raw = _download_daily()
        if daily_raw is None:
            daily_raw = _history_daily()
        if daily_raw is not None:
            df = _resample_to_monthly(daily_raw)
            if df is not None:
                print(f"[TechnoBot/hist] {ticker}/1mo: resampled from daily, "
                      f"rows={len(df)}")

    # Stage 6: external provider fallbacks (Twelve Data → Alpha Vantage)
    # Only attempt when all Yahoo-based attempts have failed.
    if df is None:
        # Try Twelve Data fallback (silent, respects TD_AVAILABLE internally)
        try:
            from data.twelvedata_fallback import fetch_twelvedata_fallback
            print(f"[TechnoBot/hist] {ticker}/{timeframe}: trying Twelve Data fallback")
            td_result = fetch_twelvedata_fallback(ticker, timeframe, "yahoo_empty")
            if td_result and td_result.get("data") is not None:
                df = td_result["data"].reset_index()
                df = _clean_df(df)
                if df is not None and not df.empty:
                    print(f"[TechnoBot/hist] {ticker}/{timeframe}: Twelve Data returned rows={len(df)}")
        except Exception:
            pass

    if df is None:
        # Try Alpha Vantage fallback
        try:
            from data.alphavantage_fetcher import fetch_from_alphavantage
            print(f"[TechnoBot/hist] {ticker}/{timeframe}: trying Alpha Vantage fallback")
            av_result = fetch_from_alphavantage(ticker, timeframe)
            if av_result and av_result.get("data") is not None:
                df = av_result["data"].reset_index()
                df = _clean_df(df)
                if df is not None and not df.empty:
                    print(f"[TechnoBot/hist] {ticker}/{timeframe}: Alpha Vantage returned rows={len(df)}")
        except Exception:
            pass

    if df is None or df.empty:
        raise RuntimeError(
            f"No historical data available for '{ticker}' on the {timeframe} timeframe. "
            "Yahoo Finance may be temporarily throttling requests for this instrument. "
            "Try again in 30–60 seconds or switch to the daily timeframe."
        )

    print(f"[TechnoBot/hist] {ticker}/{timeframe}: rows={len(df)}")

    meta = {
        "source":       "Yahoo Finance (Historical Data)",
        "ticker":       ticker,
        "timeframe":    timeframe,
        "interval":     cfg["interval"],
        "window_label": cfg["label"],
        "rows":         len(df),
        "reference":    f"https://finance.yahoo.com/quote/{ticker}/history?p={ticker}",
    }

    return df, meta


# ── Snapshot / export helpers ─────────────────────────────────────────────────

def snapshot_head_tail(df: pd.DataFrame, n: int = 10):
    """Return first n and last n rows."""
    return df.head(n), df.tail(n)


def export_csv(df: pd.DataFrame, ticker: str, timeframe: str) -> str:
    """Export DataFrame to a timestamped CSV and return the file path."""
    os.makedirs("exports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname     = f"{ticker}_{timeframe}_historical_{timestamp}.csv"
    path      = os.path.join("exports", fname)
    df.to_csv(path, index=False)
    return path
