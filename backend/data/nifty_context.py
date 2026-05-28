"""
data/nifty_context.py — Market context derivation for TechnoBot.

Calls safe_fetch() and computes TA indicators from the resulting OHLCV data.
Gracefully handles five quality levels returned by safe_fetch:

  ok / partial / resampled
      Full OHLCV available → compute all indicators normally.

  minimal (fast_info only)
      No OHLCV. Use last price + day % change to infer directional bias.
      Returns a "partial" context so the AI layer can still produce a
      meaningful (if limited) response instead of a hard error.

  rate_limited
      Yahoo Finance returned HTTP 429.  Returns a flagged context so the
      router can surface an accurate "rate limited" message rather than
      falsely reporting the ticker as delisted or unrecognised.

  failed
      Truly nothing available → return _UNKNOWN so the data validation gate
      in api_server.py shows a clean "data unavailable" message.
"""

import pandas as pd
from data.nifty_fetcher import safe_fetch
from datetime import datetime


# ── Deterministic RSI classification ─────────────────────────────────────────

def classify_rsi(
    rsi: float | None,
    trend: str,
    slope: str = "stable",
) -> tuple[str, str]:
    """Return (rsi_state, rsi_note) from RSI value, trend direction, and slope.

    rsi_state   — one of: overbought | bullish | mildly_bullish | neutral |
                           mildly_bearish | bearish | oversold | n/a
    rsi_note    — plain-English interpretation for beginners.  Varies by zone,
                  slope (rising/falling/stable), and proximity to key levels
                  (30 / 50 / 70) so the same RSI range never produces identical
                  wording across different structural conditions.

    slope       — "rising"  : RSI up >2 pts vs 3 bars ago
                  "falling" : RSI down >2 pts vs 3 bars ago
                  "stable"  : change ≤2 pts (default)
    """
    if rsi is None:
        return "n/a", ""

    trend = (trend or "").lower()
    slope = (slope or "stable").lower()

    # ── Note matrix: (state, slope) → plain-English description ─────────────
    # Rules:
    #  • Reference the RSI zone relative to key levels (30, 50, 70)
    #  • Describe what the slope means in plain terms
    #  • Never repeat wording across adjacent cells
    #  • Keep each note ≤ 18 words

    if rsi >= 70:
        state = "overbought"
        note = {
            "rising":  "climbing into overbought zone — strong upside momentum, watch for exhaustion",
            "falling": "pulling back from overbought levels — upside momentum may be peaking",
            "stable":  "holding in overbought territory — bullish momentum extended, reversal risk rising",
        }.get(slope, "overbought zone — upside momentum stretched beyond normal levels")

    elif rsi >= 60:
        state = "bullish"
        note = {
            "rising":  "rising toward overbought — bullish momentum building with growing conviction",
            "falling": "retreating from strong levels — bullish momentum fading but still intact above 60",
            "stable":  "firmly above neutral — steady bullish momentum without being stretched",
        }.get(slope, "above neutral — demand conviction present, not yet extended")

    elif rsi >= 50:
        state = "mildly_bullish"
        note = {
            "rising":  "edging above neutral with improving momentum — buyers gaining early control",
            "falling": "above neutral but softening — bullish edge narrowing, watch for test of 50",
            "stable":  "hovering just above neutral — mild upside bias, momentum not yet confirmed",
        }.get(slope, "above the 50 midpoint — slight demand tilt, conviction not yet strong")

    elif rsi >= 45:
        state = "neutral"
        note = {
            "rising":  "approaching the neutral 50 midpoint — momentum recovering, buying interest returning",
            "falling": "dipping below neutral — momentum weakening, sellers gaining a slight edge",
            "stable":  "near the neutral 50 level — balanced momentum, no clear directional edge",
        }.get(slope, "near the neutral midpoint — neither side holds a momentum advantage")

    elif rsi >= 40:
        state = "mildly_bearish"
        note = {
            "rising":  "in weak territory but recovering — selling pressure may be easing gradually",
            "falling": "slipping toward oversold territory — selling pressure building, momentum weak",
            "stable":  "below neutral with mild selling momentum — downside bias, no extreme pressure",
        }.get(slope, "below the 50 midpoint — supply has a slight edge, demand struggling")

    elif rsi >= 30:
        state = "bearish"
        note = {
            "rising":  "approaching oversold but starting to recover — a bounce becoming possible",
            "falling": "entering oversold territory — selling pressure active and still intensifying",
            "stable":  "in weak territory — sustained selling pressure, no reversal signal yet",
        }.get(slope, "deep in supply territory — downside momentum dominant, approaching oversold")

    else:  # rsi < 30
        state = "oversold"
        note = {
            "rising":  "deeply oversold but recovering — downside momentum exhausting, bounce risk rising",
            "falling": "deeply oversold and still falling — extreme selling pressure active",
            "stable":  "deeply oversold — downside momentum may be exhausting, no reversal signal yet",
        }.get(slope, "deeply oversold zone — selling climax territory, high exhaustion risk")

    return state, note


# ── Unknown sentinel ──────────────────────────────────────────────────────────

def _unknown(interval: str) -> dict:
    return {
        "timeframe":     interval,
        "trend":         "unknown",
        "momentum":      "unknown",
        "volatility":    "unknown",
        "participation": "unknown",
        "rate_limited":  False,
    }


# ── Rate-limited sentinel ─────────────────────────────────────────────────────

def _rate_limited_context(interval: str, ticker: str) -> dict:
    """Context returned when Yahoo Finance is actively rate-limiting (HTTP 429).

    Sets rate_limited=True so the router can distinguish this from a genuinely
    unavailable ticker and show a proper "please wait" message instead of
    "ticker not found" or "delisted".
    """
    return {
        "timeframe":     interval,
        "trend":         "unknown",
        "momentum":      "unknown",
        "volatility":    "unknown",
        "participation": "unknown",
        "rate_limited":  True,
        "data_note":     (
            "Live market data is temporarily unavailable due to Yahoo Finance "
            "rate limiting (HTTP 429). Please wait a few minutes and try again."
        ),
    }


# ── Minimal context (fast_info path) ─────────────────────────────────────────

def _minimal_context(interval: str, ticker: str, fetch_result: dict) -> dict:
    """Build a minimal context from fast_info data (no OHLCV available).

    Infers trend direction from the day % change: > +0.15% → up,
    < -0.15% → down, else sideways.  All TA indicators that require price
    history are set to "unclear" so the AI layer knows not to cite them.
    """
    last_price = fetch_result.get("last_price")
    pct        = fetch_result.get("pct_change")  # may be None

    if pct is not None:
        if pct > 0.15:
            trend    = "up"
            momentum = "strengthening" if abs(pct) > 1.0 else "neutral"
        elif pct < -0.15:
            trend    = "down"
            momentum = "weakening" if abs(pct) > 1.0 else "neutral"
        else:
            trend    = "sideways"
            momentum = "neutral"
    else:
        trend    = "unclear"
        momentum = "unclear"

    rsi_state, rsi_note = classify_rsi(None, trend)
    return {
        "timeframe":     interval,
        "trend":         trend,
        "momentum":      momentum,
        "volatility":    "unclear",
        "participation": "unclear",
        "rsi":           None,
        "rsi_state":     rsi_state,
        "rsi_note":      rsi_note,
        "last_price":    last_price,
        "last_date":     datetime.now().strftime("%Y-%m-%d"),
        "data_source":   "fast_info",
        "data_note":     "OHLCV unavailable — context derived from live price only",
        "rate_limited":  False,
    }


# ── Full context (OHLCV path) ─────────────────────────────────────────────────

def derive_market_context(interval: str = "1d", ticker: str = "^NSEI") -> dict:
    """Derive market context from OHLCV data for the given ticker and interval.

    Delegates data fetching to safe_fetch() which applies:
      • 60-second in-memory cache
      • 5-minute global 429 cooldown gate (skips Yahoo entirely during cooldown)
      • primary download → retry (1 s gap) → Ticker.history() fallback
      • weekly-resample fallback (1wk only: daily data → resample to W bars)
      • fast_info minimal fallback when all OHLCV paths are exhausted

    All DataFrame access is guarded so no TypeError / IndexError can escape.
    """
    result = safe_fetch(ticker, interval)
    status = result.get("status", "failed")
    df     = result.get("data")

    # ── Rate-limited: Yahoo returned 429 ─────────────────────────────────────
    # Return dedicated sentinel — do NOT say "delisted" or "unavailable ticker".
    if result.get("rate_limited") or status == "rate_limited":
        print(f"[TechnoBot/context] {ticker}/{interval}: rate-limited — returning rate_limited context")
        return _rate_limited_context(interval, ticker)

    # ── Minimal fallback: fast_info only, no OHLCV ───────────────────────────
    if df is None or (hasattr(df, "empty") and df.empty):
        if status == "minimal" and result.get("last_price") is not None:
            print(f"[TechnoBot/context] {ticker}/{interval}: using minimal fast_info context")
            return _minimal_context(interval, ticker, result)
        return _unknown(interval)

    # ── Require Close, High, Low ─────────────────────────────────────────────
    if not {"Close", "High", "Low"}.issubset(df.columns):
        print(f"[TechnoBot/context] {ticker}/{interval}: missing OHLC columns — "
              f"columns={list(df.columns)}")
        return _unknown(interval)

    # ── Extract Series, guarding against duplicate-column DataFrames ─────────
    def _to_series(col: str) -> pd.Series:
        raw = df[col].squeeze()
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]
        return raw.astype(float)

    close = _to_series("Close")
    high  = _to_series("High")
    low   = _to_series("Low")

    # ── Minimum bar requirement ───────────────────────────────────────────────
    # 20 bars needed for EMA-20 and ATR-14.  For shorter series (e.g. resampled
    # weekly from only 5 weeks of daily data) degrade gracefully to "unclear".
    if len(close) < 5:
        print(f"[TechnoBot/context] {ticker}/{interval}: too few rows ({len(close)}) — unclear context")
        return {
            "timeframe":     interval,
            "trend":         "unclear",
            "momentum":      "unclear",
            "volatility":    "unclear",
            "participation": "unclear",
            "rate_limited":  False,
        }

    # ── TREND ─────────────────────────────────────────────────────────────────
    recent_close = float(close.iloc[-1])

    if len(close) >= 20:
        ema20        = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        earlier_mean = float(close.iloc[:-5].mean())
        trend_ema    = "up" if recent_close > ema20        else "down"
        trend_mean   = "up" if recent_close > earlier_mean else "down"
        trend = (
            "up"       if trend_ema == "up"   and trend_mean == "up"   else
            "down"     if trend_ema == "down" and trend_mean == "down" else
            "sideways"
        )
    else:
        # Shorter series: use simple price direction from 5 bars ago
        lookback = min(5, len(close) - 1)
        pct5 = (close.iloc[-1] - close.iloc[-1 - lookback]) / close.iloc[-1 - lookback] * 100
        trend = "up" if pct5 > 0.5 else ("down" if pct5 < -0.5 else "sideways")

    # ── MOMENTUM ──────────────────────────────────────────────────────────────
    rsi: float | None = None
    rsi_slope: str = "stable"
    momentum = "neutral"

    if len(close) >= 6:
        roc5 = float((close.iloc[-1] / close.iloc[-6] - 1) * 100)
        momentum = "strengthening" if roc5 > 0.5 else ("weakening" if roc5 < -0.5 else "neutral")

    if len(close) >= 14:
        delta         = close.diff()
        gain          = delta.clip(lower=0)
        loss          = (-delta).clip(lower=0)
        avg_gain_s    = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss_s    = loss.ewm(alpha=1 / 14, adjust=False).mean()

        # Full RSI series — avoids division by zero via masked replace
        _rs = avg_gain_s / avg_loss_s.replace(0, float("nan"))
        rsi_series = (100 - (100 / (1 + _rs))).clip(0, 100).fillna(100)

        rsi = round(float(rsi_series.iloc[-1]), 1)

        # RSI slope: compare last value to 3 bars ago (>2 pt change = directional)
        _lookback = min(3, len(rsi_series) - 1)
        if _lookback > 0:
            _rsi_prev = float(rsi_series.iloc[-1 - _lookback])
            _rsi_diff = rsi - _rsi_prev
            rsi_slope = (
                "rising"  if _rsi_diff >  2.0 else
                "falling" if _rsi_diff < -2.0 else
                "stable"
            )
        else:
            rsi_slope = "stable"

    # ── VOLATILITY ────────────────────────────────────────────────────────────
    volatility = "unclear"
    if len(close) >= 10:
        tr          = (high - low).clip(lower=0)
        atr_recent  = float(tr.iloc[-5:].mean())
        atr_earlier = (float(tr.iloc[-20:-5].mean()) if len(tr) >= 20
                       else float(tr.iloc[:-5].mean()) if len(tr) > 5
                       else atr_recent)
        volatility = (
            "expanding"   if atr_recent > atr_earlier * 1.05 else
            "contracting" if atr_recent < atr_earlier * 0.95 else
            "stable"
        )

    # ── PARTICIPATION (volume) ────────────────────────────────────────────────
    participation = "unknown"
    if "Volume" in df.columns:
        vol_raw = df["Volume"].squeeze()
        if isinstance(vol_raw, pd.DataFrame):
            vol_raw = vol_raw.iloc[:, 0]
        vol = vol_raw.astype(float)
        if len(vol) >= 5:
            recent_vol  = float(vol.iloc[-5:].mean())
            earlier_vol = float(vol.iloc[-20:-5].mean()) if len(vol) >= 20 else recent_vol
            if recent_vol > earlier_vol * 1.1:
                participation = "increasing"
            elif recent_vol < earlier_vol * 0.9:
                participation = "decreasing"
            else:
                participation = "steady"

    # ── LAST PRICE + DATE ─────────────────────────────────────────────────────
    last_price = round(recent_close, 2)
    try:
        last_date_raw = df.index[-1]
        last_date = (
            last_date_raw.strftime("%Y-%m-%d")
            if hasattr(last_date_raw, "strftime")
            else str(last_date_raw)[:10]
        )
    except Exception:
        last_date = datetime.now().strftime("%Y-%m-%d")

    # Fix 3: Safety net — if trend is still "unknown" despite having sufficient
    # data, resolve it to "sideways" so the router's data gate never blocks
    # a valid data result.  In practice the computation above always resolves
    # trend before reaching here; this guard handles any unexpected edge case.
    if trend == "unknown" and len(close) >= 5:
        trend = "sideways"

    rsi_state, rsi_note = classify_rsi(rsi, trend, rsi_slope)

    # Fix 4: RSI fallback — when RSI cannot be computed (fewer than 14 bars),
    # classify_rsi() returns "n/a" which surfaces as a dead label in prompts.
    # Use "neutral" instead so the LLM receives a meaningful structural signal.
    if rsi is None:
        rsi_state = "neutral"
        rsi_note  = "RSI unavailable — insufficient history for computation"

    # Fix 5: Confirm acceptance before returning so logs clearly show the
    # context was accepted, not silently rejected by an upstream guard.
    print(f"[TechnoBot/context] ACCEPTED rows={len(close)} trend={trend} rsi={rsi}")

    ctx = {
        "timeframe":     interval,
        "trend":         trend,
        "momentum":      momentum,
        "volatility":    volatility,
        "participation": participation,
        "last_price":    last_price,
        "last_date":     last_date,
        "data_source":   result.get("source", "unknown"),
        "rate_limited":  False,
        "rsi_state":     rsi_state,
        "rsi_note":      rsi_note,
        "rsi_slope":     rsi_slope,
    }
    if rsi is not None:
        ctx["rsi"] = rsi
    if status == "resampled":
        ctx["data_note"] = "Weekly bars derived by resampling daily data"

    return ctx


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for sym, iv in [("^NSEI", "1d"), ("^CNXIT", "1wk"), ("RELIANCE.NS", "1d")]:
        ctx = derive_market_context(iv, sym)
        print(f"\n{sym}/{iv}:")
        for k, v in ctx.items():
            print(f"  {k}: {v}")
