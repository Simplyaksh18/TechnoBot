from __future__ import annotations

import json
import time
import requests
import re
from datetime import datetime
import os
import pandas as pd

from technobot_data.historical_fetcher import (
    fetch_historical_data,
    snapshot_head_tail,
    export_csv,
)

from technobot_data.file_parser import parse_price_file, infer_timeframe
from technobot_data.nifty_context import derive_market_context

# Prompt templates — plain functions, no external framework
from technobot_prompts import ta_mentor_prompt, indicator_prompt, comparison_prompt

# local helpers in this file (do not import from utils to avoid conflicts)
# from utils import (
#     call_ollama,
#     enforce_bullet_only,
#     generate_followup_from_last_bullet,
#     detect_timeframe,
#     detect_instrument,
#     split_confirmation_and_query,
# )

# ── Groq Cloud LLM client ─────────────────────────────────────────────────────
# Replaces local Ollama inference.  All LLM calls route through call_groq().
from llm.groq_client import call_groq, GROQ_MODEL

# ── Per-request LLM time budget ────────────────────────────────────────────────
# Safety-net timeout (seconds) passed to call_groq().  Groq typically responds
# in 1–3 s; this cap prevents runaway waits on network failures.
_REQUEST_BUDGET_S: float = 30.0


# ── Fallback quality guards (complexity-aware) ────────────────────────────────
# Applied to all models listed in _QUALITY_GUARD_MODELS (currently gemma:2b
# and qwen2.5:1.5b).  Enforces TA depth and bullet-only output format.
#
# Two tiers:
#   _FALLBACK_BASE_GUARD    — standard depth/tone enforcement (simple queries)
#   _FALLBACK_COMPLEX_GUARD — adds causal-reasoning mandate (multi-indicator /
#                             multi-timeframe / comparison queries)

_FALLBACK_BASE_GUARD = (
    "\n\n---\n"
    "QUALITY ENFORCEMENT — read before writing:\n"
    "Think carefully before each bullet. Ask yourself:\n"
    "  — Is this observation anchored to the actual values in the data above?\n"
    "  — Does this bullet explain WHY the reading matters, not just WHAT it is?\n"
    "  — Does this bullet open differently from the previous one?\n"
    "Vary sentence style: mix observations, cause-effect links, inferences, conditions.\n"
    "Analytical precision + human tone — never robotic, never template-like.\n"
    "---"
)

_FALLBACK_COMPLEX_GUARD = (
    "\n\n---\n"
    "COMPLEX QUERY QUALITY ENFORCEMENT — read before writing:\n"
    "Think carefully before each bullet. Ask yourself:\n"
    "  — Is this observation anchored to the actual values in the data above?\n"
    "  — Am I explaining the RELATIONSHIP between indicators, not each in isolation?\n"
    "  — Does this bullet open differently from the previous one?\n"
    "For multi-indicator / multi-timeframe queries:\n"
    "  — Explain what the CONVERGENCE of signals means, not each signal separately\n"
    "  — Distinguish correlation from confirmation\n"
    "Vary sentence style: mix observations, cause-effect links, inferences, conditions.\n"
    "Analytical precision + human tone — never robotic, never template-like.\n"
    "---"
)

# Legacy alias kept so api_server.py imports still resolve during transition.
_FALLBACK_QUALITY_GUARD = _FALLBACK_BASE_GUARD

# ── Follow-up non-repetition guard ─────────────────────────────────────────────
# Injected into the system prompt ONLY when conversation history is non-empty.
# Forces the LLM to add new analytical dimensions instead of restating prior analysis.
_NO_REPEAT_GUARD = (
    "\n\nFOLLOW-UP INSTRUCTION: This is a follow-up query. "
    "Follow the same concise format as the initial query: 3 to 5 bullets, "
    "one short sentence per bullet, and no multi-line elaboration. "
    "DO NOT repeat or paraphrase analysis already delivered in previous turns. "
    "Each bullet must add one new angle only:\n"
    "  - Validation conditions (what would CONFIRM the current structure)\n"
    "  - Invalidation conditions (what structural break would DENY the thesis)\n"
    "  - Deeper timeframe context (what the higher/lower TF reading adds)\n"
    "  - Participation or volume angle not yet mentioned\n"
    "Keep wording tight and crisp; do not restate trend direction, RSI level, "
    "or volatility phase already covered."
)

# ── Style variation + differentiation mandate ─────────────────────────────────
# Injected into EVERY system prompt via _build_groq_messages().
# Eliminates template-like repetition by mandating varied sentence structures,
# ticker-specific anchoring, anti-template rules, count control, and human tone.
_VARIATION_GUARD = (
    "\n\nRESPONSE STYLE RULES:\n"
    "Use bullet points BUT vary sentence structure across bullets.\n"
    "Mix these styles — do NOT use the same style for every bullet:\n"
    "  • Observation:  describe what you see in the price action directly\n"
    "  • Cause-effect: link an indicator reading to its structural consequence\n"
    "  • Inference:    draw a conclusion from the combined signals\n"
    "  • Condition:    state the structural pivot ('Unless X happens, Y remains likely')\n\n"

    "GOOD examples (varied openings, different styles):\n"
    "  — 'Price is consistently forming lower highs, showing controlled selling pressure'\n"
    "  — 'With RSI hovering near 45, momentum has not built enough to challenge the trend'\n"
    "  — 'This kind of structure typically reflects institutional distribution, not panic'\n"
    "  — 'Unless RSI reclaims 50, upside attempts are likely to fade at resistance'\n\n"

    "BAD examples (rigid template — NEVER produce):\n"
    "  — 'Structure is a downtrend...' (same opener every time)\n"
    "  — 'RSI indicates weakness...' (generic label without explanation)\n"
    "  — 'Momentum shows...' (vague, not anchored to a specific value)\n\n"

    "ANTI-TEMPLATE RULE:\n"
    "Every response must feel freshly written for THIS specific ticker. Never reuse:\n"
    "  — the same opening phrase across bullets\n"
    "  — the same reasoning chain used for a different stock\n"
    "  — the same sentence pattern repeated within one response\n"
    "If a bullet sounds like it could describe any stock, rewrite it.\n\n"

    "DIFFERENTIATION MANDATE:\n"
    "When stocks share the same trend direction, differentiate by:\n"
    "  — The EXACT RSI value and what that specific number implies structurally\n"
    "  — The specific volatility phase (contracting / expanding / stable)\n"
    "  — The momentum conviction depth (pace and decisiveness of the move)\n"
    "Each bullet must be fingerprinted to THIS ticker's own numbers.\n\n"

    "BULLET COUNT: 3 to 5 bullets ONLY.\n"
    "Each bullet must carry new information — do not repeat the same idea.\n\n"

    "TONE: Write like a human analyst thinking aloud — precise and conversational.\n"
    "Not like a report generator. No robotic phrases. No textbook definitions.\n\n"

    "HARD BULLET RULES:\n"
    "- Each bullet must be at most 2 lines — no long chained sentences\n"
    "- No filler words, padding phrases, or textbook definitions\n"
    "- Each bullet must add NEW information — never restate an idea already expressed\n"
    "- One clear structural observation per bullet, anchored to actual data\n"
    "- BAD: 'Infosys is showing a downtrend which indicates weakness in the structure of the stock...'\n"
    "- GOOD: 'Infosys's price continues to form lower highs — selling pressure still dominant.'"
)

# ── Query intent classifier ────────────────────────────────────────────────────
# Detects conversational / decision-oriented queries that need a short, direct
# answer rather than a full multi-bullet TA breakdown.
_ADVISORY_PATTERNS = (
    "should i", "is it good", "is it bad", "worth watching",
    "why is", "what does it mean", "is it strong", "is it weak",
    "is this good", "is this bad", "good setup", "bad setup",
    "worth monitoring", "worth buying", "worth holding",
    "can i buy", "should i hold", "can i avoid",
    "good entry", "good time", "right time",
    "any concerns", "look weak", "look strong",
    "how does", "how do", "what do you think",
    "looks like", "any red flags", "look today",
    "is it risky", "is it safe", "is this a good",
    "is this strong", "is this weak", "is it worth",
    "should i watch", "is it worth watching",
)


def _detect_query_intent(query: str) -> str:
    """Return 'advisory' for conversational/decision queries, 'analysis' otherwise.

    Advisory queries need a SHORT direct answer (2–3 bullets) that addresses
    the user's specific question first.  Analysis queries get the full
    structured multi-bullet TA breakdown.
    """
    q = query.lower().strip()
    return "advisory" if any(pat in q for pat in _ADVISORY_PATTERNS) else "analysis"


# Keywords used to classify query complexity
_INDICATOR_KW = frozenset({
    "rsi", "macd", "ema", "sma", "atr", "bollinger", "bb", "stoch",
    "obv", "vwap", "ichimoku", "adx", "cci", "mfi", "roc", "dmi",
})
_TIMEFRAME_KW = frozenset({
    "daily", "weekly", "monthly", "hourly", "intraday",
    "5m", "15m", "1d", "1w", "1mo",
})


def _is_complex_query(query: str) -> bool:
    """Return True when the query involves multi-indicator, multi-timeframe, or comparison."""
    q = query.lower()
    is_comparison  = any(kw in q for kw in ("vs", "versus", "compare", "compared to"))
    indicator_hits = sum(1 for kw in _INDICATOR_KW if kw in q)
    timeframe_hits = sum(1 for kw in _TIMEFRAME_KW if kw in q)
    return is_comparison or indicator_hits >= 2 or timeframe_hits >= 2


def _build_fallback_quality_guard(query: str = "") -> str:
    """Return the appropriate quality guard string for the fallback model.

    Complex queries get the richer guard that stresses causal / cross-indicator
    reasoning.  Simple queries get the base guard (depth + tone).
    Called with query="" falls back to the base guard.
    """
    return _FALLBACK_COMPLEX_GUARD if _is_complex_query(query) else _FALLBACK_BASE_GUARD


# ── Groq message builder ───────────────────────────────────────────────────────

def _build_groq_messages(
    sys_prompt: str,
    usr_msg:    str,
    history:    list | None = None,
    query:      str = "",
) -> list[dict]:
    """Assemble the messages list for call_groq().

    Applies the quality guard to the system prompt (same logic that was
    previously baked into call_ollama).  Also injects the non-repetition
    guard when conversation history is present.
    """
    effective_sys = sys_prompt + _build_fallback_quality_guard(query) + _VARIATION_GUARD
    if history:
        effective_sys += _NO_REPEAT_GUARD
    msgs = [{"role": "system", "content": effective_sys}]
    msgs.extend(history or [])
    msgs.append({"role": "user", "content": usr_msg})
    return msgs


# ── LLM error formatter ────────────────────────────────────────────────────────

def _format_llm_error(resp: dict) -> str:
    """Convert a call_groq() error dict into a user-facing ⚠️ string."""
    code = resp.get("error", "")
    if code == "rate_limit":
        return "⚠️ High demand — please retry in a few seconds"
    if code == "timeout":
        return "⚠️ Response took too long — please try again"
    if code == "no_api_key":
        return "⚠️ LLM API key not configured — contact the administrator"
    # api_error, connection_error, unexpected, unknown
    return resp.get("message") or "⚠️ LLM service unavailable — please try again"


# ── Structured response helpers ────────────────────────────────────────────────

def _determine_response_status(context: dict) -> str:
    """Return 'ok', 'partial', or 'unavailable' based on live context fields.

    Rules (in precedence order — data-driven, not flag-driven):
      1. Uploaded file / custom ticker             → 'ok'
      2. rate_limited or api_failed                → 'unavailable'
      3. last_price is None                        → 'unavailable'
      4. trend='unknown' or 'unclear'              → 'unavailable'
      5. _lightweight=True (residual, still set)   → 'partial' if known ticker
      6. Otherwise                                 → 'ok'

    Note: has_data gate in answer() blocks LLM calls before this function is
    reached for cases 2–4, so 'unavailable' here is primarily for structured
    /chat response tagging.  _lightweight (rule 5) is a secondary signal used
    only when Fix 2 could not clear it (e.g. comparison path, edge cases).
    """
    if not context:
        return "unavailable"
    ticker = context.get("ticker", "")
    # Uploaded-file sessions always report ok (data came from the user's file)
    if context.get("data_source") == "uploaded_file" or ticker == "custom":
        return "ok"
    if context.get("rate_limited") or context.get("api_failed"):
        return "unavailable"
    if context.get("last_price") is None:
        return "unavailable"
    trend = (context.get("trend") or "").lower()
    if trend in ("unknown", "unclear", ""):
        return "unavailable"
    # Secondary: _lightweight still set (comparison path / edge cases)
    if context.get("_lightweight"):
        _known_tickers: set[str] = set(INSTRUMENT_MAP.values())
        return "partial" if ticker in _known_tickers else "unavailable"
    return "ok"


def _build_evidence_dict(context: dict) -> dict:
    """Build a dynamic evidence dict from the actual live context.

    Never hardcoded — every field comes from the fetched data.
    Period labels (e.g. "Last 6 Months") are preserved when the user
    asked for a specific lookback window over daily bars.
    """
    ticker    = context.get("ticker") or "N/A"
    tf_raw    = context.get("timeframe") or "1d"

    # Prefer a context-provided label when available so evidence mirrors the
    # actual request/window instead of a fixed formatter string.
    tf_label = (
        context.get("data_window_label")
        or context.get("window_label")
        or context.get("timeframe_label")
        or (
            f"{context.get('_period_label')} (daily bars)"
            if context.get("_period_label")
            else tf_raw
        )
    )
    data_src  = context.get("data_source") or ""
    if data_src == "uploaded_file":
        source = "Uploaded CSV"
    elif context.get("_lightweight"):
        source = "Qualitative fallback (live feed unavailable)"
    elif "twelvedata" in data_src.lower():
        source = "TwelveData"
    else:
        source = "Alpha Vantage"
    link = (
        context.get("reference")
        or context.get("link")
        or (
            f"https://finance.yahoo.com/quote/{ticker}"
            if ticker not in ("N/A", "custom", "")
            else ""
        )
    )
    return {
        "source":     source,
        "ticker":     ticker,
        "instrument": context.get("instrument_name") or ticker,
        "timeframe":  tf_label,
        "link":       link,
    }


def _build_insight_dict(context: dict) -> dict | None:
    """Build a structured insight dict by parsing the output of generate_insight_layer().

    Reuses the existing 11-case insight engine — no duplicate logic.
    Returns None when the context is too sparse to produce a meaningful insight.
    """
    # generate_insight_layer is defined later in this module.
    # We call it lazily here to avoid forward-reference issues at module load.
    raw = generate_insight_layer(context)  # noqa: F821  (defined below)
    if not raw:
        return None
    return _parse_insight_text(raw)


def _parse_insight_text(raw: str) -> dict | None:
    """Parse a '📌 Contextual Insight:' block into {regime, pattern, implication}.

    Input format (from generate_insight_layer / generate_comparison_insight):
        📌 Contextual Insight:
        • Current regime: <text>
        • Pattern: <text>
        • Implication: <text>
    """
    result: dict = {"regime": "", "pattern": "", "implication": ""}
    for line in (raw or "").splitlines():
        s = line.strip()
        if s.startswith("• Current regime:"):
            result["regime"] = s[len("• Current regime:"):].strip()
        elif s.startswith("• Pattern:"):
            result["pattern"] = s[len("• Pattern:"):].strip()
        elif s.startswith("• Implication:"):
            result["implication"] = s[len("• Implication:"):].strip()
        # Comparison-insight format: "🔍 Relative Insight: ..."
        elif s.startswith("🔍 Relative Insight:"):
            # Flatten into pattern field so the comparison insight is preserved
            result["regime"]      = "comparative"
            result["pattern"]     = s[len("🔍 Relative Insight:"):].strip()
            result["implication"] = ""
            break
    if not any(result.values()):
        return None
    return result


# --- A3: Per-session state (replaces module-level globals) ---
# Keyed by chat_id so concurrent sessions do not corrupt each other.
SESSION_STORE: dict = {}

def get_session(chat_id: str) -> dict:
    """Return (and lazily create) the session dict for a given chat_id."""
    if chat_id not in SESSION_STORE:
        SESSION_STORE[chat_id] = {
            "last_followup_intent": None,
            "last_context": None,
            "last_historical": None,
            "last_uploaded_df": None,
            # D2: sliding window of last 3 user+assistant turn pairs
            "conversation_history": [],
            # D8: list of {filename, ticker, timeframe, timestamp} for export log
            "export_history": [],
            # Follow-up deduplication — track what was already delivered
            "last_response_ticker": None,   # ticker from previous response
            "last_response_tf": None,       # timeframe from previous response
            "last_insight_hash": None,      # hash of last insight text to avoid repeats
        }
    return SESSION_STORE[chat_id]


# D2: Max turns kept in the sliding window (each turn = 1 user + 1 assistant msg)
CONVERSATION_WINDOW = 3

# ── Instrument resolution cache (query → (display_name, ticker)) ──────────────
# Avoids re-scanning INSTRUMENT_MAP for identical repeated lookups.
# Keyed on lowercased, stripped query text; does NOT cache fallback-context paths.
_INSTRUMENT_RESOLUTION_CACHE: dict[str, tuple[str, str]] = {}


# --- A4: Column normalisation utility ---
def normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with all column names lowercased and stripped.

    Uploaded CSVs may arrive with 'Close', 'CLOSE', or 'close'. Calling
    this once at ingestion means every downstream function can use lowercase
    keys without dual-case checks.
    """
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    return out


def detect_timeframe(user_query: str):
    q = user_query.lower()

    if "minute" in q or "intraday" in q:
        return "5m"
    if "hour" in q:
        return "1h"
    if "week" in q or "weekly" in q:
        return "1wk"

    # Period expressions — "last 6 months", "past 3 months", "over the last year"
    # mean a lookback window over daily bars, NOT monthly-resolution bars.
    # The daily interval already uses period="6mo" in SUPPORTED_INTERVALS, which
    # covers the most common period request without changing the bar size.
    _PERIOD_RE = re.compile(
        r"\b(?:last|past|previous|over\s+(?:the\s+)?last?)\s+\d+\s*months?\b",
        re.IGNORECASE,
    )
    if _PERIOD_RE.search(q):
        return "1d"  # daily bars; yfinance fetch uses period="6mo" by default

    # "month" / "monthly" / "long term" / "year" → monthly bar chart
    if "month" in q or "long term" in q or "year" in q:
        return "1mo"

    # DEFAULT → DAILY
    return "1d"

def format_market_context(context: dict) -> str:
    # ── Lightweight fallback path ─────────────────────────────────────────────
    # When all live data providers have failed, build_lightweight_context() sets
    # _lightweight=True and populates _qualitative_note with instrument-specific
    # TA reasoning.  Inject a disclaimer + that note so the LLM can still give
    # a useful, grounded response instead of a generic "data unavailable" reply.
    if context.get("_lightweight"):
        instrument = context.get("instrument_name") or context.get("ticker", "this instrument")
        timeframe  = context.get("timeframe", "selected timeframe")
        note       = context.get("_qualitative_note", "")
        lines = [
            "⚠️ LIVE DATA UNAVAILABLE — QUALITATIVE ANALYSIS MODE",
            f"Instrument : {instrument}",
            f"Timeframe  : {timeframe}",
            "",
            "Live market data could not be retrieved from any provider.",
            "All numeric indicators (price, RSI, volume) are unavailable.",
            "Respond using general technical reasoning only — no fabricated numbers.",
            "",
            "Instrument-specific context:",
            note,
        ]
        return "\n".join(lines)

    # ── Normal live-data path ─────────────────────────────────────────────────
    # participation_label is injected by _validate_context_consistency()
    participation_display = (
        context.get("participation_label")
        or context.get("participation", "unknown")
    )
    lines = [
        "Market context:",
        f"- Instrument: {context.get('instrument_name', 'unknown')}",
        f"- Timeframe: {context.get('timeframe', 'unknown')}",
        f"- Trend: {context.get('trend', 'unknown')}",
        f"- Momentum: {context.get('momentum', 'unknown')}",
        f"- Volatility: {context.get('volatility', 'unknown')}",
        f"- Participation: {participation_display}",
    ]

    rsi       = context.get("rsi")
    rsi_state = context.get("rsi_state", "")
    rsi_note  = context.get("rsi_note",  "")

    if rsi is not None:
        # Primary RSI line — state label replaces raw direction string
        # so the LLM receives a pre-classified signal, not a raw value to interpret.
        if rsi_state and rsi_state != "n/a":
            state_display = rsi_state.replace("_", " ")
            rsi_line = f"- RSI(14): {rsi} [{state_display}]"
            if rsi_note:
                rsi_line += f" — {rsi_note}"
        else:
            direction = "above 50 (demand-led)" if float(rsi) >= 50 else "below 50 (supply-led)"
            rsi_line = f"- RSI(14): {rsi} — {direction}"
        lines.append(rsi_line)

        # Divergence note — alerts LLM when RSI and trend contradict each other
        if context.get("_rsi_divergence"):
            lines.append(
                "- ⚠ RSI / trend mixed signal: RSI reads above/below the expected "
                "zone for this trend direction — describe the divergence structurally; "
                "do NOT force a single-sided narrative."
            )

    last_price = context.get("last_price")
    if last_price:
        lines.append(f"- Last close: {last_price}")
    return "\n".join(lines)


INDICATOR_KEYWORDS = {
    "adx", "rsi", "macd", "bollinger", "bb", "stoch", "stochastic", "atr",
    "obv", "vwap", "ema", "sma", "ichimoku", "supertrend", "cci", "mfi",
    "volume", "pivot", "fibonacci", "roc", "dmi", "di+", "di-", "psar", "sar",
}

# TA concepts that qualify a query as domain-relevant
TA_CONCEPT_KEYWORDS = {
    "trend", "momentum", "volatility", "support", "resistance", "structure",
    "pattern", "consolidation", "range", "divergence", "crossover", "overbought",
    "oversold", "reversal", "rally", "correction", "analysis", "chart", "indicator",
    "signal", "level", "zone", "weekly", "daily", "monthly", "timeframe", "swing",
    "higher high", "lower low", "double top", "double bottom", "head and shoulders",
    "triangle", "flag", "cup", "handle", "breakout", "breakdown", "squeeze",
    "expansion", "contraction", "neckline", "retest", "accumulation", "distribution",
    "candle", "candlestick", "doji", "hammer", "engulfing", "trending", "sideways",
    "uptrend", "downtrend", "market", "stock", "index", "sector",
}

# Minimum data-fetch cache: (ticker, timeframe) → (fetched_at_ts, context_dict)
# TTL = 5 minutes to keep live data fresh without hammering yfinance
_CONTEXT_CACHE: dict[str, tuple[float, dict]] = {}
_CONTEXT_CACHE_TTL = 300  # seconds


def _derive_lightweight_context(ticker: str, timeframe: str) -> dict:
    """Minimal context when full market data fetch fails.

    Tries to get last price + basic trend from yfinance fast_info.
    Falls back to a structured 'partial' context if even that fails.
    Always returns a dict with at least 'trend', 'timeframe', 'data_status' keys.
    """
    import yfinance as yf  # import here to avoid top-level dep ordering issues

    base: dict = {
        "timeframe":     timeframe,
        "trend":         "unknown",
        "momentum":      "unknown",
        "volatility":    "unknown",
        "participation": "unknown",
        "data_status":   "partial",
    }
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info  # fast_info doesn't require a full download
        lp = getattr(fi, "last_price", None) or getattr(fi, "regularMarketPrice", None)
        pc = getattr(fi, "previous_close", None)
        if lp is not None:
            base["last_price"] = round(float(lp), 2)
        if lp is not None and pc is not None:
            pct = (float(lp) - float(pc)) / float(pc) * 100
            base["prev_close"] = round(float(pc), 2)
            if pct > 0.15:
                base["trend"] = "up"
                base["momentum"] = "strengthening" if abs(pct) > 1.0 else "neutral"
            elif pct < -0.15:
                base["trend"] = "down"
                base["momentum"] = "strengthening" if abs(pct) > 1.0 else "neutral"
            else:
                base["trend"] = "sideways"
                base["momentum"] = "neutral"
    except Exception:
        pass  # keep base context — evidence block will note partial data
    return base


def _cached_derive_market_context(timeframe: str, ticker: str) -> dict:
    """5-minute cache wrapper around derive_market_context.

    Design constraints (rate-limit hardening):
    • NO retry at this layer — safe_fetch already retries once internally.
      Adding a second retry here would double every Yahoo call count.
    • NO _derive_lightweight_context call — safe_fetch already has fast_info
      as a last resort; calling it again here adds an extra Yahoo call for
      every failure, which amplifies rate-limiting pressure.
    • Rate-limited results are NEVER cached so the next caller retries fresh
      once the 5-minute 429 cooldown expires.
    • Unknown/failed results are NOT cached to prevent poisoning the next
      5 minutes of queries for that ticker.
    """
    # Map unsupported intervals to daily — safe_fetch only knows 1d/1wk/1mo
    _TF_FALLBACK = {"5m": "1d", "1h": "1d"}
    effective_tf = _TF_FALLBACK.get(timeframe, timeframe)

    key = f"{ticker}:{effective_tf}"
    now = datetime.now().timestamp()
    if key in _CONTEXT_CACHE:
        fetched_at, ctx = _CONTEXT_CACHE[key]
        if now - fetched_at < _CONTEXT_CACHE_TTL:
            result = dict(ctx)
            result["timeframe"] = timeframe
            return result

    # ── Single fetch — no retry at this level ────────────────────────────────
    ctx = derive_market_context(effective_tf, ticker)

    # Rate-limited: return immediately, do not cache, do not retry.
    if ctx.get("rate_limited"):
        print(f"[TechnoBot] {ticker}/{effective_tf}: rate-limited — skip cache")
        result = dict(ctx)
        result["timeframe"] = timeframe
        return result

    # ── Lightweight context fallback ─────────────────────────────────────────
    # When ALL providers fail and trend is still "unknown", build a qualitative
    # context so the LLM can still produce a useful (if data-less) response.
    # Do NOT cache lightweight results — the next request should retry live data.
    if ctx.get("trend") == "unknown" and ctx.get("last_price") is None:
        try:
            from data.lightweight_context import build_lightweight_context
            lw_ctx = build_lightweight_context(ticker, effective_tf)
            lw_ctx["timeframe"] = timeframe
            print(f"[TechnoBot] {ticker}/{effective_tf}: all providers failed "
                  f"— using lightweight qualitative context (not cached)")
            return lw_ctx
        except Exception as _lw_exc:
            print(f"[TechnoBot] lightweight_context error: {_lw_exc}")
            # Fall through — return the unknown context as-is

    # ── Cache only usable results ─────────────────────────────────────────────
    # "unknown" and rate_limited must not be cached — they'd block all retries
    # for the next 5 minutes.
    if ctx.get("trend") != "unknown" or ctx.get("last_price") is not None:
        _CONTEXT_CACHE[key] = (now, ctx)
        print(f"[TechnoBot] {ticker}/{effective_tf}: cached context "
              f"trend={ctx.get('trend')} rsi={ctx.get('rsi')}")
    else:
        print(f"[TechnoBot] {ticker}/{effective_tf}: context unknown — not caching")

    result = dict(ctx)
    result["timeframe"] = timeframe
    return result


def is_ta_query(text: str) -> bool:
    """Return True if the query is about technical analysis or a known instrument.

    Queries that match none of the below are redirected with a scope message
    instead of running a full NIFTY analysis on unrelated topics.
    """
    q = text.lower()
    # Instrument name match
    for name in INSTRUMENT_MAP:
        if name in q:
            return True
    # Ambiguous instrument names
    for name in AMBIGUOUS_INSTRUMENT_MAP:
        if re.search(rf"\b{re.escape(name)}\b", q):
            return True
    # Indicator keywords
    if any(k in q for k in INDICATOR_KEYWORDS):
        return True
    # TA concept keywords
    if any(k in q for k in TA_CONCEPT_KEYWORDS):
        return True
    # Explicit ticker-like tokens (e.g. RELIANCE.NS, ^NSEI)
    if re.search(r"\b[A-Z]{2,12}(?:\.[A-Z]{1,4})?\b", text):
        return True
    return False

INSTRUMENT_MAP = {
    # Indices
    "nifty 50":      "^NSEI",
    "nifty50":       "^NSEI",
    "nifty":         "^NSEI",
    "sensex":        "^BSESN",
    "bse sensex":    "^BSESN",
    "nifty bank":    "^NSEBANK",
    "bank nifty":    "^NSEBANK",
    "nifty it":      "^CNXIT",
    "nifty midcap":  "^NSEMDCP50",
    # Banking
    "hdfc bank":     "HDFCBANK.NS",
    "icici bank":    "ICICIBANK.NS",
    "sbi":           "SBIN.NS",
    "state bank":    "SBIN.NS",
    "axis bank":     "AXISBANK.NS",
    "kotak":         "KOTAKBANK.NS",
    "kotak bank":    "KOTAKBANK.NS",
    "kotak mahindra":"KOTAKBANK.NS",
    "indusind":      "INDUSINDBK.NS",
    "yes bank":      "YESBANK.NS",
    "pnb":           "PNB.NS",
    "bank of baroda":"BANKBARODA.NS",
    # IT
    "tcs":           "TCS.NS",
    "infosys":       "INFY.NS",
    "wipro":         "WIPRO.NS",
    "hcl tech":      "HCLTECH.NS",
    "hcltech":       "HCLTECH.NS",
    "hcl":           "HCLTECH.NS",
    "tech mahindra": "TECHM.NS",
    "mphasis":       "MPHASIS.NS",
    "ltimindtree":   "LTIM.NS",
    # Conglomerates / Energy
    "reliance":      "RELIANCE.NS",
    "ongc":          "ONGC.NS",
    "coal india":    "COALINDIA.NS",
    "ntpc":          "NTPC.NS",
    "power grid":    "POWERGRID.NS",
    "bhel":          "BHEL.NS",
    "adani ports":   "ADANIPORTS.NS",
    "adani enterprises": "ADANIENT.NS",
    "adani power":   "ADANIPOWER.NS",
    "adani green":   "ADANIGREEN.NS",
    # Tata Group
    "tata steel":    "TATASTEEL.NS",
    "tata motors":   "TATAMOTORS.NS",  # TODO: Verify this ticker - may be delisted
    "tata power":    "TATAPOWER.NS",
    "tata consumer": "TATACONSUM.NS",
    # Financials
    "bajaj finance": "BAJFINANCE.NS",
    "bajaj finserv": "BAJAJFINSV.NS",
    "hdfc life":     "HDFCLIFE.NS",
    "sbi life":      "SBILIFE.NS",
    "icici pru":     "ICICIPRULI.NS",
    # Auto
    "bajaj auto":    "BAJAJ-AUTO.NS",
    "maruti":        "MARUTI.NS",
    "maruti suzuki": "MARUTI.NS",
    "hero motocorp": "HEROMOTOCO.NS",
    "eicher":        "EICHERMOT.NS",
    "mahindra":      "M&M.NS",
    "m&m":           "M&M.NS",
    # Consumer / FMCG
    "itc":           "ITC.NS",
    "hindustan unilever": "HINDUNILVR.NS",
    "hul":           "HINDUNILVR.NS",
    "nestle":        "NESTLEIND.NS",
    "britannia":     "BRITANNIA.NS",
    "dabur":         "DABUR.NS",
    "godrej":        "GODREJCP.NS",
    # Pharma
    "sun pharma":    "SUNPHARMA.NS",
    "dr reddy":      "DRREDDY.NS",
    "cipla":         "CIPLA.NS",
    "divis":         "DIVISLAB.NS",
    "apollo hospital":"APOLLOHOSP.NS",
    # Infra / Metals
    "l&t":           "LT.NS",
    "larsen":        "LT.NS",
    "ultratech":     "ULTRACEMCO.NS",
    "shree cement":  "SHREECEM.NS",
    "hindalco":      "HINDALCO.NS",
    "jsw steel":     "JSWSTEEL.NS",
    "jswsteel":      "JSWSTEEL.NS",
    "steel authority": "SAIL.NS",
    "sail":          "SAIL.NS",
    # Consumer / Retail
    "asian paints":  "ASIANPAINT.NS",
    "asian paint":   "ASIANPAINT.NS",
    "titan":         "TITAN.NS",
    "dmart":         "DMART.NS",
    "zomato":        "ZOMATO.NS",
    "nykaa":         "NYKAA.NS",
    "paytm":         "PAYTM.NS",
    # Task 1: common abbreviations / short-forms users type
    "ril":           "RELIANCE.NS",   # Reliance Industries Ltd
    "rjio":          "RELIANCE.NS",   # Jio = Reliance
    "reliance industries": "RELIANCE.NS",
    "infy":          "INFY.NS",       # Infosys ticker as a name
    "hdfcbank":      "HDFCBANK.NS",   # explicit key prevents "hdfcb" substring mis-match
    "hdfcb":         "HDFCBANK.NS",
    "icicib":        "ICICIBANK.NS",
    "bse":           "^BSESN",        # plain "BSE" → Sensex
    "nse":           "^NSEI",         # plain "NSE" → Nifty 50
    "midcap 50":     "^NSEMDCP50",
    "tatapower":     "TATAPOWER.NS",
    "tatasteel":     "TATASTEEL.NS",
    "tatamotors":    "TATAMOTORS.NS",  # TODO: Verify this ticker - may be delisted
    "sunpharma":     "SUNPHARMA.NS",
    "bajajfinance":  "BAJFINANCE.NS",
    "bajajfinserv":  "BAJAJFINSV.NS",
    "coalindia":     "COALINDIA.NS",
    "powergrid":     "POWERGRID.NS",
    "hindunilvr":    "HINDUNILVR.NS",
    "nestleindia":   "NESTLEIND.NS",
    "drreddy":       "DRREDDY.NS",
    "dr reddys":     "DRREDDY.NS",
    "dr. reddy":     "DRREDDY.NS",
    "apollohosp":    "APOLLOHOSP.NS",
    "apollo":        "APOLLOHOSP.NS",
    "ltim":          "LTIM.NS",
    "lt":            "LT.NS",         # plain L&T short form
    "ultratech":     "ULTRACEMCO.NS",
    "ultracemco":    "ULTRACEMCO.NS",
    "hindalco":      "HINDALCO.NS",
    "sail":          "SAIL.NS",
    "vedanta":       "VEDL.NS",
    "jsw":           "JSWSTEEL.NS",   # JSW alone → JSW Steel
}

INDICATOR_TICKER_BLOCKLIST = {
    "RSI", "MACD", "VWAP", "EMA", "SMA", "ADX", "ATR", "OBV", "BB", "MA",
    "DMI", "CCI", "MFI", "SAR",
}

# D4: Short/ambiguous terms that could map to multiple instruments.
# When detected, TechnoBot asks the user to clarify instead of guessing.
AMBIGUOUS_INSTRUMENT_MAP: dict[str, list[str]] = {
    "tata": ["Tata Steel (TATASTEEL.NS)", "Tata Motors (TATAMOTORS.NS)", "TCS (TCS.NS)", "Tata Power (TATAPOWER.NS)"],
    "adani": ["Adani Enterprises (ADANIENT.NS)", "Adani Ports (ADANIPORTS.NS)", "Adani Power (ADANIPOWER.NS)", "Adani Green (ADANIGREEN.NS)"],
    "bajaj": ["Bajaj Auto (BAJAJ-AUTO.NS)", "Bajaj Finance (BAJFINANCE.NS)", "Bajaj Finserv (BAJAJFINSV.NS)"],
    "hdfc": ["HDFC Bank (HDFCBANK.NS)", "HDFC Life (HDFCLIFE.NS)", "HDFC AMC (HDFCAMC.NS)"],
    "axis": ["Axis Bank (AXISBANK.NS)"],  # single match, no ambiguity needed here but listed for extensibility
}


BOT_NAME = "🤖TechnoBot"

# =========================================================
# GREETING DETECTION (FUZZY & HUMAN)
# =========================================================

GREETING_REGEX = re.compile(
    r"^(hi+|hello+|hey+)(\s+there|\s+technobot|\s+bot)?[!?.]*\s*$",
    re.IGNORECASE
)

def is_greeting(text: str) -> bool:
    return bool(GREETING_REGEX.match(text.strip()))

def time_based_greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 18:
        return "Good afternoon"
    else:
        return "Good evening"

def greeting_response(user_name: str | None) -> str:
    name = f", {user_name}" if user_name else ""
    return (
        f"{time_based_greeting()}{name} 👋\n"
        f"I’m **{BOT_NAME}**, a technical-analysis assistant designed to help you "
        f"understand market behavior clearly and responsibly."
    )

def is_indicator_question(q: str) -> bool:
    s = (q or "").lower()
    return any(k in s for k in INDICATOR_KEYWORDS) or "indicator" in s


def _extract_model_followup(raw: str) -> str:
    """Return the model's own follow-up question from its raw output, if any.

    The LLM sometimes appends a question after the bullet list. We pick the last
    non-empty line that ends with '?' and is NOT a bullet marker line.
    """
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.endswith("?") and not ln.startswith(("•", "*", "-")):
            return ln
    return ""


def _shorten_followup_question(text: str, target_words: int = 14) -> str:
    """Keep a follow-up question short and crisp without changing its intent."""
    if not text:
        return ""

    cleaned = re.sub(r"^(Follow.?up[:\s]|Question[:\s]|Q[:\s])", "", text, flags=re.I).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return ""

    if len(cleaned.split()) > target_words:
        for separator in (" — ", " - ", ";", ","):
            if separator in cleaned:
                cleaned = cleaned.split(separator, 1)[0].strip()
                break

    words = cleaned.split()
    if len(words) > target_words:
        cleaned = " ".join(words[:target_words]).rstrip(".?!,;:")

    if not cleaned.endswith("?"):
        cleaned = cleaned.rstrip(".!") + "?"

    return cleaned


def _compress_bullet(text: str, target: int = 35) -> str:
    """Trim a verbose bullet to ≤target words, cutting at a natural sentence boundary.

    Strategy (in priority order):
      1. Already within target → return as-is.
      2. Find an em-dash or period in the second half of the target window →
         truncate there (preserves the 'what — why' structure).
      3. Hard-cap at target words.

    Never truncates mid-clause; always returns a grammatically complete phrase.
    """
    words = text.split()
    if len(words) <= target:
        return text

    # Build a slightly overlong candidate string to search for natural breaks
    candidate = " ".join(words[: target + 6])

    # Look for natural break points AFTER the midpoint of the target window
    midpoint = len(" ".join(words[: target // 2]))
    best_idx = None
    for sep in (" — ", ". ", "; "):
        idx = candidate.find(sep, midpoint)
        if 0 < idx < len(candidate) - 3:
            # Choose the break closest to target length (not exceeding target + 3 words)
            truncated = candidate[:idx].strip()
            if len(truncated.split()) <= target + 3:
                best_idx = idx
                best_sep = sep
                break  # prefer em-dash over period over semicolon

    if best_idx is not None:
        return candidate[:best_idx].strip()

    # Fallback: hard cap
    return " ".join(words[:target])


def enforce_bullet_only(text: str) -> str:
    """Extract bullet lines, compress verbose bullets to 20-35 words, re-prefix with •."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]

    bullet_lines = []
    for ln in lines:
        if ln.startswith("•") or ln.startswith("*") or ln.startswith("-"):
            body = ln.lstrip("•*- ").strip()
            # Skip empty bullets and follow-up questions (model sometimes bullets them)
            if not body or body.endswith("?"):
                continue
            bullet_lines.append(body)

    cleaned = []
    for body in bullet_lines:
        compressed = _compress_bullet(body, target=38)
        cleaned.append("• " + compressed)

    return "\n".join(cleaned[:6])


def _enrich_bullet_semantics(text: str, context: dict) -> str:
    """Enrich bullets with context-derived nuance where meaning is genuinely absent.

    Replaces the old _enforce_min_bullet_depth (which was length-driven and
    produced mechanical, repetitive phrasing).

    Principles:
    • Triggered by semantic gaps, not word count
    • Uses actual market context values (RSI reading, momentum direction, phase)
    • Never adds filler — every addition improves analytical precision
    • Skips bullets that already contain the relevant nuance
    """
    _TF = {"1d": "daily", "1wk": "weekly", "1mo": "monthly",
           "1h": "hourly", "5m": "5-minute"}

    tf    = _TF.get(context.get("timeframe", ""), context.get("timeframe", "current"))
    instr = context.get("instrument_name", "the instrument")
    trend      = (context.get("trend")          or "").lower()
    momentum   = (context.get("momentum")        or "").lower()
    volatility = (context.get("volatility")      or "").lower()
    particip   = (context.get("participation")   or "").lower()

    try:
        rsi = float(context.get("rsi") or 0)
    except (TypeError, ValueError):
        rsi = 0.0

    # ── Derive compact qualifiers ────────────────────────────────────────────
    mom_dir = (
        "strengthening" if any(w in momentum for w in ("strengthening", "strong")) else
        "weakening"     if any(w in momentum for w in ("weakening", "weak"))       else
        "neutral"       if "neutral" in momentum                                    else ""
    )
    vol_phase = (
        "compression" if any(w in volatility for w in ("contract", "compress")) else
        "expansion"   if "expand" in volatility                                 else ""
    )
    conv_dir = (
        "rising"  if any(w in particip for w in ("rising", "increasing")) else
        "fading"  if any(w in particip for w in ("fading", "declin"))     else
        "steady"  if "steady" in particip                                  else ""
    )
    # Actual RSI value (never treat "50" as the measured RSI)
    rsi_str = str(round(rsi)) if rsi > 0 else ""

    output_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("•"):
            output_lines.append(line)
            continue

        body = stripped[1:].strip()
        body_lower = body.lower()
        original_body = body

        # ── RSI / Momentum bullet ────────────────────────────────────────────
        if "rsi" in body_lower:
            # Is the actual RSI value (not just the ">50" threshold) present?
            has_rsi_value = bool(
                rsi_str and rsi_str not in ("50", "0", "")
                and rsi_str in body
            )
            if not has_rsi_value and rsi > 0:
                # IMPORTANT: clauses must NOT contain " — " to avoid double em-dash
                if rsi >= 70 and mom_dir == "weakening":
                    clause = (f"at {rsi:.0f}, RSI is extended and softening, "
                              "meaning demand has overreached and is losing upward authority")
                elif rsi >= 70:
                    clause = (f"at {rsi:.0f}, RSI sits in extended territory, "
                              "raising exhaustion risk even as demand conviction remains elevated")
                elif rsi >= 55 and mom_dir == "strengthening":
                    clause = (f"at {rsi:.0f} and rising, momentum is actively building "
                              "rather than plateauing, signalling a developing demand phase")
                elif rsi >= 50 and mom_dir == "weakening":
                    clause = (f"at {rsi:.0f}, the demand edge is narrowing as conviction "
                              "fades without yet reversing structural bias")
                elif rsi >= 50:
                    clause = (f"at {rsi:.0f}, demand retains the momentum edge "
                              "without reaching extremes, reflecting a controlled structural phase")
                elif rsi <= 32 and mom_dir == "weakening":
                    clause = (f"at {rsi:.0f}, RSI sits deep in supply territory, "
                              "where selling pressure may be overdone relative to structural support")
                elif rsi <= 45 and mom_dir == "strengthening":
                    clause = (f"at {rsi:.0f}, supply leads but momentum is building; "
                              "a reclaim of the neutral 50 line would shift the structural reading")
                else:
                    clause = (f"at {rsi:.0f}, supply retains the structural momentum edge "
                              "without reaching oversold extremes")

                if clause and clause.lower()[:20] not in body_lower:
                    body = body.rstrip(".,") + f" — {clause}."

        # ── Structure bullet ─────────────────────────────────────────────────
        elif body_lower.startswith("structure") or body_lower.startswith("price structure"):
            _QUALITY_WORDS = {
                "controlled", "choppy", "accelerating", "decelerating",
                "consolidating", "compressing", "ascending", "descending",
                "coiling", "impulsive", "irregular", "orderly",
            }
            has_quality = any(w in body_lower for w in _QUALITY_WORDS)

            if not has_quality and trend and mom_dir:
                # Clauses must NOT contain " — "
                if trend == "up" and mom_dir == "strengthening" and vol_phase == "expansion":
                    clause = ("with momentum and range expanding in tandem, "
                              "this is an accelerating demand phase rather than a plateauing one")
                elif trend == "up" and mom_dir == "weakening":
                    clause = ("the uptrend remains intact, yet weakening momentum "
                              "signals demand conviction is softening at current levels")
                elif trend == "down" and mom_dir == "strengthening":
                    clause = ("supply is intensifying rather than stabilising, "
                              "marking an accelerating decline rather than a controlled correction")
                elif trend == "down" and mom_dir == "weakening":
                    clause = ("selling momentum is decelerating within the downtrend; "
                              "structure stays bearish but sellers face diminishing follow-through")
                elif vol_phase == "compression":
                    direction = "demand" if trend == "up" else "supply" if trend == "down" else "directional"
                    clause = (f"the narrowing range signals structural compression, "
                              f"with contraction in an established trend typically resolving "
                              f"via {direction}-side expansion")
                else:
                    clause = ""

                if clause and clause.lower()[:20] not in body_lower:
                    body = body.rstrip(".,") + f" — {clause}."

        # ── Volatility bullet ────────────────────────────────────────────────
        elif body_lower.startswith("volatility") or (
            body_lower.startswith("range") and vol_phase
        ):
            _PHASE_WORDS = {
                "compression", "expansion", "exhaustion", "coiling",
                "breakout", "squeeze", "narrowing", "widening",
            }
            has_phase = any(w in body_lower for w in _PHASE_WORDS)

            if not has_phase and vol_phase:
                # Clauses must NOT contain " — "
                if vol_phase == "compression" and trend:
                    side = "demand" if trend == "up" else "supply"
                    clause = (f"a contracting range within an established {trend}trend "
                              f"typically resolves with {side}-driven expansion rather than reversal")
                elif vol_phase == "expansion" and trend and mom_dir == "strengthening":
                    dom = "demand" if trend == "up" else "supply"
                    clause = (f"range widening aligned with {dom} momentum "
                              "signals structural conviction, as expansion in trend direction "
                              "rarely reverses immediately")
                elif vol_phase == "expansion":
                    clause = ("range widening signals the market is committing to a direction; "
                              "the dominant side's ability to sustain the expansion determines durability")
                else:
                    clause = ""

                if clause and clause.lower()[:20] not in body_lower:
                    body = body.rstrip(".,") + f" — {clause}."

        # ── Participation bullet ─────────────────────────────────────────────
        elif body_lower.startswith("participation"):
            _CONV_WORDS = {
                "rising", "fading", "declining", "increasing",
                "steady", "growing", "shrinking", "building",
            }
            has_conv = any(w in body_lower for w in _CONV_WORDS)

            if not has_conv and conv_dir:
                # Clauses must NOT contain " — "
                if conv_dir == "rising":
                    clause = ("rising conviction signals the dominant side is actively "
                              "extending its structural footprint, not merely holding ground")
                elif conv_dir == "fading":
                    clause = ("fading conviction warns that the dominant side is losing "
                              "structural authority; the trend is intact but increasingly fragile")
                elif conv_dir == "steady":
                    clause = ("steady conviction reflects a controlled phase where neither "
                              "side is accelerating, supporting continuation rather than reversal")
                else:
                    clause = ""

                if clause and clause.lower()[:20] not in body_lower:
                    body = body.rstrip(".,") + f" — {clause}."

        # ── Interpretation bullet (if / then) ────────────────────────────────
        elif body_lower.startswith("if "):
            # Handle both "if … then …" and "if …, …" (comma-separated) formats.
            then_m = re.search(r"(?:\bthen\b|,\s*)(.+?)$", body, re.IGNORECASE)
            then_words = len(then_m.group(1).split()) if then_m else 0
            total_words = len(body.split())

            # Enrich when the "then" clause is thin OR the whole bullet is short
            if (then_words < 8 or total_words < 20) and trend:
                # Clauses must NOT contain " — "
                if trend == "up" and vol_phase == "expansion":
                    clause = "confirming the demand phase is entering an acceleration stage"
                elif trend == "down" and vol_phase == "expansion":
                    clause = "signalling supply is entering an impulsive continuation rather than a corrective bounce"
                elif vol_phase == "compression":
                    clause = ("resolving the compression phase, with directional bias "
                              "established post-contraction tending to extend rapidly")
                elif trend == "up":
                    clause = "affirming the structural demand bias remains intact and in control"
                elif trend == "down":
                    clause = "confirming structural supply control has not yet been challenged"
                else:
                    clause = ""

                if clause and clause.lower()[:20] not in body_lower:
                    body = body.rstrip(".,") + f", {clause}."

        # ── Generic depth fallback ───────────────────────────────────────────
        # Fires only when no specific trigger matched AND bullet is still short.
        # Handles unusual bullet starts produced by multi-TF / multi-indicator
        # queries (e.g. "Daily: structure is…", "Weekly RSI holds…") that escape
        # the keyword-prefix checks above.
        if body == original_body and len(body.split()) < 20:
            if rsi > 0 and rsi_str not in body and rsi_str not in ("50", "0"):
                if trend == "up":
                    clause = (f"with RSI at {rsi:.0f}, structural momentum supports "
                              "demand-side continuation on this timeframe")
                elif trend == "down":
                    clause = (f"with RSI at {rsi:.0f}, the structural bias remains "
                              "supply-led on this timeframe")
                else:
                    clause = (f"with RSI at {rsi:.0f}, the structural reading "
                              "is reinforced by momentum context")
            elif trend and vol_phase:
                side = "demand" if trend == "up" else "supply"
                clause = (f"the {vol_phase} regime in this {trend}trend "
                          f"supports {side}-side structural continuity")
            else:
                clause = ""

            if clause and clause.lower()[:20] not in body_lower:
                body = body.rstrip(".,") + f" — {clause}."

        output_lines.append(("• " + body) if body != original_body else line)

    return "\n".join(output_lines)


def _ensure_naturalness(text: str) -> str:
    """Remove mechanical expansion artifacts and reduce robotic repetition.

    Pure string heuristics — zero LLM calls, zero added latency.
    Applied as the final post-processing step before response assembly.

    Fixes:
      1. Old _enforce_min_bullet_depth mechanical suffix patterns
      2. Double em-dash chains where the last segment redundantly repeats the second
      3. Confirming/signalling/indicating word duplication across a bullet
      4. Orphaned punctuation artifacts
    """
    _OLD_MECHANICAL = [
        r"\s*—\s*implying \S+(?:'s|s') \S+ supply[/\\]demand balance remains structurally intact\.?",
        r"\s*—\s*confirming the dominant side retains momentum control on the \S+ chart\.?",
        r"\s*—\s*suggesting a directional resolution is forming in \S+(?:'s|s') \S+ structure\.?",
        r"\s*—\s*affirming structural conviction is aligned with the prevailing \S+ bias\.?",
        r"\s*—\s*signalling a potential shift in \S+(?:'s|s') \S+ demand[/\\]supply equilibrium\.?",
    ]
    _STOPWORDS = frozenset({
        "the", "a", "an", "is", "it", "in", "on", "of", "and", "or",
        "not", "but", "so", "as", "at", "to", "for", "be", "are",
    })

    lines = text.splitlines()
    result: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("•"):
            result.append(line)
            continue

        body = stripped[1:].strip()

        # ── 1. Remove old mechanical suffix artifacts ────────────────────────
        for pat in _OLD_MECHANICAL:
            body = re.sub(pat, "", body, flags=re.IGNORECASE)

        # ── 2. Collapse any double em-dash chains ────────────────────────────
        # After enrichment clauses are written without internal em-dashes,
        # this handles any residual cases from model output or conversation history.
        parts = body.split(" — ")
        if len(parts) >= 3:
            # Strategy: keep first two segments; merge the third into the second
            # only if it adds new content.  Otherwise drop it.
            def _content_words(s: str) -> set[str]:
                return {w for w in s.lower().split()
                        if w not in _STOPWORDS and len(w) > 3}
            # Drop last segment if it semantically overlaps the second (≥2 shared words)
            overlap = _content_words(parts[-1]) & _content_words(parts[-2])
            if len(overlap) >= 2:
                body = " — ".join(parts[:-1])
            elif len(parts) == 3:
                # Three distinct segments: merge 2nd and 3rd with comma
                body = f"{parts[0]} — {parts[1]}, {parts[2]}"

        # ── 3. Remove within-bullet word duplication ─────────────────────────
        # "confirming X — confirming Y" → "confirming X — Y"
        for verb in ("confirming", "signalling", "indicating", "suggesting"):
            body = re.sub(
                rf"\b{verb}\b([^—.{{0,80}}]) — {verb}\b",
                rf"{verb}\1 —",
                body, flags=re.IGNORECASE,
            )

        # ── 4. Normalise punctuation artifacts ───────────────────────────────
        body = re.sub(r"\.{2,}", ".", body)
        body = re.sub(r"\.,", ".", body)
        body = re.sub(r"\s{2,}", " ", body)
        body = re.sub(r"\s+\.", ".", body)
        body = body.strip()
        if body and not body[-1] in ".?!":
            body += "."

        result.append("• " + body)

    return "\n".join(result)


# Kept for import compatibility — delegates to the new function.
def _enforce_min_bullet_depth(text: str, context: dict, min_words: int = 20) -> str:
    """Deprecated: delegates to _enrich_bullet_semantics + _ensure_naturalness."""
    return _ensure_naturalness(_enrich_bullet_semantics(text, context))


# ── Deterministic participation labelling ─────────────────────────────────────

def _derive_participation_label(trend: str, participation: str) -> str:
    """Return a demand/supply-aware participation label grounded in trend direction.

    Prevents LLM from mis-labelling participation as 'demand-led' in a downtrend
    (or vice versa) by computing the dominant side deterministically.

    Returns a compact label suitable for injection into format_market_context.
    """
    t = (trend or "").lower()
    p = (participation or "").lower()

    if t == "up":
        conv_map = {
            "increasing": "demand-led with rising conviction",
            "decreasing": "demand-led with fading conviction",
            "steady":     "demand-led with steady conviction",
        }
        return conv_map.get(p, f"demand-led ({p})" if p else "demand-led")

    if t == "down":
        conv_map = {
            "increasing": "supply-led with rising conviction",
            "decreasing": "supply-led with fading conviction",
            "steady":     "supply-led with steady conviction",
        }
        return conv_map.get(p, f"supply-led ({p})" if p else "supply-led")

    # Sideways / unclear — neutral label
    conv_map = {
        "increasing": "increasing (no directional bias confirmed)",
        "decreasing": "decreasing (no directional bias confirmed)",
        "steady":     "balanced — neither side asserting conviction",
    }
    return conv_map.get(p, p or "unclear")


# ── Contradiction validation ───────────────────────────────────────────────────

def _validate_context_consistency(ctx: dict) -> dict:
    """Detect and annotate logical contradictions in derived context.

    Does NOT alter raw data values (RSI, trend, momentum) — they come from
    real market data and may genuinely reflect mixed conditions.

    What it DOES do:
      1. Injects `participation_label` — demand/supply-aware participation string
         that prevents LLM from inverting the dominant side.
      2. Detects RSI / trend divergence and flags it with `_rsi_divergence` so
         format_market_context can relay the mixed signal to the LLM.
      3. Re-computes `rsi_state` from the *existing* `classify_rsi()` function
         (imported from data.nifty_context) when trend contradicts the raw state,
         ensuring the label accurately reflects the mixed condition.

    Returns a new dict (caller's dict is never mutated).
    """
    ctx = dict(ctx)

    trend       = (ctx.get("trend")         or "").lower()
    momentum    = (ctx.get("momentum")      or "").lower()
    participation = (ctx.get("participation") or "").lower()
    rsi         = ctx.get("rsi")

    # ── 1. Deterministic participation label ─────────────────────────────────
    ctx["participation_label"] = _derive_participation_label(trend, participation)

    # ── 2. RSI / trend divergence detection ──────────────────────────────────
    if rsi is not None:
        rsi_f = float(rsi)
        divergent = (
            (trend == "down" and rsi_f > 55) or
            (trend == "up"   and rsi_f < 45)
        )
        if divergent:
            ctx["_rsi_divergence"] = True
            print(
                f"[TechnoBot/validate] RSI {rsi_f:.1f} diverges from trend '{trend}' "
                f"— mixed signal injected into context"
            )

        # ── 3. Re-derive rsi_state label when trend contradicts naive zone ────
        # classify_rsi is already called during data fetch; we call it again here
        # only when the first pass produced a state that conflicts with trend direction.
        existing_state = ctx.get("rsi_state", "n/a")
        demand_states  = {"overbought", "bullish", "mildly_bullish"}
        supply_states  = {"oversold", "bearish", "mildly_bearish"}

        _slope = ctx.get("rsi_slope", "stable")

        if trend == "down" and existing_state in demand_states:
            # Supply-led trend but RSI reads bullish — re-classify with trend + slope
            try:
                from data.nifty_context import classify_rsi
                state, note = classify_rsi(rsi_f, trend, _slope)
                ctx["rsi_state"] = state
                ctx["rsi_note"]  = note
            except Exception:
                pass  # keep existing state; don't crash on import error

        elif trend == "up" and existing_state in supply_states:
            try:
                from data.nifty_context import classify_rsi
                state, note = classify_rsi(rsi_f, trend, _slope)
                ctx["rsi_state"] = state
                ctx["rsi_note"]  = note
            except Exception:
                pass

    # ── 4. Momentum / trend sanity (no fix — just annotate) ──────────────────
    # "strengthening" momentum in a downtrend = acceleration (valid, not a bug)
    # "weakening" momentum in an uptrend = deceleration (valid, not a bug)
    # No correction needed — these are real market conditions, not data errors.

    return ctx



def is_market_query(q: str) -> bool:
    keywords = [
        "market", "nifty", "index", "overall",
        "confusing", "weak", "strong", "behaving"
    ]
    q = q.lower()
    return any(k in q for k in keywords)


def generate_followup_from_last_bullet(bullets: str, intent: str, context: dict) -> str:
    """Return a short mentor-style follow-up question.

    Selection is driven by the content of the last bullet and the user's intent.
    The follow-up must stay concise, single-sentence, and non-predictive.
    """
    import random
    lines = [ln.lower() for ln in bullets.splitlines() if ln.strip()]
    tf_raw = context.get("timeframe", "")
    tf = {"1d": "daily", "1wk": "weekly", "1mo": "monthly", "1h": "hourly", "5m": "5-minute"}.get(tf_raw, tf_raw or "current")
    instr = context.get("instrument_name", "this instrument")
    intent_lower = (intent or "").lower()

    trend = (context.get("trend") or "").lower()

    trend_label = "constructive" if trend == "up" else ("weakening" if trend == "down" else "sideways")

    confirmation = [
        f"What would confirm {instr}'s {tf} bias?",
        f"What level would keep this {trend_label} structure valid?",
        f"Which signal would show conviction is rising?",
    ]
    invalidation = [
        f"What would invalidate this reading on {instr}'s {tf} chart?",
        f"What level would break the current {trend_label} bias?",
        f"Which shift would show the structure is no longer valid?",
    ]
    scenario = [
        f"How would {instr}'s {tf} structure change if participation shifts?",
        f"What happens if momentum reverses from here?",
        f"How would the reading change if volatility shifts?",
    ]

    def pick(options: list[str]) -> str:
        return _shorten_followup_question(random.choice(options))

    if any(w in intent_lower for w in ["buy", "sell", "entry", "exit", "should i"]):
        return pick(confirmation)

    if "divergence" in intent_lower or (lines and "diverge" in lines[-1]):
        return pick(scenario)

    pattern_words = ["double top", "double bottom", "triangle", "flag", "head",
                     "shoulder", "cup", "handle", "wedge", "channel", "neckline"]
    if any(p in intent_lower for p in pattern_words):
        return pick(invalidation)

    if not lines:
        return pick(confirmation)

    last = lines[-1]

    if "interpret" in last or "if" in last or "scenario" in last:
        return pick(confirmation + invalidation + scenario)

    if "participation" in last or "volume" in last or "supply" in last or "demand" in last:
        return pick(invalidation)

    if "momentum" in last or "rsi" in last or "macd" in last or "weakening" in last:
        return pick(scenario)

    if "volatility" in last or "range" in last or "compress" in last or "contract" in last:
        return pick(scenario)

    if "trend" in last or "uptrend" in last or "downtrend" in last or "sideways" in last:
        return pick(confirmation)

    return pick(confirmation)

    if "support" in last or "resistance" in last or "structure" in last or "level" in last:
        return random.choice(INVALIDATION)

    # ── default: rotate through all three ────────────────────────────────────
    return random.choice(CONFIRMATION + INVALIDATION + SCENARIO)


def generate_insight_layer(context: dict) -> str:
    """Return a single 📌 contextual insight grounded in pattern, regime, and directional reasoning.

    Deterministic — no LLM call. Reads the computed context dict and applies
    experience-based structural pattern recognition across three dimensions:

      • Pattern similarity  — what historical structure does this resemble?
      • Regime framing      — what market phase is this? (compression / expansion / transition)
      • Directional implication — continuation, exhaustion, or potential reversal?

    Priority (first match wins):
      1. RSI extreme (≥70 or ≤30)              — extended / exhausted regime
      2. Uptrend + weakening momentum           — distribution / exhaustion pattern
      3. Downtrend + weakening momentum         — deceleration / base-building transition
      4. Downtrend + strengthening momentum     — acceleration / impulsive continuation
      5. Volatility compression                 — coiling / pre-expansion regime
      6. Volatility expansion + trend aligned   — breakout / impulse regime
      7. Participation vs trend alignment       — conviction health check
      8. RSI near neutral (45–55)               — transition / indecision regime
    """
    trend         = (context.get("trend")          or "").lower()
    momentum      = (context.get("momentum")        or "").lower()
    volatility    = (context.get("volatility")      or "").lower()
    participation = (context.get("participation")   or "").lower()
    instr         = context.get("instrument_name")  or "this instrument"
    tf_raw        = context.get("timeframe", "")
    tf = {"1d": "daily", "1wk": "weekly", "1mo": "monthly",
          "1h": "hourly", "5m": "5-minute"}.get(tf_raw, tf_raw or "current")

    try:
        rsi = float(context.get("rsi") or 0)
    except (TypeError, ValueError):
        rsi = 0.0

    # ── Rotation index — cycles phrase sets so same context yields different wording
    # Derived from RSI + trend + momentum so it's deterministic but varies per state.
    _rot_seed = (int(rsi) * 7 + len(trend) * 3 + len(momentum)) % 3

    # ── Helper: build the standard 3-bullet insight block ──────────────────────
    def _insight(
        regime: str | list[str],
        pattern: str | list[str],
        implication: str | list[str],
    ) -> str:
        """Format: header line + 3 bullet lines (parsed by frontend block classifier).

        Each argument may be a single string or a list of 2-3 alternative phrasings.
        When a list is provided, `_rot_seed` picks which phrasing to emit — same
        context will emit different wording across sessions/follow-ups.
        """
        def _pick(v: "str | list[str]") -> str:
            if isinstance(v, list):
                return v[_rot_seed % len(v)]
            return v

        return (
            "📌 Contextual Insight:\n"
            f"• Current regime: {_pick(regime)}\n"
            f"• Pattern: {_pick(pattern)}\n"
            f"• Implication: {_pick(implication)}"
        )

    # If context is too sparse (full + lightweight both failed), emit NO insight.
    # Data-error messaging belongs in the evidence block, NOT in the insight layer.
    # Emitting a fake "data unavailable" insight block pollutes the insight section
    # with non-analytical content — Task 4 mandates an empty return here.
    if trend in ("unknown", "unclear", "") and rsi == 0.0:
        return ""

    is_weakening   = "weaken" in momentum or momentum == "weakening"
    is_strengthen  = "strength" in momentum or momentum == "strengthening"
    is_contracting = "contract" in volatility or volatility == "contracting"
    is_expanding   = "expand" in volatility or volatility == "expanding"

    rsi_str = f"RSI {rsi:.0f}" if rsi else "RSI unconfirmed"

    # ── 1. RSI extended ≥70 — late-rally exhaustion ───────────────────────────
    if rsi >= 70:
        return _insight(
            regime=[
                f"Extended demand — {rsi_str} signals overbought territory on {tf} chart",
                f"Late-rally phase — {rsi_str} approaching historically congested RSI territory on {tf} chart",
                f"Demand saturation — {rsi_str} in historically supply-reactive zone on {tf} chart",
            ],
            pattern=[
                f"{instr} late-rally exhaustion: advances historically compress near RSI ≥70 before supply asserts",
                f"{instr} supply-reactive zone: RSI ≥70 has historically preceded range compression or demand-testing pullbacks",
                f"{instr} extended structure: prior RSI ≥70 phases resolved with supply-side pressure within 3–8 candles",
            ],
            implication=[
                f"If momentum begins fading at current RSI levels, supply-led consolidation becomes the structural risk",
                f"If RSI fails to print a new high on the next leg, bearish divergence becomes the structural alert",
                f"If price extends further without RSI confirmation, the risk of a sharp mean-reversion move rises materially",
            ],
        )

    # ── 2. RSI exhausted ≤30 — capitulation / demand re-engagement ───────────
    if 0 < rsi <= 30:
        return _insight(
            regime=[
                f"Supply exhaustion — {rsi_str} in deeply oversold territory on {tf} chart",
                f"Capitulation phase — {rsi_str} approaching historically demand-reactive zone on {tf} chart",
                f"Demand re-engagement potential — {rsi_str} at structurally oversold depth on {tf} chart",
            ],
            pattern=[
                f"{instr} capitulation structure: prior phases at RSI ≤30 attracted demand after sustained selling",
                f"{instr} demand-reactive zone: RSI ≤30 historically preceded brief demand impulses even within downtrends",
                f"{instr} exhaustion signal: sustained supply conviction near RSI ≤30 raises risk of sharp technical bounces",
            ],
            implication=[
                f"If demand conviction begins building from this depth, early base-building becomes the structural scenario",
                f"If price prints a higher low from current levels, demand re-engagement receives structural confirmation",
                f"If RSI forms a positive divergence on the next low, it marks the first condition for a potential reversal",
            ],
        )

    # ── 3. Uptrend + weakening momentum — distribution / top ─────────────────
    if trend == "up" and is_weakening:
        return _insight(
            regime=[
                f"Demand-led but decelerating — uptrend intact with fading momentum on {tf} chart",
                f"Distribution phase — uptrend losing internal conviction on {tf} chart",
                f"Late-trend fatigue — price advancing but momentum diverging on {tf} chart",
            ],
            pattern=[
                f"{instr} distribution-phase structure: successive highs with diminishing RSI conviction",
                f"{instr} momentum divergence: price continuing higher while RSI prints lower highs — classic distribution signal",
                f"{instr} deceleration pattern: demand still dominant but losing buying urgency with each successive swing high",
            ],
            implication=[
                f"If momentum continues fading without a new high, consolidation or a structural demand test becomes likely",
                f"If the next high fails to recover RSI toward prior peaks, the structural case for a trend pause strengthens",
                f"If price fails to hold the last swing low after momentum fails, demand exhaustion becomes the primary scenario",
            ],
        )

    # ── 4. Downtrend + weakening momentum — deceleration / base-building ─────
    if trend == "down" and is_weakening:
        return _insight(
            regime=[
                f"Supply-dominant but decelerating — downtrend intact with fading conviction on {tf} chart",
                f"Deceleration phase — downtrend losing supply urgency on {tf} chart",
                f"Base-building precursor — supply conviction diminishing within the downtrend on {tf} chart",
            ],
            pattern=[
                f"{instr} deceleration structure: supply momentum fading, precursor to base-building phases historically",
                f"{instr} supply exhaustion pattern: lower lows printing with diminishing RSI conviction — structural caution for bears",
                f"{instr} loss-of-momentum structure: each downswing requires more effort for less price movement — classic deceleration signal",
            ],
            implication=[
                f"If supply conviction continues fading without a new low, the first structural condition for demand recovery is forming",
                f"If RSI prints a higher low while price makes a lower low, positive divergence marks the first structural sign of reversal",
                f"If price stabilises above the prior low with fading momentum, a demand-side base becomes the structural scenario",
            ],
        )

    # ── 5. Downtrend + strengthening momentum — impulsive continuation ────────
    if trend == "down" and is_strengthen:
        return _insight(
            regime=[
                f"Supply expansion — impulsive downtrend with accelerating conviction on {tf} chart",
                f"Impulsive decline — supply accelerating with no structural demand resistance on {tf} chart",
                f"Trend continuation — supply dominating decisively as momentum accelerates on {tf} chart",
            ],
            pattern=[
                f"{instr} continuation structure: supply dominates without resistance; range widening on downswings",
                f"{instr} impulsive decline: each supply wave travels further than the last — hallmark of an accelerating downtrend",
                f"{instr} supply dominance: demand attempts being absorbed decisively; structurally bearish continuation pattern",
            ],
            implication=[
                f"If momentum sustains without a volatility climax or RSI extreme, the downtrend continuation is the structural base case",
                f"If supply momentum reaches RSI ≤30 without a climax candle, watch for a countertrend bounce before continuation",
                f"If range expansion sustains into the next swing low, downside targets from prior structure become active",
            ],
        )

    # ── 6. Uptrend + strengthening momentum — demand impulse ─────────────────
    if trend == "up" and is_strengthen:
        return _insight(
            regime=[
                f"Demand expansion — uptrend with accelerating conviction on {tf} chart",
                f"Impulse phase — demand accelerating with decisive supply absorption on {tf} chart",
                f"Trend acceleration — demand dominates as momentum builds on {tf} chart",
            ],
            pattern=[
                f"{instr} impulse structure: demand absorbing supply decisively; hallmark of early trend acceleration",
                f"{instr} demand impulse: each advance travels further with less RSI pullback — structurally strong continuation",
                f"{instr} expanding demand: supply levels being absorbed on each retest; momentum confirming structural bullish bias",
            ],
            implication=[
                f"If momentum sustains, continuation is the structural base case until RSI reaches extended territory",
                f"If each pullback holds above the prior swing high, the impulse structure remains intact for continuation",
                f"If RSI remains below 70 while momentum accelerates, the trend has runway before entering supply-reactive territory",
            ],
        )

    # ── 7. Volatility compression — coiling / pre-expansion ──────────────────
    if is_contracting:
        phase = "within the downtrend" if trend == "down" else (
                "within the uptrend" if trend == "up" else "at a structural inflection point")
        return _insight(
            regime=f"Compression — pre-expansion contraction {phase} on {tf} chart",
            pattern=f"{instr} range contraction: volatility compressing, historically precedes a directional expansion",
            implication=f"If the compression resolves with a supply-side break, the downtrend receives confirmation; demand-side resolution favours continuation",
        )

    # ── 8. Volatility expansion — aligned with trend ──────────────────────────
    if is_expanding and trend == "down":
        return _insight(
            regime=f"Supply expansion — impulsive range widening in a downtrend on {tf} chart",
            pattern=f"{instr} breakout continuation: supply absorbing demand at every attempt, range expanding decisively",
            implication=f"If range expansion sustains without a volume climax, supply dominance continues; watch for exhaustion signals",
        )
    if is_expanding and trend == "up":
        return _insight(
            regime=f"Demand expansion — impulsive range widening in an uptrend on {tf} chart",
            pattern=f"{instr} breakout structure: demand overwhelms supply decisively; structurally healthy expansion",
            implication=f"If RSI remains below extended territory and range stays wide, continuation is the structural base case",
        )

    # ── 9. RSI near neutral — transition / indecision ────────────────────────
    if 0 < rsi <= 55 and rsi >= 45:
        dominant = "supply" if trend == "down" else ("demand" if trend == "up" else "neither side")
        return _insight(
            regime=f"Transition — RSI near neutral 50 on {tf} chart; {dominant} holds structural edge",
            pattern=f"{instr} indecision phase: momentum unconfirmed at midpoint; structurally inconclusive",
            implication=f"If the next directional move arrives with RSI confirming above/below 50, that direction establishes the structural bias",
        )

    # ── 10. Sideways / ranging — balance / consolidation ─────────────────────
    if trend in ("sideways", "unclear"):
        if rsi > 55:
            regime_detail = f"RSI {rsi:.0f} tilts toward demand within the range"
            impl = "If range resolves with demand-side expansion, an upside structural leg becomes the base case"
        elif 0 < rsi < 45:
            regime_detail = f"RSI {rsi:.0f} tilts toward supply within the range"
            impl = "If range resolves with supply-side expansion, the downside structural leg becomes the base case"
        else:
            regime_detail = "RSI near neutral; neither side holds a structural edge"
            impl = "If momentum confirms a directional break with expanding range, that direction sets the next structural leg"
        return _insight(
            regime=f"Consolidation / balance — price oscillating without directional resolution on {tf} chart; {regime_detail}",
            pattern=f"{instr} balance structure: price compressing within a range; breakout direction will define next structural leg",
            implication=impl,
        )

    # ── 11. Catch-all — always return something meaningful ────────────────────
    direction = "constructive" if trend == "up" else ("supply-dominant" if trend == "down" else "neutral")
    return _insight(
        regime=[
            f"Transitional — {direction} structure on {tf} chart with {rsi_str}",
            f"Regime unclear — {direction} bias tentative; awaiting momentum confirmation on {tf} chart",
            f"Structural indecision — {direction} tilt with {rsi_str}; regime confirmation pending on {tf} chart",
        ],
        pattern=[
            f"{instr} structural positioning: monitor momentum conviction and volatility phase for regime confirmation",
            f"{instr} mixed signals: trend and momentum not yet aligned — regime transition is the base scenario",
            f"{instr} pre-resolution phase: price action lacks decisive directional follow-through; volume participation is the key tell",
        ],
        implication=[
            f"If momentum resolves directionally with an expanding range, the {direction} bias receives structural confirmation",
            f"If the next swing closes with expanding range and volume, directional regime emerges; wait for that confirmation",
            f"If RSI and price align directionally on the next move, the structural bias clarifies; premature positioning carries higher risk",
        ],
    )


def generate_multitf_context(ticker: str, primary_tf: str, primary_ctx: dict) -> str:
    """Compare the primary timeframe context against the next-higher timeframe.

    Adds a '🔭 Broader Context' line that tells analysts whether the primary-tf
    move is aligned with or counter to the larger structural bias.

    No extra LLM call — fully deterministic from the two context dicts.
    Uses the same 5-min LRU cache as all other context fetches, so if the higher
    TF was already fetched this session the cost is ~0ms.

    Returns an empty string when:
      - Primary TF is already monthly (no upgrade available)
      - Higher TF data fetch fails
      - Both contexts are 'unknown' / 'unclear'
    """
    # Fix 3: Multi-TF fetch DISABLED to stay within free-tier API limits.
    # Fetching an additional timeframe per query doubles the API call rate,
    # pushing the system over the 8 req/min Twelve Data ceiling.
    # Single-TF mode (Fix 2) is sufficient for production stability.
    # Re-enable by removing this guard once a paid API tier is available.
    print(f"[TechnoBot/fetch] MULTI-TF DISABLED (Fix 3) — {ticker}/{primary_tf}")
    return ""

    # Timeframe upgrade ladder
    _UPGRADE = {"5m": "1d", "1h": "1d", "1d": "1wk", "1wk": "1mo"}
    higher_tf = _UPGRADE.get(primary_tf)
    if not higher_tf:
        return ""  # already at monthly — no higher TF to compare

    # Use pre-fetched higher-TF context if build_prompt already fetched it (0 ms).
    # Falls back to a fresh fetch only when not pre-fetched (e.g. comparison path).
    try:
        ctx_h = primary_ctx.get("_prefetched_higher_ctx") or \
                _cached_derive_market_context(higher_tf, ticker)
    except Exception:
        return ""

    p_trend = (primary_ctx.get("trend") or "").lower()
    h_trend = (ctx_h.get("trend")       or "").lower()
    p_mom   = (primary_ctx.get("momentum") or "").lower()

    instr = primary_ctx.get("instrument_name") or "this instrument"

    # Human-readable labels
    _TF = {"1d": "daily", "1wk": "weekly", "1mo": "monthly", "1h": "hourly", "5m": "5-min"}
    ptf = _TF.get(primary_tf, primary_tf)
    htf = _TF.get(higher_tf,  higher_tf)

    # Task 3: if primary trend is known but higher TF is unknown, give a partial note
    if p_trend and p_trend not in ("unknown", "unclear") and h_trend in ("unknown", "unclear", ""):
        direction = "constructive" if p_trend == "up" else ("supply-dominant" if p_trend == "down" else "range-bound")
        return (
            f"🔭 Broader Context: {htf.title()} data for {instr} is currently limited — "
            f"the {ptf} structure shows a {direction} bias; "
            "confirm on the higher timeframe once data is available for alignment check."
        )

    if not p_trend or not h_trend or h_trend in ("unknown", "unclear"):
        return ""

    # ── Aligned trends ───────────────────────────────────────────────────────
    if p_trend == h_trend == "down":
        suffix = (
            "suggesting this is a structurally deep move with conviction across timeframes, "
            "not an isolated short-term decline."
        )
        if "weaken" in p_mom:
            suffix = (
                "though the weakening daily momentum suggests the rate of decline may be "
                "decelerating within the broader supply-dominant structure."
            )
        return (
            f"🔭 Broader Context: The {htf} trend also remains supply-dominant — "
            f"the {ptf} weakness is structurally aligned with the higher timeframe, {suffix}"
        )

    if p_trend == h_trend == "up":
        suffix = (
            "suggesting demand conviction extends beyond the short-term and the broader "
            "structural bias supports the current move."
        )
        if "weaken" in p_mom:
            suffix = (
                "though fading daily momentum suggests the {ptf} advance is losing conviction "
                "even as the {htf} bias remains constructive."
            ).format(ptf=ptf, htf=htf)
        return (
            f"🔭 Broader Context: The {htf} structure is also constructive — "
            f"the {ptf} uptrend is reinforced by the higher timeframe alignment, {suffix}"
        )

    # ── Counter-trend (divergent timeframes) ────────────────────────────────
    if p_trend == "down" and h_trend == "up":
        return (
            f"🔭 Broader Context: While the {ptf} structure shows supply pressure, "
            f"the {htf} trend remains constructive — this is structurally consistent with "
            "a short-term pullback or correction within a larger demand-led structure, "
            "rather than a structural trend reversal."
        )

    if p_trend == "up" and h_trend == "down":
        return (
            f"🔭 Broader Context: The {ptf} shows a demand-led recovery, but the "
            f"{htf} structure remains supply-dominant — structural caution is warranted; "
            "the higher timeframe bias has not confirmed a shift, making this recovery "
            "a potential counter-trend move until the broader structure confirms."
        )

    # ── Sideways / unclear on higher TF ─────────────────────────────────────
    if h_trend in ("sideways", ""):
        return (
            f"🔭 Broader Context: The {htf} structure is consolidating — "
            f"the {ptf} move is occurring within a broader range without a clear directional "
            "bias on the higher timeframe, reducing structural conviction for either side."
        )

    return ""


def generate_comparison_insight(
    inst1: str, ctx1: dict,
    inst2: str, ctx2: dict,
) -> str:
    """Return a '🔍 Relative Insight' comparing two instruments' structural positioning.

    Deterministic — no LLM call. Derives comparative reasoning from the two
    context dicts: momentum exhaustion, RSI differential, trend alignment, and
    conviction depth.
    """
    t1 = (ctx1.get("trend")     or "").lower()
    t2 = (ctx2.get("trend")     or "").lower()
    m1 = (ctx1.get("momentum")  or "").lower()
    m2 = (ctx2.get("momentum")  or "").lower()
    v1 = (ctx1.get("volatility") or "").lower()
    v2 = (ctx2.get("volatility") or "").lower()
    p1 = (ctx1.get("participation") or "").lower()
    p2 = (ctx2.get("participation") or "").lower()

    # If either instrument has no usable trend data, skip the insight entirely —
    # generating misleading "sideways" labels for unknown data pollutes the output.
    if t1 in ("unknown", "unclear", "") or t2 in ("unknown", "unclear", ""):
        return ""

    try:
        rsi1 = float(ctx1.get("rsi") or 0)
        rsi2 = float(ctx2.get("rsi") or 0)
    except (TypeError, ValueError):
        rsi1 = rsi2 = 0.0

    weak1 = "weaken" in m1
    weak2 = "weaken" in m2
    str1  = "strength" in m1
    str2  = "strength" in m2

    # Volatility phase helpers
    exp1  = "expand" in v1
    exp2  = "expand" in v2
    comp1 = "contract" in v1
    comp2 = "contract" in v2

    # RSI state labels for compact display
    rs1 = (ctx1.get("rsi_state") or "").replace("_", " ")
    rs2 = (ctx2.get("rsi_state") or "").replace("_", " ")

    # ── Both in same trend direction ─────────────────────────────────────────
    if t1 == t2:
        trend_label = (
            "downtrend" if t1 == "down" else
            "uptrend"   if t1 == "up"   else
            "sideways phase"
        )

        # RSI differential — primary differentiator (threshold lowered to 3 pts)
        if rsi1 > 0 and rsi2 > 0 and abs(rsi1 - rsi2) >= 3:
            stronger, weaker_inst, rsi_s, rsi_w, rs_s, rs_w = (
                (inst2, inst1, rsi2, rsi1, rs2, rs1) if rsi2 > rsi1 else
                (inst1, inst2, rsi1, rsi2, rs1, rs2)
            )
            # Volatility addendum when phases differ
            vol_add = ""
            if exp1 and not exp2:
                vol_add = f", with {inst1} also showing higher volatility expansion"
            elif exp2 and not exp1:
                vol_add = f", with {inst2} also showing higher volatility expansion"
            elif comp1 and not comp2:
                vol_add = f", with {inst1} in a tighter compression phase"
            elif comp2 and not comp1:
                vol_add = f", with {inst2} in a tighter compression phase"

            if t1 == "up":
                return (
                    f"🔍 Relative Insight: Both instruments are in a {trend_label}, but "
                    f"{stronger} maintains stronger bullish conviction — RSI {rsi_s:.0f} "
                    f"({rs_s}) vs {rsi_w:.0f} ({rs_w}) — making {stronger} the relatively "
                    f"stronger structure with more momentum depth{vol_add}."
                )
            elif t1 == "down":
                # Lower RSI in a downtrend = deeper supply = weaker structure
                deeper, shallower, rsi_deep, rsi_shal = (
                    (inst1, inst2, rsi1, rsi2) if rsi1 < rsi2 else (inst2, inst1, rsi2, rsi1)
                )
                return (
                    f"🔍 Relative Insight: Both are in a {trend_label}, but "
                    f"{deeper} shows sharper downside pressure — RSI {rsi_deep:.0f} "
                    f"vs {rsi_shal:.0f} — indicating deeper supply entrenchment "
                    f"compared to {shallower}, which shows relatively earlier signs "
                    f"of supply deceleration{vol_add}."
                )

        # Momentum exhaustion differentiator (no RSI available or diff < 3)
        if weak1 and not weak2:
            vol_note = ""
            if rsi1 > 0 and rsi2 > 0:
                vol_note = f" (RSI {rsi1:.0f} vs {rsi2:.0f})"
            return (
                f"🔍 Relative Insight: Both {inst1} and {inst2} share a similar {trend_label}, "
                f"but {inst1} is showing earlier momentum exhaustion{vol_note} — conviction fading "
                f"while {inst2} maintains a steadier structural pace, making {inst2} the "
                "relatively stronger structure in the current phase."
            )
        if weak2 and not weak1:
            vol_note = ""
            if rsi1 > 0 and rsi2 > 0:
                vol_note = f" (RSI {rsi1:.0f} vs {rsi2:.0f})"
            return (
                f"🔍 Relative Insight: Both {inst1} and {inst2} share a similar {trend_label}, "
                f"but {inst2} is showing earlier momentum exhaustion{vol_note} — {inst1} maintains "
                "stronger structural conviction, suggesting relative strength in its current phase."
            )

        # Volatility differentiator
        if exp1 and not exp2:
            rsi_note = (f" (RSI {rsi1:.0f} vs {rsi2:.0f})" if rsi1 > 0 and rsi2 > 0 else "")
            return (
                f"🔍 Relative Insight: Both instruments share a {trend_label}, but "
                f"{inst1} is in a range expansion phase while {inst2} is not{rsi_note} — "
                f"{inst1} is committing to the directional move more decisively, "
                "making it the more structurally active instrument right now."
            )
        if comp1 and not comp2:
            rsi_note = (f" (RSI {rsi1:.0f} vs {rsi2:.0f})" if rsi1 > 0 and rsi2 > 0 else "")
            return (
                f"🔍 Relative Insight: Both instruments share a {trend_label}, but "
                f"{inst1} is in a compression phase while {inst2} is not{rsi_note} — "
                f"{inst1} is building structural tension for a range break, "
                "while the other is expressing the move more openly."
            )

        # Aligned on all dimensions — note parity with RSI context
        rsi_parity = (
            f" (RSI {rsi1:.0f} vs {rsi2:.0f}, both in {rs1 or 'similar'} territory)"
            if rsi1 > 0 and rsi2 > 0 else ""
        )
        return (
            f"🔍 Relative Insight: {inst1} and {inst2} are in broad structural alignment "
            f"within the {trend_label}{rsi_parity}. "
            "The relative distinction is subtle — monitor participation conviction and "
            "volatility on the next directional swing to identify which diverges first."
        )

    # ── Divergent trend directions ────────────────────────────────────────────
    label1 = "constructive" if t1 == "up" else ("supply-dominant" if t1 == "down" else "sideways")
    label2 = "constructive" if t2 == "up" else ("supply-dominant" if t2 == "down" else "sideways")
    rsi_divergence = ""
    if rsi1 > 0 and rsi2 > 0:
        rsi_divergence = (
            f" RSI {rsi1:.0f} ({rs1}) vs {rsi2:.0f} ({rs2}) confirms the "
            "directional divergence — the gap in momentum conviction supports "
            "a rotation dynamic between these structures."
        )
    return (
        f"🔍 Relative Insight: {inst1} and {inst2} are in divergent structural phases — "
        f"{inst1} is {label1} while {inst2} is {label2}.{rsi_divergence} "
        "Capital tends to rotate from the weaker structure toward the stronger one "
        "when the divergence is this pronounced."
    )


def detect_instrument(query, fallback_context: dict | None = None):
    q = query.lower()

    # ── Resolution cache (skip when fallback_context is active to avoid staleness) ──
    if not fallback_context:
        cache_key = q.strip()
        if cache_key in _INSTRUMENT_RESOLUTION_CACHE:
            return _INSTRUMENT_RESOLUTION_CACHE[cache_key]

    # ── Display-name overrides ─────────────────────────────────────────────────
    # Keys whose title-case form is wrong (abbreviations, compound words, acronyms).
    # Map: INSTRUMENT_MAP key → human-readable display name.
    _DISPLAY_NAMES: dict[str, str] = {
        # Abbreviations / acronyms that must stay ALL-CAPS
        "tcs":           "TCS",
        "sbi":           "SBI",
        "infy":          "Infosys",
        "ril":           "Reliance",
        "rjio":          "Reliance",
        "bse":           "BSE Sensex",
        "nse":           "Nifty 50",
        # Compound ticker-style keys (single token, should split at word boundary)
        "hdfcbank":      "HDFC Bank",
        "hdfcb":         "HDFC Bank",
        "icicibank":     "ICICI Bank",
        "icicib":        "ICICI Bank",
        "axisbank":      "Axis Bank",
        "kotakbank":     "Kotak Bank",
        "tatasteel":     "Tata Steel",
        "tatamotors":    "Tata Motors",
        "tatapower":     "Tata Power",
        "adanient":      "Adani Enterprises",
        "adaniports":    "Adani Ports",
        "adanigreen":    "Adani Green",
        "adanipower":    "Adani Power",
        "bajajfinance":  "Bajaj Finance",
        "bajajfinserv":  "Bajaj Finserv",
        "coalindia":     "Coal India",
        "powergrid":     "Power Grid",
        "hindunilvr":    "Hindustan Unilever",
        "nestleindia":   "Nestle India",
        "sunpharma":     "Sun Pharma",
        "drreddy":       "Dr. Reddy's",
        "reliance industries": "Reliance Industries",
        # Index names
        "nifty bank":    "Nifty Bank",
        "nifty it":      "Nifty IT",
        "nifty midcap":  "Nifty Midcap 50",
        "nifty 50":      "Nifty 50",
        "sensex":        "BSE Sensex",
        "midcap 50":     "Nifty Midcap 50",
    }

    # Sort by name length descending — "nifty bank" must match before "nifty"
    _ALL_CAPS = {"it", "etf", "sbi", "tcs", "hdfc", "icici", "rsi", "ema", "sma"}
    for name in sorted(INSTRUMENT_MAP.keys(), key=len, reverse=True):
        if name in q:
            # Prefer explicit override; fall back to title-case heuristic
            if name in _DISPLAY_NAMES:
                display = _DISPLAY_NAMES[name]
            else:
                parts = name.split()
                display = " ".join(p.upper() if p in _ALL_CAPS else p.title()
                                   for p in parts)
            result = display, INSTRUMENT_MAP[name]
            if not fallback_context:
                _INSTRUMENT_RESOLUTION_CACHE[q.strip()] = result
            return result

    # attempt to detect explicit ticker tokens (skip indicator-like tokens)
    matches = re.findall(r"\b[A-Z]{2,12}(?:\.[A-Z]{1,4})?\b", query)
    for token in matches:
        if token in INDICATOR_TICKER_BLOCKLIST:
            continue
        ticker = token if "." in token else f"{token}.NS"
        result = token, ticker
        if not fallback_context:
            _INSTRUMENT_RESOLUTION_CACHE[q.strip()] = result
        return result

    # Use fallback_context only when the user appears to refer to the previous
    # instrument (words like 'it', 'that', 'this', 'same', 'previous', etc.).
    if (
        fallback_context
        and fallback_context.get("instrument_name")
        and fallback_context.get("ticker")
    ):
        if re.search(r"\b(it|that|this|same|previous|current|here|there)\b", q):
            return fallback_context["instrument_name"], fallback_context["ticker"]

    default = "NIFTY 50", "^NSEI"
    if not fallback_context:
        _INSTRUMENT_RESOLUTION_CACHE[q.strip()] = default
    return default


def derive_context_from_dataframe(df, timeframe: str) -> dict:
    """Derive market context from an uploaded dataframe.

    A4: Expects df with normalised (lowercase) column names.
    Call normalize_df_columns() before passing df here.
    """
    # A4: always normalise so this function is safe regardless of caller
    df = normalize_df_columns(df)

    if df.empty or not {"close", "high", "low"}.issubset(df.columns):
        return {
            "timeframe": timeframe,
            "trend": "unknown",
            "momentum": "unknown",
            "volatility": "unknown",
            "data_source": "uploaded_file"
        }

    close = df["close"].astype(float)

    if len(close) < 20:
        return {
            "timeframe": timeframe,
            "trend": "unclear",
            "momentum": "unclear",
            "volatility": "unclear",
            "data_source": "uploaded_file"
        }

    # TREND
    recent_close = float(close.iloc[-1])
    earlier_mean = float(close.iloc[:-5].mean())
    trend = "up" if recent_close > earlier_mean else "down"

    # MOMENTUM
    recent_change = float(close.diff().iloc[-5:].abs().mean())
    earlier_change = float(close.diff().iloc[:-5].abs().mean())
    momentum = "strengthening" if recent_change > earlier_change else "weakening"

    # VOLATILITY
    recent_range = float((df["high"] - df["low"]).iloc[-5:].mean())
    earlier_range = float((df["high"] - df["low"]).iloc[:-5].mean())
    volatility = "expanding" if recent_range > earlier_range else "contracting"

    return {
        "timeframe": timeframe,
        "trend": trend,
        "momentum": momentum,
        "volatility": volatility,
        "data_source": "uploaded_file",
        "rows_used": len(df)
    }


def format_evidence_summary(context: dict) -> str:
    """Build the evidence block shown below every analysis response.

    Always includes all 7 required fields:
      Source · Instrument · Ticker · Timeframe · Data Range ·
      Last Price (with date) · Yahoo Finance link (dynamic from ticker)
    """
    # Task 2: never return an empty evidence block — always show at minimum
    # the instrument, ticker, timeframe, and data note so the user can verify.
    if not context:
        return ""

    data_status = context.get("data_status", "available")
    data_source = context.get("data_source", "yahoo")

    # Partial / unknown context — build a minimal evidence block instead of ""
    if context.get("trend") == "unknown":
        ticker_u   = context.get("ticker")      or "N/A"
        instr_u    = context.get("instrument_name") or "Unknown instrument"
        tf_u       = context.get("timeframe",   "")
        lp_u       = context.get("last_price")
        price_line = f"• Last Close: ₹{lp_u:,.2f}\n" if lp_u else ""
        yurl_u = f"https://finance.yahoo.com/quote/{ticker_u}" if ticker_u != "N/A" else ""
        ref_line = f"• Reference: {yurl_u}\n" if yurl_u else ""
        note_line = (
            "• Data note: OHLCV history temporarily unavailable — analysis draws on "
            "price structure context. Try rephrasing the query or checking the ticker symbol.\n"
        )
        return (
            "\nEvidence summary:\n"
            f"• Source: Yahoo Finance (live quote — historical OHLCV limited)\n"
            f"• Instrument: {instr_u}\n"
            f"• Ticker: {ticker_u}\n"
            f"• Timeframe: {tf_u or 'daily (default)'}\n"
            + price_line
            + ref_line
            + note_line
        )

    note = (
        "Intraday data unavailable; interpretation reflects higher timeframe structure."
        if data_status == "unavailable" else None
    )

    # ── Uploaded CSV path ────────────────────────────────────────────────────
    if data_source == "uploaded_file":
        instrument = context.get("instrument_name", "Uploaded file")
        rows       = context.get("rows_used", "N/A")
        tf_raw     = context.get("timeframe", "detected")
        last_price = context.get("last_price")
        last_date  = context.get("last_date", "")
        price_line = f"• Last Close: {last_price:,} as of {last_date}\n" if last_price and last_date else (
                     f"• Last Close: {last_price:,}\n" if last_price else "")
        summary = (
            "\nEvidence summary:\n"
            f"• Source: Uploaded CSV\n"
            f"• Instrument: {instrument}\n"
            f"• Timeframe: {tf_raw}\n"
            f"• Data Range: {rows} rows\n"
            + price_line +
            f"• Data used: price structure, volatility behavior, momentum profile\n"
        )
        return summary

    # ── Live market data path ────────────────────────────────────────────────
    ticker = context.get("ticker") or "^NSEI"

    # Dynamic Yahoo Finance link — always generated from the resolved ticker.
    yahoo_url = f"https://finance.yahoo.com/quote/{ticker}"

    _TF_LABELS = {
        "5m":  "Intraday · 5-min bars · last 5 days",
        "1h":  "Intraday · 1-hour bars · last 60 days",
        "1d":  "Daily · last 6 months",
        "1wk": "Weekly · last 2 years",
        "1mo": "Monthly · last 5 years",
    }
    _TF_SESSIONS = {
        "5m":  "~120 bars",
        "1h":  "~480 bars",
        "1d":  "~125 sessions",
        "1wk": "~104 weeks",
        "1mo": "~60 months",
    }
    tf_raw   = context.get("timeframe", "")
    tf_label = context.get("data_window_label") or _TF_LABELS.get(tf_raw, f"{tf_raw} chart")
    data_range = _TF_SESSIONS.get(tf_raw, "recent sessions")

    instrument = context.get("instrument_name") or "NIFTY 50"
    data_src = (context.get("data_source") or "").lower()
    if data_src == "uploaded_file":
        source_label = "Uploaded CSV"
    elif "twelvedata" in data_src:
        source_label = "Twelve Data"
    elif "alphavantage" in data_src:
        source_label = "Alpha Vantage"
    else:
        source_label = "Yahoo Finance"

    # Last price line — ₹ prefix for Indian instruments
    last_price = context.get("last_price")
    last_date  = context.get("last_date", "")
    last_updated = (
        f" · updated {last_date}" if last_date
        else f" · updated {datetime.now().strftime('%Y-%m-%d')}"
    )
    price_line = ""
    if last_price:
        is_indian = (
            ".NS" in ticker or ".BO" in ticker
            or ticker.startswith("^N") or ticker.startswith("^B")
        )
        currency = "₹" if is_indian else ""
        price_line = f"• Last Close: {currency}{last_price:,}{last_updated}\n"
    else:
        # Always include the field — fall back to "N/A" so the block is complete
        price_line = f"• Last Close: N/A\n"

    summary = (
        "\nEvidence summary:\n"
        f"• Source: {source_label}\n"
        f"• Instrument: {instrument}\n"
        f"• Ticker: {ticker}\n"
        f"• Timeframe: {tf_label}\n"
        f"• Data Range: {data_range}\n"
        + price_line
        + f"• Reference: {yahoo_url}\n"
    )

    if note:
        summary += f"• Note: {note}\n"

    return summary


def format_uploaded_file_evidence_summary(
    context: dict,
    uploaded_df: pd.DataFrame | None = None,
) -> str:
    """Build a CSV-only evidence block for user-uploaded files."""
    if not context or context.get("data_source") != "uploaded_file":
        return ""

    df = uploaded_df.copy() if uploaded_df is not None else None
    if df is not None:
        df = normalize_df_columns(df)

    timeframe = context.get("timeframe") or "unknown"
    date_range = ""
    rows_used = context.get("rows_used")

    if df is not None and not df.empty:
        rows_used = len(df)
        if "date" in df.columns:
            try:
                date_series = pd.to_datetime(df["date"], errors="coerce").dropna()
                if not date_series.empty:
                    inferred_timeframe = infer_timeframe(df)
                    if inferred_timeframe:
                        timeframe = inferred_timeframe
                    start_date = date_series.iloc[0].strftime("%Y-%m-%d")
                    end_date = date_series.iloc[-1].strftime("%Y-%m-%d")
                    date_range = f"• Data Range: {start_date} to {end_date}\n"
            except Exception:
                pass
    elif rows_used is None:
        rows_used = "N/A"

    timeframe_label = {
        "intraday": "intraday (derived from file dates)",
        "daily": "daily (derived from file dates)",
        "weekly": "weekly (derived from file dates)",
        "unknown": "unknown (insufficient date spacing)",
    }.get(timeframe, f"{timeframe} (derived from file dates)")

    last_price = context.get("last_price")
    last_date = context.get("last_date")
    price_line = ""
    if last_price is not None:
        price_line = (
            f"• Last Close: {last_price:,} as of {last_date}\n"
            if last_date
            else f"• Last Close: {last_price:,}\n"
        )

    lines = [
        "\nEvidence summary:",
        "• Source: CSV file",
        f"• Timeframe: {timeframe_label}",
    ]

    ticker = context.get("ticker")
    if ticker and ticker not in {"", "custom", "N/A"}:
        lines.append(f"• Ticker: {ticker}")

    if rows_used is not None:
        lines.append(f"• Rows analyzed: {rows_used}")

    if date_range:
        lines.append(date_range.rstrip())

    if price_line:
        lines.append(price_line.rstrip())

    return "\n".join(lines) + "\n"


def build_evidence_summary_for_context(
    context: dict,
    uploaded_df: pd.DataFrame | None = None,
) -> str:
    if context and context.get("data_source") == "uploaded_file":
        return format_uploaded_file_evidence_summary(context, uploaded_df=uploaded_df)
    return format_evidence_summary(context)



def is_data_valid(ctx: dict) -> bool:
    """Return True only when context has real OHLCV-backed indicator data.

    Gate used by both answer() and the streaming path in api_server.py
    before calling the LLM.  Any context that fails this check must be handled
    by _format_data_unavailable() — never sent to the LLM.

    Driven by ACTUAL data fields — NOT the _lightweight flag, which can be
    stale from a prior failed fetch that was cached before AV/TD recovered.

    Failure conditions:
      • rate_limited = True  — AV / TD returned 429
      • last_price is None   — no price anchor; no real OHLCV was processed
      • trend is unknown / unclear / empty — OHLCV failed to resolve direction
    """
    if ctx.get("rate_limited"):
        return False
    if ctx.get("last_price") is None:
        return False
    trend = (ctx.get("trend") or "").lower()
    if trend in ("unknown", "unclear", ""):
        return False
    return True


def build_prompt(user_query: str, fallback_context: dict | None) -> tuple[str, str, dict]:
    """Return (system_prompt, user_message, context).

    Splitting system rules from the user message lets Ollama cache the KV pairs
    of the large system prompt across requests, cutting per-request latency
    from ~2 min to under ~20 s on a warm model.
    """
    # If fallback_context has uploaded file data, use it directly
    if fallback_context and fallback_context.get("data_source") == "uploaded_file":
        context = fallback_context
    else:
        # Fix 4: Purge any stale _lightweight flag from the previous session
        # context before it can bleed into the new fetch.  If AV/TD failed last
        # turn but succeeds this turn, the cached flag must NOT survive into the
        # new context dict and falsely trigger the unavailability gate in answer().
        if fallback_context:
            fallback_context.pop("_lightweight", None)

        # Normal AV → TD pipeline flow — use cache to avoid re-fetching hot data
        interval = detect_timeframe(user_query)

        # ── Period label extraction ───────────────────────────────────────────
        # When the user says "last 6 months" / "past 3 months" etc., we store a
        # human-readable label so _build_evidence_dict() can show e.g.
        # "Last 6 Months (daily bars)" instead of the generic "Daily".
        _PERIOD_LABEL_RE = re.compile(
            r"\b((?:last|past|previous)\s+\d+\s*(?:months?|years?|weeks?))\b",
            re.IGNORECASE,
        )
        _pm = _PERIOD_LABEL_RE.search(user_query)
        _period_label: str | None = _pm.group(1).title() if _pm else None

        instrument_name, ticker = detect_instrument(user_query, fallback_context)
        print(f"[TechnoBot] Resolved ticker: instrument='{instrument_name}' "
              f"ticker='{ticker}' tf='{interval}'")

        # ── Single fetch via AV → TD pipeline ────────────────────────────────
        context = _cached_derive_market_context(interval, ticker)

        # Fix 2: If the pipeline returned a real price anchor, the _lightweight
        # flag is definitively wrong — clear it so the answer() gate never blocks
        # valid data.  Covers the race where a previously-cached lightweight
        # context is returned before AV/TD has had a chance to refresh.
        if context and context.get("last_price") is not None:
            context["_lightweight"] = False

        context["instrument_name"] = instrument_name
        context["ticker"] = ticker

        # Persist period label for evidence display (only meaningful on 1d bars)
        if _period_label and interval == "1d":
            context["_period_label"] = _period_label

        # ── Contradiction validation + participation labelling ────────────────
        # Must run AFTER instrument_name / ticker are set so logs are clear.
        context = _validate_context_consistency(context)

        print(f"[TechnoBot] fetch complete | "
              f"trend='{context.get('trend')}' rsi={context.get('rsi')} "
              f"rsi_state='{context.get('rsi_state')}' "
              f"src={context.get('source') or 'lightweight'}")
        print(f"[TechnoBot] data_gate | "
              f"ticker={ticker} provider={context.get('data_source') or context.get('source') or 'unknown'} "
              f"lightweight={context.get('_lightweight', False)} "
              f"is_valid={is_data_valid(context)}")

    # ── Phase 12: Gate in answer() uses has_data, not _lightweight ───────────
    # Fix 2 above clears _lightweight whenever last_price is set, so by the time
    # answer() reads the context the flag accurately reflects provider state.
    # format_market_context() still reads _lightweight for the qualitative path
    # (used in comparison and edge cases) — do not remove it from that function.

    market_block = format_market_context(context)

    # System prompt = TA mentor rules (long, stable — Ollama will KV-cache this)
    system_prompt = (
        indicator_prompt(user_query)
        if is_indicator_question(user_query)
        else ta_mentor_prompt(user_query)
    )

    # ── Ticker-specific analysis directive ───────────────────────────────────
    # Injected for all live/qualitative paths (NOT uploaded files) so the LLM
    # receives an explicit, numbered snapshot it must anchor every bullet to.
    # This eliminates template-like responses by forcing ticker-fingerprinted output.
    _ticker_directive = ""
    if context.get("data_source") != "uploaded_file":
        _t_name  = context.get("instrument_name") or context.get("ticker", "?")
        _t_sym   = context.get("ticker", "?")
        _tf_raw  = context.get("timeframe") or "1d"
        _tf_lbl  = {"1d": "daily", "1wk": "weekly", "1mo": "monthly",
                    "1h": "1-hour", "5m": "5-minute"}.get(_tf_raw, _tf_raw)
        _trend   = context.get("trend")    or "unknown"
        _mom     = context.get("momentum") or "unknown"
        _vol     = context.get("volatility") or "unknown"
        _rsi_v   = context.get("rsi")
        _rsi_st  = (context.get("rsi_state") or "").replace("_", " ")
        if _rsi_v is not None:
            _rsi_str = f"{_rsi_v:.1f}" + (f" [{_rsi_st}]" if _rsi_st else "")
        else:
            _rsi_str = "unavailable (qualitative mode)"
        _lp      = context.get("last_price")
        _lp_str  = str(_lp) if _lp is not None else "unavailable"

        _ticker_directive = (
            f"\n\n--- TICKER-SPECIFIC ANALYSIS DIRECTIVE ---\n"
            f"Instrument : {_t_name} ({_t_sym})\n"
            f"Timeframe  : {_tf_lbl}\n"
            f"Trend      : {_trend}\n"
            f"RSI(14)    : {_rsi_str}\n"
            f"Volatility : {_vol}\n"
            f"Momentum   : {_mom}\n"
            f"Last close : {_lp_str}\n"
            "---\n"
            "MANDATORY RULES for this specific response:\n"
            f"  1. Name '{_t_name}' explicitly in at least two bullets\n"
            f"  2. Reference RSI {_rsi_str} and explain what it implies for {_t_name}\n"
            f"  3. Explain what the {_trend} trend means for {_t_name}'s structure — "
            "not for stocks in general\n"
            "  4. Every bullet must use the numbers above — no generic placeholders\n"
            "--- END DIRECTIVE ---"
        )

    # ── Query intent directive ────────────────────────────────────────────────
    # Detects advisory (conversational/decision) vs analysis (full breakdown).
    # Advisory mode: 2–3 bullets, direct answer first, no technical dump.
    _intent_mode = _detect_query_intent(user_query)
    context["_query_intent"] = _intent_mode  # expose to downstream (/chat, insight)

    _intent_directive = ""
    if _intent_mode == "advisory":
        _intent_directive = (
            "\n\n--- QUERY INTENT: ADVISORY ---\n"
            "The user asked a conversational or decision-oriented question.\n"
            "They do NOT want a full technical analysis dump.\n"
            "FORMAT RULES (mandatory — override bullet count from other rules):\n"
            "  - Write EXACTLY 2 to 3 bullets — no more, no less\n"
            "  - Bullet 1: direct answer (yes / cautious / not yet / no) + one-line reason\n"
            "  - Bullet 2: one supporting signal — RSI level + trend\n"
            "  - Bullet 3 (optional): one forward-looking implication or key watchpoint\n"
            "  - Conversational, mentor-like tone — answer the question directly\n"
            "  - Do NOT list every indicator — only what answers this specific question\n"
            "--- END INTENT ---"
        )

    # User message = market snapshot + ticker directive + intent directive + user query
    user_message = f"{market_block}{_ticker_directive}{_intent_directive}\n\nUser query: {user_query}"

    return system_prompt, user_message, context


def should_use_uploaded_context(user_query: str) -> bool:
    """Return True when the user's query clearly refers to an uploaded file
    or indicators where uploaded data should take precedence.
    """
    q = (user_query or "").lower()
    file_kw = ["file", "uploaded", "this file", "csv", "export", "rows", "uploaded csv"]
    indicator_kw = ["rsi", "divergence", "sma", "ema", "macd", "indicator", "levels", "confirmation", "invalidation"]
    if any(k in q for k in file_kw + indicator_kw):
        return True
    return False


def is_affirmative_reply(text: str) -> bool:
    t = text.lower().strip()
    return t in {
        "yes", "yeah", "yep", "ok", "okay",
        "sure", "go ahead", "please do", "do it", "lets go"
    }

def normalize_query(q: str) -> str:
    """Fix common misspellings so routing and keyword detection work on typo-heavy input."""
    fixes = {
        # TA concepts
        "voltility":  "volatility",
        "volaility":  "volatility",
        "momemtum":   "momentum",
        "momentun":   "momentum",
        "analaysis":  "analysis",
        "analyisis":  "analysis",
        "anlysis":    "analysis",
        "bullisch":   "bullish",
        "bearisch":   "bearish",
        # timeframes
        "wekly":      "weekly",
        "weeky":      "weekly",
        "montly":     "monthly",
        "monthely":   "monthly",
        "daliy":      "daily",
        "daly":       "daily",
        "daiy":       "daily",
        "intrday":    "intraday",
        "intarday":   "intraday",
        # instruments
        "nfity":      "nifty",
        "nify":       "nifty",
        "nfty":       "nifty",
        "sensext":    "sensex",
        "releiance":  "reliance",
        "reilance":   "reliance",
        "infoys":     "infosys",
        "infosyss":   "infosys",
        "tatasteel":  "tata steel",
        "hdfc bnk":   "hdfc bank",
        # affirmative
        "yea":        "yes",
        "yess":       "yes",
    }
    q = q.lower()
    for k, v in fixes.items():
        q = q.replace(k, v)
    return q

def split_confirmation_and_query(text: str):
    orig = text or ""
    trimmed = orig.strip()

    confirmations = ["yes", "yeah", "yep", "ok", "okay",
        "sure", "go ahead", "please do", "do it", "lets go"]

    for c in confirmations:
        # match case-insensitively but capture remaining from original text to preserve casing
        pattern = r"^\s*" + re.escape(c) + r"(?:[\s,]*(.*))?$"
        m = re.match(pattern, trimmed, flags=re.IGNORECASE)
        if m:
            rem = m.group(1)
            if rem:
                # strip leading connective words like 'and' and punctuation
                rem = re.sub(r"^[\s,]*and\b[\s,]*", "", rem, flags=re.IGNORECASE).strip()
                return c, rem
            return c, None

    return None, orig.strip()


def asks_for_historical_data(query: str) -> bool:
    keywords = ["historical", "past", "last", "period", "from", "to", "data", "export", "csv"]
    q = query.lower()
    return any(k in q for k in keywords)


def asks_for_csv(query: str) -> bool:
    keywords = ["csv", "export", "download", "file"]
    q = query.lower()
    return any(k in q for k in keywords)


def has_uploaded_file(user_input: str) -> str | None:
    import os
    if os.path.isfile(user_input.strip()):
        return user_input.strip()
    return None


FORBIDDEN_REPLACEMENTS = {
    # Financial advice language → structural observation language
    "bullish":              "constructive",
    "bearish":              "weakening",
    "breakout":             "range expansion",
    "entry":                "setup point",
    "target":               "reference level",
    "gain":                 "increase",
    "gains":                "increases",
    "profit":               "return",
    "profits":              "returns",
    "upside":               "upward potential",
    "downside":             "downward risk",
    "strongly":             "notably",
    "significant gains":    "notable increases",
    # Buy/sell language → demand/supply language
    "buying pressure":      "demand pressure",
    "selling pressure":     "supply pressure",
    "buying conviction":    "demand conviction",
    "selling conviction":   "supply conviction",
    "buyers are in control":"demand is in control",
    "sellers are in control":"supply is in control",
    "buy signal":           "demand signal",
    "sell signal":          "supply signal",
}

def sanitize_language(text: str) -> str:
    """Replace sensitive phrasing with neutral alternatives while preserving readability."""
    if not text:
        return text

    out = text
    # Replace longer phrases first
    for phrase in sorted(FORBIDDEN_REPLACEMENTS.keys(), key=len, reverse=True):
        repl = FORBIDDEN_REPLACEMENTS[phrase]
        out = re.sub(rf"\b{re.escape(phrase)}\b", repl, out, flags=re.IGNORECASE)

    return out


# ---------------------------------------------------------
# HISTORICAL QUERY DETECTION
# ---------------------------------------------------------
def is_historical_request(text: str) -> bool:
    """Return True when the user explicitly wants a data table, not an analysis.

    Triggers on explicit data/fetch/show phrasing. Does NOT trigger on timeframe
    words alone ("weekly analysis", "daily trend") — those route to LLM analysis.
    """
    t = text.lower()
    exact_keywords = [
        "historical",
        "history",
        "price data",
        "show data",
        "fetch data",
        "get data",
        "show me data",
        "show prices",
        "raw data",
        "price history",
        "ohlcv",
        "ohlc",
        "past prices",
        "last prices",
    ]
    if any(k in t for k in exact_keywords):
        return True
    # "show me [word] data/prices" or "show [word] data/prices" pattern
    if re.search(r"\bshow\b.{0,20}\b(data|prices)\b", t):
        return True
    # "get me [timeframe/instrument] data"
    if re.search(r"\bget\b.{0,20}\bdata\b", t):
        return True
    return False


# ---------------------------------------------------------
# HISTORICAL RESPONSE HANDLER
# ---------------------------------------------------------
def handle_historical_request(user_query: str, sess: dict, export_now=False):
    """A3: sess is the per-chat session dict — no global state written."""

    def _historical_query_evidence_summary(
        query_text: str,
        instrument_name: str,
        resolved_ticker: str,
        resolved_timeframe: str,
        rows: int | None = None,
        data_frame=None,
        meta_info: dict | None = None,
    ) -> str:
        """Historical-only evidence summary sourced from user query intent.

        This helper is intentionally local to historical flow so it does not
        affect any other evidence-summary architecture.
        """
        q = (query_text or "").lower()

        # Prefer explicit horizon from query, e.g. "last 10 years"
        requested_scope = "recent history"
        m = re.search(
            r"\b(?:last|past|previous|over\s+the\s+last)\s+(\d+)\s+"
            r"(day|days|week|weeks|month|months|year|years)\b",
            q,
        )
        if m:
            n = m.group(1)
            unit = m.group(2)
            requested_scope = f"last {n} {unit}"
        else:
            # Support forms like "fetch 10 years historical data ..."
            m2 = re.search(
                r"\b(\d+)\s+(day|days|week|weeks|month|months|year|years)\b",
                q,
            )
            if m2 and any(k in q for k in ("historical", "history", "data", "prices")):
                requested_scope = f"last {m2.group(1)} {m2.group(2)}"

        # Also support explicit date-range intent in user query.
        from_to = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:$|\?|\.|,)", query_text, flags=re.IGNORECASE)
        if from_to:
            requested_scope = f"from {from_to.group(1).strip()} to {from_to.group(2).strip()}"

        timeframe_hint = re.search(
            r"\b(intraday|hourly|daily|weekly|monthly|1d|1wk|1mo|1h|5m)\b",
            q,
        )
        if timeframe_hint:
            timeframe_label = timeframe_hint.group(1)
        else:
            timeframe_label = {
                "1d": "daily",
                "1wk": "weekly",
                "1mo": "monthly",
                "1h": "hourly",
                "5m": "5-minute",
            }.get(resolved_timeframe, resolved_timeframe)

        # Prefer actual returned data span when available.
        coverage_line = None
        if data_frame is not None and not data_frame.empty and "Date" in data_frame.columns:
            try:
                date_series = pd.to_datetime(data_frame["Date"], errors="coerce").dropna()
                if not date_series.empty:
                    start_date = date_series.iloc[0].strftime("%Y-%m-%d")
                    end_date = date_series.iloc[-1].strftime("%Y-%m-%d")
                    coverage_line = f"• Data coverage: {start_date} to {end_date}"
            except Exception:
                coverage_line = None

        if rows is None and meta_info:
            rows = meta_info.get("rows")

        if requested_scope == "recent history" and meta_info and meta_info.get("window_label"):
            requested_scope = str(meta_info.get("window_label"))

        yahoo_link = f"https://finance.yahoo.com/quote/{resolved_ticker}"

        lines = [
            "\nEvidence summary:",
            "• Source: Yahoo Finance",
            f"• Instrument: {instrument_name}",
            f"• Ticker: {resolved_ticker}",
            f"• Requested scope: {requested_scope}",
            f"• Timeframe: {timeframe_label}",
            f"• Link: {yahoo_link}",
        ]
        if rows is not None:
            lines.append(f"• Rows: {rows}")
        if coverage_line:
            lines.append(coverage_line)
        return "\n".join(lines) + "\n"

    instrument, ticker = detect_instrument(user_query)
    timeframe = detect_timeframe(user_query) or "1d"

    # try live fetch, otherwise fallback to local cached CSVs
    try:
        df, meta = fetch_historical_data(ticker, timeframe)
    except Exception as e:
        import glob
        import pandas as pd

        exports_dir = os.path.join(os.path.dirname(__file__), "technobot_data", "exports")
        pattern = f"*{ticker}*{timeframe}*.csv"
        matches = sorted(glob.glob(os.path.join(exports_dir, pattern)))
        if matches:
            path = matches[-1]
            try:
                df = pd.read_csv(path)
                meta = {
                    "window_label": f"cached {timeframe}",
                    "rows": len(df),
                    "reference": path,
                }
            except Exception as read_err:
                return f"Failed to read cached file {path}: {read_err}"
        else:
            evidence_fail = _historical_query_evidence_summary(
                user_query,
                instrument,
                ticker,
                timeframe,
                rows=None,
                data_frame=None,
                meta_info=None,
            )
            return (
                f"Failed to fetch historical data for {ticker}: {e}. "
                "No cached CSV fallback found."
                + evidence_fail
            )

    # If user asked to export now, just export and return the path line
    if export_now:
        path = export_csv(df, ticker, timeframe)
        # D8: log export to session history
        sess["export_history"].append({
            "filename": os.path.basename(path),
            "ticker": ticker,
            "instrument": instrument,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
        })
        return f"📁 CSV exported to: {path}"

    head, tail = snapshot_head_tail(df, n=10)

    # -------- Interpretation summary (expanded) --------
    try:
        first_date = df["Date"].iloc[0]
        last_date = df["Date"].iloc[-1]
        first_close = float(df["Close"].iloc[0])
        last_close = float(df["Close"].iloc[-1])
        pct_change = (last_close - first_close) / first_close * 100.0
        trend_desc = "shows a broadly rising structure" if last_close > first_close else "shows a broadly weakening structure"
        extra = (
            f"Between {first_date} and {last_date} the close moved from {first_close:.2f} to {last_close:.2f} "
            f"(~{pct_change:.1f}% change). Over this span we see multi-year up/down phases, "
            "periods of consolidation, and episodes of volatility expansion that define the larger structure."
        )
    except Exception:
        trend_desc = "shows an unclear structure"
        extra = "Detailed span statistics unavailable."

    interpretation = (
        f"\n📈 Historical structure summary:\n"
        f"The {timeframe} data {trend_desc}, with periods of consolidation and volatility expansion visible over time.\n"
        f"{extra}\n"
    )

    response = []
    response.append(f"📊 Historical price data snapshot for {instrument}\n")
    response.append("First 10 rows:")
    response.append(head.to_string(index=False))
    response.append("\nLast 10 rows:")
    response.append(tail.to_string(index=False))

    response.append(interpretation)

    evidence = _historical_query_evidence_summary(
        user_query,
        instrument,
        ticker,
        timeframe,
        rows=meta.get("rows"),
        data_frame=df,
        meta_info=meta,
    )

    response.append(evidence)

    # A3: store in session instead of global
    sess["last_historical"] = {
        "df": df,
        "ticker": ticker,
        "instrument": instrument,
        "timeframe": timeframe,
    }

    response.append(
        "\nWould you like me to export this historical data as a CSV file?"
    )

    # Emit chart sentinel so frontend renders a price chart for this ticker
    chart_label = meta.get("window_label") or timeframe
    response.append(f"\n__CHART__{ticker}:{chart_label}")

    return "\n".join(response)


# ---------------------------------------------------------
# MAIN ANSWER FUNCTION
# ---------------------------------------------------------
def decompose_paragraph_query(text: str) -> list[str]:
    """F2: Split a paragraph with multiple embedded questions into sub-queries.

    Triggers when the input has more than one '?'.  The word-count threshold
    was removed (was 40 words) because realistic multi-question paragraphs
    can be 25-35 words and were not being split.
    Returns the original text unchanged for single-question inputs.
    """
    question_count = text.count("?")
    if question_count <= 1:
        return [text]

    # Split on sentence boundaries: period/question mark followed by space+capital
    parts = re.split(r"(?<=[.?])\s+(?=[A-Z])", text.strip())
    # Filter out empty strings and very short fragments (< 4 words)
    parts = [p.strip() for p in parts if len(p.split()) >= 4]
    return parts if len(parts) > 1 else [text]


def detect_comparison_query(text: str):
    """Return (name1, ticker1, name2, ticker2, timeframe) if query compares two instruments, else None.

    Phase 2: expanded trigger patterns beyond just 'vs' and 'compare'.
    Phase 3: two-pass instrument resolution so raw-ticker queries like
             "RELIANCE.NS vs TATASTEEL.NS" are detected in addition to the
             name-based form "Reliance vs Tata Steel".
    """
    q = text.lower()
    is_vs          = bool(re.search(r"\bvs\.?\b|\bversus\b", q))
    is_cmp         = bool(re.search(r"\bcompar\w*\b", q))
    is_diff        = bool(re.search(r"\bdifference\s+between\b", q))
    is_against     = bool(re.search(r"\bagainst\b", q))
    is_compared_to = bool(re.search(r"\bcompared\s+to\b", q))
    is_stack       = bool(re.search(r"\bstack\s+up\b", q))
    is_better      = bool(re.search(r"\bwhich\s+is\s+(better|stronger|weaker)\b", q))

    if not any([is_vs, is_cmp, is_diff, is_against, is_compared_to, is_stack, is_better]):
        return None

    # ── Pass 1: instrument name matching ──────────────────────────────────────
    # Sort by match position so order reflects user's phrasing.
    found: list[tuple[str, str]] = []
    seen_tickers: set[str] = set()
    name_positions: list[tuple[int, str, str]] = []
    for name, ticker in INSTRUMENT_MAP.items():
        idx = q.find(name)
        if idx != -1 and ticker not in seen_tickers:
            name_positions.append((idx, name.title(), ticker))
            seen_tickers.add(ticker)
    name_positions.sort(key=lambda x: x[0])
    found = [(n, t) for _, n, t in name_positions]

    # ── Pass 2: raw ticker token matching ─────────────────────────────────────
    # Handles queries like "Compare RELIANCE.NS vs TATASTEEL.NS" where the user
    # types the Yahoo Finance ticker directly instead of the instrument name.
    # "tata steel" is never found inside "tatasteel.ns" (no space), so Pass 1
    # misses it.  Pass 2 extracts uppercase token groups and reverse-maps them.
    if len(found) < 2:
        # Build reverse map: "RELIANCE.NS" → "Reliance", etc.
        _ticker_to_name: dict[str, str] = {}
        for name, ticker in INSTRUMENT_MAP.items():
            if ticker not in _ticker_to_name:          # keep first (most specific) name
                _ticker_to_name[ticker] = name.title()

        # Extract tokens that look like tickers: 2+ uppercase letters, optional .XX suffix
        # Use the original text (case-preserved) so "RELIANCE.NS" stays uppercase.
        raw_tokens = re.findall(r'\b([A-Z]{2,}(?:\.[A-Z]{1,4})?)\b', text)
        for tok in raw_tokens:
            if tok in seen_tickers:
                continue
            # Direct match (ticker already has exchange suffix, e.g. RELIANCE.NS)
            if tok in _ticker_to_name:
                found.append((_ticker_to_name[tok], tok))
                seen_tickers.add(tok)
            else:
                # Try appending common exchange suffixes
                for sfx in (".NS", ".BO"):
                    candidate = tok + sfx
                    if candidate in _ticker_to_name and candidate not in seen_tickers:
                        found.append((_ticker_to_name[candidate], candidate))
                        seen_tickers.add(candidate)
                        break

    if len(found) < 2:
        # ── Partial-match error: comparison intent clear, but ≤1 instrument resolved ─
        # Look for uppercase tokens in the original text that weren't matched to any
        # known instrument.  These are likely the user's intended (but unrecognised)
        # tickers.  Common TA / stop-word caps are excluded so "RSI", "VS", "EMA"
        # don't show up as spurious unrecognised tickers.
        _UPPER_STOP = {
            "VS", "IN", "ON", "AT", "OF", "TO", "BY", "BE", "IS", "ARE",
            "AND", "FOR", "THE", "ALL", "RSI", "EMA", "SMA", "MACD", "ATR",
            "PE", "PB", "ATH", "ATL", "NSE", "BSE", "IPO", "OI", "CE", "PE",
        }
        resolved_tickers: set[str] = {t for _, t in found}
        raw_upper = re.findall(r'\b([A-Z]{2,}(?:\.[A-Z]{1,4})?)\b', text)
        unrecognized: list[str] = []
        _seen_unrecog: set[str] = set()
        for tok in raw_upper:
            if tok in _UPPER_STOP or tok in _seen_unrecog:
                continue
            # Skip if this token IS a known ticker (just already resolved)
            if tok in resolved_tickers or tok + ".NS" in resolved_tickers or tok + ".BO" in resolved_tickers:
                continue
            # Skip if it directly maps to a known instrument (case-insensitive key lookup)
            if tok.lower() in INSTRUMENT_MAP:
                continue
            unrecognized.append(tok)
            _seen_unrecog.add(tok)

        if unrecognized:
            unrecog_str = ", ".join(f"**{u}**" for u in unrecognized)
            if found:
                found_name = found[0][0]
                _msg = (
                    f"I understood you want to compare **{found_name}** against another "
                    f"instrument, but I couldn't recognise {unrecog_str} as a supported "
                    f"symbol.\n\n"
                    f"Please use a name from my supported list — for example:\n"
                    f"*\"Compare {found_name} vs TCS\"* or *\"Compare {found_name} vs BankNifty\"*"
                )
            else:
                _msg = (
                    f"I noticed a comparison request, but couldn't recognise "
                    f"{unrecog_str} as supported instrument(s).\n\n"
                    f"Try: *\"Compare Reliance vs TCS\"* or *\"Nifty vs BankNifty on weekly\"*"
                )
            return {"error": "unrecognized_comparison", "message": _msg}

        # No suspicious uppercase tokens → ambiguous query (e.g. "compare daily vs weekly"),
        # fall through to normal single-instrument routing.
        return None

    timeframe = detect_timeframe(text)
    inst1, t1 = found[0]
    inst2, t2 = found[1]
    print(f"[TechnoBot/comparison] Detected: '{inst1}' ({t1}) vs '{inst2}' ({t2}) tf={timeframe}")
    return inst1, t1, inst2, t2, timeframe


def _parse_comparison_rows(raw: str) -> list[dict]:
    """Extract pipe-separated table rows from LLM output."""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    table_rows = [ln for ln in lines if "|" in ln and not ln.endswith("?")]
    rows = []
    for row in table_rows:
        parts = [p.strip() for p in row.split("|")]
        if len(parts) >= 3:
            aspect = re.sub(r"^[-*•]\s*", "", parts[0]).strip()
            # Skip header-like separators (---|---|---)
            if re.match(r"^[-:]+$", aspect.replace(" ", "")):
                continue
            rows.append({"aspect": aspect, "a": parts[1], "b": parts[2]})
    return rows


def _bullet_comparison_fallback(
    inst1: str, ctx1: dict,
    inst2: str, ctx2: dict,
    timeframe: str, sess: dict,
) -> str:
    """Comparison fallback: always emits the __COMPARISON__ JSON sentinel so the
    frontend renders a proper table even when the LLM ignores the pipe format.

    Previously returned plain bullets — that caused plain-text rendering instead
    of the structured table. Now matches the same output contract as
    handle_comparison_request so the frontend ComparisonBlock always fires.
    """
    import json as _json

    _TF_LABEL = {"1d": "daily", "1wk": "weekly", "1mo": "monthly",
                 "1h": "hourly", "5m": "5-min"}
    tf_label = _TF_LABEL.get(timeframe, timeframe)

    def _cell(ctx: dict, key: str) -> str:
        val = (ctx.get(key) or "unclear").title()
        return val

    rsi1 = ctx1.get("rsi")
    rsi2 = ctx2.get("rsi")

    rows = [
        {"aspect": "Trend structure",    "a": _cell(ctx1, "trend"),         "b": _cell(ctx2, "trend")},
        {"aspect": "Momentum",           "a": _cell(ctx1, "momentum"),      "b": _cell(ctx2, "momentum")},
        {"aspect": "Volatility",         "a": _cell(ctx1, "volatility"),    "b": _cell(ctx2, "volatility")},
        {"aspect": "Participation",      "a": _cell(ctx1, "participation"), "b": _cell(ctx2, "participation")},
        {"aspect": "RSI (14)",           "a": f"{rsi1:.1f}" if rsi1 else "N/A",
                                          "b": f"{rsi2:.1f}" if rsi2 else "N/A"},
        {"aspect": "Momentum bias",
         "a": "Demand-led" if (ctx1.get("trend") == "up") else ("Supply-led" if ctx1.get("trend") == "down" else "Neutral"),
         "b": "Demand-led" if (ctx2.get("trend") == "up") else ("Supply-led" if ctx2.get("trend") == "down" else "Neutral")},
    ]

    payload = _json.dumps({
        "inst1": inst1, "inst2": inst2,
        "timeframe": tf_label,
        "ticker1": ctx1.get("ticker", ""),
        "ticker2": ctx2.get("ticker", ""),
        "rows": rows,
    }, separators=(",", ":"))

    rel_insight = generate_comparison_insight(inst1, ctx1, inst2, ctx2)
    followup = generate_followup_from_last_bullet(
        " ".join(r["a"] for r in rows), f"{inst1} vs {inst2}", ctx1
    )
    sess["last_context"] = ctx1
    sess["last_followup_intent"] = followup

    result = f"__COMPARISON__{payload}"
    if rel_insight:
        result += f"\n\n{rel_insight}"
    if followup:
        result += f"\n\nFollow-up: {followup}"
    return result


def handle_comparison_request(
    inst1: str, t1: str,
    inst2: str, t2: str,
    timeframe: str,
    sess: dict,
) -> str:
    """Fetch context for both instruments (in parallel) and generate a comparison table.

    Phase 3: Parallel data fetch cuts latency by ~50% versus sequential calls.
    Phase 9: Retry at temperature 0.05 if first call returns no pipe rows.
             Falls back to structured bullets if retry also fails.
    """
    # ── Parallel fetch — both tickers fetched simultaneously ─────────────────
    # Using ThreadPoolExecutor eliminates the sequential 1 s gap, cutting
    # comparison latency by ~50 %.  safe_fetch / _cached_derive_market_context
    # are both thread-safe (all shared state is guarded by locks).
    # If the global Yahoo 429 cooldown is already active when this function is
    # called, nifty_fetcher will route both fetches through AV → TD → stale-cache
    # automatically — no special handling needed here.
    import concurrent.futures as _futures

    _EMPTY_CTX = {"trend": "unknown", "momentum": "unknown", "volatility": "unknown",
                  "participation": "unknown", "rsi": None, "rate_limited": False}

    def _safe_ctx(tf: str, ticker: str) -> dict:
        try:
            return _cached_derive_market_context(tf, ticker)
        except Exception:
            return dict(_EMPTY_CTX)

    print(f"[TechnoBot] Comparison: fetching {t1} and {t2} in parallel")
    with _futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="cmp") as _pool:
        _f1 = _pool.submit(_safe_ctx, timeframe, t1)
        _f2 = _pool.submit(_safe_ctx, timeframe, t2)
        try:
            ctx1 = _f1.result(timeout=10)
        except Exception:
            ctx1 = dict(_EMPTY_CTX)
        try:
            ctx2 = _f2.result(timeout=10)
        except Exception:
            ctx2 = dict(_EMPTY_CTX)

    print(f"[TechnoBot] Comparison fetch complete: "
          f"{t1}(src={ctx1.get('source') or 'lightweight'}) "
          f"vs {t2}(src={ctx2.get('source') or 'lightweight'})")

    # ── Data validation: rate-limit and unknown checks ───────────────────────
    t1_rl = ctx1.get("rate_limited", False)
    t2_rl = ctx2.get("rate_limited", False)
    either_rate_limited = t1_rl or t2_rl
    t1_unknown = ctx1.get("trend") == "unknown"
    t2_unknown = ctx2.get("trend") == "unknown"
    both_unknown = t1_unknown and t2_unknown
    one_unknown  = (t1_unknown or t2_unknown) and not both_unknown

    if either_rate_limited:
        # Identify which ticker(s) are rate-limited so we can be specific
        if t1_rl and not t2_rl and not t2_unknown:
            # ticker2 has valid data — show partial result
            print(f"[TechnoBot] Comparison: {inst1} rate-limited, {inst2} OK — partial")
            ctx1["instrument_name"] = inst1;  ctx1["ticker"] = t1
            ctx2["instrument_name"] = inst2;  ctx2["ticker"] = t2
            partial_note = (
                f"\n\n> ⚠️ **{inst1}** data is temporarily unavailable (Yahoo rate limit). "
                f"Only **{inst2}** data is shown."
            )
            return _bullet_comparison_fallback(inst1, ctx1, inst2, ctx2, timeframe, sess) + partial_note
        if t2_rl and not t1_rl and not t1_unknown:
            # ticker1 has valid data — show partial result
            print(f"[TechnoBot] Comparison: {inst1} OK, {inst2} rate-limited — partial")
            ctx1["instrument_name"] = inst1;  ctx1["ticker"] = t1
            ctx2["instrument_name"] = inst2;  ctx2["ticker"] = t2
            partial_note = (
                f"\n\n> ⚠️ **{inst2}** data is temporarily unavailable (Yahoo rate limit). "
                f"Only **{inst1}** data is shown."
            )
            return _bullet_comparison_fallback(inst1, ctx1, inst2, ctx2, timeframe, sess) + partial_note
        # Both rate-limited (or neither has data) — full fallback message
        return (
            f"⚠️ Live market data for **{inst1}** and/or **{inst2}** is temporarily "
            "unavailable due to Yahoo Finance rate limiting (HTTP 429).\n\n"
            "**Please wait 3–5 minutes and try again.**\n\n"
            "In the meantime:\n"
            "  • Ask about each stock separately once data is available\n"
            "  • Upload CSV files for offline comparison"
        )

    if both_unknown:
        return (
            f"⚠️ Live data for both **{inst1}** and **{inst2}** could not be retrieved "
            f"on the {timeframe} timeframe.\n\n"
            "This usually means Yahoo Finance is temporarily throttling requests.\n\n"
            "Suggested alternatives:\n"
            "  • Try again in a minute\n"
            "  • Ask about each instrument separately: 'Analyse Reliance on daily'\n"
            "  • Upload CSV files for offline comparison"
        )

    if one_unknown:
        # One ticker has real data — still produce a table using the fallback
        # (partial cells will show 'Unclear' for the missing instrument).
        # This is better UX than refusing entirely.
        failed  = inst1 if t1_unknown else inst2
        working = inst2 if t1_unknown else inst1
        print(f"[TechnoBot] Partial comparison: {failed} data unavailable — "
              f"using fallback table with {working} data only")
        ctx1["instrument_name"] = inst1
        ctx1["ticker"] = t1
        ctx2["instrument_name"] = inst2
        ctx2["ticker"] = t2
        partial_note = (
            f"\n\n> ⚠️ **{failed}** data could not be retrieved right now. "
            f"The table above shows **{working}** data only. "
            f"Try asking about {failed} separately."
        )
        return _bullet_comparison_fallback(inst1, ctx1, inst2, ctx2, timeframe, sess) + partial_note

    # Enrich context with instrument names for evidence / follow-ups
    ctx1["instrument_name"] = inst1
    ctx1["ticker"] = t1
    ctx2["instrument_name"] = inst2
    ctx2["ticker"] = t2

    # Run contradiction validation on both contexts before LLM call
    ctx1 = _validate_context_consistency(ctx1)
    ctx2 = _validate_context_consistency(ctx2)

    def ctx_summary(name: str, ctx: dict) -> str:
        if ctx.get("_lightweight"):
            note = ctx.get("_qualitative_note", "")
            return (
                f"{name}: [live data unavailable — qualitative reasoning only]\n"
                f"  {note}"
            )
        rsi_val   = ctx.get("rsi")
        rsi_state = ctx.get("rsi_state", "")
        # Comparison queries get rsi_state label only (no note) — keeps table cells concise
        rsi_display = (
            f"{rsi_val:.1f} [{rsi_state.replace('_',' ')}]"
            if rsi_val is not None and rsi_state and rsi_state != "n/a"
            else (f"{rsi_val:.1f}" if rsi_val is not None else "n/a")
        )
        part_label = ctx.get("participation_label") or ctx.get("participation", "?")
        divergence_note = (
            " ⚠[RSI/trend divergence detected]"
            if ctx.get("_rsi_divergence") else ""
        )
        return (
            f"{name}: trend={ctx.get('trend','?')}, "
            f"momentum={ctx.get('momentum','?')}, "
            f"volatility={ctx.get('volatility','?')}, "
            f"participation={part_label}, "
            f"rsi={rsi_display}{divergence_note}"
        )

    # Determine if either side is lightweight (no live data)
    _lw1 = ctx1.get("_lightweight", False)
    _lw2 = ctx2.get("_lightweight", False)
    _live_label = "Live" if not (_lw1 or _lw2) else "Qualitative"
    _disclaimer  = (
        "\n⚠️ Note: live market data unavailable for one or both instruments. "
        "Analysis is based on general technical reasoning only."
        if (_lw1 or _lw2) else ""
    )

    sys_p = comparison_prompt(inst1, inst2, timeframe)
    usr_m = (
        f"{_live_label} {timeframe} context:{_disclaimer}\n"
        f"{ctx_summary(inst1, ctx1)}\n"
        f"{ctx_summary(inst2, ctx2)}\n\n"
        f"Compare {inst1} vs {inst2} on {timeframe}."
    )

    # ── Single LLM call via Groq (no retry — Groq is reliable at 1–3 s) ────────
    _cmp_t0 = time.monotonic()
    print(f"[LLM START] comparison {inst1} vs {inst2} tf={timeframe}")
    _cmp_msgs = _build_groq_messages(sys_p, usr_m)
    _cmp_resp = call_groq(_cmp_msgs, temperature=0.1, timeout=_REQUEST_BUDGET_S)
    print(f"[LLM END] comparison duration={time.monotonic() - _cmp_t0:.2f}s")
    if _cmp_resp.get("error"):
        return _format_llm_error(_cmp_resp)
    raw = _cmp_resp["content"]

    rows = _parse_comparison_rows(raw)

    # ── Structured bullet fallback if Groq output is unparseable ─────────────
    if not rows:
        return _bullet_comparison_fallback(inst1, ctx1, inst2, ctx2, timeframe, sess)

    # ── Extract follow-up from the model's own output (last ? line) ──────────
    raw_followup = _extract_model_followup(raw)
    # Filter out prediction-based follow-ups
    _PREDICTION_PHRASES = ["probability", "will happen", "forecast", "target", "predict"]
    if any(p in (raw_followup or "").lower() for p in _PREDICTION_PHRASES):
        raw_followup = ""
    followup = _shorten_followup_question(raw_followup) or generate_followup_from_last_bullet(
        "\n".join(r["a"] for r in rows), f"{inst1} vs {inst2}", ctx1
    )

    import json as _json
    payload = _json.dumps({
        "inst1": inst1, "inst2": inst2,
        "timeframe": timeframe,
        "ticker1": t1,  "ticker2": t2,
        "rows": rows,
    }, separators=(",", ":"))

    sess["last_context"] = ctx1
    sess["last_followup_intent"] = followup

    # Relative Insight — deterministic comparative reasoning from the two context dicts
    rel_insight = generate_comparison_insight(inst1, ctx1, inst2, ctx2)
    # Store for /chat structured response insight field
    sess["_last_comparison_insight"] = rel_insight
    sess["_last_followup"] = followup

    result = f"__COMPARISON__{payload}"
    if rel_insight:
        result += f"\n\n{rel_insight}"
    if followup:
        result += f"\n\n{followup}"
    return result


_OOT_RESPONSE = (
    "TechnoBot is a technical analysis assistant. "
    "Try asking about a specific instrument (e.g. 'How is Reliance trending on weekly?') "
    "or a TA concept (e.g. 'What does RSI divergence mean for Infosys?')."
)

# ── Data unavailability response builder ──────────────────────────────────────

def _format_data_unavailable(context: dict) -> str:
    """Return a structured data-unavailability response that bypasses the LLM.

    Exactly 3 cases — selected by failure mode, not a ticker-lookup heuristic:

      Case A — rate_limited or api_failed:
        Provider returned HTTP 429 or explicit API error.
        Message: temporarily rate-limited, wait and retry.

      Case B — known supported ticker, all providers returned no data:
        Ticker IS in INSTRUMENT_MAP (officially supported) but every live
        provider failed to return OHLCV (transient outage, quota exhaustion,
        or network interruption).
        Message: known ticker, service temporarily down, retry or upload CSV.

      Case C — unknown / unsupported ticker:
        Ticker is NOT in INSTRUMENT_MAP — wrong format, delisted, or outside
        the supported universe entirely.
        Message: unsupported ticker, suggest supported alternatives.

    Compliance (Rule 3):
      ✓ Never calls the LLM             ✓ Always includes Reference link
      ✓ Never fabricates OHLCV values   ✓ Response is fully structured
      ✓ Tells user what to do next      ✓ Explains the actual failure reason
    """
    ticker   = context.get("ticker") or "N/A"
    instr    = context.get("instrument_name") or ticker
    tf_raw   = context.get("timeframe", "")
    tf_label = {
        "1d": "daily", "1wk": "weekly", "1mo": "monthly",
        "1h": "hourly", "5m": "5-minute",
    }.get(tf_raw, tf_raw or "selected timeframe")
    ref = f"https://finance.yahoo.com/quote/{ticker}"

    _known_tickers: set[str] = set(INSTRUMENT_MAP.values())
    is_known = ticker in _known_tickers

    # ── Deterministic failure-reason derivation ───────────────────────────────
    reason = (
        "rate_limited"   if context.get("rate_limited")   else
        "api_failed"     if context.get("api_failed")     else
        "fetch_failed"   if context.get("fetch_failed")   else
        "lightweight"    if context.get("_lightweight")   else
        "unknown_ticker" if not is_known                  else
        "provider_down"
    )
    print(
        f"[TechnoBot] DATA INVALID → ticker={ticker} reason={reason} "
        f"tf={tf_raw or 'unknown'} is_known={is_known} "
        f"lightweight={context.get('_lightweight')} "
        f"rate_limited={context.get('rate_limited')}"
    )

    # ── Case A: Rate-limited or explicit API failure ──────────────────────────
    if context.get("rate_limited") or context.get("api_failed"):
        return (
            f"**{instr}** · {tf_label} timeframe\n\n"
            "---\n"
            "**• Data Status:** Market data temporarily rate-limited\n\n"
            "**• Source:** Alpha Vantage / TwelveData API\n\n"
            "**• Explanation:** The data provider has issued a temporary rate-limit "
            "(HTTP 429). This is a transient restriction — no data has been lost "
            "and the service will recover automatically within a few minutes.\n\n"
            "**• Structural Guidance:** No technical analysis generated — "
            "fabricating indicators from missing data would be misleading.\n\n"
            "**• What you can do:**\n"
            "  — Wait 3–5 minutes and retry\n"
            "  — Upload a CSV file for offline analysis (drag and drop)\n"
            "  — Ask about TA concepts while data recovers\n\n"
            f"**• Reference:** [{ticker}]({ref})"
        )

    # ── Case B: Known supported ticker — providers temporarily down ───────────
    if is_known:
        return (
            f"**{instr}** · {tf_label} timeframe\n\n"
            "---\n"
            "**• Data Status:** Market data service temporarily unavailable\n\n"
            "**• Source:** Alpha Vantage / TwelveData API\n\n"
            f"**• Explanation:** **{instr}** is a supported instrument, but live "
            f"OHLCV data could not be retrieved from either provider right now. "
            "This is likely a transient outage or network interruption — "
            "the ticker and symbol mapping are correct.\n\n"
            "**• Structural Guidance:** No technical analysis generated — "
            "fabricating indicators from missing data would be misleading.\n\n"
            "**• What you can do:**\n"
            "  — Retry in 1–2 minutes (most provider outages resolve quickly)\n"
            "  — Check your internet connection\n"
            "  — Upload a CSV file for offline analysis (drag and drop a price file)\n"
            "  — Ask TechnoBot about TA concepts while data recovers\n\n"
            f"**• Reference:** [{ticker}]({ref})"
        )

    # ── Case C: Unknown / unsupported ticker ─────────────────────────────────
    return (
        f"**{instr}** · {tf_label} timeframe\n\n"
        "---\n"
        "**• Data Status:** Instrument not recognised or unsupported\n\n"
        "**• Source:** Alpha Vantage / TwelveData (free-tier coverage)\n\n"
        f"**• Explanation:** The ticker **{ticker}** could not be matched to a "
        "supported instrument. The symbol may be outside the free-tier coverage, "
        "incorrectly formatted, or not traded on NSE/BSE.\n\n"
        "**• Structural Guidance:** No technical analysis generated — "
        "verified market data is required before any indicator conclusions.\n\n"
        "**• What you can do:**\n"
        "  — Try a supported instrument: Nifty 50, Reliance, TCS, HDFC Bank, Infosys\n"
        "  — Refine the ticker format (e.g. 'RELIANCE.NS', 'TCS.NS', '^NSEI')\n"
        "  — Upload a CSV file for offline analysis\n"
        "  — Verify the instrument exists at the reference link below\n\n"
        f"**• Reference:** [{ticker}]({ref})"
    )


def answer(user_query: str, chat_id: str = "default") -> str:
    """A3: chat_id scopes all session state so concurrent chats don't interfere."""
    sess = get_session(chat_id)
    _req_t0 = time.monotonic()   # per-request LLM budget clock
    uploaded_df = sess.get("last_uploaded_df")

    # Default response type — overridden at structured return paths.
    # /chat reads this to decide how to assemble the structured response.
    sess["_last_response_type"] = "text"

    # Greeting shortcut — return immediately, no analysis, no evidence block
    if is_greeting(user_query):
        return greeting_response(None)

    # File upload path — always handle before any text-based checks
    if os.path.exists(user_query.strip()):
        return parse_uploaded_file(user_query.strip(), sess=sess)

    # Normalise typos before any keyword-based checks
    normalized_query = normalize_query(user_query)

    # Phase 1: Off-topic guard — must come BEFORE instrument detection
    # Affirmative replies ("yes", "ok") are follow-up continuations, not OOT
    if not is_affirmative_reply(user_query.strip()) and not is_ta_query(normalized_query):
        return _OOT_RESPONSE

    # Comparison query — detect before paragraph decomposition so "X vs Y" isn't split
    comparison = detect_comparison_query(user_query)
    if isinstance(comparison, dict) and comparison.get("error") == "unrecognized_comparison":
        # Comparison intent was clear but one instrument was unrecognised — tell the user.
        return comparison["message"]
    if comparison:
        inst1, t1, inst2, t2, timeframe = comparison
        sess["_last_response_type"] = "comparison"
        return handle_comparison_request(inst1, t1, inst2, t2, timeframe, sess)

    # F2: detect and decompose paragraph queries with multiple questions
    sub_queries = decompose_paragraph_query(user_query)
    if len(sub_queries) > 1:
        responses = []
        for i, sub in enumerate(sub_queries, 1):
            section_response = answer(sub, chat_id=chat_id)
            responses.append(f"[{i}] {section_response}")
        return "\n\n".join(responses)

    # D4: Instrument disambiguation — detect ambiguous short names before routing
    q_lower = user_query.lower().strip()
    for ambig_term, options in AMBIGUOUS_INSTRUMENT_MAP.items():
        if len(options) > 1 and re.search(rf"\b{re.escape(ambig_term)}\b", q_lower):
            # Only disambiguate if no more specific instrument name is present
            specific_match = any(
                name in q_lower for name in INSTRUMENT_MAP if ambig_term in name and name != ambig_term
            )
            if not specific_match:
                options_text = "\n".join(f"  {i+1}. {o}" for i, o in enumerate(options))
                return (
                    f"I noticed you mentioned '{ambig_term.title()}'. "
                    f"There are several instruments under this name:\n\n"
                    f"{options_text}\n\n"
                    f"Could you specify which one you'd like to analyse? "
                    f"For example: 'Analyse Tata Steel on daily' or 'How is Tata Motors doing weekly?'"
                )

    # Phase 1: Unknown instrument detection
    # If user query mentions a TA timeframe/concept but no recognisable instrument,
    # AND no ambiguous term was caught above, gently ask for clarification.
    # Only trigger on queries that look like they expect a specific instrument.
    _INSTRUMENT_INTENT_MARKERS = re.compile(
        r"\b(analys\w+|trend|trending|chart|how\s+is|how\s+does|show\s+me|what\s+is)\b",
        re.IGNORECASE,
    )
    if _INSTRUMENT_INTENT_MARKERS.search(user_query):
        detected_name, detected_ticker = detect_instrument(user_query)
        # If we defaulted to NIFTY 50 but the user's query doesn't mention nifty/sensex/market
        if detected_ticker == "^NSEI" and not re.search(
            r"\b(nifty|sensex|market|index|nsei)\b", user_query.lower()
        ):
            # Try a fuzzy match against known instrument names
            q_lower2 = user_query.lower()
            # Improved fuzzy match: check multi-token overlap, deduplicate by ticker
            q_tokens = [tok for tok in q_lower2.split() if len(tok) > 3]
            seen_tickers_fuzzy: set[str] = set()
            close_matches = []
            for name, ticker in INSTRUMENT_MAP.items():
                if name in ("nifty", "nifty 50", "sensex", "market"):
                    continue
                if ticker in seen_tickers_fuzzy:
                    continue
                if any(tok in name for tok in q_tokens):
                    seen_tickers_fuzzy.add(ticker)
                    close_matches.append(name.title())
            if close_matches:
                top = close_matches[:3]
                suggestions = "\n".join(f"  • {m}" for m in top)
                return (
                    f"I noticed you're asking about a specific instrument, "
                    f"but couldn't confirm which one. Did you mean one of these?\n\n"
                    f"{suggestions}\n\n"
                    f"Try: 'Analyse {top[0]} on daily' or 'How is {top[0]} trending on weekly?'"
                )
            else:
                return (
                    "I couldn't identify the instrument in your query.\n\n"
                    "TechnoBot covers 60+ Indian stocks and indices. "
                    "Try naming it explicitly — for example:\n"
                    "  • 'How is Infosys trending on daily?'\n"
                    "  • 'Analyse Reliance on weekly'\n"
                    "  • 'What does RSI show for TCS?'"
                )

    confirm, remaining = split_confirmation_and_query(user_query)

    # -------- YES → export historical --------
    if confirm and sess["last_historical"]:
        export_result = handle_historical_request(
            sess["last_historical"]["instrument"],
            sess=sess,
            export_now=True,
        )
        if remaining:
            return answer(remaining, chat_id=chat_id)
        return export_result

    # (File upload path handled at top of answer() before OOT guard)

    # -------- Historical mode --------
    if is_historical_request(user_query):
        return handle_historical_request(user_query, sess=sess)

    # -------- Follow-up branching (fixed) --------
    responses = []

    # Only run the branching logic when the user explicitly confirmed (e.g., "yes ...").
    # This prevents fresh queries (confirm is None) from being treated as "follow-up" answers.
    if confirm:
        # track whether the 'remaining' part of a "yes and ..." reply
        # was already consumed when answering the follow-up, so we
        # don't produce a duplicate "additional query" response.
        remaining_consumed = False

        # Fast-path: "Yes, explain more about X" — the user's own follow-up
        # text is more specific than the stored intent.  Answer it directly
        # using last_context as grounding, then return immediately so we
        # never produce a second evidence block.
        if remaining and sess["last_context"]:
            sys_r, usr_r, ctx_r = build_prompt(remaining, fallback_context=sess["last_context"])
            _t_fu1 = time.monotonic()
            print(f"[LLM START] follow-up(remaining) budget_used={_t_fu1 - _req_t0:.1f}s")
            _resp_r = call_groq(_build_groq_messages(sys_r, usr_r), temperature=0.1, timeout=_REQUEST_BUDGET_S)
            print(f"[LLM END] follow-up(remaining) duration={time.monotonic() - _t_fu1:.2f}s")
            if _resp_r.get("error"):
                return _format_llm_error(_resp_r)
            raw_r = _resp_r["content"]
            bullets_r = enforce_bullet_only(sanitize_language(raw_r)) or raw_r.strip()
            evidence_r = ctx_r.get("evidence") or build_evidence_summary_for_context(ctx_r, sess.get("last_uploaded_df"))
            followup_r = generate_followup_from_last_bullet(bullets_r, remaining, ctx_r)
            sess["last_context"] = ctx_r
            sess["last_followup_intent"] = followup_r
            sess["conversation_history"].append({"role": "user", "content": user_query})
            body_r = bullets_r + (("\n\n" + evidence_r) if evidence_r else "") + (("\n\n" + followup_r) if followup_r else "")
            sess["conversation_history"].append({"role": "assistant", "content": body_r})
            max_entries = CONVERSATION_WINDOW * 2
            if len(sess["conversation_history"]) > max_entries:
                sess["conversation_history"] = sess["conversation_history"][-max_entries:]
            return body_r

        # If user confirmed the last follow-up intent, answer that intent first (clear it after)
        if sess["last_followup_intent"]:
            # If the last follow-up was a request to map confirmation/invalidation
            # levels and we have an uploaded file, produce deterministic numeric
            # levels rather than calling the LLM. This prevents generic filler
            # and ensures specific price levels are returned.
            mapping_keywords = ["map", "confirmation", "invalidation", "levels"]
            uploaded_df = sess["last_uploaded_df"]
            if (
                sess["last_context"]
                and sess["last_context"].get("data_source") == "uploaded_file"
                and uploaded_df is not None
                and any(k in sess["last_followup_intent"].lower() for k in mapping_keywords)
            ):
                levels = find_recent_swings(uploaded_df)
                bullets_followup = format_level_bullets(sess["last_context"], levels, uploaded_df=uploaded_df)
                evidence_followup = build_evidence_summary_for_context(sess["last_context"], uploaded_df)

                # clear the answered intent
                sess["last_followup_intent"] = None

                # If the user confirmed and attached an extra question like
                # "is this stock in an uptrend?", answer that deterministically
                # and combine into a single response to avoid duplication.
                remaining_handled = False
                if remaining:
                    rq = remaining.lower()
                    if "uptrend" in rq or "is this stock in an uptrend" in rq or "is this an uptrend" in rq:
                        # compute deterministic trend evidence (A4: df already lowercase)
                        try:
                            df = normalize_df_columns(uploaded_df)
                            c = df["close"]
                            last_close = float(c.iloc[-1])
                            earlier_mean = float(c.iloc[:-5].mean()) if len(c) > 5 else None
                            sma50 = compute_sma(c, 50)
                            sma200 = compute_sma(c, 200)

                            trend_yes = False
                            reasons = []
                            if earlier_mean is not None and last_close > earlier_mean:
                                trend_yes = True
                                reasons.append(f"Last close {last_close:.2f} > earlier mean {earlier_mean:.2f}")
                            if sma50 is not None and last_close > sma50:
                                trend_yes = True
                                reasons.append(f"Last close {last_close:.2f} > 50-day SMA {sma50:.2f}")
                            if sma200 is not None and last_close > sma200:
                                trend_yes = True
                                reasons.append(f"Last close {last_close:.2f} > 200-day SMA {sma200:.2f}")

                            obs_line = "Yes — structure is up." if trend_yes else "No — structure is not clearly up."
                            why_lines = "; ".join(reasons) if reasons else "Price does not exceed reference averages strongly."

                            trend_block = (
                                f"\n\nObservation: {obs_line}\n\nThe 'Why': {why_lines}\n\nVerification:\nLast close: {last_close:.2f}\n"
                                + (f"50-day SMA: {sma50:.2f}\n" if sma50 is not None else "")
                                + (f"200-day SMA: {sma200:.2f}\n" if sma200 is not None else "")
                            )

                            body_followup = bullets_followup + (("\n\n" + evidence_followup) if evidence_followup else "") + trend_block
                            responses.append("Answering your follow-up question:\n\n" + body_followup)
                            remaining_handled = True
                            remaining_consumed = True
                        except Exception:
                            body_followup = bullets_followup + (("\n\n" + evidence_followup) if evidence_followup else "")
                            responses.append("Answering your follow-up question:\n\n" + body_followup)
                            remaining_handled = True

                if not remaining or not remaining_handled:
                    body_followup = bullets_followup + (("\n\n" + evidence_followup) if evidence_followup else "")
                    responses.append("Answering your follow-up question:\n\n" + body_followup)
            else:
                sys_followup, usr_followup, context_followup = build_prompt(sess["last_followup_intent"], fallback_context=sess["last_context"])
                _t_fu2 = time.monotonic()
                print(f"[LLM START] follow-up(intent) budget_used={_t_fu2 - _req_t0:.1f}s")
                _resp_fu2 = call_groq(_build_groq_messages(sys_followup, usr_followup), temperature=0.1, timeout=_REQUEST_BUDGET_S)
                print(f"[LLM END] follow-up(intent) duration={time.monotonic() - _t_fu2:.2f}s")
                if _resp_fu2.get("error"):
                    return _format_llm_error(_resp_fu2)
                raw_followup = _resp_fu2["content"]
                bullets_followup = enforce_bullet_only(raw_followup)
                followup_for_memory = generate_followup_from_last_bullet(bullets_followup, sess["last_followup_intent"], context_followup)
                evidence_followup = context_followup.get("evidence") or build_evidence_summary_for_context(context_followup, uploaded_df)

                # update memory and clear the answered intent
                sess["last_context"] = context_followup
                sess["last_followup_intent"] = None
                body_followup = (
                    bullets_followup
                    + (("\n\n" + evidence_followup) if evidence_followup else "")
                )

                responses.append("Answering your follow-up question:\n\n" + body_followup)

        # If there's remaining text after the confirmation (a fresh extra query), answer it separately
        uploaded_df = sess["last_uploaded_df"]
        if remaining and not remaining_consumed:
            # If we have uploaded-file context, attempt a deterministic numeric reply
            if sess["last_context"] and sess["last_context"].get("data_source") == "uploaded_file" and uploaded_df is not None:
                # Simple deterministic answers for common follow-ups like trend or levels
                q = remaining.lower()
                if "trend" in q or "uptrend" in q or "downtrend" in q or "is this stock in an uptrend" in q:
                    ctx = sess["last_context"]
                    recent_close = None
                    try:
                        df_norm = normalize_df_columns(uploaded_df)
                        c = df_norm["close"]
                        recent_close = float(c.iloc[-1])
                        earlier_mean = float(c.iloc[:-5].mean()) if len(c) > 5 else None
                    except Exception:
                        recent_close = None
                        earlier_mean = None

                    trend = ctx.get('trend', 'unclear')
                    bullets_remaining = "Key observations\n" + f"• Daily trend: {'up' if trend == 'up' else trend}\n"
                    if recent_close is not None and earlier_mean is not None:
                        bullets_remaining += f"• Last close: {recent_close:.2f}; earlier mean: {earlier_mean:.2f}\n"

                    followup_remaining = generate_followup_from_last_bullet(bullets_remaining, remaining, ctx)
                    evidence_remaining = ctx.get("evidence") or build_evidence_summary_for_context(ctx, uploaded_df)

                    sess["last_context"] = ctx
                    sess["last_followup_intent"] = followup_remaining

                    body_remaining = bullets_remaining + (("\n\n" + evidence_remaining) if evidence_remaining else "")
                    responses.append("Answering your additional query:\n\n" + body_remaining)
                else:
                    # fallback to LLM for other remaining queries
                    # Only use the uploaded-file `LAST_CONTEXT` as fallback when the
                    # remaining text explicitly references the uploaded file or indicators.
                    fb_for_remaining = sess["last_context"] if not (
                        sess["last_context"]
                        and sess["last_context"].get("data_source") == "uploaded_file"
                        and not should_use_uploaded_context(remaining)
                    ) else None

                    sys_remaining, usr_remaining, context_remaining = build_prompt(remaining, fallback_context=fb_for_remaining)
                    _t_rem1 = time.monotonic()
                    _rem1_budget = _REQUEST_BUDGET_S - (_t_rem1 - _req_t0)
                    if _rem1_budget < 5.0:
                        print(f"[LLM SKIP] remaining(uploaded) — budget exhausted "
                              f"({_t_rem1 - _req_t0:.1f}s elapsed)")
                        responses.append("Answering your additional query:\n\n"
                                         "• Budget exhausted — please ask this as a new message.")
                    else:
                        print(f"[LLM START] remaining(uploaded) budget_remaining={_rem1_budget:.1f}s")
                        _resp_rem1 = call_groq(_build_groq_messages(sys_remaining, usr_remaining), temperature=0.1, timeout=_REQUEST_BUDGET_S)
                        print(f"[LLM END] remaining(uploaded) duration={time.monotonic() - _t_rem1:.2f}s")
                        if _resp_rem1.get("error"):
                            responses.append("Answering your additional query:\n\n" + _format_llm_error(_resp_rem1))
                        else:
                            raw_remaining = _resp_rem1["content"]
                        bullets_remaining = enforce_bullet_only(raw_remaining if not _resp_rem1.get("error") else "")
                        followup_remaining = generate_followup_from_last_bullet(bullets_remaining, remaining, context_remaining)
                        evidence_remaining = context_remaining.get("evidence") or build_evidence_summary_for_context(context_remaining, uploaded_df)

                        sess["last_context"] = context_remaining
                        sess["last_followup_intent"] = followup_remaining

                        body_remaining = (
                            bullets_remaining
                            + (("\n\n" + evidence_remaining) if evidence_remaining else "")
                        )

                        responses.append("Answering your additional query:\n\n" + body_remaining)
            else:
                sys_remaining, usr_remaining, context_remaining = build_prompt(remaining, fallback_context=sess["last_context"])
                _t_rem2 = time.monotonic()
                _rem2_budget = _REQUEST_BUDGET_S - (_t_rem2 - _req_t0)
                if _rem2_budget < 5.0:
                    print(f"[LLM SKIP] remaining(live) — budget exhausted "
                          f"({_t_rem2 - _req_t0:.1f}s elapsed)")
                    responses.append("Answering your additional query:\n\n"
                                     "• Budget exhausted — please ask this as a new message.")
                else:
                    print(f"[LLM START] remaining(live) budget_remaining={_rem2_budget:.1f}s")
                    _resp_rem2 = call_groq(_build_groq_messages(sys_remaining, usr_remaining), temperature=0.1, timeout=_REQUEST_BUDGET_S)
                    print(f"[LLM END] remaining(live) duration={time.monotonic() - _t_rem2:.2f}s")
                    if _resp_rem2.get("error"):
                        responses.append("Answering your additional query:\n\n" + _format_llm_error(_resp_rem2))
                    else:
                        raw_remaining = _resp_rem2["content"]
                    bullets_remaining = enforce_bullet_only(raw_remaining if not _resp_rem2.get("error") else "")
                    followup_remaining = generate_followup_from_last_bullet(bullets_remaining, remaining, context_remaining)
                    evidence_remaining = context_remaining.get("evidence") or build_evidence_summary_for_context(context_remaining, uploaded_df)

                    sess["last_context"] = context_remaining
                    sess["last_followup_intent"] = followup_remaining

                    body_remaining = (
                        bullets_remaining
                        + (("\n\n" + evidence_remaining) if evidence_remaining else "")
                    )

                    responses.append("Answering your additional query:\n\n" + body_remaining)

        # After answering both the original follow-up and any additional
        # query, present a single follow-up prompt (prefer the additional
        # query's follow-up if available). This ensures follow-ups are
        # raised only after all requested answers are delivered.
        if responses:
            final_followup = None
            if 'followup_remaining' in locals() and followup_remaining:
                final_followup = followup_remaining
            elif 'followup_for_memory' in locals() and followup_for_memory:
                final_followup = followup_for_memory

            combined = "\n\n".join(responses)
            if final_followup:
                combined = combined + "\n\n" + final_followup

            return combined

    # -------- Analysis mode --------
    # If the user asked about an indicator (RSI/divergence) and we have an
    # uploaded dataframe in memory, perform deterministic numeric analysis
    # and return a structured answer rather than routing to the LLM.
    # Handle combined requests like "analyze this and check RSI/divergence"
    qlow = user_query.lower()
    uploaded_df = sess["last_uploaded_df"]

    if (
        sess["last_context"]
        and sess["last_context"].get("data_source") == "uploaded_file"
        and uploaded_df is not None
        and (("analyze" in qlow or "analysis" in qlow or "structure" in qlow) and ("rsi" in qlow or "divergence" in qlow))
    ):
        try:
            sys_comb, usr_comb, context_comb = build_prompt(user_query, fallback_context=sess["last_context"])
            _t_comb = time.monotonic()
            print(f"[LLM START] analyze+rsi budget_used={_t_comb - _req_t0:.1f}s")
            _resp_comb = call_groq(_build_groq_messages(sys_comb, usr_comb), temperature=0.1, timeout=_REQUEST_BUDGET_S)
            print(f"[LLM END] analyze+rsi duration={time.monotonic() - _t_comb:.2f}s")
            if _resp_comb.get("error"):
                bullets_comb = "• Structural analysis unavailable (LLM error)."
            else:
                raw_comb = sanitize_language(_resp_comb["content"])
                bullets_comb = enforce_bullet_only(raw_comb)
        except Exception:
            bullets_comb = "• Structural analysis unavailable (LLM error)."

        numeric_comb = analyze_uploaded_rsi_divergence(uploaded_df)
        evidence_comb = build_evidence_summary_for_context(sess["last_context"], uploaded_df)

        followup_comb = generate_followup_from_last_bullet(bullets_comb + "\n" + numeric_comb, user_query, sess["last_context"])
        sess["last_followup_intent"] = followup_comb

        combined = bullets_comb + "\n\n" + numeric_comb
        if evidence_comb:
            combined = combined + "\n\n" + evidence_comb
        if followup_comb:
            combined = combined + "\n\n" + followup_comb

        return combined

    if (
        sess["last_context"]
        and sess["last_context"].get("data_source") == "uploaded_file"
        and uploaded_df is not None
        and ("rsi" in qlow or "divergence" in qlow)
    ):
        numeric_answer = analyze_uploaded_rsi_divergence(uploaded_df)
        evidence = build_evidence_summary_for_context(sess["last_context"], uploaded_df)
        followup = generate_followup_from_last_bullet(numeric_answer, user_query, sess["last_context"])
        sess["last_followup_intent"] = followup

        combined = numeric_answer
        if evidence:
            combined = combined + "\n\n" + evidence
        if followup:
            combined = combined + "\n\n" + followup

        return combined

    # Deterministic trend check for uploaded files
    if (
        sess["last_context"]
        and sess["last_context"].get("data_source") == "uploaded_file"
        and uploaded_df is not None
        and ("uptrend" in qlow or "is this stock in an uptrend" in qlow or "is this an uptrend" in qlow)
    ):
        try:
            df_norm = normalize_df_columns(uploaded_df)
            if "close" not in df_norm.columns:
                return "Error: No 'close' column found in the uploaded file."
            close = df_norm["close"].astype(float).reset_index(drop=True)
            last_close = float(close.iloc[-1])
            earlier_mean = float(close.iloc[:-5].mean()) if len(close) > 5 else None

            sma50 = compute_sma(close, 50)
            sma200 = compute_sma(close, 200)

            trend_yes = False
            reasons = []
            if earlier_mean is not None and last_close > earlier_mean:
                trend_yes = True
                reasons.append(f"Last close {last_close:.2f} > earlier mean {earlier_mean:.2f}")
            if sma50 is not None and last_close > sma50:
                trend_yes = True
                reasons.append(f"Last close {last_close:.2f} > 50-day SMA {sma50:.2f}")
            if sma200 is not None and last_close > sma200:
                trend_yes = True
                reasons.append(f"Last close {last_close:.2f} > 200-day SMA {sma200:.2f}")

            levels = find_recent_swings(df_norm)
            level_bullets = format_level_bullets(sess["last_context"], levels, uploaded_df=df_norm)

            obs_line = "Yes — structure is up." if trend_yes else "No — structure is not clearly up."
            why_lines = "; ".join(reasons) if reasons else "Price does not exceed reference averages strongly."
            evidence = build_evidence_summary_for_context(sess["last_context"], uploaded_df)

            combined = (
                f"Observation: {obs_line}\n\nThe 'Why': {why_lines}\n\nVerification:\n"
                f"Last close: {last_close:.2f}\n"
                + (f"50-day SMA: {sma50:.2f}\n" if sma50 is not None else "")
                + (f"200-day SMA: {sma200:.2f}\n" if sma200 is not None else "")
                + "\n"
                + level_bullets
            )
            if evidence:
                combined = combined + "\n\n" + evidence

            sess["last_followup_intent"] = generate_followup_from_last_bullet(combined, user_query, sess["last_context"])

            return combined
        except Exception as e:
            return f"Error computing deterministic trend: {e}"

    # Default: LLM-backed analysis
    fb = sess["last_context"] if not (
        sess["last_context"]
        and sess["last_context"].get("data_source") == "uploaded_file"
        and not should_use_uploaded_context(user_query)
    ) else None

    sys_prompt, usr_msg, context = build_prompt(user_query, fallback_context=fb)

    # ── Phase 12: Hard data gate — data-validity driven routing ──────────────
    # Fix 1: Gate is driven by ACTUAL data fields (last_price + trend), NOT by
    #        the _lightweight flag.  _lightweight can be stale from a previous
    #        failed fetch that was cached before AV/TD recovered.  Using it as
    #        the gate means valid data is blocked after a transient failure.
    #
    # has_data = True  → AV/TD returned real OHLCV → full LLM analysis.
    # has_data = False → All providers failed      → unavailability message.
    #
    # The old known-ticker / unknown-ticker split is no longer needed here:
    #   • When providers fail for ANY ticker, last_price is None → has_data=False.
    #   • When providers succeed for ANY ticker, last_price is set → has_data=True.

    _hd_ticker = context.get("ticker", "")
    _hd_trend  = (context.get("trend") or "").lower()
    _hd_price  = context.get("last_price")
    _hd_lw     = context.get("_lightweight", False)

    # Fix 3: Debug log — exact values the router sees before the gate decision.
    print(
        f"[TechnoBot/router] DATA CHECK → "
        f"ticker={_hd_ticker} "
        f"trend={_hd_trend!r} "
        f"price={_hd_price} "
        f"lightweight={_hd_lw}"
    )

    has_data = (
        context is not None
        and _hd_price is not None
        and _hd_trend not in ("", "unknown", "unclear")
    )

    # Single source of truth — stamp _final_status onto context NOW,
    # before any session storage or response path diverges.  Every downstream
    # reader (/chat endpoint, streaming path) reads THIS value — no independent
    # recomputation allowed.
    #
    # Three-way status:
    #   ok          — live OHLCV data present; full indicator analysis.
    #   partial     — lightweight qualitative fallback; LLM still runs.
    #   unavailable — no data and no qualitative fallback possible.
    if has_data:
        context["_final_status"] = "ok"
    elif context.get("_lightweight"):
        context["_final_status"] = "partial"
    else:
        context["_final_status"] = "unavailable"

    print(f"[TechnoBot/status] final_status={context['_final_status']} "
          f"lightweight={context.get('_lightweight', False)}")

    if not has_data:
        # Hard validation log — live data confirmed unavailable.
        print(
            f"[TechnoBot/router] BLOCKED → no valid data "
            f"ticker={_hd_ticker} trend={_hd_trend!r} price={_hd_price}"
        )
        # Fix 2: DO NOT RETURN — continue into the LLM path with lightweight context.
        # format_market_context() detects _lightweight=True and injects the qualitative
        # disclaimer + _qualitative_note so the LLM produces historical/educational
        # analysis (structural tendencies, indicator behaviour, typical cycles).
        context["_lightweight"] = True
        print(f"[TechnoBot/router] FALLBACK → running qualitative LLM")
        # Clear stale turn-level fields so the LLM output below populates them fresh.
        for _k in ("_last_analysis", "_last_followup", "_last_comparison_insight",
                   "last_response_ticker", "last_response_tf", "last_insight_hash"):
            sess.pop(_k, None)
        # _final_status stays "unavailable"; /chat reads it from context.
        # _last_response_type is set after LLM completes — NOT here.

    # ── Latency: trim history for follow-ups ──────────────────────────────────
    # First query gets the full window (3 pairs) for maximum context.
    # Follow-up queries only need the immediately prior turn — sending less
    # history reduces token processing overhead per request.
    is_followup = bool(sess["conversation_history"])
    if is_followup:
        # Only last 1 pair (2 entries) for follow-ups — sufficient for continuity
        history_window = sess["conversation_history"][-2:]
    else:
        history_window = sess["conversation_history"][-(CONVERSATION_WINDOW * 2):]

    # ── Data validation gate ──────────────────────────────────────────────────
    # Check rate-limit first — never report a 429 as "ticker not found".
    # Then check full failure (no trend, no RSI, no price).
    if context.get("rate_limited"):
        instr_name = context.get("instrument_name") or "this instrument"
        # Clear stale analysis fields — rate-limit response must not carry old analysis
        for _k in ("_last_analysis", "_last_followup", "_last_comparison_insight",
                   "last_response_ticker", "last_response_tf", "last_insight_hash"):
            sess.pop(_k, None)
        sess["last_context"]        = context.copy()  # Fix 5: snapshot
        sess["_last_response_type"] = "unavailable"
        return (
            f"⚠️ Live market data for **{instr_name}** is temporarily unavailable "
            "due to Yahoo Finance rate limiting (HTTP 429).\n\n"
            "This happens when multiple requests are made in a short time. "
            "**Please wait 3–5 minutes and try again.**\n\n"
            "In the meantime you can:\n"
            "• **Upload a CSV** with price data to analyse offline\n"
            "• **Ask about TA concepts** (RSI, MACD, support/resistance) without live data\n"
            "• **Come back shortly** — the data provider cooldown clears automatically"
        )

    # Fix 4: This gate is only reached when the lightweight path did NOT already
    # set _lightweight=True (i.e. a context with unknown trend reached here via
    # an unexpected code path — genuinely unresolvable, even qualitatively).
    # When _lightweight=True the block above already decided to continue to LLM;
    # this gate must not re-intercept and return a static error message.
    if (
        not context.get("_lightweight")       # skip for lightweight — LLM handles it
        and context.get("trend") == "unknown"
        and context.get("rsi") is None
        and context.get("last_price") is None
    ):
        instr_name = context.get("instrument_name") or "this instrument"
        tf_label   = {"1d": "daily", "1wk": "weekly", "1mo": "monthly"}.get(
            context.get("timeframe", "1d"), context.get("timeframe", "daily")
        )
        # Clear stale analysis fields — no-data response must not carry old ticker
        for _k in ("_last_analysis", "_last_followup", "_last_comparison_insight",
                   "last_response_ticker", "last_response_tf", "last_insight_hash"):
            sess.pop(_k, None)
        sess["last_context"]        = context.copy()
        sess["_last_response_type"] = "unavailable"
        return (
            f"⚠️ Live market data for **{instr_name}** on the {tf_label} timeframe "
            f"could not be retrieved right now.\n\n"
            "What you can do:\n"
            "• **Try again** in 30–60 seconds — transient network delays resolve quickly\n"
            "• **Check the ticker** — e.g., use 'Reliance' or 'RELIANCE.NS' explicitly\n"
            "• **Switch timeframe** — try 'daily' if you asked for weekly/monthly\n"
            "• **Upload a CSV** to analyse offline data without a live feed"
        )

    _t_main = time.monotonic()
    # Fix 5: Debug log — confirms whether LLM runs with live data or qualitative mode.
    print(f"[TechnoBot/router] LLM EXECUTION → lightweight={context.get('_lightweight', False)}")
    print(f"[LLM START] main ticker={context.get('ticker','?')} "
          f"tf={context.get('timeframe','?')} budget_used={_t_main - _req_t0:.1f}s")
    _msgs_main = _build_groq_messages(sys_prompt, usr_msg, history=history_window, query=user_query)
    _resp_main = call_groq(_msgs_main, temperature=0.1, timeout=_REQUEST_BUDGET_S)
    print(f"[LLM END] main duration={time.monotonic() - _t_main:.2f}s")

    if _resp_main.get("error"):
        return _format_llm_error(_resp_main)
    raw = _resp_main["content"]

    analysis_text = enforce_bullet_only(sanitize_language(raw))
    if not analysis_text:
        analysis_text = raw.strip()

    # ── Post-processing quality pipeline (zero LLM calls) ────────────────────
    analysis_text = _enrich_bullet_semantics(analysis_text, context)
    analysis_text = _ensure_naturalness(analysis_text)

    # Store pure bullets for /chat structured response (before evidence/insight are appended)
    sess["_last_analysis"] = analysis_text
    # Both live-data and lightweight paths reach here — response type is always
    # "analysis" (LLM ran).  Status comes from context["_final_status"], not rtype.
    sess["_last_response_type"] = "analysis"

    followup = generate_followup_from_last_bullet(analysis_text, user_query, context)

    # Store followup so /chat structured response can expose it as its own field
    sess["_last_followup"] = followup

    # ── Follow-up verbosity control ───────────────────────────────────────────
    # If this is a follow-up on the SAME instrument + timeframe, suppress the
    # evidence block and contextual insight (they were already shown and haven't
    # changed). This keeps follow-up responses concise and signal-focused.
    prev_ticker = sess.get("last_response_ticker")
    prev_tf     = sess.get("last_response_tf")
    curr_ticker = context.get("ticker", "")
    curr_tf     = context.get("timeframe", "")
    context_unchanged = (
        is_followup
        and bool(prev_ticker)
        and prev_ticker == curr_ticker
        and prev_tf == curr_tf
        and context.get("data_source") != "uploaded_file"
    )

    if context_unchanged:
        # Follow-up on same instrument: skip repeated evidence, multitf, insight
        evidence = ""
        multitf  = ""
        # Only regenerate insight if the insight hash would differ (phrase rotation)
        prev_insight_hash = sess.get("last_insight_hash")
        raw_insight = generate_insight_layer(context)
        new_hash = str(hash(raw_insight[:80]))
        insight = "" if new_hash == prev_insight_hash else raw_insight
    else:
        # First query or new instrument — show everything
        evidence = context.get("evidence") or build_evidence_summary_for_context(context, sess.get("last_uploaded_df"))
        insight  = generate_insight_layer(context)
        multitf  = ""
        if context.get("data_source") != "uploaded_file":
            ticker_mtf = curr_ticker
            tf_mtf     = curr_tf
            if ticker_mtf and tf_mtf:
                try:
                    multitf = generate_multitf_context(ticker_mtf, tf_mtf, context)
                except Exception:
                    multitf = ""

    print(f"[TechnoBot] followup_mode={context_unchanged} | evidence={bool(evidence)} | insight={bool(insight)} | multitf={bool(multitf)}")

    # Update follow-up tracking state
    sess["last_followup_intent"] = followup
    sess["last_context"]         = context.copy()  # Fix 5: snapshot, not reference
    sess["last_response_ticker"] = curr_ticker
    sess["last_response_tf"]     = curr_tf
    if insight:
        sess["last_insight_hash"] = str(hash(insight[:80]))

    final_response = (
        analysis_text
        + (("\n\n" + evidence)  if evidence  else "")
        + (("\n\n" + multitf)   if multitf   else "")
        + (("\n\n" + insight)   if insight   else "")
        + (("\n\n" + followup)  if followup  else "")
    )

    # D2: record this turn in the sliding window
    sess["conversation_history"].append({"role": "user",      "content": user_query})
    sess["conversation_history"].append({"role": "assistant", "content": final_response})
    max_entries = CONVERSATION_WINDOW * 2
    if len(sess["conversation_history"]) > max_entries:
        sess["conversation_history"] = sess["conversation_history"][-max_entries:]

    return final_response


def parse_uploaded_file(file_path: str, sess: dict) -> str:
    """A3: sess is the per-chat session dict — all state written here is scoped."""
    try:
        df = parse_price_file(file_path)
        # A4: normalise column names immediately so all downstream code uses lowercase
        df = normalize_df_columns(df)
        timeframe = infer_timeframe(df)

        # A3: store in session instead of global
        sess["last_uploaded_df"] = df.copy()

        context = derive_context_from_dataframe(df, timeframe)
        context["instrument_name"] = "Uploaded CSV"
        context["ticker"] = "custom"
        context["data_source"] = "uploaded_file"
        context["rows_used"] = len(df)

        sys_up, usr_up, _ = build_prompt(f"Analyze {timeframe} price structure", fallback_context=context)
        _t_up = time.monotonic()
        print(f"[LLM START] uploaded-file analyze timeframe={timeframe}")
        _resp_up = call_groq(_build_groq_messages(sys_up, usr_up), temperature=0.1, timeout=_REQUEST_BUDGET_S)
        print(f"[LLM END] uploaded-file analyze duration={time.monotonic() - _t_up:.2f}s")
        if _resp_up.get("error"):
            return _format_llm_error(_resp_up)
        raw = sanitize_language(_resp_up["content"])
        bullets = enforce_bullet_only(raw)
        followup = generate_followup_from_last_bullet(bullets, file_path, context)
        evidence = build_evidence_summary_for_context(context, sess.get("last_uploaded_df"))

        sess["last_followup_intent"] = followup
        sess["last_context"] = context

        return bullets + (evidence if evidence else "") + "\n\n" + followup

    except ValueError as ve:
        # D5: structured validation error — tell the user exactly what to fix
        msg = str(ve)
        return (
            "⚠️ Your file could not be parsed.\n\n"
            f"Reason: {msg}\n\n"
            "TechnoBot expects a CSV with at minimum these columns:\n"
            "• Date (or Datetime / Timestamp)\n"
            "• Open\n"
            "• High\n"
            "• Low\n"
            "• Close\n\n"
            "Column names are case-insensitive. Volume is optional.\n"
            "Please correct your file and upload again."
        )
    except Exception as e:
        return f"⚠️ Unexpected error reading your file: {e}"


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """F3: Wilder's smoothing (correct RSI) instead of simple rolling mean.

    Wilder's method uses alpha = 1/period, equivalent to an EMA with
    com = period - 1. The initial seed is the SMA of the first `period`
    gains/losses (standard convention used by TradingView, MT4, etc.).
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Seed with SMA of first `period` values, then apply Wilder's smoothing
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()

    # Let pandas handle division naturally:
    #   avg_loss = 0, avg_gain > 0  →  rs = +inf  →  RSI = 100  (correct: strong uptrend)
    #   avg_loss > 0, avg_gain = 0  →  rs =  0    →  RSI = 0    (correct: strong downtrend)
    #   both = 0 (flat data)        →  rs = NaN   →  RSI = NaN  (acceptable: indeterminate)
    # Using .replace(0, NaN) was wrong: it turned avg_loss=0 into NaN making rs=NaN for uptrends
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_sma(series: pd.Series, window: int) -> float | None:
    try:
        vals = series.astype(float)
        if len(vals) < window:
            return None
        return float(vals.rolling(window=window).mean().iloc[-1])
    except Exception:
        return None


def find_recent_swings(df: pd.DataFrame):
    """Return recent swing high (confirmation) and swing low (invalidation).
    Heuristic: find the last local peak and last local trough in the close series.
    A4: df is expected to have lowercase columns after normalize_df_columns().
    """
    # A4: normalise defensively so this function always works
    df = normalize_df_columns(df)
    close_col = "close" if "close" in df.columns else None
    if close_col is None:
        raise ValueError("Uploaded data must contain a 'close' column.")

    close = df[close_col]
    dates = df.get("date")
    close = close.astype(float).reset_index(drop=True)
    n = len(close)
    # find last peak
    peak_idx = None
    for i in range(n - 2, 0, -1):
        if close.iloc[i] > close.iloc[i - 1] and close.iloc[i] > close.iloc[i + 1]:
            peak_idx = i
            break
    # find last trough
    trough_idx = None
    for i in range(n - 2, 0, -1):
        if close.iloc[i] < close.iloc[i - 1] and close.iloc[i] < close.iloc[i + 1]:
            trough_idx = i
            break

    # fallback to simple last high/low if no local extremum found
    if peak_idx is None:
        peak_idx = close.idxmax()
    if trough_idx is None:
        trough_idx = close.idxmin()

    def idx_to_date(i):
        if dates is None:
            return str(i)
        try:
            return pd.to_datetime(dates.reset_index(drop=True).iloc[i]).strftime("%Y-%m-%d")
        except Exception:
            return str(i)

    peak_idx_int = int(peak_idx)
    trough_idx_int = int(trough_idx)
    return {
        "peak_idx": peak_idx_int,
        "peak_price": float(close.iloc[peak_idx_int]),
        "peak_date": idx_to_date(peak_idx_int),
        "trough_idx": trough_idx_int,
        "trough_price": float(close.iloc[trough_idx_int]),
        "trough_date": idx_to_date(trough_idx_int),
    }


def format_level_bullets(context: dict, levels: dict, uploaded_df=None) -> str:
    """A3/A4: uploaded_df passed explicitly — no global access."""
    lines = []
    lines.append("Key levels (daily):")
    lines.append(f"• Confirmation (recent swing high): {levels['peak_price']:.2f} ({levels['peak_date']})")
    lines.append(f"• Invalidation (recent swing low): {levels['trough_price']:.2f} ({levels['trough_date']})")
    if context and uploaded_df is not None:
        df_norm = normalize_df_columns(uploaded_df)
        close_series = df_norm.get("close")
        if close_series is not None:
            sma50 = compute_sma(close_series, 50)
            sma200 = compute_sma(close_series, 200)
            if sma50:
                lines.append(f"• 50-day SMA: {sma50:.2f}")
            if sma200:
                lines.append(f"• 200-day SMA: {sma200:.2f}")

    return "\n".join(lines)


def analyze_uploaded_rsi_divergence(df: pd.DataFrame) -> str:
    """Return structured analysis (Observation / The 'Why' / Verification)
    using the uploaded dataframe numeric values. If no clear divergence is
    found, explain what the data shows.
    A4: normalises columns before any access.
    """
    try:
        df = normalize_df_columns(df)
        close = df["close"]
        dates = df["date"] if "date" in df.columns else None
        close = close.astype(float).reset_index(drop=True)
        if dates is not None:
            dates = pd.to_datetime(dates).reset_index(drop=True)

        if len(close) < 20:
            last_rsi = None
            rsi = compute_rsi(close)
            last_rsi = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else None
            last_close = float(close.iloc[-1])
            obs = f"Data insufficient for robust divergence detection (only {len(close)} rows). Last close: {last_close:.2f}."
            why = "Divergence requires at least two swings and a reliable RSI series; provide more history for stronger signals."
            verification = (
                "Numbers used:\n" +
                (f"Last close: {last_close:.2f}\n" if last_close is not None else "") +
                (f"Last RSI: {last_rsi:.2f}\n" if last_rsi is not None else "")
            )
            return f"Observation: {obs}\n\nThe 'Why': {why}\n\nVerification:\n{verification}"

        rsi = compute_rsi(close)

        # Helper to find previous two peaks / troughs
        def find_peaks(vals, n=2):
            peaks = []
            for i in range(1, len(vals) - 1):
                if vals.iloc[i] > vals.iloc[i - 1] and vals.iloc[i] > vals.iloc[i + 1]:
                    peaks.append(i)
            # take last n peaks
            if len(peaks) >= n:
                return peaks[-n:]
            # fallback: choose top n values' indices in recent window
            window = vals.iloc[-60:]
            idxs = window.nlargest(n).index.tolist()
            return idxs

        def find_troughs(vals, n=2):
            troughs = []
            for i in range(1, len(vals) - 1):
                if vals.iloc[i] < vals.iloc[i - 1] and vals.iloc[i] < vals.iloc[i + 1]:
                    troughs.append(i)
            if len(troughs) >= n:
                return troughs[-n:]
            window = vals.iloc[-60:]
            idxs = window.nsmallest(n).index.tolist()
            return idxs

        peak_idxs = find_peaks(close, n=2)
        trough_idxs = find_troughs(close, n=2)

        analysis_lines = []

        # check bearish divergence (higher high in price, lower high in RSI)
        divergence_found = False
        if len(peak_idxs) >= 2:
            p1, p2 = peak_idxs[0], peak_idxs[1]
            c1, c2 = float(close.iloc[p1]), float(close.iloc[p2])
            r1, r2 = float(rsi.iloc[p1]) if not pd.isna(rsi.iloc[p1]) else None, float(rsi.iloc[p2]) if not pd.isna(rsi.iloc[p2]) else None
            d1 = dates.iloc[p1].strftime("%Y-%m-%d") if dates is not None else str(p1)
            d2 = dates.iloc[p2].strftime("%Y-%m-%d") if dates is not None else str(p2)
            if c2 > c1 and r1 is not None and r2 is not None and r2 < r1:
                divergence_found = True
                obs = (
                    f"Price made a higher high: {c1:.2f} on {d1} → {c2:.2f} on {d2}. "
                    f"RSI fell from {r1:.2f} to {r2:.2f} over the same swings (bearish divergence)."
                )
                why = (
                    "Bearish divergence suggests weakening momentum despite higher prices, "
                    "increasing the probability of a corrective move or trend pause."
                )
                verification = (
                    f"Price & RSI points used:\n{d1}: Price={c1:.2f}, RSI={r1:.2f}\n{d2}: Price={c2:.2f}, RSI={r2:.2f}\n"
                )
                return f"Observation: {obs}\n\nThe 'Why': {why}\n\nVerification:\n{verification}"

        # check bullish divergence (lower low in price, higher low in RSI)
        if len(trough_idxs) >= 2:
            t1, t2 = trough_idxs[0], trough_idxs[1]
            c1, c2 = float(close.iloc[t1]), float(close.iloc[t2])
            r1, r2 = float(rsi.iloc[t1]) if not pd.isna(rsi.iloc[t1]) else None, float(rsi.iloc[t2]) if not pd.isna(rsi.iloc[t2]) else None
            d1 = dates.iloc[t1].strftime("%Y-%m-%d") if dates is not None else str(t1)
            d2 = dates.iloc[t2].strftime("%Y-%m-%d") if dates is not None else str(t2)
            if c2 < c1 and r1 is not None and r2 is not None and r2 > r1:
                divergence_found = True
                obs = (
                    f"Price made a lower low: {c1:.2f} on {d1} → {c2:.2f} on {d2}. "
                    f"RSI rose from {r1:.2f} to {r2:.2f} over the same swings (bullish divergence)."
                )
                why = (
                    "Bullish divergence suggests momentum is improving despite lower prices, "
                    "increasing the probability of a reversal or corrective bounce."
                )
                verification = (
                    f"Price & RSI points used:\n{d1}: Price={c1:.2f}, RSI={r1:.2f}\n{d2}: Price={c2:.2f}, RSI={r2:.2f}\n"
                )
                return f"Observation: {obs}\n\nThe 'Why': {why}\n\nVerification:\n{verification}"

        # No clear divergence found — report current RSI and last swings
        last_idx = len(close) - 1
        last_close = float(close.iloc[last_idx])
        last_rsi = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else None
        rsi_desc = (
            "overbought" if last_rsi and last_rsi > 70 else
            ("oversold" if last_rsi and last_rsi < 30 else "neutral")
        )
        obs = f"No clear price/RSI divergence detected in the uploaded data. Last close: {last_close:.2f}."
        why = (
            "Either the swings do not form the higher-high/lower-high or lower-low/higher-low pattern required, "
            "or the RSI series is not showing opposite momentum on the compared swings."
        )
        verification = (
            f"Last values used:\nLast date: {dates.iloc[-1].strftime('%Y-%m-%d') if dates is not None else last_idx}\n" +
            f"Last price: {last_close:.2f}\n" +
            (f"Last RSI: {last_rsi:.2f}\n" if last_rsi is not None else "Last RSI: N/A\n")
        )

        return f"Observation: {obs}\n\nThe 'Why': {why}\n\nVerification:\n{verification}"
    except Exception as e:
        return f"Error analyzing uploaded RSI/divergence: {e}"


if __name__ == "__main__":
    while True:
        q = input("\nAsk TechnoBot (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        print("\n🤖TechnoBot is thinking🤔...\n")
        print(answer(q))

