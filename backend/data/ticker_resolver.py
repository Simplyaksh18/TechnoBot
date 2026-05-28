"""
data/ticker_resolver.py — Lightweight natural-language → ticker resolution.

Provides two independent utilities:

  resolve_to_yahoo(name)        → Yahoo Finance ticker string (e.g. "RELIANCE.NS")
  yahoo_to_twelvedata(ticker)   → Twelve Data symbol string  (e.g. "RELIANCE:NSE")

This module is self-contained — it does NOT import from technobot_router,
nifty_fetcher, or any other TechnoBot module.  It can be imported by:
  • twelvedata_fallback.py  (format conversion only)
  • Any caller that wants a lightweight pre-fetch name resolver

Resolution strategy for resolve_to_yahoo()
───────────────────────────────────────────
1. Normalise input: lowercase, strip punctuation, collapse spaces.
2. Exact match in _ALIAS_MAP.
3. Prefix scan (longest match wins) so "bajaj finance" beats "bajaj".
4. Fuzzy token overlap — catches "reliance industries limited" → RELIANCE.NS.
5. Return None if unresolved (caller falls back to treating input as raw ticker).

Twelve Data format for Indian equities
────────────────────────────────────────
NSE stocks : "RELIANCE:NSE"
BSE stocks : "RELIANCE:BSE"
Indices    : explicit map below (Twelve Data uses its own index symbols)
"""

from __future__ import annotations
import re


# ── Alias → Yahoo ticker map ──────────────────────────────────────────────────
# Keys are ALL lowercase, normalised (no punctuation, single spaces).
# Longer keys must come first in ambiguous prefix groups so the scan finds the
# most specific match.  The dict is sorted by key length (longest first) at
# module load; users of _ALIAS_MAP_SORTED rely on that order.

_ALIAS_MAP: dict[str, str] = {
    # ── Indices ──────────────────────────────────────────────────────────────
    "nifty 50":             "^NSEI",
    "nifty50":              "^NSEI",
    "nifty":                "^NSEI",
    "nse nifty":            "^NSEI",
    "sensex":               "^BSESN",
    "bse sensex":           "^BSESN",
    "bank nifty":           "^NSEBANK",
    "banknifty":            "^NSEBANK",
    "nifty bank":           "^NSEBANK",
    "cnx it":               "^CNXIT",
    "nifty it":             "^CNXIT",
    "cnxit":                "^CNXIT",
    "nifty pharma":         "^CNXPHARMA",
    "nifty auto":           "^CNXAUTO",
    "nifty fmcg":           "^CNXFMCG",
    "nifty metal":          "^CNXMETAL",
    "nifty midcap":         "^NSEMDCP50",
    "nifty smallcap":       "^CNXSC",
    "india vix":            "^INDIAVIX",
    "vix":                  "^INDIAVIX",

    # ── Reliance group ───────────────────────────────────────────────────────
    "reliance industries":  "RELIANCE.NS",
    "reliance industries limited": "RELIANCE.NS",
    "reliance":             "RELIANCE.NS",
    "ril":                  "RELIANCE.NS",
    "rjio":                 "RELIANCE.NS",
    "mukesh ambani company":"RELIANCE.NS",

    # ── TCS ──────────────────────────────────────────────────────────────────
    "tata consultancy services": "TCS.NS",
    "tata consultancy":     "TCS.NS",
    "tcs":                  "TCS.NS",

    # ── Infosys ──────────────────────────────────────────────────────────────
    "infosys limited":      "INFY.NS",
    "infosys":              "INFY.NS",
    "infy":                 "INFY.NS",
    "infosys ltd":          "INFY.NS",

    # ── HDFC group ───────────────────────────────────────────────────────────
    "hdfc bank limited":    "HDFCBANK.NS",
    "hdfc bank":            "HDFCBANK.NS",
    "hdfcbank":             "HDFCBANK.NS",
    "hdfcb":                "HDFCBANK.NS",
    "hdfc":                 "HDFCBANK.NS",
    "hdfc life":            "HDFCLIFE.NS",
    "hdfc amc":             "HDFCAMC.NS",

    # ── ICICI group ───────────────────────────────────────────────────────────
    "icici bank":           "ICICIBANK.NS",
    "icicibank":            "ICICIBANK.NS",
    "icicib":               "ICICIBANK.NS",
    "icici":                "ICICIBANK.NS",
    "icici prudential":     "ICICIPRULI.NS",

    # ── Tata group ────────────────────────────────────────────────────────────
    "tata motors limited":  "TATAMOTORS.NS",
    "tata motors":          "TATAMOTORS.NS",
    "tatamotors":           "TATAMOTORS.NS",
    "tata steel":           "TATASTEEL.NS",
    "tatasteel":            "TATASTEEL.NS",
    "tata power":           "TATAPOWER.NS",
    "tatapower":            "TATAPOWER.NS",
    "tata elxsi":           "TATAELXSI.NS",
    "titan":                "TITAN.NS",
    "titan company":        "TITAN.NS",

    # ── SBI group ─────────────────────────────────────────────────────────────
    "state bank of india":  "SBIN.NS",
    "state bank":           "SBIN.NS",
    "sbi":                  "SBIN.NS",
    "sbin":                 "SBIN.NS",
    "sbi life":             "SBILIFE.NS",
    "sbi card":             "SBICARD.NS",
    "sbi cards":            "SBICARD.NS",

    # ── Bajaj group ───────────────────────────────────────────────────────────
    "bajaj finance limited":"BAJFINANCE.NS",
    "bajaj finance":        "BAJFINANCE.NS",
    "bajfinance":           "BAJFINANCE.NS",
    "bajaj finserv":        "BAJAJFINSV.NS",
    "bajaj auto":           "BAJAJ-AUTO.NS",
    "bajaj":                "BAJFINANCE.NS",   # most-queried Bajaj

    # ── Kotak ─────────────────────────────────────────────────────────────────
    "kotak mahindra bank":  "KOTAKBANK.NS",
    "kotak bank":           "KOTAKBANK.NS",
    "kotak":                "KOTAKBANK.NS",
    "kotakbank":            "KOTAKBANK.NS",

    # ── Axis Bank ─────────────────────────────────────────────────────────────
    "axis bank":            "AXISBANK.NS",
    "axisbank":             "AXISBANK.NS",
    "axis":                 "AXISBANK.NS",

    # ── Mahindra group ────────────────────────────────────────────────────────
    "mahindra and mahindra":"M&M.NS",
    "mahindra mahindra":    "M&M.NS",
    "mahindra":             "M&M.NS",
    "m and m":              "M&M.NS",
    "m&m":                  "M&M.NS",
    "tech mahindra":        "TECHM.NS",
    "techm":                "TECHM.NS",

    # ── IT sector ─────────────────────────────────────────────────────────────
    "wipro":                "WIPRO.NS",
    "hcl technologies":     "HCLTECH.NS",
    "hcl tech":             "HCLTECH.NS",
    "hcltech":              "HCLTECH.NS",
    "hcl":                  "HCLTECH.NS",
    "ltimindtree":          "LTIM.NS",
    "lti mindtree":         "LTIM.NS",
    "mphasis":              "MPHASIS.NS",

    # ── Pharma ────────────────────────────────────────────────────────────────
    "sun pharma":           "SUNPHARMA.NS",
    "sun pharmaceutical":   "SUNPHARMA.NS",
    "sunpharma":            "SUNPHARMA.NS",
    "dr reddy":             "DRREDDY.NS",
    "dr reddys":            "DRREDDY.NS",
    "cipla":                "CIPLA.NS",
    "divi laboratories":    "DIVISLAB.NS",
    "divi labs":            "DIVISLAB.NS",
    "divislab":             "DIVISLAB.NS",
    "lupin":                "LUPIN.NS",

    # ── FMCG ──────────────────────────────────────────────────────────────────
    "hindustan unilever":   "HINDUNILVR.NS",
    "hul":                  "HINDUNILVR.NS",
    "hindunilvr":           "HINDUNILVR.NS",
    "nestle india":         "NESTLEIND.NS",
    "nestle":               "NESTLEIND.NS",
    "nestleind":            "NESTLEIND.NS",
    "itc":                  "ITC.NS",
    "itc limited":          "ITC.NS",
    "dabur":                "DABUR.NS",
    "britannia":            "BRITANNIA.NS",
    "marico":               "MARICO.NS",

    # ── Energy / Power ────────────────────────────────────────────────────────
    "ntpc":                 "NTPC.NS",
    "power grid":           "POWERGRID.NS",
    "powergrid":            "POWERGRID.NS",
    "adani green":          "ADANIGREEN.NS",
    "adani ports":          "ADANIPORTS.NS",
    "adani enterprises":    "ADANIENT.NS",
    "adani":                "ADANIENT.NS",
    "coal india":           "COALINDIA.NS",
    "coalindia":            "COALINDIA.NS",
    "ongc":                 "ONGC.NS",
    "oil and natural gas":  "ONGC.NS",
    "ioc":                  "IOC.NS",
    "indian oil":           "IOC.NS",
    "bpcl":                 "BPCL.NS",

    # ── Auto / Industrials ────────────────────────────────────────────────────
    "maruti":               "MARUTI.NS",
    "maruti suzuki":        "MARUTI.NS",
    "hero motocorp":        "HEROMOTOCO.NS",
    "hero":                 "HEROMOTOCO.NS",
    "eicher motors":        "EICHERMOT.NS",
    "royal enfield":        "EICHERMOT.NS",
    "larsen and toubro":    "LT.NS",
    "larsen toubro":        "LT.NS",
    "l and t":              "LT.NS",
    "lt":                   "LT.NS",
    "bhel":                 "BHEL.NS",

    # ── Metals / Cement ───────────────────────────────────────────────────────
    "jsw steel":            "JSWSTEEL.NS",
    "jswsteel":             "JSWSTEEL.NS",
    "hindalco":             "HINDALCO.NS",
    "ultratech cement":     "ULTRACEMCO.NS",
    "ultratech":            "ULTRACEMCO.NS",
    "shree cement":         "SHREECEM.NS",
    "ambuja cement":        "AMBUJACEM.NS",
    "acc":                  "ACC.NS",
    "vedanta":              "VEDL.NS",

    # ── Paints / Specialty ────────────────────────────────────────────────────
    "asian paints":         "ASIANPAINT.NS",
    "asianpaint":           "ASIANPAINT.NS",
    "asian paint":          "ASIANPAINT.NS",
    "berger paints":        "BERGEPAINT.NS",

    # ── Financial / Insurance / NBFCs ─────────────────────────────────────────
    "bajaj allianz":        "BAJAJFINSV.NS",
    "shriram finance":      "SHRIRAMFIN.NS",
    "muthoot finance":      "MUTHOOTFIN.NS",
    "cholamandalam":        "CHOLAFIN.NS",
    "lt finance":           "L&TFH.NS",
    "pnb":                  "PNB.NS",
    "punjab national bank": "PNB.NS",
    "canara bank":          "CANBK.NS",
    "bank of baroda":       "BANKBARODA.NS",
    "bob":                  "BANKBARODA.NS",
    "yes bank":             "YESBANK.NS",
    "indusind bank":        "INDUSINDBK.NS",
    "indusind":             "INDUSINDBK.NS",

    # ── Telecom / Media ───────────────────────────────────────────────────────
    "bharti airtel":        "BHARTIARTL.NS",
    "airtel":               "BHARTIARTL.NS",
    "bhartiartl":           "BHARTIARTL.NS",
    "vodafone idea":        "IDEA.NS",
    "vi":                   "IDEA.NS",
    "idea":                 "IDEA.NS",
    "zomato":               "ZOMATO.NS",
    "nykaa":                "FSN.NS",
    "paytm":                "PAYTM.NS",
    "dmart":                "DMART.NS",
    "avenue supermarts":    "DMART.NS",
    "irctc":                "IRCTC.NS",

    # ── Common misspellings / variations ──────────────────────────────────────
    "relaince":             "RELIANCE.NS",   # common typo
    "infosis":              "INFY.NS",        # common typo
    "infosy":               "INFY.NS",
    "tatamtor":             "TATAMOTORS.NS",
    "hdfc bk":              "HDFCBANK.NS",
}

# Pre-sort by key length descending so longest/most-specific match wins
_ALIAS_MAP_SORTED: list[tuple[str, str]] = sorted(
    _ALIAS_MAP.items(), key=lambda kv: len(kv[0]), reverse=True
)


# ── Yahoo Finance → Twelve Data symbol map (indices and special cases) ────────
# For regular .NS stocks the conversion is mechanical: strip ".NS", append ":NSE".
# For .BO stocks: strip ".BO", append ":BSE".
# Indices and special symbols need an explicit override.

_YAHOO_TO_TD_OVERRIDE: dict[str, str] = {
    "^NSEI":      "NIFTY:NSE",
    "^BSESN":     "BSE:BSE",         # BSE Sensex
    "^NSEBANK":   "BANKNIFTY:NSE",
    "^CNXIT":     "NIFTYIT:NSE",     # Twelve Data name for CNX IT
    "^CNXPHARMA": "NIFTYPHARMA:NSE",
    "^CNXAUTO":   "NIFTYAUTO:NSE",
    "^CNXFMCG":   "NIFTYFMCG:NSE",
    "^CNXMETAL":  "NIFTYMETAL:NSE",
    "^NSEMDCP50": "NIFTYMDCP50:NSE",
    "^INDIAVIX":  "INDIAVIX:NSE",
    # BSE index — Twelve Data uses a different format
    "BAJAJ-AUTO.NS": "BAJAJAUTO:NSE",  # hyphen not valid in TD symbol
    "L&TFH.NS":   "LTFH:NSE",         # & not valid in TD symbol
    "M&M.NS":     "MM:NSE",           # & not valid in TD symbol
}


def _normalise(text: str) -> str:
    """Lowercase, remove punctuation except '&', collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s&]", " ", text)   # keep word chars, spaces, &
    text = re.sub(r"\s+", " ", text).strip()
    return text


def resolve_to_yahoo(name: str) -> str | None:
    """Resolve a natural language name/alias/ticker to a Yahoo Finance ticker.

    Returns None if the name cannot be resolved — the caller should treat the
    original string as a raw ticker and let Yahoo validate it.

    Examples
    --------
    resolve_to_yahoo("reliance industries")  → "RELIANCE.NS"
    resolve_to_yahoo("Infosys")              → "INFY.NS"
    resolve_to_yahoo("TCS")                  → "TCS.NS"
    resolve_to_yahoo("nifty 50")             → "^NSEI"
    resolve_to_yahoo("RELIANCE.NS")          → "RELIANCE.NS"  (pass-through)
    resolve_to_yahoo("UNKNOWN_XYZ")          → None
    """
    # Already looks like a Yahoo ticker — pass through
    if (name.endswith(".NS") or name.endswith(".BO")
            or name.startswith("^")
            or (name.isupper() and "." in name)):
        return name

    normalised = _normalise(name)

    # 1. Exact match
    hit = _ALIAS_MAP.get(normalised)
    if hit:
        return hit

    # 2. Longest-prefix scan (handles "tata motors limited quarterly" → tata motors)
    for alias, ticker in _ALIAS_MAP_SORTED:
        if normalised.startswith(alias) or alias.startswith(normalised):
            # Require at least 60% overlap to avoid false positives on short aliases
            overlap = min(len(alias), len(normalised))
            longer  = max(len(alias), len(normalised))
            if overlap / longer >= 0.60:
                return ticker

    # 3. Token overlap — for queries like "infosys limited quarterly analysis"
    query_tokens = set(normalised.split())
    best_score   = 0
    best_ticker  = None
    for alias, ticker in _ALIAS_MAP_SORTED:
        alias_tokens = set(alias.split())
        if not alias_tokens:
            continue
        shared = query_tokens & alias_tokens
        score  = len(shared) / len(alias_tokens)
        if score > best_score and score >= 0.75:  # ≥75% of alias tokens present
            best_score  = score
            best_ticker = ticker
    if best_ticker:
        return best_ticker

    # 4. Unresolved — return None so caller handles as raw ticker
    return None


def yahoo_to_twelvedata(yahoo_ticker: str) -> str | None:
    """Convert a Yahoo Finance ticker to Twelve Data symbol format.

    Returns None if conversion is not possible (e.g. unsupported index).

    Examples
    --------
    yahoo_to_twelvedata("RELIANCE.NS")  → "RELIANCE:NSE"
    yahoo_to_twelvedata("INFY.NS")      → "INFY:NSE"
    yahoo_to_twelvedata("^NSEI")        → "NIFTY:NSE"
    yahoo_to_twelvedata("RELIANCE.BO")  → "RELIANCE:BSE"
    """
    # Check explicit overrides first
    override = _YAHOO_TO_TD_OVERRIDE.get(yahoo_ticker)
    if override:
        return override

    # NSE stocks — strip ".NS", append ":NSE"
    if yahoo_ticker.endswith(".NS"):
        base = yahoo_ticker[:-3]
        # Reject symbols with characters curl/TD doesn't accept
        if re.match(r"^[A-Z0-9]+$", base):
            return f"{base}:NSE"

    # BSE stocks — strip ".BO", append ":BSE"
    if yahoo_ticker.endswith(".BO"):
        base = yahoo_ticker[:-3]
        if re.match(r"^[A-Z0-9]+$", base):
            return f"{base}:BSE"

    # Unrecognised format — cannot convert
    return None
