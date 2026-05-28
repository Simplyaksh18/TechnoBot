"""
data/lightweight_context.py — Qualitative market context when all live providers fail.

Called ONLY by _cached_derive_market_context() when every live data source
(Yahoo, Alpha Vantage, Twelve Data, stale cache) has been exhausted and the
context would otherwise be returned as trend="unknown".

Design rules
─────────────
• NO fake prices, NO invented indicator values.  Every numeric field that
  requires live data (rsi, last_price, pct_change) is explicitly None.
• trend is set to "consolidating" (never "unknown") so it passes the
  validation gate in _cached_derive_market_context().
• The _lightweight flag tells the response builder to prepend a disclaimer
  instead of presenting the analysis as live TA.
• The _qualitative_note field carries instrument-specific reasoning that the
  LLM can reference in its response.

Disclaimer injected by the caller:
  "Live market data is temporarily unavailable.
   The following is general technical reasoning — not live analysis."
"""

from __future__ import annotations

# ── Ticker classification ─────────────────────────────────────────────────────

_BROAD_INDICES = frozenset({
    "^NSEI", "^NSEBANK", "^BSESN", "^CNXIT", "^CNXPHARMA",
    "^CNXFMCG", "^CNXMETAL", "^CNXAUTO", "^CNXREALTY",
    "^CNXMIDCAP", "^CNXSMALLCAP", "^CNXENERGY", "^CNXINFRA",
    "^CNXPSUBANK", "^CNXFINANCE", "CNXMIDCAP.NS", "CNXSMALLCAP.NS",
})

_IT_SUFFIXES = ("TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
                "MPHASIS.NS", "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS")
_BANK_SUFFIXES = ("HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS",
                  "SBIN.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS",
                  "INDUSINDBK.NS", "BANDHANBNK.NS", "FEDERALBNK.NS")
_ENERGY_SUFFIXES = ("RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS",
                    "TATAPOWER.NS", "ADANIGREEN.NS", "ADANIPORTS.NS",
                    "ADANIPOWER.NS", "BPCL.NS", "IOC.NS", "GAIL.NS")
_PHARMA_SUFFIXES = ("SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
                    "APOLLOHOSP.NS", "AUROPHARMA.NS", "BIOCON.NS")
_AUTO_SUFFIXES   = ("TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS",
                    "HEROMOTOCO.NS", "EICHERMOT.NS", "ASHOKLEY.NS")
_FMCG_SUFFIXES   = ("HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",
                    "DABUR.NS", "MARICO.NS", "COLPAL.NS")


def _classify_ticker(ticker: str) -> str:
    """Return a coarse asset-class label for qualitative framing."""
    t = ticker.upper()
    if t in _BROAD_INDICES or t.startswith("^"):
        if "BANK" in t:
            return "bank_index"
        if "IT" in t or "CNXIT" in t:
            return "it_index"
        return "broad_index"
    if t in _IT_SUFFIXES:
        return "it_stock"
    if t in _BANK_SUFFIXES:
        return "bank_stock"
    if t in _ENERGY_SUFFIXES:
        return "energy_stock"
    if t in _PHARMA_SUFFIXES:
        return "pharma_stock"
    if t in _AUTO_SUFFIXES:
        return "auto_stock"
    if t in _FMCG_SUFFIXES:
        return "fmcg_stock"
    if t.endswith(".NS") or t.endswith(".BO"):
        return "indian_equity"
    return "unknown_equity"


# ── Qualitative note library ───────────────────────────────────────────────────

_NOTES: dict[str, dict[str, str]] = {
    "broad_index": {
        "1d":  (
            "Nifty 50 / broad-market indices are driven by FII/DII flows, global "
            "risk sentiment, RBI policy signals, and rupee movement. In the absence "
            "of live data, assess directional bias from: US market overnight close, "
            "Asian peer performance, India VIX level, and recent sector rotation. "
            "Support/resistance is best marked at round-number psychological levels."
        ),
        "1wk": (
            "On the weekly chart, Nifty respects multi-month consolidation zones and "
            "prior swing highs/lows. Key macro triggers — RBI MPC outcomes, Union "
            "Budget, Q results season — often define the weekly trend. Broader market "
            "breadth (advance-decline ratio) is a reliable secondary confirmation."
        ),
        "1mo": (
            "Monthly Nifty trends reflect multi-quarter earnings cycles, central bank "
            "rate trajectories, and election-year political sentiment. Long-term "
            "support clusters form at major 200-week SMA intersections and "
            "Fibonacci retracement levels from the 2020 COVID low to current highs."
        ),
    },
    "bank_index": {
        "1d":  (
            "Bank Nifty is one of the most volatile Indian indices — beta > 1.5 vs "
            "Nifty 50. Intraday direction is driven by RBI rate decisions, bond yield "
            "moves, and heavyweight HDFC Bank / ICICI Bank price action. "
            "Options OI concentration (PCR, max-pain) is an important sentiment gauge."
        ),
        "1wk": (
            "Weekly Bank Nifty trend follows credit-growth data, NPA cycle position, "
            "and NIM (net interest margin) outlook. Private-sector banks typically "
            "lead rallies; PSU banks outperform during policy-driven risk-on phases."
        ),
        "1mo": (
            "Monthly Bank Nifty performance correlates closely with India's credit "
            "cycle and global rate environment. Look for NPA resolution milestones "
            "and RBI's monetary policy stance as primary trend determinants."
        ),
    },
    "it_index": {
        "1d":  (
            "IT indices (CNXIT, Nifty IT) react to USD/INR rate, Nasdaq futures, "
            "and US tech sentiment. Earnings guidance from TCS/Infosys/Wipro sets the "
            "sector tone for weeks after results. Daily moves above ±2% often signal "
            "a macro trigger rather than a technical breakout."
        ),
        "1wk": (
            "Weekly IT trend aligns with global tech cycle and INR direction. "
            "A weak rupee is broadly positive for IT export earnings. "
            "Watch for deal-win announcements and quarterly revenue-growth commentary."
        ),
        "1mo": (
            "Multi-month IT sector direction is driven by US enterprise IT spending "
            "sentiment, US recession probability, and domestic hiring/margin trends. "
            "Valuation re-ratings (PE expansion/contraction) amplify the trend."
        ),
    },
    "it_stock": {
        "1d":  (
            "Indian IT large-caps shadow USD/INR and Nasdaq closely. Intraday "
            "momentum is usually front-loaded; fades are common in the last hour. "
            "Relative strength vs CNXIT is a useful filter for stock selection."
        ),
        "1wk": (
            "Weekly IT stock trends reflect deal-pipeline visibility and quarterly "
            "earnings beat/miss cycles. INR depreciation tailwind and US BFSI "
            "spending remain the two primary drivers over a 4–8 week horizon."
        ),
        "1mo": (
            "Multi-month IT stock performance depends on revenue-growth visibility, "
            "margin trajectory, and valuation multiple relative to US software peers."
        ),
    },
    "bank_stock": {
        "1d":  (
            "Private-sector bank stocks react sharply to RBI commentary, bond yields, "
            "and NBFC stress signals. HDFC Bank and ICICI Bank are high-liquidity "
            "proxies — their intraday action often leads the sector."
        ),
        "1wk": (
            "Weekly bank-stock direction tracks credit-growth momentum, NIM guidance, "
            "and system-level liquidity. Watch for RBI policy minutes and SEBI "
            "regulatory announcements for directional catalysts."
        ),
        "1mo": (
            "Multi-month bank stock trends reflect the full NPA cycle and capex "
            "lending environment. A falling yield curve typically pressures NIMs "
            "while a rising curve supports them."
        ),
    },
    "energy_stock": {
        "1d":  (
            "Energy stocks (Reliance, ONGC, BPCL) track Brent crude closely. "
            "Refining margin data (Singapore GRM) and government fuel pricing "
            "decisions are key intraday catalysts for OMCs."
        ),
        "1wk": (
            "Weekly energy sector direction follows crude oil cycle, OPEC production "
            "decisions, and INR impact on import costs. Reliance Industries' "
            "diversified businesses (Jio, Retail) provide a partial buffer."
        ),
        "1mo": (
            "Multi-month energy stock performance reflects global crude supply-demand "
            "balance, domestic fuel subsidies, and capex cycle for new exploration."
        ),
    },
    "pharma_stock": {
        "1d":  (
            "Pharma stocks are relatively defensive intraday. Key catalysts are USFDA "
            "inspection outcomes, ANDA approvals, and global generic pricing trends. "
            "API cost inflation can pressure margins in a rising commodity environment."
        ),
        "1wk": (
            "Weekly pharma direction is driven by USFDA import alert resolution "
            "timelines, US generics pricing environment, and domestic formulation "
            "volume growth. Sun Pharma / Dr Reddy's lead the sector by market cap."
        ),
        "1mo": (
            "Multi-month pharma performance reflects US patent cliff opportunities, "
            "complex generic launches, and specialty pharma pipeline execution. "
            "INR depreciation boosts export-heavy pharma earnings."
        ),
    },
    "auto_stock": {
        "1d":  (
            "Auto stocks react to monthly SIAM volume data, commodity input costs, "
            "and EV transition announcements. Tata Motors (JLR exposure) and M&M "
            "(tractor/SUV mix) often diverge from Maruti's passenger car trend."
        ),
        "1wk": (
            "Weekly auto direction tracks monthly volume growth, inventory levels "
            "at dealer networks, and semiconductor availability. Festival-season "
            "demand visibility (Sep–Nov) drives pre-positioning by institutional buyers."
        ),
        "1mo": (
            "Multi-month auto trends follow the capex/consumer credit cycle, rural "
            "income (linked to agricultural output), and EV adoption ramp. "
            "Commodity cost pass-through ability is a key margin indicator."
        ),
    },
    "fmcg_stock": {
        "1d":  (
            "FMCG stocks (HUL, ITC, Nestle) are defensive — low beta vs Nifty. "
            "Volume-growth commentary in quarterly results drives the biggest "
            "single-day moves. Rural demand proxies (2W sales, tractor volumes) "
            "serve as leading indicators."
        ),
        "1wk": (
            "Weekly FMCG direction tracks raw material cost (palm oil, crude "
            "derivatives) and urban vs rural consumption mix. Premium portfolio "
            "performance is a key differentiator at current valuation levels."
        ),
        "1mo": (
            "Multi-month FMCG performance is driven by volume recovery in rural "
            "markets, GST policy stability, and pricing power relative to input "
            "cost inflation. ITC benefits from a distinct tobacco-plus-FMCG mix."
        ),
    },
    "indian_equity": {
        "1d":  (
            "Indian equity follows the broader Nifty direction intraday. Key "
            "daily inputs: FII/DII provisional flow data (NSE), sector-specific "
            "catalyst (results, regulatory news), and global risk-on/risk-off mood."
        ),
        "1wk": (
            "Weekly direction for Indian equities tracks earnings season trajectory, "
            "macro data prints (CPI, IIP, PMI), and RBI policy stance. "
            "Mid-cap and small-cap segments show higher volatility than large-caps."
        ),
        "1mo": (
            "Multi-month Indian equity trends reflect GDP growth expectations, "
            "corporate earnings cycle, and global emerging-market flows. "
            "Political event risk (elections, budget) can dominate over several months."
        ),
    },
    "unknown_equity": {
        "1d":  (
            "Without live data, assess intraday bias from: sector peers, Nifty "
            "trend, and recent stock-specific news. Volume confirmation on any "
            "directional move is essential before entering a position."
        ),
        "1wk": (
            "On a weekly basis, track the stock's relative performance vs its "
            "sector index and broad Nifty. Earnings-driven gaps often set the "
            "weekly trend; monitor for analyst rating changes."
        ),
        "1mo": (
            "Multi-month analysis requires understanding the company's earnings "
            "growth cycle, valuation relative to peers, and sector tailwinds "
            "or headwinds. Promoter holding changes are a useful insider signal."
        ),
    },
}

# Default note when interval_key is not found
_DEFAULT_NOTE = (
    "Live data is temporarily unavailable. General market principles apply: "
    "confirm any directional bias with volume, sector peers, and macro backdrop "
    "before drawing conclusions."
)


def _get_qualitative_note(asset_class: str, interval_key: str) -> str:
    """Return the interval-specific qualitative note for the given asset class."""
    bucket = _NOTES.get(asset_class, _NOTES["unknown_equity"])
    return bucket.get(interval_key, _DEFAULT_NOTE)


# ── Public API ────────────────────────────────────────────────────────────────

def build_lightweight_context(ticker: str, interval_key: str = "1d") -> dict:
    """Build a qualitative market context dict when all live providers have failed.

    The returned dict is schema-compatible with derive_market_context() output:
      • trend = "consolidating"  (NOT "unknown" — passes the validation gate)
      • All numeric live-data fields (rsi, last_price, pct_change) are None
      • _lightweight = True      — caller should prepend a disclaimer
      • _qualitative_note        — instrument-specific reasoning for the LLM

    This is ONLY called after Yahoo Finance, Alpha Vantage, Twelve Data, and the
    stale cache have all been exhausted for the requested ticker/interval.
    """
    asset_class = _classify_ticker(ticker)
    note = _get_qualitative_note(asset_class, interval_key)

    return {
        # ── Core context fields (schema-identical to derive_market_context) ──
        "trend":         "consolidating",
        "momentum":      "neutral",
        "volatility":    "moderate",
        "participation": "average",
        "rsi":           None,
        "last_price":    None,
        "pct_change":    None,
        # ── Status flags ────────────────────────────────────────────────────
        "rate_limited":  False,
        "error_type":    None,
        "fail_reason":   None,
        "data":          None,
        "status":        "lightweight",
        "source":        "lightweight",
        # ── Lightweight-specific fields ──────────────────────────────────────
        "_lightweight":       True,
        "_asset_class":       asset_class,
        "_qualitative_note":  note,
    }
