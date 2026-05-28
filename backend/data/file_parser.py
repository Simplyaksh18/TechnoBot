import pandas as pd

# We require OHLC at minimum; date is strongly preferred but can be inferred for some files.
REQUIRED_OHLC = {"open", "high", "low", "close"}

# Common column name variants we see in exports
COLUMN_ALIASES = {
    "date": ["date", "datetime", "timestamp", "time"],
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c", "adj close", "adj_close", "adjusted close", "adjusted_close"],
    "volume": ["volume", "vol", "v"],
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                rename_map[a] = canonical
                break

    df = df.rename(columns=rename_map)
    return df

def parse_price_file(file_path: str) -> pd.DataFrame:
    """
    Parses a user-uploaded OHLC CSV into a normalized dataframe:
    columns: date (optional), open, high, low, close, volume (optional)
    """
    df = pd.read_csv(file_path)
    df = _normalize_columns(df)

    # Validate OHLC exists
    if not REQUIRED_OHLC.issubset(df.columns):
        missing = sorted(list(REQUIRED_OHLC - set(df.columns)))
        raise ValueError(f"CSV missing required OHLC columns: {missing}")

    # Parse date if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

    # Force numeric OHLC
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Keep only relevant columns (volume optional)
    keep = ["open", "high", "low", "close"]
    if "date" in df.columns:
        keep = ["date"] + keep
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        keep = keep + ["volume"]

    return df[keep].reset_index(drop=True)

def infer_timeframe(df: pd.DataFrame) -> str:
    """
    Returns: 'intraday' | 'daily' | 'weekly' | 'unknown'
    """
    if "date" not in df.columns:
        return "unknown"

    deltas = df["date"].diff().dropna()
    if deltas.empty:
        return "unknown"

    median = deltas.median()

    # pandas Timedelta supports .days; for intraday it may be 0
    if getattr(median, "days", 0) >= 7:
        return "weekly"
    if getattr(median, "days", 0) >= 1:
        return "daily"
    return "intraday"
