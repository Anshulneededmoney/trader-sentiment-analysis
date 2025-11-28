from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # adjust if needed
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed_data"

HISTORICAL_RAW = RAW_DATA_DIR / "historical_data.csv"
FEAR_GREED_RAW = RAW_DATA_DIR / "fear_greed_index.csv"

HISTORICAL_PROCESSED = PROCESSED_DATA_DIR / "historical_data_processed.csv"
FEAR_GREED_PROCESSED = PROCESSED_DATA_DIR / "fear_greed_index_processed.csv"


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def _ensure_processed_dir() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names to snake_case-ish: lower, replace spaces and '/' with '_'.
    """
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
                  .str.replace(" ", "_")
                  .str.replace("/", "_", regex=False)
                  .str.replace("(", "", regex=False)
                  .str.replace(")", "", regex=False)
                  .str.lower()
    )
    return df


# ---------------------------------------------------------------------
# Historical trades preprocessing
# ---------------------------------------------------------------------

def process_historical_data(
    input_path: Path = HISTORICAL_RAW,
    output_path: Path = HISTORICAL_PROCESSED,
) -> pd.DataFrame:
    """
    Clean Hyperliquid historical trades data and save a processed CSV.

    Expected raw columns:
    Account,Coin,Execution Price,Size Tokens,Size USD,Side,
    Timestamp IST,Start Position,Direction,Closed PnL,Transaction Hash,
    Order ID,Crossed,Fee,Trade ID,Timestamp
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Historical data not found at {input_path}")

    df = pd.read_csv(input_path)

    # Standardize column names
    df = _standardize_columns(df)

    # Rename some to clearer names (optional but nice)
    rename_map = {
        "execution_price": "execution_price",
        "size_tokens": "size_tokens",
        "size_usd": "size_usd",
        "timestamp_ist": "timestamp_ist",
        "closed_pnl": "closed_pnl",
        "transaction_hash": "transaction_hash",
        "order_id": "order_id",
        "trade_id": "trade_id",
    }
    df = df.rename(columns=rename_map)

    # Parse timestamp (IST) -> datetime
    if "timestamp_ist" in df.columns:
        # Example format: "02-12-2024 22:50"
        df["timestamp_ist"] = pd.to_datetime(
            df["timestamp_ist"],
            format="%d-%m-%Y %H:%M",
            errors="coerce",
            dayfirst=True,
        )
        # Derive date for daily-level merging
        df["date"] = df["timestamp_ist"].dt.date

    # If numeric timestamp present, keep as datetime too (ms-ish)
    if "timestamp" in df.columns:
        # Often in ms; if you see weird dates you can divide by 1000 outside
        # For now, try as seconds; easy to change.
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="ignore")

    # Clean string columns
    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.upper().str.strip()

    if "direction" in df.columns:
        df["direction"] = df["direction"].astype(str).str.capitalize().str.strip()

    if "coin" in df.columns:
        # Example: "@107" -> "107"
        df["coin"] = df["coin"].astype(str).str.strip()
        df["coin_clean"] = df["coin"].str.replace("@", "", regex=False)

    # Boolean-like columns
    if "crossed" in df.columns:
        df["crossed"] = (
            df["crossed"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({"TRUE": True, "FALSE": False})
        )

    # Cast numeric columns
    for col in ["execution_price", "size_tokens", "size_usd", "closed_pnl", "fee", "start_position"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort data
    sort_cols = [c for c in ["account", "timestamp_ist", "trade_id"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # Save processed
    _ensure_processed_dir()
    df.to_csv(output_path, index=False)

    return df


# ---------------------------------------------------------------------
# Fear-Greed index preprocessing
# ---------------------------------------------------------------------

def process_fear_greed_index(
    input_path: Path = FEAR_GREED_RAW,
    output_path: Path = FEAR_GREED_PROCESSED,
) -> pd.DataFrame:
    """
    Clean Fear-Greed index data and save a processed CSV.

    Expected raw columns:
    timestamp,value,classification,date
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Fear-Greed data not found at {input_path}")

    fg = pd.read_csv(input_path)

    # Standardize column names
    fg = _standardize_columns(fg)

    # Parse unix timestamp (seconds) to datetime
    if "timestamp" in fg.columns:
        fg["timestamp"] = pd.to_datetime(fg["timestamp"], unit="s", errors="coerce")
    else:
        fg["timestamp"] = pd.NaT

    # Parse 'date' column (fallback if given as string)
    if "date" in fg.columns:
        fg["date"] = pd.to_datetime(fg["date"], errors="coerce").dt.date
    else:
        # derive from timestamp if needed
        fg["date"] = fg["timestamp"].dt.date

    # Standardize classification text
    if "classification" in fg.columns:
        fg["classification"] = (
            fg["classification"]
            .astype(str)
            .str.strip()
            .str.title()
        )

    # Ensure value is numeric
    if "value" in fg.columns:
        fg["value"] = pd.to_numeric(fg["value"], errors="coerce")

    # Drop rows with no date (if any)
    fg = fg.dropna(subset=["date"])

    # Sort by date
    fg = fg.sort_values("date").reset_index(drop=True)

    # Save processed
    _ensure_processed_dir()
    fg.to_csv(output_path, index=False)

    return fg


# ---------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------

def main() -> None:
    print(f"Reading from: {RAW_DATA_DIR}")
    print(f"Writing to:   {PROCESSED_DATA_DIR}")

    hist = process_historical_data()
    print(f"Historical data processed: {hist.shape[0]} rows, {hist.shape[1]} columns")

    fg = process_fear_greed_index()
    print(f"Fear-Greed index processed: {fg.shape[0]} rows, {fg.shape[1]} columns")


if __name__ == "__main__":
    main()
