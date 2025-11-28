from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------

# This assumes: project_root/modules/features.py
# and data/ is at project_root/data
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed_data"

HISTORICAL_PROCESSED = PROCESSED_DATA_DIR / "historical_data_processed.csv"
FEAR_GREED_PROCESSED = PROCESSED_DATA_DIR / "fear_greed_index_processed.csv"

FEATURES_DAILY_OUTPUT = PROCESSED_DATA_DIR / "trades_daily_with_sentiment.csv"


# ---------------------------------------------------------------------
# Core feature engineering
# ---------------------------------------------------------------------

def load_processed_data(
    historical_path: Path = HISTORICAL_PROCESSED,
    fg_path: Path = FEAR_GREED_PROCESSED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load already processed trade & fear-greed CSVs."""
    if not historical_path.exists():
        raise FileNotFoundError(f"Historical data not found at {historical_path}")
    if not fg_path.exists():
        raise FileNotFoundError(f"Fear-Greed data not found at {fg_path}")

    trades = pd.read_csv(historical_path, parse_dates=["timestamp_ist"], dayfirst=True)
    fg = pd.read_csv(fg_path)

    # date column may have been saved as object -> convert back to date
    if "date" in trades.columns:
        trades["date"] = pd.to_datetime(trades["date"]).dt.date
    if "date" in fg.columns:
        fg["date"] = pd.to_datetime(fg["date"]).dt.date

    return trades, fg


def build_daily_trade_features(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate trade-level data into daily per-account features.

    Input columns (from your processed file):
    account, coin, execution_price, size_tokens, size_usd, side,
    timestamp_ist, start_position, direction, closed_pnl, transaction_hash,
    order_id, crossed, fee, trade_id, timestamp, date, coin_clean
    """

    df = trades.copy()

    # -----------------------------------------------------------------
    # 1) Basic cleaning / helper columns
    # -----------------------------------------------------------------

    # Ensure side is standardized
    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.upper().str.strip()
    else:
        raise KeyError("Column 'side' not found in trades dataframe")

    # Boolean: is this a BUY trade?
    df["is_buy"] = df["side"].eq("BUY")

    # Ensure timestamp_ist is datetime
    if "timestamp_ist" in df.columns:
        df["timestamp_ist"] = pd.to_datetime(
            df["timestamp_ist"],
            errors="coerce",
            dayfirst=True,
        )
    else:
        raise KeyError("Column 'timestamp_ist' not found in trades dataframe")

    # Ensure numeric types
    for col in ["size_usd", "closed_pnl", "fee", "start_position"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Boolean for crossed (market) orders
    if "crossed" in df.columns:
        # if it's not already boolean, coerce from strings / ints
        df["crossed"] = df["crossed"].astype(str).str.upper().map(
            {"TRUE": True, "FALSE": False}
        ).fillna(False)

    # Ensure date is parsed correctly
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Sort trades properly before aggregation
    df = df.sort_values(["account", "date", "timestamp_ist"]).reset_index(drop=True)


    # -----------------------------------------------------------------
    # 2) Group by (account, date) to get daily behavior & performance
    # -----------------------------------------------------------------

    group_cols = ["account", "date"]

    daily = df.groupby(group_cols).agg(
        daily_pnl=("closed_pnl", "sum"),             # total PnL for the day
        num_trades=("trade_id", "count"),            # number of trades
        volume_usd=("size_usd", "sum"),              # total notional
        avg_trade_size_usd=("size_usd", "mean"),     # average trade size
        long_fraction=("is_buy", "mean"),            # fraction of BUY trades
        total_fee_usd=("fee", "sum"),                # total fees paid
        crossed_fraction=("crossed", "mean"),        # % of market orders
        avg_start_position=("start_position", "mean"),
        first_trade_time=("timestamp_ist", "min"),
        last_trade_time=("timestamp_ist", "max"),
    ).reset_index()

    # -----------------------------------------------------------------
    # 3) Derived features at daily level
    # -----------------------------------------------------------------

    # Daily win flag: was the account profitable today?
    daily["win_flag"] = (daily["daily_pnl"] > 0).astype(int)

    # Daily efficiency: PnL per USD traded
    daily["pnl_per_usd_traded"] = daily["daily_pnl"] / daily["volume_usd"].replace(0, np.nan)

    # Trading span (in hours) within the day
    # Make sure these are datetime64 before subtracting
    if np.issubdtype(daily["first_trade_time"].dtype, np.datetime64) and \
       np.issubdtype(daily["last_trade_time"].dtype, np.datetime64):

        trading_span = (
            daily["last_trade_time"] - daily["first_trade_time"]
        ) / pd.Timedelta(hours=1)

        daily["trading_span_hours"] = trading_span
    else:
        # fallback if something went wrong; you can inspect later
        daily["trading_span_hours"] = np.nan

    # Account-level baselines to build relative features
    account_volume_mean = daily.groupby("account")["volume_usd"].transform("mean")
    daily["relative_volume_usd"] = daily["volume_usd"] / account_volume_mean.replace(0, np.nan)

    account_trades_mean = daily.groupby("account")["num_trades"].transform("mean")
    daily["relative_num_trades"] = daily["num_trades"] / account_trades_mean.replace(0, np.nan)

    # Fee as proportion of PnL magnitude (cost vs outcome)
    daily["fee_to_pnl_abs"] = daily["total_fee_usd"] / daily["daily_pnl"].abs().replace(0, np.nan)

    # Sort final daily dataset by date, then account
    daily = daily.sort_values(["date", "account"]).reset_index(drop=True)




    return daily


def merge_with_fear_greed(
    daily: pd.DataFrame,
    fg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge daily account-level features with Fear-Greed index on date.
    Adds sentiment features: value, classification, is_greed, is_fear.
    """
    fg_local = fg.copy()

    # Standardize classification text
    if "classification" in fg_local.columns:
        fg_local["classification"] = fg_local["classification"].astype(str).str.strip().str.title()

    # Merge on date
    merged = daily.merge(fg_local[["date", "value", "classification"]], on="date", how="left")

    # Sentiment convenience flags
    merged["is_greed"] = merged["classification"].str.contains("Greed", case=False, na=False).astype(int)
    merged["is_fear"] = merged["classification"].str.contains("Fear", case=False, na=False).astype(int)

    # For models, you may want a simpler binary sentiment:
    # 1 = greed/optimistic, 0 = fear/pessimistic
    # Here we treat any "Greed" (including Extreme Greed) as 1, otherwise 0.
    merged["sentiment_binary"] = merged["is_greed"]

    return merged


def build_and_save_features(
    historical_path: Path = HISTORICAL_PROCESSED,
    fg_path: Path = FEAR_GREED_PROCESSED,
    output_path: Path = FEATURES_DAILY_OUTPUT,
) -> pd.DataFrame:
    """
    Orchestrates the full pipeline:
      1. Load processed trades + fear-greed index
      2. Build daily account-level features
      3. Merge with sentiment
      4. Save to CSV
    """
    trades, fg = load_processed_data(historical_path, fg_path)
    daily = build_daily_trade_features(trades)
    merged = merge_with_fear_greed(daily, fg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def main() -> None:
    print(f"Reading processed data from: {PROCESSED_DATA_DIR}")
    features = build_and_save_features()
    print(f"Daily features with sentiment saved to: {FEATURES_DAILY_OUTPUT}")
    print(f"Shape: {features.shape[0]} rows x {features.shape[1]} columns")


if __name__ == "__main__":
    main()
