"""Load Binance spot trade ticks from the public CSV dump."""
import pathlib

import numpy as np
import pandas as pd

DEFAULT_CSV = pathlib.Path(__file__).resolve().parent.parent / "0GUSDT-trades-2026-08.csv"

COLUMNS = ["trade_id", "price", "qty", "quote_qty",
           "timestamp_us", "is_buyer_maker", "is_best_match"]

DTYPES = {
    "trade_id": "int64",
    "price": "float64",
    "qty": "float64",
    "quote_qty": "float64",
    "timestamp_us": "int64",
    "is_buyer_maker": "bool",
    "is_best_match": "bool",
}


def load_trades(path=None, nrows=None):
    """Return trades in time order, indexed by UTC timestamp.

    side_sign is +1 when the buyer was the aggressor, -1 when the seller was.
    """
    path = pathlib.Path(path) if path else DEFAULT_CSV
    if not path.exists():
        raise FileNotFoundError(f"trades file not found: {path}")

    df = pd.read_csv(path, header=None, names=COLUMNS, dtype=DTYPES,
                     nrows=nrows, engine="c")
    df = df.sort_values("trade_id")
    df.index = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    df.index.name = "ts"

    # buyer is the maker -> the seller crossed the spread
    df["side_sign"] = np.where(df["is_buyer_maker"], -1, 1).astype("int8")
    return df.drop(columns=["trade_id", "timestamp_us"])
