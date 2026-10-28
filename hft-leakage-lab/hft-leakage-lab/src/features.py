"""Trade-flow features and the forward-looking label.

Every feature at row t uses rows <= t only. The label looks `horizon` ticks ahead.
"""
import numpy as np
import pandas as pd


def build_xy(df, horizon=10):
    """Build (X, y) from the output of data.load_trades.

    y is 1 only if the price is strictly higher `horizon` ticks later;
    an unchanged price counts as 0.
    """
    price = df["price"]
    qty = df["qty"]
    signed_qty = df["side_sign"] * qty

    x = pd.DataFrame(index=df.index)

    for w in (10, 50, 200):
        flow = signed_qty.rolling(w, min_periods=w).sum()
        volume = qty.rolling(w, min_periods=w).sum()
        x[f"ofi_{w}"] = flow / (volume + 1e-9)

    for w in (10, 50):
        x[f"ret_{w}"] = np.log(price / price.shift(w))

    vwap = (price * qty).rolling(50, min_periods=50).sum() / (
        qty.rolling(50, min_periods=50).sum() + 1e-9)
    x["vwap_dev_50"] = (price - vwap) / (vwap + 1e-9)

    x["buy_ratio_30"] = (df["side_sign"] == 1).astype(int).rolling(30, min_periods=30).mean()
    x["vol_30"] = qty.rolling(30, min_periods=30).std()

    future_move = price.shift(-horizon) - price
    y = (future_move > 0).astype(int)

    keep = x.notna().all(axis=1) & future_move.notna()
    return x[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
