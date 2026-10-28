import numpy as np
import pandas as pd

from features import build_xy

WARMUP = 199  # first row where every rolling window is full


def make_trades(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "price": 1 + np.cumsum(rng.normal(0, 0.001, n)),
        "qty": rng.uniform(1, 10, n),
        "side_sign": rng.choice([-1, 1], n),
    })


def test_features_ignore_the_future():
    trades = make_trades()
    changed = trades.copy()
    changed.loc[1500:, "price"] += 5
    X1, _ = build_xy(trades)
    X2, _ = build_xy(changed)
    safe_rows = 1500 - WARMUP - 1
    assert np.allclose(X1.iloc[:safe_rows].values, X2.iloc[:safe_rows].values)


def test_label_is_strict_price_rise_after_horizon():
    trades = make_trades()
    X, y = build_xy(trades, horizon=10)
    price = trades["price"].to_numpy()
    expected = (price[WARMUP + 10:WARMUP + 10 + 50] > price[WARMUP:WARMUP + 50]).astype(int)
    assert (y.iloc[:50].to_numpy() == expected).all()


def test_no_missing_values_and_aligned_lengths():
    X, y = build_xy(make_trades())
    assert len(X) == len(y)
    assert not X.isna().any().any()
