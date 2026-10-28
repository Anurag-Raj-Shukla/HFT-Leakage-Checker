import pathlib
import tempfile

from data import load_trades

ROWS = [
    "3,1.02,5,5.1,1700000000000300,False,True",
    "1,1.00,10,10,1700000000000100,True,True",
    "2,1.01,20,20.2,1700000000000200,False,True",
]


def write_csv(lines):
    path = pathlib.Path(tempfile.mkdtemp()) / "trades.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_sorted_by_time_and_columns_present():
    df = load_trades(write_csv(ROWS))
    assert list(df["price"]) == [1.00, 1.01, 1.02]
    assert df.index.is_monotonic_increasing
    assert {"price", "qty", "side_sign"} <= set(df.columns)


def test_buyer_maker_means_seller_aggressor():
    df = load_trades(write_csv(ROWS))
    assert list(df["side_sign"]) == [-1, 1, 1]
