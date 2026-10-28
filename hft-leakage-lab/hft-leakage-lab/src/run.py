"""Shuffled K-fold vs purged walk-forward on real trade data.

    cd src
    python run.py            # LogReg + GBM
    python run.py --quick    # LogReg only, much faster
"""
import argparse
import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import load_trades
from features import build_xy
from splits import walk_forward_splits

warnings.filterwarnings("ignore")

HORIZON = 10
N_FOLDS = 5
EMBARGO = 10
GAP_FLAG = 0.03


def logreg():
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=500, C=0.1))])


def gbm():
    return GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=42)


def mean_auc(make_model, X, y, splits):
    scores = []
    for train, test in splits:
        if y.iloc[test].nunique() < 2:
            continue
        model = make_model().fit(X.iloc[train], y.iloc[train])
        scores.append(roc_auc_score(y.iloc[test], model.predict_proba(X.iloc[test])[:, 1]))
    return float(np.mean(scores)), float(np.std(scores))


def label_breakdown(trades, horizon):
    move = (trades["price"].shift(-horizon) - trades["price"]).dropna()
    return (move > 0).mean(), (move == 0).mean(), (move < 0).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="path to the trades CSV")
    parser.add_argument("--nrows", type=int, help="only read the first N rows")
    parser.add_argument("--quick", action="store_true", help="logistic regression only")
    args = parser.parse_args()

    trades = load_trades(args.data, args.nrows)
    print(f"trades: {len(trades):,}  |  {trades.index[0]}  ->  {trades.index[-1]}")
    up, flat, down = label_breakdown(trades, HORIZON)
    print(f"price after {HORIZON} ticks: up {up:.1%}, flat {flat:.1%}, down {down:.1%}")

    X, y = build_xy(trades, horizon=HORIZON)
    print(f"samples: {len(X):,}  |  label=1: {y.mean():.1%}  |  features: {list(X.columns)}\n")

    naive = list(KFold(N_FOLDS, shuffle=True, random_state=0).split(X))
    walk = list(walk_forward_splits(len(X), N_FOLDS, horizon=HORIZON, embargo=EMBARGO))

    models = {"LogReg": logreg} if args.quick else {"LogReg": logreg, "GBM": gbm}
    print(f"{'model':<8}{'shuffled K-fold AUC':>24}{'walk-forward AUC':>24}{'gap':>9}")
    for name, make_model in models.items():
        n_mean, n_std = mean_auc(make_model, X, y, naive)
        w_mean, w_std = mean_auc(make_model, X, y, walk)
        gap = n_mean - w_mean
        flag = "  large gap" if gap > GAP_FLAG else ""
        print(f"{name:<8}{n_mean:>15.4f} +/- {n_std:.4f}{w_mean:>15.4f} +/- {w_std:.4f}{gap:>+9.4f}{flag}")


if __name__ == "__main__":
    main()
