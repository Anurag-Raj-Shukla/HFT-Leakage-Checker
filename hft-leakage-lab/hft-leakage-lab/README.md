# hft-leakage-lab

Short-horizon price prediction on real trade data, validated two ways, to show how much a
shuffled cross-validation can flatter a model and what an honest validation looks like.

I built this after reading HFT Architecture's Junior Machine Learning Engineer posting.
The posting asks for people who can explain why ordinary cross-validation fails on time
series and who keep the future out of the training set. This repo is my attempt at that
problem, with the numbers and the limits written down.

## The question

Given the last few hundred trades, can a model tell whether the price will be higher
10 ticks from now? And, more importantly: how much of the score I see depends on how I
split the data?

## Data

- Binance spot trades for 0GUSDT, August 2026 (`0GUSDT-trades-2026-08.csv`, from Binance public data).
- Trade ticks only. There is no order book depth, so "order flow" here means signed trade volume.
- 453,751 samples after warm-up and label trimming.
- The CSV is not committed. Put it in the repo root or pass `--data path/to/file.csv`.

## Features (all use rows up to t only)

| Feature | Meaning |
|---|---|
| `ofi_10`, `ofi_50`, `ofi_200` | Signed volume / total volume over the last N trades |
| `ret_10`, `ret_50` | Log return over N ticks |
| `vwap_dev_50` | Distance of price from the 50-tick VWAP |
| `buy_ratio_30` | Share of buyer-initiated trades in the last 30 |
| `vol_30` | Standard deviation of trade size over 30 trades |

**Label:** 1 if the price is strictly higher 10 ticks later, otherwise 0 (an unchanged
price counts as 0, so the label is not balanced even in a flat market).

## Validation

- **Shuffled K-fold (5 folds):** the common mistake. Neighbouring rows land on both sides
  of the split, and their labels and rolling features overlap.
- **Purged walk-forward (5 folds):** always train on the past, test on the next block.
  The last `horizon` training rows are purged because their labels reach into the test
  block, and an extra embargo of 10 rows is left as a buffer for overlapping features.

Same models, same features, same data for both. Only the split changes.

## Results

One run, horizon = 10 ticks, 5 folds, embargo = 10. Values are mean ROC-AUC, +/- is the
standard deviation across folds.

| Model | Shuffled K-fold AUC | Walk-forward AUC | Gap |
|---|---|---|---|
| Logistic regression | 0.5356 +/- 0.0023 | 0.4966 +/- 0.0157 | +0.039 |
| Gradient boosting | 0.6525 +/- 0.0011 | 0.5282 +/- 0.0108 | +0.124 |

What this shows:

- The same gradient boosting model looks strong (0.65) under shuffled K-fold and close to
  random (0.53) once the split respects time. The validation scheme alone moves the
  score by 0.12.
- Logistic regression has no measurable skill under walk-forward (0.4966, within the fold-to-fold noise of 0.5).
- The gap is larger for the more flexible model. A plausible reason is that it memorizes
  local structure shared between neighbouring rows, but I have not isolated which part
  (label overlap, overlapping rolling features, or regime effects) causes most of it.

What this does not show:

- AUC is a ranking metric, not profit. A walk-forward AUC of 0.528 is a small signal and
  I have not tested whether it is statistically distinguishable from noise or whether it
  survives fees, spread and slippage. I make no claim of a tradable edge.
- It is one month of one pair, one run, no repeated seeds.
- Trades only: no order book depth, no queue position.

## Run it

```
pip install -r requirements.txt
cd src
python run.py            # logistic regression + gradient boosting (GBM is slow on ~450k rows)
python run.py --quick    # logistic regression only
```

`run.py` also prints how often the price goes up, stays flat or goes down after 10 ticks,
so the label balance can be checked directly.

Tests (from the repo root): `pytest`

- splits never let a training row's label window reach the test block
- features at row t do not change when future prices are changed
- the label equals "strictly higher after `horizon` ticks"
- the loader sorts by time and maps `is_buyer_maker` to the right side

## Layout

```
src/data.py       trade loader
src/features.py   features and label
src/splits.py     walk-forward split with purge and embargo
src/run.py        comparison script
tests/            pytest suite
```

## Next steps

- Vary horizon and embargo and plot the gap, to see which one drives it
- Add transaction costs and report a cost-adjusted metric
- Repeat on other months and pairs; add a significance test for the walk-forward AUC
