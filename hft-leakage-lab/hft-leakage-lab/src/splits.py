"""Walk-forward splits with purge and embargo."""
import numpy as np


def walk_forward_splits(n, n_folds=5, min_train=0.3, horizon=10, embargo=10):
    """Yield (train_idx, test_idx) for time-ordered rows.

    Train always ends `horizon + embargo` rows before the test block starts.
    horizon removes train rows whose labels reach into the test block;
    embargo adds a buffer for features that overlap across the boundary.
    """
    start = int(n * min_train)
    fold_size = (n - start) // n_folds
    for k in range(n_folds):
        test_start = start + k * fold_size
        test_end = test_start + fold_size if k < n_folds - 1 else n
        train_end = test_start - horizon - embargo
        if train_end <= 0:
            continue
        yield np.arange(train_end), np.arange(test_start, test_end)
