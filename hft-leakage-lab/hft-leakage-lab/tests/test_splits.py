from splits import walk_forward_splits


def test_train_ends_before_test_with_purge_and_embargo():
    for train, test in walk_forward_splits(5000, 5, horizon=10, embargo=10):
        assert train.max() + 10 + 10 <= test.min()
        assert not set(train) & set(test)


def test_last_train_label_ends_before_test_starts():
    for train, test in walk_forward_splits(5000, 5, horizon=10, embargo=0):
        assert train.max() + 10 < test.min()


def test_test_blocks_move_forward_and_cover_the_end():
    blocks = list(walk_forward_splits(5000, 5))
    starts = [test.min() for _, test in blocks]
    assert starts == sorted(starts)
    assert blocks[-1][1].max() == 4999
