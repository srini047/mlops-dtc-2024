@test
def test_dataset(
    X: csr_matrix,
    X_train: csr_matrix,
    X_val: csr_matrix,
    y: Series,
    y_train: Series,
    y_val: Series,
    *args,
) -> None:
    assert (
        X.shape[0] == 105870
    ), f'Entire dataset should have 105870 examples, but has {X.shape[0]}'
    assert (
        X.shape[1] == 7027
    ), f'Entire dataset should have 7027 features, but has {X.shape[1]}'
    assert (
        len(y.index) == X.shape[0]
    ), f'Entire dataset should have {X.shape[0]} examples, but has {len(y.index)}'