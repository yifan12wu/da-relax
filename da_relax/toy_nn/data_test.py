from predprob.pytorch.toy_nn import data


def test_relative():
    dataset = data.Relative(6, 3, seed=0)
    x_tr, y_tr = dataset.x_train, dataset.y_train
    print('train 1:', x_tr, y_tr)
    dataset = data.Relative(6, 3, seed=1)
    x_tr, y_tr = dataset.x_train, dataset.y_train
    print('train 2:', x_tr, y_tr)
    x_te, y_te = dataset.x, dataset.y
    print('full test:', x_te, y_te)


def test_basis(inv):
    dataset = data.Basis(6, 4, freq=inv, seed=0)
    x_tr, y_tr = dataset.x_train, dataset.y_train
    print('train 1:', x_tr, y_tr)
    dataset = data.Basis(6, 4, freq=inv, seed=1)
    x_tr, y_tr = dataset.x_train, dataset.y_train
    print('train 2:', x_tr, y_tr)
    x_te, y_te = dataset.x, dataset.y
    print('full test:', x_te, y_te)


def main():
    test_relative()
    for inv in range(6):
        test_basis(inv + 1)


if __name__ == '__main__':
    main()
