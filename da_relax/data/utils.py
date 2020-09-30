import collections
import numpy as np

from . import data_base

def subsample(x, y, classes, n_samples=None):
    """Subsample a data batch based on given set of classes."""
    rand = np.random.RandomState(0)
    per_class_indices = []
    for c in classes:
        class_indices = np.arange(x.shape[0])[y == c]
        per_class_indices.append(class_indices)
    if n_samples is None:
        selected = np.concatenate(per_class_indices, axis=-1)
        n = selected.shape[0]
        selected = selected[rand.permutation(n)]
        x_selected = x[selected]
        y_selected = y[selected]
        for i, c in enumerate(classes):
            y_selected[y_selected == c] = i
        return x_selected, y_selected
    if isinstance(n_samples, int):
        n_list = [n_samples]
    else:
        n_list = n_samples  # must be a list of integers
    selected = []  # selected indices for each batches
    st = []  # number of already-selected samples for each batch
    for n in n_list:
        selected.append(np.zeros(n).astype(np.int64))
        st.append(0)
    k = len(classes)
    for i in range(k):
        # select samples for the i-th class
        m = []
        # get number of samples to be collected for each batch
        for n in n_list:
            if i < k - 1:
                m.append(int(n // k))
            else:
                _m = n - int(n // k) * (k - 1)
                m.append(_m)
        _m_total = 0
        # select indices for each batch
        for j in range(len(st)):
            selected[j][st[j]:st[j] + m[j]] = per_class_indices[i][
                    _m_total:_m_total + m[j]]
            _m_total += m[j]
            st[j] = st[j] + m[j]
    result = []
    for i in range(len(selected)):
        # shuffle
        selected[i] = selected[i][rand.permutation(selected[i].shape[0])]
        x_selected = x[selected[i]]
        y_selected = y[selected[i]]
        for j, c in enumerate(classes):
            y_selected[y_selected == c] = j
        assert x_selected.shape[0] == n_list[i]
        result.append((x_selected, y_selected))
    assert len(result) == len(n_list)
    if isinstance(n_samples, int):
        return result[0]
    else:
        return result
 

def combine_batches(batches):
    """
    Combine multiple batches from different datasets. 
    Assume prepros are the same accross different batches.
    Add a dataset labelfor each sample. 
    No shuffling at the current point.
    """
    xs = []
    ys = []
    yds = []
    names = []
    n_classes = []
    x_prepro = batches[0].data.x.prepro
    y_prepro = batches[0].data.y.prepro
    for i, batch in enumerate(batches):
        xs.append(batch.data.x.val)
        ys.append(batch.data.y.val)
        names.append(batch.info.name)
        n_classes.append(batch.info.n_classes)
        n_samples = batch.data.y.val.shape[0]
        yd = np.zeros(n_samples).astype(np.int64) + i
        yds.append(yd)
    x_new = np.concatenate(xs, axis=0)
    y_new = np.concatenate(ys, axis=0)
    yd_new = np.concatename(yds, axis=0)
    data_dict = collections.OrderedDict()
    data_dict['x'] = (x_new, x_prepro)
    data_dict['y'] = (y_new, y_prepro)
    data_dict['yd'] = (yd_new, None)
    info_dict = collections.OrderedDict()
    info_dict['names'] = names
    info_dict['n_classes'] = n_classes
    return data_base.Batch(data_dict=data_dict, info_dict=info_dict)

         

