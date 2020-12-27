import collections
import numpy as np


def select_classes(batch, classes, relabel=False):
    """Select samples from a given list of classes.
    
    If relabel is true, transform labels to {1,...,len(classes)}.
    """
    x, y = batch
    per_class_indices = []
    for c in classes:
        indices = np.arange(x.shape[0])[y == c]
        per_class_indices.append(indices)
    selected_indices = np.concatenate(per_class_indices, axis=-1)
    x_selected = x[selected_indices]
    y_selected = y[selected_indices]
    if relabel:
        for i, c in enumerate(classes):
            y_selected[y_selected == c] = i
    return x_selected, y_selected


def subsample(batch, sizes, seed=0):
    """Subsample non-overlapping batches.

    batch: a list of np arrays with the same batch size.
    sizes: a list of integers.
    seed: random seed for sampling.

    return corresponding subsampled batches.
    """
    n = batch[0].shape[0]
    rand = np.random.RandomState(0)
    shuffled_indices = rand.permutation(n)
    assert sum(sizes) <= n
    sampled_batches = []
    start = 0
    for m in sizes:
        indices = shuffled_indices[start:start + m]
        sampled_batch = []
        for x in batch:
            sampled_batch.append(x[indices])
        sampled_batches.append(sampled_batch)
        start += m
    return sampled_batches



