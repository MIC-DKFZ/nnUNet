from typing import List

import numpy as np


def generate_crossval_split(train_identifiers: List[str], seed=12345, n_splits=5) -> List[dict[str, List[str]]]:
    splits = []
    from sklearn.model_selection import KFold  # lazy: ~0.2 s per process, only needed to create splits
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (train_idx, test_idx) in enumerate(kfold.split(train_identifiers)):
        train_keys = np.array(train_identifiers)[train_idx]
        test_keys = np.array(train_identifiers)[test_idx]
        splits.append({})
        splits[-1]['train'] = list(train_keys)
        splits[-1]['val'] = list(test_keys)
    return splits
