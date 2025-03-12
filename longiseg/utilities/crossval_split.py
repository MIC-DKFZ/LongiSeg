from typing import List

import numpy as np
from sklearn.model_selection import KFold, GroupKFold


def generate_crossval_split(train_identifiers: List[str], seed=12345, n_splits=5) -> List[dict[str, List[str]]]:
    splits = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (train_idx, test_idx) in enumerate(kfold.split(train_identifiers)):
        train_keys = np.array(train_identifiers)[train_idx]
        test_keys = np.array(train_identifiers)[test_idx]
        splits.append({})
        splits[-1]['train'] = list(train_keys)
        splits[-1]['val'] = list(test_keys)
    return splits


def generate_crossval_split_longi(train_patients, seed=12345, n_splits=5):
    splits = []
    all_keys = []
    groups = []
    for idx, patient in enumerate(train_patients):
        all_keys = all_keys + train_patients[patient]
        groups = groups + [idx] * len(train_patients[patient])
    groups = np.array(groups)
    group_kfold = GroupKFold(n_splits=n_splits)

    for i, (train_idx, test_idx) in enumerate(group_kfold.split(all_keys, groups=groups)):
        train_keys = np.array(all_keys)[train_idx]
        test_keys = np.array(all_keys)[test_idx]
        splits.append({})
        splits[-1]['train'] = list(train_keys)
        splits[-1]['val'] = list(test_keys)
    return splits