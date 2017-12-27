import os

import pytest
from copy import copy
import numpy as np

from surprise import Dataset
from surprise import Reader
from surprise import KFold


np.random.seed(1)


def test_KFold():

    reader = Reader(line_format='user item rating', sep=' ', skip_lines=3,
                    rating_scale=(1, 5))
    custom_dataset_path = (os.path.dirname(os.path.realpath(__file__)) +
                           '/custom_dataset')
    data = Dataset.load_from_file(file_path=custom_dataset_path, reader=reader)

    # Test n_folds parameter
    kf = KFold(n_splits=5)
    assert len(list(kf.split(data))) == 5

    with pytest.raises(ValueError):
        kf = KFold(n_splits=10)
        next(kf.split(data))  # Too big (greater than number of ratings)

    with pytest.raises(ValueError):
        kf = KFold(n_splits=1)
        next(kf.split(data))  # Too low (must be >= 2)

    # Make sure data has not been shuffled. If not shuffled, the users in the
    # testsets are 0, 1, 2... 4 (in that order).
    kf = KFold(n_splits=5, shuffle=False)
    users = [int(testset[0][0][-1]) for (_, testset) in kf.split(data)]
    assert users == list(range(5))

    # Make sure that when called two times without shuffling, folds are the
    # same.
    kf = KFold(n_splits=5, shuffle=False)
    testsets_a = [testset for (_, testset) in kf.split(data)]
    testsets_b = [testset for (_, testset) in kf.split(data)]
    assert testsets_a == testsets_b
    # test once again with another KFold instance
    kf = KFold(n_splits=5, shuffle=False)
    testsets_a = [testset for (_, testset) in kf.split(data)]
    assert testsets_a == testsets_b

    # We'll now shuffle b and check that folds are different.
    # (this is conditioned by seed setting at the beginning of file)
    kf = KFold(n_splits=5, seed=None, shuffle=True)
    testsets_b = [testset for (_, testset) in kf.split(data)]
    assert testsets_a != testsets_b
    # test once again: two calls to kf.split make different splits when
    # seed=None
    testsets_a = [testset for (_, testset) in kf.split(data)]
    assert testsets_a != testsets_b

    # Make sure that folds are the same when same KFold instance is used with
    # suffle is True but seed is set to some value
    kf = KFold(n_splits=5, seed=1, shuffle=True)
    testsets_a = [testset for (_, testset) in kf.split(data)]
    testsets_b = [testset for (_, testset) in kf.split(data)]
    assert testsets_a == testsets_b

    # Make sure raw ratings are not shuffled by KFold
    old_raw_ratings = copy(data.raw_ratings)
    kf = KFold(n_splits=5, shuffle=True)
    next(kf.split(data))
    assert old_raw_ratings == data.raw_ratings

    # Make sure kf.split() and the old data.split() have the same folds.
    np.random.seed(3)
    data.split(2, shuffle=True)
    testsets_a = [testset for (_, testset) in data.folds()]
    kf = KFold(n_splits=2, seed=3, shuffle=True)
    testsets_b = [testset for (_, testset) in kf.split(data)]
