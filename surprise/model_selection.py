import numpy as np


class KFold():

    def __init__(self, n_splits=5, seed=None, shuffle=True):

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed

    def split(self, data):

        if self.n_splits > len(data.raw_ratings) or self.n_splits < 2:
            raise ValueError('Incorrect value for n_splits. Must be >=2 and '
                             'less than the number or entries')

        indices = np.arange(len(data.raw_ratings))

        if self.shuffle:
            # if seed is None, use RandomState singleton from numpy
            # else use RandomState initialized with the seed. This guaranties
            # several calls to kf.split() yield the same splits when seed is
            # not None.
            if self.seed is None:
                rng = np.random.mtrand._rand
            else:
                rng = np.random.RandomState(self.seed)

            rng.shuffle(indices)

        start, stop = 0, 0
        for fold_i in range(self.n_splits):
            start = stop
            stop += len(indices) // self.n_splits
            if fold_i < len(indices) % self.n_splits:
                stop += 1

            train = np.concatenate([indices[:start], indices[stop:]])
            test = indices[start:stop]
            raw_trainset = [data.raw_ratings[i] for i in train]
            raw_testset = [data.raw_ratings[i] for i in test]

            trainset = data.construct_trainset(raw_trainset)
            testset = data.construct_testset(raw_testset)

            yield trainset, testset
