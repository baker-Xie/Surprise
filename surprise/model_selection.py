from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from itertools import chain
from math import ceil, floor

import numpy as np


def get_rng(seed):
    # if seed is None, use RandomState singleton from numpy
    # else use RandomState initialized with the seed. This guaranties
    # several calls to kf.split() yield the same splits when seed is
    # not None.
    if seed is None:
        rng = np.random.mtrand._rand
    else:
        rng = np.random.RandomState(seed)

    return rng

class KFold():

    def __init__(self, n_splits=5, seed=None, shuffle=True):

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed

    def split(self, data):

        if self.n_splits > len(data.raw_ratings) or self.n_splits < 2:
            raise ValueError('Incorrect value for n_splits. Must be >=2 and '
                             'less than the number or entries')

        # We use indices to avoid shuffling the original data.raw_ratings list.
        indices = np.arange(len(data.raw_ratings))

        if self.shuffle:
            get_rng(self.seed).shuffle(indices)

        start, stop = 0, 0
        for fold_i in range(self.n_splits):
            start = stop
            stop += len(indices) // self.n_splits
            if fold_i < len(indices) % self.n_splits:
                stop += 1

            raw_trainset = [data.raw_ratings[i] for i in chain(indices[:start],
                                                               indices[stop:])]
            raw_testset = [data.raw_ratings[i] for i in indices[start:stop]]

            trainset = data.construct_trainset(raw_trainset)
            testset = data.construct_testset(raw_testset)

            yield trainset, testset

class ShuffleSplit():
    '''Note: setting shuffle to false defeats the purpose of ShuffleSplit but
    it's useful for train_test_split.'''

    def __init__(self, n_splits=5, test_size=.2, train_size=None, seed=None,
                 shuffle=True):

        if n_splits <= 0:
            raise ValueError('n_splits = {0} should be strictly greater than '
                             '0.'.format(n_splits))
        if test_size is not None and test_size <= 0:
            raise ValueError('test_size={0} should be strictly greater than '
                             '0'.format(test_size))

        if train_size is not None and train_size <= 0:
            raise ValueError('train_size={0} should be strictly greater than '
                             '0'.format(train_size))

        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.seed = seed
        self.shuffle = shuffle

    def validate_train_test_sizes(self, test_size, train_size, n_ratings):

        if test_size is not None and test_size >= n_ratings:
            raise ValueError('test_size={0} should be less than the number of '
                             'ratings {1}'.format(test_size, n_ratings))

        if train_size is not None and train_size >= n_ratings:
            raise ValueError('train_size={0} should be less than the number of'
                             ' ratings {1}'.format(train_size, n_ratings))

        if np.asarray(test_size).dtype.kind == 'f':
            test_size = ceil(test_size * n_ratings)

        if train_size is None:
            train_size = n_ratings - test_size
        elif np.asarray(train_size).dtype.kind == 'f':
            train_size = floor(train_size * n_ratings)

        if test_size is None:
            test_size = n_ratings - train_size

        if train_size + test_size > n_ratings:
            raise ValueError('The sum of train_size and test_size ({0}) '
                             'should be smaller than the number of '
                             'ratings {1}.'.format(train_size + test_size,
                                                   n_ratings))

        return int(train_size), int(test_size)

    def split(self, data):

        test_size, train_size = self.validate_train_test_sizes(
            self.test_size, self.train_size, len(data.raw_ratings))
        rng = get_rng(self.seed)

        for _ in range(self.n_splits):

            if self.shuffle:
                permutation = rng.permutation(len(data.raw_ratings))
            else:
                permutation = np.arange(len(data.raw_ratings))

            raw_trainset = [data.raw_ratings[i] for i in
                            permutation[:test_size]]
            raw_testset = [data.raw_ratings[i] for i in
                           permutation[test_size:(test_size + train_size)]]

            trainset = data.construct_trainset(raw_trainset)
            testset = data.construct_testset(raw_testset)

            yield trainset, testset


def train_test_split(data, test_size=.2, train_size=None, seed=None,
                     shuffle=True):
    ss = ShuffleSplit(n_splits=1, test_size=test_size, train_size=train_size,
                      seed=seed, shuffle=shuffle)
    return next(ss.split(data))
