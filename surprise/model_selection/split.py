from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from itertools import chain
from math import ceil, floor
import numbers
from collections import defaultdict

from six import iteritems
from six import string_types

import numpy as np


def get_rng(random_state):
    # if random_state is None, use RandomState singleton from numpy.
    # Else if it's an integer, consider it's a seed and initialized an rng with
    # that seed. If it's already an rng, return it.
    if random_state is None:
        return np.random.mtrand._rand
    elif isinstance(random_state, (numbers.Integral, np.integer)):
        return np.random.RandomState(random_state)
    if isinstance(random_state, np.random.RandomState):
        return random_state
    raise ValueError('Wrong random state. Expecting None, an int or a numpy '
                     'RandomState instance, got a '
                     '{}'.format(type(random_state)))


def get_cv(cv):

    if cv is None:
        return KFold(n_splits=5)
    if isinstance(cv, numbers.Integral):
        return KFold(n_splits=cv)
    if hasattr(cv, 'split') and not isinstance(cv, string_types):
        return cv  # str have split

    raise ValueError('Wrong CV object. Expecting None, an int or CV iterator, '
                     'got a {}'.format(type(cv)))


class KFold():

    def __init__(self, n_splits=5, random_state=None, shuffle=True):

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, data):

        if self.n_splits > len(data.raw_ratings) or self.n_splits < 2:
            raise ValueError('Incorrect value for n_splits={0}. '
                             'Must be >=2 and less than the number '
                             'of ratings'.format(len(data.raw_ratings)))

        # We use indices to avoid shuffling the original data.raw_ratings list.
        indices = np.arange(len(data.raw_ratings))

        if self.shuffle:
            get_rng(self.random_state).shuffle(indices)

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

    def __init__(self, n_splits=5, test_size=.2, train_size=None,
                 random_state=None, shuffle=True):

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
        self.random_state = random_state
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
        rng = get_rng(self.random_state)

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


def train_test_split(data, test_size=.2, train_size=None, random_state=None,
                     shuffle=True):
    ss = ShuffleSplit(n_splits=1, test_size=test_size, train_size=train_size,
                      random_state=random_state, shuffle=shuffle)
    return next(ss.split(data))


class RepeatedKFold():

    def __init__(self, n_splits=5, n_repeats=10, random_state=None):

        self.n_repeats = n_repeats
        self.random_state = random_state
        self.n_splits = n_splits

    def split(self, data):

        rng = get_rng(self.random_state)

        for _ in range(self.n_repeats):
            cv = KFold(n_splits=self.n_splits, random_state=rng, shuffle=True)
            for trainset, testset in cv.split(data):
                yield trainset, testset


class LeaveOneOut():

    def __init__(self, n_splits=5, random_state=None):

        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, data):

        user_ratings = defaultdict(list)
        for uid, iid, r_ui, _ in data.raw_ratings:
            user_ratings[uid].append((uid, iid, r_ui, None))

        rng = get_rng(self.random_state)

        for _ in range(self.n_splits):
            raw_trainset, raw_testset = [], []
            for uid, ratings in iteritems(user_ratings):
                i = rng.randint(0, len(ratings))
                raw_testset.append(ratings[i])
                raw_trainset += [rating for (j, rating) in enumerate(ratings)
                                 if j != i]

            if not raw_trainset:
                raise ValueError('Each user only has one rating. Cannot '
                                 'Run LOO cross-validation')
            trainset = data.construct_trainset(raw_trainset)
            testset = data.construct_testset(raw_testset)

            yield trainset, testset


class PredefinedKFold():

    def split(self, data):

        self.n_splits = len(data.folds_files)
        for train_file, test_file in data.folds_files:

            raw_trainset = data.read_ratings(train_file)
            raw_testset = data.read_ratings(test_file)
            trainset = data.construct_trainset(raw_trainset)
            testset = data.construct_testset(raw_testset)

            yield trainset, testset
