from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time
from collections import defaultdict

import numpy as np

from .split import get_cv
from .. import accuracy


def cross_validate(algo, data, measures=['rmse', 'mae'], cv=None,
                   with_dump=False, dump_dir=None, verbose=1):

    measures = [m.lower() for m in measures]

    cv = get_cv(cv)

    ret = defaultdict(list)

    if verbose:
        print('Evaluating {0} of algorithm {1}.'.format(
              ', '.join((m.upper() for m in measures)),
              algo.__class__.__name__))
        print()

    for fold_i, (trainset, testset) in enumerate(cv.split(data)):

        if verbose:
            print('-' * 12)
            print('Fold ' + str(fold_i + 1))

        test_measures, fit_time, test_time = fit_and_score(algo, trainset,
                                                           testset, measures)
        ret['fit_time'].append(fit_time)
        ret['test_time'].append(test_time)
        for m in measures:
            ret['test_' + m].append(test_measures[m])

    if verbose:
        print('-' * 12)
        print('-' * 12)
        for measure in measures:
            print('Mean {0:4s}: {1:1.4f}'.format(
                  measure.upper(), np.mean(ret['test_' + measure])))
        print('-' * 12)
        print('-' * 12)

    return ret


def fit_and_score(algo, trainset, testset, measures):

    start_time = time.time()
    algo.fit(trainset)
    fit_time = time.time() - start_time
    predictions = algo.test(testset)
    test_time = time.time() - fit_time

    test_measures = dict()
    for measure in measures:
        f = getattr(accuracy, measure.lower())
        test_measures[measure] = f(predictions, verbose=0)

    return test_measures, fit_time, test_time
