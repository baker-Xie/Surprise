from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time

import numpy as np
from joblib import Parallel
from joblib import delayed
from six import iteritems

from .split import get_cv
from .. import accuracy


def cross_validate(algo, data, measures=['rmse', 'mae'], cv=None, n_jobs=-1,
                   pre_dispatch='2*n_jobs', verbose=0):
    'bbobobob'

    measures = [m.lower() for m in measures]

    cv = get_cv(cv)

    delayed_list = (delayed(fit_and_score)(algo, trainset, testset, measures)
                    for (trainset, testset) in cv.split(data))
    out = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)(delayed_list)
    test_measures_dicts, fit_times, test_times = zip(*out)

    # transform list of dicts into dict of lists
    # Same as in GridSearchCV.fit()
    test_measures = dict()
    for m in test_measures_dicts[0]:
        test_measures[m] = np.asarray([d[m] for d in test_measures_dicts])

    ret = dict()
    for m in measures:
        ret['test_' + m] = test_measures[m]

    ret['fit_time'] = fit_times
    ret['test_time'] = test_times

    if verbose:
        print_summary(algo, measures, test_measures, fit_times, test_times,
                      cv.n_splits)

    return ret


def fit_and_score(algo, trainset, testset, measures):

    start_fit = time.time()
    algo.fit(trainset)
    fit_time = time.time() - start_fit
    start_test = time.time()
    predictions = algo.test(testset)
    test_time = time.time() - start_test

    test_measures = dict()
    for measure in measures:
        f = getattr(accuracy, measure.lower())
        test_measures[measure] = f(predictions, verbose=0)

    return test_measures, fit_time, test_time


def print_summary(algo, measures, test_measures, fit_times, test_times,
                  n_splits):

    print('Evaluating {0} of algorithm {1} on {2} split(s).'.format(
          ', '.join((m.upper() for m in measures)),
          algo.__class__.__name__, n_splits))
    print()

    row_format = '{:<12}' + '{:<8}' * (n_splits + 2)
    s = row_format.format(
        '',
        *['Fold {0}'.format(i + 1) for i in range(n_splits)] + ['Mean'] +
        ['Std'])
    s += '\n'
    s += '\n'.join(row_format.format(
        key.upper(),
        *['{:1.4f}'.format(v) for v in vals] +
        ['{:1.4f}'.format(np.mean(vals))] +
        ['{:1.4f}'.format(np.std(vals))])
        for (key, vals) in iteritems(test_measures))
    s += '\n'
    s += row_format.format('Fit time',
                           *['{:.2f}'.format(t) for t in fit_times] +
                           ['{:.2f}'.format(np.mean(fit_times))] +
                           ['{:.2f}'.format(np.std(fit_times))])
    s += '\n'
    s += row_format.format('Test time',
                           *['{:.2f}'.format(t) for t in test_times] +
                           ['{:.2f}'.format(np.mean(test_times))] +
                           ['{:.2f}'.format(np.std(test_times))])
    print(s)
