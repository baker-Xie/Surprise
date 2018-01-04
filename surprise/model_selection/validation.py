from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time

import numpy as np
from joblib import Parallel
from joblib import delayed

from .split import get_cv
from .. import accuracy


def cross_validate(algo, data, measures=['rmse', 'mae'], cv=None, n_jobs=-1,
                   pre_dispatch='2*n_jobs', verbose=1):

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
