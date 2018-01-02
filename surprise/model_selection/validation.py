from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time
from collections import defaultdict
import os

import numpy as np

from . import KFold
from .. import accuracy
from ..dump import dump


def cross_validate(algo, data, measures=['rmse', 'mae'], cv=None,
                   with_dump=False, dump_dir=None, verbose=1):

    if verbose:
        print('Evaluating {0} of algorithm {1}.'.format(
              ', '.join((m.upper() for m in measures)),
              algo.__class__.__name__))
        print()

    if cv is None:
        cv = KFold(n_splits=5)

    ret = defaultdict(list)
    for fold_i, (trainset, testset) in enumerate(cv.split(data)):

        if verbose:
            print('-' * 12)
            print('Fold ' + str(fold_i + 1))

        # train and test algorithm. Keep all rating predictions in a list
        start_time = time.time()
        algo.fit(trainset)
        fit_time = time.time() - start_time
        predictions = algo.test(testset, verbose=(verbose == 2))
        test_time = time.time() - fit_time

        ret['fit_time'].append(fit_time)
        ret['test_time'].append(test_time)

        # compute needed performance statistics
        for measure in measures:
            f = getattr(accuracy, measure.lower())
            ret['test_' + measure].append(f(predictions, verbose=verbose))

        if with_dump:

            if dump_dir is None:
                dump_dir = os.path.join(get_dataset_dir(), 'dumps/')

            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir)

            date = time.strftime('%y%m%d-%Hh%Mm%S', time.localtime())
            file_name = date + '-' + algo.__class__.__name__
            file_name += '-fold{0}'.format(fold_i + 1)
            file_name = os.path.join(dump_dir, file_name)

            dump(file_name, predictions, trainset, algo)

    if verbose:
        print('-' * 12)
        print('-' * 12)
        for measure in measures:
            print('Mean {0:4s}: {1:1.4f}'.format(
                  measure.upper(), np.mean(ret['test_' + measure])))
        print('-' * 12)
        print('-' * 12)

    return ret
