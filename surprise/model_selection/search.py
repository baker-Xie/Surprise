from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from itertools import product

import numpy as np
from joblib import Parallel
from joblib import delayed

from .split import get_cv
from .validation import fit_and_score


class GridSearchCV:
    def __init__(self, algo_class, param_grid, measures=['rmse', 'mae'],
                 cv=None, n_jobs=-1, pre_dispatch='2*n_jobs',
                 joblib_verbose=0):

        self.algo_class = algo_class
        self.param_grid = param_grid.copy()
        self.measures = [measure.lower() for measure in measures]
        self.cv = cv
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.joblib_verbose = joblib_verbose

        # As sim_options and bsl_options are dictionaries, they require a
        # special treatment.
        if 'sim_options' in self.param_grid:
            sim_options = self.param_grid['sim_options']
            sim_options_list = [dict(zip(sim_options, v)) for v in
                                product(*sim_options.values())]
            self.param_grid['sim_options'] = sim_options_list

        if 'bsl_options' in self.param_grid:
            bsl_options = self.param_grid['bsl_options']
            bsl_options_list = [dict(zip(bsl_options, v)) for v in
                                product(*bsl_options.values())]
            self.param_grid['bsl_options'] = bsl_options_list

        self.param_combinations = [dict(zip(self.param_grid, v)) for v in
                                   product(*self.param_grid.values())]

    def fit(self, data):

        cv = get_cv(self.cv)

        delayed_list = (
            delayed(fit_and_score)(self.algo_class(**params), trainset,
                                   testset, self.measures)
            for params, (trainset, testset) in product(self.param_combinations,
                                                       cv.split(data))
        )

        out = Parallel(n_jobs=self.n_jobs,
                       pre_dispatch=self.pre_dispatch,
                       verbose=self.joblib_verbose)(delayed_list)

        test_measures_dicts, fit_times, test_times = zip(*out)

        # test_measures_dicts is a list of dict like this:
        # [{'mae': 1, 'rmse': 2}, {'mae': 2, 'rmse': 3} ...]
        # E.g. for 5 splits, the first 5 dicts are for the first param
        # combination, the next 5 dicts are for the second param combination,
        # etc...
        # We convert it into a dict of list:
        # {'mae': [1, 2, ...], 'rmse': [2, 3, ...]}
        # Each list is still of size n_parameters_combinations * n_splits.
        # Then, reshape each list to have 2-D arrays of shape
        # (n_parameters_combinations, n_splits). This way we can easily compute
        # the mean and std dev over all splits or over all param comb.
        test_measures = dict()
        new_shape = (len(self.param_combinations), cv.n_splits)
        for m in self.measures:
            test_measures[m] = np.asarray([d[m] for d in test_measures_dicts])
            test_measures[m] = test_measures[m].reshape(new_shape)

        cv_results = dict()
        best_index = dict()
        best_params = dict()
        best_score = dict()
        best_estimator = dict()
        for m in self.measures:
            # cv_results: set measures for each split and each param comb
            for split in range(cv.n_splits):
                cv_results['split{0}_test_{1}'.format(split, m)] = \
                    test_measures[m][:, split]

            # cv_results: set mean and std over all splits (testset) for each
            # param comb
            mean_measures = test_measures[m].mean(axis=1)
            cv_results['mean_test_{}'.format(m)] = mean_measures
            cv_results['std_test_{}'.format(m)] = test_measures[m].std(axis=1)

            # cv_results: set rank of each param comb
            indices = cv_results['mean_test_{}'.format(m)].argsort()
            cv_results['rank_test_{}'.format(m)] = np.empty_like(indices)
            cv_results['rank_test_{}'.format(m)][indices] = np.arange(
                len(indices)) + 1  # sklearn starts rankings at 1 as well.

            # set best_index, and best_xxxx attributes
            if m in ('mae', 'rmse'):
                best_index[m] = mean_measures.argmin()
            elif m in ('fcp', ):
                best_index[m] = mean_measures.argmax()
            best_params[m] = self.param_combinations[best_index[m]]
            best_score[m] = mean_measures[best_index[m]]
            best_estimator[m] = self.algo_class(**best_params[m])

        # Cv results: set fit and train times (mean, std)
        fit_times = np.array(fit_times).reshape(new_shape)
        test_times = np.array(test_times).reshape(new_shape)
        for s, times in zip(('fit', 'test'), (fit_times, test_times)):
            cv_results['mean_{}_time'.format(s)] = times.mean(axis=1)
            cv_results['std_{}_time'.format(s)] = times.std(axis=1)

        # cv_results: set params key
        cv_results['params'] = self.param_combinations

        self.best_index = best_index
        self.best_params = best_params
        self.best_score = best_score
        self.best_estimator = best_estimator
        self.cv_results = cv_results
