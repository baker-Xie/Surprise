"""
Module for testing the model_selection.search module.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import random

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import KNNBasic
from surprise.model_selection import KFold
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate


def test_parameter_combinations():
    """Make sure that parameter_combinations attribute is correct (has correct
    size).  Dict parameters like bsl_options and sim_options require special
    treatment in the param_grid argument. We here test both in one shot with
    KNNBaseline."""

    param_grid = {'bsl_options': {'method': ['als', 'sgd'],
                                  'reg': [1, 2]},
                  'k': [2, 3],
                  'sim_options': {'name': ['msd', 'cosine'],
                                  'min_support': [1, 5],
                                  'user_based': [False]}
                  }

    gs = GridSearchCV(SVD, param_grid)
    assert len(gs.param_combinations) == 32


def test_best_estimator():
    """Ensure that the best estimator is the one giving the best score (by
    re-running it)"""

    train_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
    test_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_test')
    data = Dataset.load_from_folds([(train_file, test_file)],
                                   Reader('ml-100k'))

    param_grid = {'n_epochs': [5], 'lr_all': [0.002, 0.005],
                  'reg_all': [0.4, 0.6], 'n_factors': [1], 'init_std_dev': [0]}
    gs = GridSearchCV(SVD, param_grid, measures=['mae'],
                      cv=PredefinedKFold(), verbose=0)
    gs.fit(data)
    best_estimator = gs.best_estimator['mae']

    # recompute MAE of best_estimator
    mae = cross_validate(best_estimator, data, measures=['MAE'],
                           cv=PredefinedKFold())['test_mae']

    assert mae == gs.best_score['mae']


def test_same_splits():
    """Ensure that all parameter combinations are tested on the same splits (we
    check that average RMSE scores are the same, when run on the same set of
    parameters, which should be enough). We use as much parallelism as
    possible."""

    data_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_test')
    data = Dataset.load_from_file(data_file, reader=Reader('ml-100k'))
    kf = KFold(3, shuffle=True, random_state=4)

    # all RMSE should be the same (as param combinations are the same)
    param_grid = {'k': [2, 2], 'min_k': [1, 1]}
    gs = GridSearchCV(KNNBasic, param_grid, measures=['RMSE'], cv=kf, n_jobs=-1)
    gs.fit(data)

    rmse_scores = [m for m in gs.mean_measures['rmse']]
    assert len(set(rmse_scores)) == 1  # assert rmse_scores are all equal

    # Note: actually, even when setting random_state=None in kf, the same folds
    # are used because we use product(param_comb, kf.split(...)). However, it's
    # needed to have the same folds when calling fit again:
    gs.fit(data)
    rmse_scores += [m for m in gs.mean_measures['rmse']]
    assert len(set(rmse_scores)) == 1  # assert rmse_scores are all equal
