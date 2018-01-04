.. _getting_started:

Getting Started
===============


.. _load_builtin_example:

Basic usage
-----------

`Surprise <https://nicolashug.github.io/Surprise/>`_ has a set of built-in
:ref:`algorithms<prediction_algorithms>` and :ref:`datasets <dataset>` for you
to play with. In its simplest form, it only takes a few lines of code to
run a cross-validation procedure:

.. literalinclude:: ../../examples/basic_usage.py
    :caption: From file ``examples/basic_usage.py``
    :name: basic_usage.py
    :lines: 9-

The result should be as follows (actual values may vary due to randomization):

.. parsed-literal::

    Evaluating RMSE, MAE of algorithm SVD on 5 split(s).

                Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std
    RMSE        0.9311  0.9370  0.9320  0.9317  0.9391  0.9342  0.0032
    MAE         0.7350  0.7375  0.7341  0.7342  0.7375  0.7357  0.0015
    Fit time    6.53    7.11    7.23    7.15    3.99    6.40    1.23
    Test time   0.26    0.26    0.25    0.15    0.13    0.21    0.06


The :meth:`load_builtin() <surprise.dataset.Dataset.load_builtin>` method will
offer to download the `movielens-100k dataset
<http://grouplens.org/datasets/movielens/>`_ if it has not already been
downloaded, and it will save it in the ``.surprise_data`` folder in your home
directory (you can also choose to save it :ref:`somewhere else <data_folder>`).

We are here using the well-known
:class:`SVD<surprise.prediction_algorithms.matrix_factorization.SVD>`
algorithm, but many other algorithms are available. See
:ref:`prediction_algorithms` for more details.

The :func:`cross_validate()<surprise.model_selection.validation.cross_validate>`
function runs a cross-validation procedure according to the ``cv`` argument,
and computes some :mod:`accuracy <surprise.accuracy>` measures. We are here
using a classical 5-fold cross-validation, but fancier iterators can be used
(see :ref:`here <cross_validation_iterators_api>`).

--------------

TODO: make example with train_test_split here.
Then say that if train and test files are already defined, refer to
load_from_folds. Then fo on to train on whole trainset.

.. _train_on_whole_trainset:

Obviously, we could also simply fit our algorithm to the whole dataset, rather
than running cross-validation. This can be done by using the
:meth:`build_full_trainset()
<surprise.dataset.DatasetAutoFolds.build_full_trainset>` method which will
build a :class:`trainset <surprise.Trainset>` object, and the :meth:`fit()
<surprise.prediction_algorithms.algo_base.AlgoBase.fit>` method which will
train the algorithm on the trainset:

.. literalinclude:: ../../examples/predict_ratings.py
    :caption: From file ``examples/predict_ratings.py``
    :name: predict_ratings.py
    :lines: 9-20

We can now predict ratings by directly calling the :meth:`predict()
<surprise.prediction_algorithms.algo_base.AlgoBase.predict>` method.  Let's say
you're interested in user 196 and item 302 (make sure they're in the
trainset!), and you know that the true rating :math:`r_{ui} = 4`:

.. literalinclude:: ../../examples/predict_ratings.py
    :caption: From file ``examples/predict_ratings.py``
    :name: predict_ratings2.py
    :lines: 23-27

The result should be:

.. parsed-literal::

    user: 196        item: 302        r_ui = 4.00   est = 4.06   {'actual_k': 40, 'was_impossible': False}

.. note::

    The :meth:`predict()
    <surprise.prediction_algorithms.algo_base.AlgoBase.predict>` uses **raw**
    ids (please read :ref:`this <raw_inner_note>` about raw and inner ids). As
    the dataset we have used has been read from a file, the raw ids are strings
    (even if they represent numbers).

We have here used a built-in datasets but you can of course use your own, which
is explained in the next section.

.. _load_custom:

Load a custom dataset
---------------------

`Surprise <https://nicolashug.github.io/Surprise/>`_ has a set of  builtin
:ref:`datasets <dataset>`, but you can of course use a custom dataset.
Loading a rating dataset can be done either from a file (e.g. a csv file), or
from a pandas dataframe.  Either way, you will need to define a :class:`Reader
<surprise.reader.Reader>` object for `Surprise
<https://nicolashug.github.io/Surprise/>`_ to be able to parse the file or the
dataframe.

.. _load_from_file_example:

- To load a dataset from a file (e.g. a csv file), you will need the
  :meth:`load_from_file() <surprise.dataset.Dataset.load_from_file>` method:

  .. literalinclude:: ../../examples/load_custom_dataset.py
      :caption: From file ``examples/load_custom_dataset.py``
      :name: load_custom_dataset.py
      :lines: 12-28

  For more details about readers and how to use them, see the :class:`Reader
  class <surprise.reader.Reader>` documentation.

  .. note::
      As you already know from the previous section, the Movielens-100k dataset
      is built-in so a much quicker way to load the dataset is to do ``data =
      Dataset.load_builtin('ml-100k')``. We will of course ignore this here.

.. _load_from_df_example:

- To load a dataset from a pandas dataframe, you will need the
  :meth:`load_from_df() <surprise.dataset.Dataset.load_from_df>` method. You
  will also need a :class:`Reader<surprise.reader.Reader>` object, but only
  the ``rating_scale`` parameter must be specified. The dataframe must have
  three columns, corresponding to the user (raw) ids, the item (raw) ids, and
  the ratings in this order. Each row thus corresponds to a given rating. This
  is not restrictive as you can reorder the columns of your dataframe easily.

  .. literalinclude:: ../../examples/load_from_dataframe.py
      :caption: From file ``examples/load_from_dataframe.py``
      :name: load_dom_dataframe.py
      :lines: 8-29

  The dataframe initially looks like this:

  .. parsed-literal::

            itemID  rating    userID
      0       1       3         9
      1       1       2        32
      2       1       4         2
      3       2       3        45
      4       2       1  user_foo


Using cross-validation iterators
--------------------------------

We have so far used the :func:`cross_validate()
<surprise.model_selection.validation.cross_validate>`
function that does all the hard work for us. We could also instanciate a
cross-validation iterator, and make predictions over each split using the
``split()`` method of the iterator, and the
:meth:`test()<surprise.prediction_algorithms.algo_base.AlgoBase.test>` method
of the algorithm. Here is an example where we use a classical K-fold
cross-validation procedure with 3 splits:

.. literalinclude:: ../../examples/use_cross_validation_iterators.py
    :caption: From file ``examples/use_cross_validation_iterators.py``
    :name: use_cross_validation_iterators.py
    :lines: 8-

Result could be, e.g.:

.. parsed-literal::
    RMSE: 0.9374
    RMSE: 0.9476
    RMSE: 0.9478

Other cross-validation iterator can be used, like LeaveOneOut or ShuffleSplit.
See all the available iterators :ref:`here <cross_validation_iterators_api>`.

.. _load_from_folds_example:

A special case of cross-validation is when the folds are already predefined by
some files. For instance, the movielens-100K dataset already provides 5 train
and test files (u1.base, u1.test ... u5.base, u5.test). Surprise can handle
this case by using a :class:`surprise.model_selection.split.PredefinedKFold`
object:

.. literalinclude:: ../../examples/load_custom_dataset_predefined_folds.py
    :caption: From file ``examples/load_custom_dataset_predefined_folds.py``
    :name: load_custom_dataset_predefined_folds.py
    :lines: 13-

Of course, nothing prevents you from only loading a single file for training
and a single file for testing. However, the ``folds_files`` parameter still
needs to be a ``list``.

.. _tuning_algorithm_parameters:

Tune algorithm parameters with GridSearch
-----------------------------------------

The :func:`evaluate() <surprise.evaluate.evaluate>` function gives us the
results on one set of parameters given to the algorithm. If the user wants
to try the algorithm on a different set of parameters, the
:class:`GridSearch <surprise.evaluate.GridSearch>` class comes to the rescue.
Given a ``dict`` of parameters, this
class exhaustively tries all the combination of parameters and helps get the
best combination for an accuracy measurement. It is analogous to
`GridSearchCV <http://scikit-learn.org/stable/modules/generated/sklearn.model
_selection.GridSearchCV.html>`_ from scikit-learn.

For instance, suppose that we want to tune the parameters of the
:class:`SVD <surprise.prediction_algorithms.matrix_factorization.SVD>`. Some of
the parameters of this algorithm are ``n_epochs``, ``lr_all`` and ``reg_all``.
Thus we define a parameters grid as follows

.. literalinclude:: ../../examples/grid_search_usage.py
    :caption: From file ``examples/grid_search_usage.py``
    :name: grid_search_usage.py
    :lines: 13-14

Next we define a :class:`GridSearch <surprise.evaluate.GridSearch>` instance
and give it the class
:class:`SVD <surprise.prediction_algorithms.matrix_factorization.SVD>` as an
algorithm, and ``param_grid``. We will compute both the
RMSE and FCP values for all the combination. Thus the following definition:

.. literalinclude:: ../../examples/grid_search_usage.py
    :caption: From file ``examples/grid_search_usage.py``
    :name: grid_search_usage2.py
    :lines: 16

Now that :class:`GridSearch <surprise.evaluate.GridSearch>` instance is ready,
we can evaluate the algorithm on any data with the
:meth:`GridSearch.evaluate()<surprise.evaluate.GridSearch.evaluate>` method,
exactly like with the regular
:func:`evaluate() <surprise.evaluate.evaluate>` function:

.. literalinclude:: ../../examples/grid_search_usage.py
    :caption: From file ``examples/grid_search_usage.py``
    :name: grid_search_usage3.py
    :lines: 19-22

Everything is ready now to read the results. For example, we get the best RMSE
and FCP scores and parameters as follows:

.. literalinclude:: ../../examples/grid_search_usage.py
    :caption: From file ``examples/grid_search_usage.py``
    :name: grid_search_usage4.py
    :lines: 24-38

For further analysis, we can easily read all the results in a pandas
``DataFrame`` as follows:

.. literalinclude:: ../../examples/grid_search_usage.py
    :caption: From file ``examples/grid_search_usage.py``
    :name: grid_search_usage5.py
    :lines: 40-

.. _grid_search_note:
.. note::

    Dictionary parameters such as ``bsl_options`` and ``sim_options`` require
    particular treatment. See usage example below:

    .. parsed-literal::

        param_grid = {'k': [10, 20],
                      'sim_options': {'name': ['msd', 'cosine'],
                                      'min_support': [1, 5],
                                      'user_based': [False]}
                      }

    Naturally, both can be combined, for example for the
    :class:`KNNBaseline <surprise.prediction_algorithms.knns.KNNBaseline>`
    algorithm:

    .. parsed-literal::
        param_grid = {'bsl_options': {'method': ['als', 'sgd'],
                                      'reg': [1, 2]},
                      'k': [2, 3],
                      'sim_options': {'name': ['msd', 'cosine'],
                                      'min_support': [1, 5],
                                      'user_based': [False]}
                      }




Command line usage
------------------

Surprise can also be used from the command line, for example:

.. code::

    surprise -algo SVD -params "{'n_epochs': 5, 'verbose': True}" -load-builtin ml-100k -n-folds 3

See detailed usage by running:

.. code::

    surprise -h
