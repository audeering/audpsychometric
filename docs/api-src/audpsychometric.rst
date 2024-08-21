audpsychometric
===============

.. automodule:: audpsychometric

Library to facilitate evaluation and processing of annotated speech.

The input data format for all functions in this module is the same:
a :class:`pd.DataFrame` is expected.

This dataframe is assumed to

- have an index that identifies the unit of observation,
  e.g. a psychometric item to be rated
- have a separate column for each rater

So the entry in the frame at (irow, icol)
identifies the rating of unit irow by rather icol.
Note that these assumptions are not checked
and are under responsibility of the user.

Pychometric Analysis
--------------------

.. autosummary::
    :toctree:
    :nosignatures:

    cronbachs_alpha
    congeneric_reliability
    intra_class_correlation

The module currently contains two reliability coefficients
from the family of structural equation model (SEM)-based
reliability coefficients.
One of them is Cronbach's alphas
in the function :func:`audpsychometric.cronbachs_alpha`.
This classical coefficient assumes *tau equivalence*
which requires factor loadings to be homogeneous.
The second coefficient
in the function :func:`audpsychometric.congeneric_reliability`
relaxes this assumption
and only assumes a `one-dimensional congeneric reliability`_ model:
congeneric measurement models are characterized by the fact
that the factor loadings of the indicators
do not have to be homogeneous,
i.e. they can differ.

In addition,
the module implements *Intraclass Correlation (ICC)* analysis.
ICC is based on the analysis of variance of a class of coefficients
that are based on ANOVA
with ratings as the dependent variable,
and terms for targets
(like e.g rated audio chunks),
raters and their interaction are estimated.
Different flavors of ICC are then computed
based on these sum of squares terms.

Note that the CCC_ is conceptually and numerically related to the ICC.
We do not implement it here,
as there are other implementations available,
e.g. :func:`audmetric.concordance_cc`.


Gold Standard Calculation
-------------------------

.. autosummary::
    :toctree:
    :nosignatures:

    confidence_categorical
    confidence_numerical
    evaluator_weighted_estimator
    mode
    rater_confidence_pearson


Demo Datasets
-------------

.. autosummary::
    :toctree:
    :nosignatures:

    list_datasets
    read_dataset

Currently these datasets are defined:

.. jupyter-execute::

    from audpsychometric import datasets
    df_datasets = datasets.list_datasets()
    print(df_datasets)


.. _one-dimensional congeneric reliability: https://en.wikipedia.org/wiki/Congeneric_reliability
.. _CCC: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
