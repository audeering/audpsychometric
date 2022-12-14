audpsychometric
===============

.. automodule:: audpsychometric

Pychometric Analysis
--------------------

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

cronbachs_alpha
^^^^^^^^^^^^^^^

.. autofunction:: cronbachs_alpha

congeneric_reliability
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: congeneric_reliability

intra_class_correlation
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: intra_class_correlation


Gold Standard Calculation
-------------------------

evaluator_weighted_estimator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: evaluator_weighted_estimator

gold_standard_mean
^^^^^^^^^^^^^^^^^^

.. autofunction:: gold_standard_mean

gold_standard_median
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gold_standard_median

gold_standard_mode
^^^^^^^^^^^^^^^^^^

.. autofunction:: gold_standard_mode

rater_confidence_pearson
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: rater_confidence_pearson


Demo Datasets
-------------

Currently these datasets are defined:

.. jupyter-execute::

    from audpsychometric import datasets
    df_datasets = datasets.list_datasets()
    print(df_datasets)

list_datasets
^^^^^^^^^^^^^

.. autofunction:: list_datasets

read_dataset
^^^^^^^^^^^^

.. autofunction:: read_dataset


.. _one-dimensional congeneric reliability: https://en.wikipedia.org/wiki/Congeneric_reliability
.. _CCC: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
