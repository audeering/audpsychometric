import typing

import numpy as np
import numpy.typing as npt
import pandas as pd

import audmetric


def confidence_categorical(
    ratings: npt.ArrayLike,
    *,
    axis: int = 1,
) -> typing.Union[float, np.ndarray]:
    r"""Confidence score for categorical ratings.

    The confidence for categorical data the fraction of raters per item
    with the rating being equal to that of the gold standard

    TODO: add equation

    Args:
        ratings: one row of the table containing raters' values
        axis: axis along which the confidences are computed.
            A value of ``1``
            assumes stimuli as rows
            and raters as columns

    Returns:
        categorical confidence score

    """
    ratings = np.atleast_2d(np.array(ratings))

    def _confidence(x):
        x = [val for val in x if not pd.isnull(val)]
        return np.sum(x == _mode(x)) / len(x)

    return _value_or_array(np.apply_along_axis(_confidence, axis, ratings))


def confidence_numerical(
    ratings: npt.ArrayLike,
    minimum: float,
    maximum: float,
    *,
    axis: int = 1,
) -> typing.Union[float, np.ndarray]:
    r"""Confidence score for numerical ratings.

    .. math::
        \text{confidence}(\text{ratings}) =
        \max(
        0, 1 - \frac{\text{std}(\text{ratings})}
        {\text{maximum} - \frac{1}{2} (\text{minimum} + \text{maximum})}
        )

    with :math:`\text{std}` the standard deviation of the ratings.

    Args:
        ratings: ratings,
            whereas a one dimensional ratings
            are treated as a row vector
        minimum: lower limit of possible rating value
        maximum: upper limit of possible rating value
        axis: axis along which the confidences are computed.
            A value of ``1``
            assumes stimuli as rows
            and raters as columns

    Returns:
        numerical confidence score(s)

    """
    ratings = np.atleast_2d(np.array(ratings))
    cutoff_max = maximum - 1 / 2 * (minimum + maximum)
    std = ratings.std(ddof=0, axis=axis)
    return _value_or_array(
        np.max([np.zeros(std.shape), np.ones(std.shape) - std / cutoff_max])
    )


def evaluator_weighted_estimator(
    ratings: npt.ArrayLike,
    *,
    axis: int = 1,
) -> typing.Union[float, np.ndarray]:
    r"""Evaluator weighted estimator (EWE) of raters' votes.

    The EWE is described in
    :cite:`coutinho-etal-2016-assessing` as follows:

    The EWE average of the individual ratings considers
    that each evaluator is subject to an individual amount of disturbance
    during the evaluation,
    by introducing evaluator-dependent weights
    that correspond to the correlation
    between the listener’s responses
    and the average ratings of all evaluators.

    See also `audformat#102`_ for implementation details.

    .. _audformat#102: https://github.com/audeering/audformat/issues/102

    Args:
        ratings: ratings
        axis: axis along which the EWE is computed.
            A value of ``1``
            assumes stimuli as rows
            and raters as columns

    Returns:
        EWE over raters

    """
    ratings = np.array(ratings)
    confidences = rater_confidence_pearson(ratings, axis=axis)
    # Ensure columns represents different raters
    if axis == 0:
        ratings = ratings.T
    return _value_or_array(np.inner(ratings, confidences) / np.sum(confidences))


def mode(
    ratings: npt.ArrayLike,
    *,
    axis: int = 1,
) -> typing.Union[float, np.ndarray]:
    r"""Mode of categorical ratings.

    Args:
        ratings: ratings
        axis: axis along which the mode is computed.
            A value of ``1``
            assumes stimuli as rows
            and raters as columns

    Returns:
        mode over raters

    """
    ratings = _remove_empty(np.atleast_2d(np.array(ratings)))
    return _value_or_array(np.apply_along_axis(_mode, axis, ratings))


def rater_confidence_pearson(
    ratings: npt.ArrayLike,
    *,
    axis: int = 1,
) -> np.ndarray:
    """Calculate rater confidences.

    Calculate the confidence of a rater
    by the correlation of a rater
    with the mean score of all other raters.

    This should not be confused with the confidence value
    that relates to a rated stimulus,
    e.g. :func:`audspychometric.confidence_numerical`.

    Args:
        ratings: ratings of each rater.
            Has to contain more than one rater
        axis: axis along which the rater confidence is computed.
            A value of ``1``
            assumes stimuli as rows
            and raters as columns

    Returns:
        rater confidences

    """
    ratings = np.array(ratings)

    # Ensure columns represents different raters
    if axis == 0:
        ratings = ratings.T

    # Remove stimuli (rows),
    # which miss ratings for one rater or more
    ratings = ratings[:, ~np.isnan(ratings).any(axis=0)]

    # Calculate confidence as Pearson Correlation Coefficient
    # between the raters' ratings
    # and the average ratings of all other raters
    confidences = []
    for n in range(ratings.shape[1]):
        ratings_selected_rater = ratings[:, n]
        average_ratings_other_raters = np.delete(ratings, n, axis=1).mean(axis=1)
        confidences.append(
            audmetric.pearson_cc(ratings_selected_rater, average_ratings_other_raters)
        )
    return np.array(confidences)


def _value_or_array(values: np.ndarray) -> typing.Union[float, np.ndarray]:
    r"""Convert single valued arrays to value.

    Squeeze array,
    and convert to single value,
    if it contains only a single entry.

    Args:
        values: input array

    Returns:
        converted array / object

    """
    values = values.squeeze()
    if values.ndim == 0 or values.shape == (1,):
        values = values.item()
    return values


def _mode(x: np.ndarray) -> typing.Any:
    """Mode of categorical values.

    Args:
        x: 1-dimensional values

    Returns:
        mode

    """
    values, counts = np.unique(x, return_counts=True)
    print(f"{values=}")
    print(f"{counts=}")
    # Find indices with maximum count
    idx = np.flatnonzero(counts == np.max(counts))
    try:
        # Take average over values with same count
        # and round to next integer
        mode = int(np.floor(np.mean(values[idx]) + 0.5))
    except TypeError:
        # If we cannot take the mean,
        # take the first occurrence
        first_occurence = np.min([np.where(x == value) for value in values[idx]])
        mode = x[first_occurence]
    return mode


def _remove_empty(ratings: np.ndarray) -> np.ndarray:
    r"""Remove empty ratings.

    Args:
        ratings: 2-dimensional array

    Returns:
        ratings without ``None`` and ``nan`` entries

    """
    return np.array([[x for x in row if not pd.isnull(x)] for row in ratings])
