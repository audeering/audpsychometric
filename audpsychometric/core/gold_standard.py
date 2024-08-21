import typing

import numpy as np
import numpy.typing as npt

import audmetric


def confidence_categorical(ratings: npt.ArrayLike) -> float:
    r"""Confidence score for categorical ratings.

    The confidence for categorical data the fraction of raters per item
    with the rating being equal to that of the gold standard

    TODO: add equation

    Args:
        ratings: one row of the table containing raters' values

    Returns:
        categorical confidence score

    """
    ratings = np.array(ratings).squeeze()
    gold_standard = mode_numerical(ratings)
    number_of_nonzero_ratings = np.count_nonzero(~np.isnan(ratings))
    return np.sum(ratings == gold_standard) / number_of_nonzero_ratings


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
    # Ensure 2D with vector as row vector
    ratings = np.atleast_2d(np.array(ratings))
    cutoff_max = maximum - 1 / 2 * (minimum + maximum)
    std = ratings.std(ddof=0, axis=axis)
    return _float_or_array(
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

    # if axis == 0:
    #     df = df.T

    # raters = df.columns.tolist()

    # def ewe(row):
    #     """Functional to determine ewe per row."""
    #     total = sum([row[x] * confidences[x] for x in raters])
    #     total /= np.sum([confidences[x] for x in raters])
    #     return total

    # y = df.apply(ewe, axis=1)
    # y.name = "EWE"

    return _float_or_array(
        np.sum(ratings * confidences, axis=axis) / np.sum(confidences)
    )


def mode_numerical(
    ratings: npt.ArrayLike,
    *,
    axis: int = 1,
) -> typing.Union[float, np.ndarray]:
    r"""Mode of for numerical ratings.

    Args:
        ratings: ratings
        axis: axis along which the mode is computed.
            A value of ``1``
            assumes stimuli as rows
            and raters as columns

    Returns:
        mode over raters

    """
    ratings = np.atleast_2d(np.array(ratings))

    def _mode(x):
        values, counts = np.unique(x, return_counts=True)
        # Find indices with maximum count
        idx = np.flatnonzero(counts == np.max(counts))
        # Take average over values with same count
        # and round to next integer
        return int(np.floor(np.mean(values[idx]) + 0.5))

    print(f"{ratings=}")
    print(f"{np.apply_along_axis(_mode, axis, ratings)=}")
    return _float_or_array(np.apply_along_axis(_mode, axis, ratings))


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
        ratings: matrix of ratings of each rater.
            Has to contain more than one rater
        axis: axis to calculate mean and confidences.
            A value of ``1`` expects raters to be columns

    Returns:
        rater confidences

    """
    ratings = np.array(ratings)

    # Ensure columns represents different raters
    if axis == 0:
        ratings = ratings.T
    elif axis > 1:
        raise ValueError(f"axis has to be 0 or 1, not {axis}.")

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


def _float_or_array(values: np.ndarray) -> typing.Union[float, np.ndarray]:
    r"""Convert single valued arrays as float.

    Squeeze array,
    and convert to single float value,
    if it contains only a single entry.

    Args:
        values: input array

    Returns:
        converted array / float

    """
    values = values.squeeze()
    if values.ndim == 0 or values.shape == (1,):
        values = float(values)
    return values
