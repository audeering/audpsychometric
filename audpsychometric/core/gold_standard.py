import typing

import numpy as np
import numpy.typing as npt

import audmetric


def mean(
    ratings: npt.ArrayLike,
    *,
    axis: int = 1,
) -> typing.Union[float, np.ndarray]:
    r"""Mean of raters' votes.

    Args:
        ratings: ratings
        axis: axis to calculate mean.
            A value of ``1`` expects raters to be columns

    Returns:
        mean over raters

    """
    ratings = np.array(ratings)
    return _float_or_array(ratings.mean(axis=axis))


def median(
    ratings: npt.ArrayLike,
    *,
    axis: int = 1,
) -> typing.Union[float, np.ndarray]:
    r"""Median of raters' votes.

    Args:
        ratings: ratings
        axis: axis to calculate mean and confidences.
            A value of ``1`` expects raters to be columns

    Returns:
        median over raters

    """
    ratings = np.array(ratings)
    return _float_or_array(ratings.median(axis=axis))


def mode(
    ratings: npt.ArrayLike,
    *,
    axis: int = 1,
) -> typing.Union[float, np.ndarray]:
    r"""Mode of raters' votes.

    Args:
        ratings: ratings
        axis: axis to calculate mode.
            A value of ``1`` expects raters to be columns

    Returns:
        mode over raters

    """
    ratings = np.array(ratings)
    return _float_or_array(np.floor(_mode(ratings, axis=axis).mean(axis=axis) + 0.5))


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
        axis: axis to calculate EWE.
            A value of ``1`` expects raters to be columns


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


def confidence_categorical(ratings: npt.ArrayLike) -> float:
    r"""Confidence score for categorical ratings.

    The confidence for categorical data the fraction of raters per item
    with the rating being equal to that of the gold standard

    TODO: add equation

    Args:
        row: one row of the table containing raters' values

    Returns:
        categorical confidence score

    """
    ratings = np.array(ratings).squeeze()
    gold_standard = gold_standard_mode(ratings)
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

    with :math:`\text{std}` the standard deviation of the ratings

    Args:
        ratings: ratings for a given stimuli
        minimum: minimum value of the ratings to calculate cut off
        maximum: maximum value of the ratings to calculate cut off
        axis: axis to calculate confidence

    Returns:
        numerical confidence score(s)

    """
    ratings = np.array(ratings)
    cutoff_max = maximum - 1 / 2 * (minimum + maximum)
    print(f"{1 - ratings.std(ddof=0, axis=axis)=}")
    std = ratings.std(ddof=0, axis=axis)
    return _float_or_array(
        # np.max([0.0, 1 - ratings.std(ddof=0, axis=axis) / cutoff_max])
        np.max([np.zeros(std.shape), np.ones(std.shape) - std / cutoff_max])
    )


def rater_confidence_pearson(
    ratings: npt.ArrayLike,
    *,
    axis: int = 1,
) -> np.ndarray:
    """Calculate the rater confidence.

    Calculate the confidence of a rater
    as the correlation of a rater
    with the mean score of all other raters.

    This should not be confused with the value
    that relates to a rated stimulus.

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

    # Remove examples,
    # which miss at least one rater
    ratings = ratings[:, ~np.isnan(ratings).any(axis=0)]

    confidences = []
    for n in range(ratings.shape[1]):
        ratings_selected_rater = ratings[:, n]
        average_ratings_other_raters = np.delete(ratings, n, axis=1).mean(axis=0)
        print(f"{ratings_selected_rater.shape=}")
        print(f"{average_ratings_other_raters.shape=}")
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
    if values.shape == (1,):
        values = float(values)
    return values


def _mode(ratings: np.ndarray, *, axis: int = 1) -> typing.Union[float, np.ndarray]:
    r"""Mode of ratings.

    Implements :meth:`pandas.DataFrame.mode` with :mod:`numpy`.

    Args:
        ratings: ratings
        axis: axis along to calculate the mode

    Returns:
        mode of ratings

    """
    values, counts = np.unique(ratings, return_counts=True, axis=axis)
    idx = np.argmax(counts)
    if len(values.shape) > 1 and axis == 1:
        values = np.array([column[index] for column in values])
    else:
        values = values[index]
    return _float_or_array(values)
