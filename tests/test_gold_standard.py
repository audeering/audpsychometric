import typing

import numpy as np
import pandas as pd
import pytest

import audpsychometric


def to_list_array_frame_series(
    x: list,
) -> typing.Tuple[list, np.ndarray, pd.DataFrame, typing.Optional[pd.Series]]:
    r"""Converts list to other input objects.

    It converts a list to an array,
    dataframe,
    and if is not a nested list (1-dimensional),
    to a series.
    For a 1-dimensional input,
    a dataframe with a single row is created.

    Args:
        x: values

    Returns:
        tuple containing values as list, array, dataframe,
        and maybe series

    """
    # list, array, dataframe
    outputs = [x, np.array(x), pd.DataFrame(np.atleast_2d(x))]
    # series
    if np.array(x).ndim == 1:
        outputs.append(pd.Series(x))
    return tuple(outputs)


@pytest.mark.parametrize(
    "ratings, axis, expected",
    [
        # axis = 0
        ([0], 0, 1.0),
        (["a"], 0, 1.0),
        ([0, 0], 0, np.array([1.0, 1.0])),
        ([[0, 0]], 0, np.array([1.0, 1.0])),
        ([[0], [0]], 0, 1.0),
        ([[None], ["a"]], 0, 1.0),
        # axis = 1
        ([0], 1, 1.0),
        (["a"], 1, 1.0),
        ([0, 0], 1, 1.0),
        ([[0, 0]], 1, 1.0),
        ([[0], [0]], 1, np.array([1.0, 1.0])),
        ([0, 0, 1], 1, 2 / 3),
        ([0, 0, 1, 1], 1, 0.5),
        (["a", "a", "b"], 1, 2 / 3),
        (["a", "a", "b", "b"], 1, 0.5),
        ([0, 1, 2], 1, 1 / 3),
        ([0, 1, 2, 2], 1, 0.5),
        (["a", "b", "c"], 1, 1 / 3),
        (["a", "b", "c", "c"], 1, 0.5),
        ([np.nan, 1, 1], 1, 1.0),
        ([np.nan, 0, 1], 1, 0.5),
        ([None, "a", "a"], 1, 1.0),
        ([None, "a", "b"], 1, 0.5),
        ([None, np.nan, 1], 1, 1.0),
    ],
)
def test_agreement_categorical(ratings, axis, expected):
    """Test agreement for categorical ratings.

    Args:
        ratings: ratings as list
        axis: axis along to compute agreement
        expected: expected agreement score(s)

    """
    for x in to_list_array_frame_series(ratings):
        np.testing.assert_equal(
            audpsychometric.agreement_categorical(x, axis=axis),
            expected,
            strict=True,
        )


# The expected agreement value for this test
# can be calculated by:
#
# def agreement(rating, minimum, maximum):
#     max_std = (maximum - minimum) / 2
#     std = np.std(rating)
#     std_norm = np.clip(std/max_std, 0, 1)
#     return 1 - std_norm
#
@pytest.mark.parametrize(
    "ratings, minimum, maximum, axis, expected",
    [
        # axis = 0
        ([0], 0, 1, 0, 1.0),
        ([0, 0], 0, 1, 0, np.array([1.0, 1.0])),
        ([[0, 0]], 0, 1, 0, np.array([1.0, 1.0])),
        ([[0], [0]], 0, 1, 0, 1.0),
        ([[0.3, 0.3, 0.3]], 0, 1, 0, np.array([1.0, 1.0, 1.0])),
        ([0, 1], 0, 1, 0, np.array([1.0, 1.0])),
        ([[0], [1]], 0, 1, 0, 0.0),
        ([[0], [1]], 0, 2, 0, 0.5),
        ([[1, 1]], 0, 1, 0, np.array([1.0, 1.0])),
        ([[1, 2, 3], [3, 4, 5]], 0, 10, 0, np.array([0.8, 0.8, 0.8])),
        ([[1, 2, 3], [3, 4, 4]], 0, 10, 0, np.array([0.8, 0.8, 0.9])),
        # axis = 1
        ([0], 0, 1, 1, 1.0),
        ([0, 0], 0, 1, 1, 1.0),
        ([[0, 0]], 0, 1, 1, 1.0),
        ([[0], [0]], 0, 1, 1, np.array([1.0, 1.0])),
        ([[1, 1]], 0, 1, 1, 1.0),
        ([[0.3, 0.3, 0.3]], 0, 1, 1, 1.0),
        ([0, 1], 0, 1, 1, 0.0),
        ([np.nan, 1], 0, 1, 1, 1.0),
        ([[0, 0, 0.1, 0.2]], 0, 1, 1, 0.83416876048223),
        ([[0, 0, 0.2, 0.4]], 0, 1, 1, 0.66833752096446),
        ([[0, 0, 0, 0, 0.2, 0.2, 0.4, 0.4]], 0, 1, 1, 0.66833752096446),
        ([[0, 0.4, 0.6, 1]], 0, 1, 1, 0.2788897449072021),
        ([[0, 0.33, 0.67, 1]], 0, 1, 1, 0.2531399060064863),
        ([[0, 1]], 0, 1, 1, 0.0),
        ([[0, 0, 1, 1]], 0, 1, 1, 0.0),
        (
            [[1, 2, 3], [3, 4, 5]],
            0,
            10,
            1,
            np.array([0.8367006838144548, 0.8367006838144548]),
        ),
        (
            [[1, 2, 3], [3, 4, 4]],
            0,
            10,
            1,
            np.array([0.8367006838144548, 0.9057190958417937]),
        ),
    ],
)
def test_agreement_numerical(ratings, minimum, maximum, axis, expected):
    """Test agreement for numerical ratings.

    If only a vector is given for ``ratings``,
    it should be treated as column vector.
    An value of 0 for ``axis``
    should compute the agreement scores along rows.

    Args:
        ratings: ratings as list
        minimum: lower limit of ratings
        maximum: upper limit of ratings
        axis: axis along to compute agreement
        expected: expected agreement score(s)

    """
    for x in to_list_array_frame_series(ratings):
        np.testing.assert_equal(
            audpsychometric.agreement_numerical(x, minimum, maximum, axis=axis),
            expected,
            strict=True,
        )


def test_rater_agreement(df_holzinger_swineford):
    """Test rater agreement."""
    # there is a very unrealible rater in this set with .24
    expected = np.array(
        [
            0.52203673,
            0.27524307,
            0.37017212,
            0.58070663,
            0.52538537,
            0.59513902,
            0.24573167,
            0.36905549,
            0.49478097,
        ],
    )
    np.testing.assert_allclose(
        audpsychometric.rater_agreement(df_holzinger_swineford),
        expected,
        strict=True,
    )


@pytest.mark.parametrize(
    "ratings, axis, expected",
    [
        # axis = 0
        ([0], 0, 0),
        ([0, 0], 0, np.array([0, 0])),
        ([[0, 0]], 0, np.array([0, 0])),
        ([[0], [0]], 0, 0),
        (["a"], 0, "a"),
        (["a", "a"], 0, np.array(["a", "a"])),
        ([["a", "a"]], 0, np.array(["a", "a"])),
        ([["a"], ["a"]], 0, "a"),
        ([[np.nan, 1], [2, np.nan]], 0, np.array([2, 1])),
        # axis = 1
        ([0], 1, 0),
        ([0, 0], 1, 0),
        ([0, 1], 1, 1),
        ([0, 0, 1], 1, 0),
        ([0, 1, 2], 1, 1),
        ([0, 1, 1], 1, 1),
        ([0, 2, 2, 1], 1, 2),
        ([0, 2, 2, 1, 1], 1, 2),
        ([0, 2, 2, 1, 1, 1], 1, 1),
        ([0, 2], 1, 1),
        ([[0, 0]], 1, 0),
        ([[0], [0]], 1, np.array([0, 0])),
        ([[1, 1]], 1, 1),
        (["a"], 1, "a"),
        (["a", "a"], 1, "a"),
        (["a", "b"], 1, "a"),
        (["b", "a"], 1, "b"),
        (["a", "a", "b"], 1, "a"),
        (["b", "a", "a"], 1, "a"),
        (["a", "b", "c"], 1, "a"),
        (["a", "c", "c", "b"], 1, "c"),
        (["a", "c", "c", "b", "b"], 1, "c"),
        (["a", "c", "c", "b", "b", "b"], 1, "b"),
        ([["a", "a"]], 1, "a"),
        ([["a"], ["a"]], 1, np.array(["a", "a"])),
        ([np.nan, np.nan, 1], 1, 1),
        ([np.nan, np.nan, None, None, 1], 1, 1),
        ([[np.nan, 1], [2, np.nan]], 1, np.array([1, 2])),
    ],
)
def test_mode(ratings, axis, expected):
    """Test mode over ratings.

    If the categories are integers,
    and there is no winner,
    mode should return the average of all winners.
    For other categories,
    the first item should be returned
    of the winning categories.

    Args:
        ratings: ratings as list
        axis: axis along to compute mode
        expected: expected mode values

    """
    for x in to_list_array_frame_series(ratings):
        np.testing.assert_equal(
            audpsychometric.mode(x, axis=axis),
            expected,
            strict=True,
        )


@pytest.mark.parametrize("axis", [0, 1])
def test_evaluator_weighted_estimator(df_holzinger_swineford, axis):
    """Test EWE over ratings.

    Args:
        df_holzinger_swineford: df_holzinger_swineford fixture
        axis: axis along to compute EWE

    """
    ewe = audpsychometric.evaluator_weighted_estimator(
        df_holzinger_swineford,
        axis=axis,
    )
    if axis == 0:
        expected = df_holzinger_swineford.values.shape[1]
    elif axis == 1:
        expected = df_holzinger_swineford.values.shape[0]
    assert ewe.shape[0] == expected
