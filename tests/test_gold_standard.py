"""Tests of the module calculating gold standard and item confidence.

Examples:
    pytest tests/test_goldstandard.py -k test_evaluator_weighted_estimator

"""


import numpy as np
import pandas as pd
import pytest

import audpsychometric


def test_confidence_categorical():
    pass


# The expected confidence value for this test
# can be calculated by
#
# def confidence(rating, minimum, maximum):
#     max_std = (maximum - minimum) / 2
#     std = np.std(rating)
#     std_norm = np.clip(std/max_std, 0, 1)
#     return 1 - std_norm
#
@pytest.mark.parametrize(
    "ratings, minimum, maximum, axis, expected",
    [
        (pd.DataFrame([0]), 0, 1, 1, 1.0),
        (pd.DataFrame([[0, 0]]), 0, 1, 1, 1.0),
        (pd.DataFrame([[1, 1]]), 0, 1, 1, 1.0),
        (pd.DataFrame([[0.3, 0.3, 0.3]]), 0, 1, 1, 1.0),
        (pd.DataFrame([[0, 0, 0.1, 0.2]]), 0, 1, 1, 0.83416876048223),
        (pd.DataFrame([[0, 0, 0.2, 0.4]]), 0, 1, 1, 0.66833752096446),
        (pd.DataFrame([[0, 0, 0, 0, 0.2, 0.2, 0.4, 0.4]]), 0, 1, 1, 0.66833752096446),
        (pd.DataFrame([[0, 0.4, 0.6, 1]]), 0, 1, 1, 0.2788897449072021),
        (pd.DataFrame([[0, 0.33, 0.67, 1]]), 0, 1, 1, 0.2531399060064863),
        (pd.DataFrame([[0, 1]]), 0, 1, 1, 0.0),
        (pd.DataFrame([[0, 0, 1, 1]]), 0, 1, 1, 0.0),
        (pd.DataFrame([[1, 2, 3], [3, 4, 5]]), 0, 10, 0, np.array([0.8, 0.8, 0.8])),
        (
            pd.DataFrame([[1, 2, 3], [3, 4, 5]]),
            0,
            10,
            1,
            np.array([0.8367006838144548, 0.8367006838144548]),
        ),
    ],
)
def test_confidence_numerical(ratings, minimum, maximum, axis, expected):
    """Test confidence for numerical ratings."""
    np.testing.assert_equal(
        audpsychometric.confidence_numerical(ratings, minimum, maximum, axis=axis),
        expected,
    )


def test_rater_confidence_pearson(df_holzinger_swineford):
    """Happy Flow test for mode for rater based consistency."""
    result = audpsychometric.rater_confidence_pearson(df_holzinger_swineford)
    result_values = np.array([x for x in result.values()])
    # there is a very unrealible rater in this set with .24
    assert all(x > 0.2 for x in result_values)


@pytest.mark.parametrize(
    "ratings, axis, expected",
    [
        (pd.DataFrame([0]), 1, 0),
        (pd.DataFrame([[0, 0]]), 1, 0),
        (pd.DataFrame([[1, 1]]), 1, 1),
        (pd.DataFrame([[0, 0]]), 0, np.array([0, 0])),
    ],
)
def test_mode(ratings, axis, expected):
    """Test mode over ratings."""
    np.testing.assert_equal(
        audpsychometric.mode(ratings, axis=axis),
        expected,
    )


@pytest.mark.parametrize(
    "ratings, axis, expected",
    [
        (pd.DataFrame([0]), 1, 0.0),
        (pd.DataFrame([[0, 0]]), 1, 0.0),
        (pd.DataFrame([[1, 1]]), 1, 1.0),
        (pd.DataFrame([[0.3, 0.3, 0.3]]), 1, 0.3),
        (pd.DataFrame([[0, 0, 0.1, 0.2]]), 1, 0.075),
        (pd.DataFrame([[0, 0, 0.2, 0.4]]), 1, 0.15),
        (pd.DataFrame([[0, 0, 0, 0, 0.2, 0.2, 0.4, 0.4]]), 1, 0.15),
        (pd.DataFrame([[0, 0.4, 0.6, 1]]), 1, 0.5),
        (pd.DataFrame([[0, 0.33, 0.67, 1]]), 1, 0.5),
        (pd.DataFrame([[0, 1]]), 1, 0.5),
        (pd.DataFrame([[0, 0, 1, 1]]), 1, 0.5),
        (pd.DataFrame([[1, 2, 3], [3, 4, 5]]), 0, np.array([2, 3, 4])),
        (pd.DataFrame([[1, 2, 3], [3, 4, 5]]), 1, np.array([2, 4])),
    ],
)
def test_mean(ratings, axis, expected):
    """Test mean over ratings.

    Args:
        ratings: ratings
        axis: axis to calculate the mean
        expected: expected mean

    """
    np.testing.assert_equal(
        audpsychometric.mean(ratings, axis=axis),
        expected,
    )


@pytest.mark.parametrize(
    "ratings, axis, expected",
    [
        (pd.DataFrame([0]), 1, 0.0),
        (pd.DataFrame([[0, 0]]), 1, 0.0),
        (pd.DataFrame([[1, 1]]), 1, 1.0),
        (pd.DataFrame([[0.3, 0.3, 0.3]]), 1, 0.3),
        (pd.DataFrame([[0, 0, 0.1, 0.2]]), 1, 0.05),
        (pd.DataFrame([[0, 0, 0.2, 0.4]]), 1, 0.1),
        (pd.DataFrame([[0, 0, 0, 0, 0.2, 0.2, 0.4, 0.4]]), 1, 0.1),
        (pd.DataFrame([[0, 0.4, 0.6, 1]]), 1, 0.5),
        (pd.DataFrame([[0, 0.33, 0.67, 1]]), 1, 0.5),
        (pd.DataFrame([[0, 1]]), 1, 0.5),
        (pd.DataFrame([[0, 0, 1, 1]]), 1, 0.5),
        (pd.DataFrame([[1, 2, 3], [3, 4, 5]]), 0, np.array([2.0, 3.0, 4.0])),
        (pd.DataFrame([[1, 2, 3], [3, 4, 5]]), 1, np.array([2.0, 4.0])),
    ],
)
def test_median(ratings, axis, expected):
    """Test median over ratings.

    Args:
        ratings: ratings
        axis: axis to calculate median
        expected: expected median

    """
    np.testing.assert_equal(
        audpsychometric.median(ratings, axis=axis),
        expected,
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_evaluator_weighted_estimator(df_holzinger_swineford, axis):
    """Test EWE over ratings.

    Args:
        df_holzinger_swineford: df_holzinger_swineford fixture
        axis: axis to calculate EWE

    """
    if axis == 0:
        df_holzinger_swineford = df_holzinger_swineford.T

    ewe = audpsychometric.evaluator_weighted_estimator(
        df_holzinger_swineford,
        axis=axis,
    )

    # results obtained from reference implementation
    expected = np.array(
        [
            3.834844,
            3.890689,
            2.681920,
            4.143616,
            3.895072,
            3.723935,
            3.580962,
            4.853387,
            3.946110,
            4.602326,
        ]
    )
    assert np.allclose(ewe, expected)
