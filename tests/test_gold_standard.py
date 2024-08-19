"""Tests of the module calculating gold standard and item confidence

Usage Example(s):

    pytest tests/test_goldstandard.py -k test_evaluator_weighted_estimator

"""

import io

import numpy as np
import pandas as pd
import pytest

import audpsychometric
from audpsychometric.core.gold_standard import _confidence_categorical


def test_rater_confidence_pearson(df_holzinger_swineford):
    """Happy Flow test for mode for rater based consistency"""
    result = audpsychometric.rater_confidence_pearson(df_holzinger_swineford)
    result_values = np.array([x for x in result.values()])
    # there is a very unrealible rater in this set with .24
    assert all(x > 0.2 for x in result_values)


def test_mode_based_gold_standard():
    """Happy Flow test for mode based gold standard"""
    df = pd.DataFrame([[4, 9, np.nan]] * 3, columns=["A", "B", "C"])
    df = audpsychometric.gold_standard_mode(df)
    assert isinstance(df, pd.DataFrame)
    assert "gold_standard" in df.columns
    assert "confidence" in df.columns
    assert np.all((df["confidence"] >= 0.0) & (df["confidence"] <= 1.0).values)


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
    "df, minimum, maximum, axis, df_expected",
    [
        (
            pd.DataFrame([0]),
            0,
            1,
            1,
            pd.DataFrame(
                [[0.0, 1.0]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame([[0, 0]]),
            0,
            1,
            1,
            pd.DataFrame(
                [[0.0, 1.0]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame([[1, 1]]),
            0,
            1,
            1,
            pd.DataFrame(
                [[1.0, 1.0]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame([[0.3, 0.3, 0.3]]),
            0,
            1,
            1,
            pd.DataFrame(
                [[0.3, 1.0]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame([[0, 0, 0.1, 0.2]]),
            0,
            1,
            1,
            pd.DataFrame(
                [[0.075, 0.83416876048223]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame([[0, 0, 0.2, 0.4]]),
            0,
            1,
            1,
            pd.DataFrame(
                [[0.150, 0.66833752096446]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame([[0, 0, 0, 0, 0.2, 0.2, 0.4, 0.4]]),
            0,
            1,
            1,
            pd.DataFrame(
                [[0.150, 0.66833752096446]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame([[0, 0.4, 0.6, 1]]),
            0,
            1,
            1,
            pd.DataFrame(
                [[0.5, 0.2788897449072021]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame([[0, 0.33, 0.67, 1]]),
            0,
            1,
            1,
            pd.DataFrame(
                [[0.5, 0.2531399060064863]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame([[0, 1]]),
            0,
            1,
            1,
            pd.DataFrame(
                [[0.5, 0.0]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame([[0, 0, 1, 1]]),
            0,
            1,
            1,
            pd.DataFrame(
                [[0.5, 0.0]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame(
                [
                    [1, 2, 3],
                    [3, 4, 5],
                ]
            ),
            0,
            10,
            0,
            pd.DataFrame(
                [
                    [2.0, 0.8],
                    [3.0, 0.8],
                    [4.0, 0.8],
                ],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame(
                [
                    [1, 2, 3],
                    [3, 4, 5],
                ]
            ),
            0,
            10,
            1,
            pd.DataFrame(
                [
                    [2.0, 0.8367006838144548],
                    [4.0, 0.8367006838144548],
                ],
                columns=["gold_standard", "confidence"],
            ),
        ),
    ],
)
def test_mean_based_gold_standard(df, minimum, maximum, axis, df_expected):
    """Happy Flow test for mode based gold standard"""
    pd.testing.assert_frame_equal(
        audpsychometric.gold_standard_mean(df, minimum, maximum, axis=axis),
        df_expected,
    )


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
    "df, minimum, maximum, axis, df_expected",
    [
        (
            pd.DataFrame([0]),
            0,
            1,
            1,
            pd.DataFrame(
                [[0.0, 1.0]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame([[0, 0]]),
            0,
            1,
            1,
            pd.DataFrame(
                [[0.0, 1.0]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame([[1, 1]]),
            0,
            1,
            1,
            pd.DataFrame(
                [[1.0, 1.0]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame([[0.3, 0.3, 0.3]]),
            0,
            1,
            1,
            pd.DataFrame(
                [[0.3, 1.0]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame([[0, 0, 0.1, 0.2]]),
            0,
            1,
            1,
            pd.DataFrame(
                [[0.05, 0.83416876048223]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame([[0, 0, 0.2, 0.4]]),
            0,
            1,
            1,
            pd.DataFrame(
                [[0.1, 0.66833752096446]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame([[0, 0, 0, 0, 0.2, 0.2, 0.4, 0.4]]),
            0,
            1,
            1,
            pd.DataFrame(
                [[0.1, 0.66833752096446]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame([[0, 0.4, 0.6, 1]]),
            0,
            1,
            1,
            pd.DataFrame(
                [[0.5, 0.2788897449072021]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame([[0, 0.33, 0.67, 1]]),
            0,
            1,
            1,
            pd.DataFrame(
                [[0.5, 0.2531399060064863]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame([[0, 1]]),
            0,
            1,
            1,
            pd.DataFrame(
                [[0.5, 0.0]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame([[0, 0, 1, 1]]),
            0,
            1,
            1,
            pd.DataFrame(
                [[0.5, 0.0]],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame(
                [
                    [1, 2, 3],
                    [3, 4, 5],
                ]
            ),
            0,
            10,
            0,
            pd.DataFrame(
                [
                    [2.0, 0.8],
                    [3.0, 0.8],
                    [4.0, 0.8],
                ],
                columns=["gold_standard", "confidence"],
            ),
        ),
        (
            pd.DataFrame(
                [
                    [1, 2, 3],
                    [3, 4, 5],
                ]
            ),
            0,
            10,
            1,
            pd.DataFrame(
                [
                    [2.0, 0.8367006838144548],
                    [4.0, 0.8367006838144548],
                ],
                columns=["gold_standard", "confidence"],
            ),
        ),
    ],
)
def test_median_based_gold_standard(df, minimum, maximum, axis, df_expected):
    """Test that  median gold standard returns df"""
    pd.testing.assert_frame_equal(
        audpsychometric.gold_standard_median(df, minimum, maximum, axis=axis),
        df_expected,
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_evaluator_weighted_estimator(df_holzinger_swineford, axis):
    """Happy Flow test for mode based gold standard"""
    if axis == 0:
        df_holzinger_swineford = df_holzinger_swineford.T

    df_ewe = audpsychometric.evaluator_weighted_estimator(
        df_holzinger_swineford,
        0,
        10,
        axis=axis,
    )

    # results obtained from reference implementation
    test_data = io.StringIO(
        """idx,ewe
    0,3.834844
    1,3.890689
    2,2.681920
    3,4.143616
    4,3.895072
    296,3.723935
    297,3.580962
    298,4.853387
    299,3.946110
    300,4.602326"""
    )

    df_test_data = pd.read_csv(test_data, sep=",", index_col="idx")
    true_results = df_test_data.copy(deep=True).values.flatten()
    data_results = df_ewe["EWE"].loc[df_test_data.index].values.flatten()
    assert np.allclose(data_results, true_results)


@pytest.mark.xfail(raises=ValueError)
def test_f_categorical(df_holzinger_swineford):
    """Test that functional raises when no gold column"""
    _ = _confidence_categorical(df_holzinger_swineford)
