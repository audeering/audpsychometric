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
    assert np.alltrue((df['confidence'] >= 0.) & (df['confidence'] <= 1.).values)


def test_mean_based_gold_standard(df_holzinger_swineford):
    """Happy Flow test for mode based gold standard"""
    df = audpsychometric.gold_standard_mean(df_holzinger_swineford, 0, 10)
    assert df.confidence.min() != 0


def test_median_based_gold_standard(df_holzinger_swineford):
    """Test that  median gold standard returns df"""
    df = audpsychometric.gold_standard_median(df_holzinger_swineford, 0, 10)
    assert isinstance(df, pd.DataFrame)
    assert "gold_standard" in df.columns
    assert "confidence" in df.columns
    assert np.alltrue((df['confidence'] >= 0.) & (df['confidence'] <= 1.).values)


def test_evaluator_weighted_estimator(df_holzinger_swineford):
    """Happy Flow test for mode based gold standard"""
    df_ewe = audpsychometric.evaluator_weighted_estimator(df_holzinger_swineford, 0, 10)

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
    """test that functional raises when no gold column"""
    _ = _confidence_categorical(df_holzinger_swineford)
