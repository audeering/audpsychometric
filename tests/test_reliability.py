import numpy as np
import pandas as pd
import pytest

import audpsychometric


def test_icc():
    """Test icc basic result validity."""
    df_dataset = audpsychometric.read_dataset("wine")

    data_wide = df_dataset.pivot_table(index="Wine", columns="Judge", values="Scores")

    icc_sm, _ = audpsychometric.intra_class_correlation(
        data_wide, anova_method="statsmodels"
    )
    for ratings in [data_wide, data_wide.values]:
        icc_pingouin, _ = audpsychometric.intra_class_correlation(ratings)
        assert np.isclose(icc_pingouin, 0.727, atol=1e-3)
        assert np.isclose(icc_sm, icc_pingouin, atol=1e-10)
        icc_pingouin_t, _ = audpsychometric.intra_class_correlation(ratings.T, axis=0)
        assert np.isclose(icc_pingouin_t, icc_pingouin, atol=1e-10)


def test_cronbachs_alpha():
    """Test cronbach's alpha return values for three raters."""
    df_dataset = audpsychometric.read_dataset("hallgren-table3")
    df = df_dataset[["Dep_Rater1", "Dep_Rater2", "Dep_Rater3"]]
    for ratings in [df, df.values]:
        alpha, result = audpsychometric.cronbachs_alpha(ratings)
        assert isinstance(result, dict)
        assert np.isclose(alpha, 0.8516, atol=1e-4)
        alpha_t, _ = audpsychometric.cronbachs_alpha(ratings.T, axis=0)
        assert np.isclose(alpha_t, alpha, atol=1e-10)


def test_congeneric_reliability(df_holzinger_swineford):
    """Test congeneric reliability."""
    for ratings in [df_holzinger_swineford, df_holzinger_swineford.values]:
        coefficient, result = audpsychometric.congeneric_reliability(ratings)
    assert np.isclose(coefficient, 0.9365, atol=1e-4)
    assert np.isclose(result["var. explained"][0], 0.3713, atol=1e-4)
    for ratings in [df_holzinger_swineford, df_holzinger_swineford.values]:
        coefficient, result = audpsychometric.congeneric_reliability(ratings)
        assert np.isclose(coefficient, 0.9365, atol=1e-4)
        assert np.isclose(result["var. explained"][0], 0.3713, atol=1e-4)
        coefficient_t, _ = audpsychometric.congeneric_reliability(ratings.T, axis=0)
        assert np.isclose(coefficient_t, coefficient, atol=1e-10)


@pytest.mark.xfail(raises=ValueError)
def test_anova_helper():
    """Test that unknown anova parametrization raises exception."""
    audpsychometric.intra_class_correlation(pd.DataFrame(), anova_method="bbbb")


def test_icc_nanremoval():
    """Cover nan removal if statement."""
    df_dataset = audpsychometric.read_dataset("HolzingerSwineford1939")
    df_dataset = df_dataset[[x for x in df_dataset.columns if x.startswith("x")]]
    nan_mat = np.random.random(df_dataset.shape) < 0.1
    audpsychometric.intra_class_correlation(df_dataset.mask(nan_mat))
