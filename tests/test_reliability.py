import numpy as np
import pandas as pd
import pytest

import audpsychometric


def test_icc():
    """Test icc basic result validity"""
    df_dataset = audpsychometric.datasets.read_dataset("wine")

    data_wide = df_dataset.pivot_table(index="Wine", columns="Judge", values="Scores")

    icc_sm, _ = audpsychometric.intra_class_correlation(
        data_wide, anova_method="statsmodels"
    )
    icc_pingouin, _ = audpsychometric.intra_class_correlation(data_wide)
    assert np.isclose(icc_pingouin, 0.727, atol=1e-3)
    assert np.isclose(icc_sm, icc_pingouin, atol=1e-10)


def test_cronbachs_alpha():
    """Test cronbach's alpha return values for three raters."""
    df_dataset = audpsychometric.datasets.read_dataset("hallgren-table3")
    df = df_dataset[["Dep_Rater1", "Dep_Rater2", "Dep_Rater3"]]
    alpha, result = audpsychometric.cronbachs_alpha(df)
    assert isinstance(result, dict)
    assert np.isclose(alpha, 0.8516, atol=1e-4)


def test_congeneric_reliability(df_holzinger_swineford):
    """Test congeneric reliability"""
    coefficient, result = audpsychometric.congeneric_reliability(df_holzinger_swineford)
    assert np.isclose(coefficient, 0.9365, atol=1e-4)
    assert np.isclose(result["var. explained"][0], 0.3713, atol=1e-4)


@pytest.mark.xfail(raises=ValueError)
def test_anova_helper():
    """Test that unknown anova parametrization raises exception"""
    audpsychometric.intra_class_correlation(pd.DataFrame(), anova_method="bbbb")


def test_icc_nanremoval():
    """Cover nan removal if statement"""
    df_dataset = audpsychometric.datasets.read_dataset("HolzingerSwineford1939")
    df_dataset = df_dataset[[x for x in df_dataset.columns if x.startswith("x")]]
    nan_mat = np.random.random(df_dataset.shape) < 0.1
    audpsychometric.intra_class_correlation(df_dataset.mask(nan_mat))
