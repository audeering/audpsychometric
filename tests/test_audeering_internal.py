"""Tests that can only be run audeering-internally.

These are marked by a pytest mark and can be selected with.

    python -m pytest -k "internal"

"""

import os

import numpy as np
import pandas as pd
import pingouin as pg
import pytest

import audmetric

import audpsychometric


pd.options.display.width = 0
N_ANSWERS_THRESH = 25
pytestmark = pytest.mark.internal


def generate_coreset_df(coreset_name="set133"):
    """Create the coreset df using audb.

    This coreset is prior to any mappint to n classes

    Args:
        coreset_name: The corset nets to be known

    Returns:
        table containing the ratings in wide format

    """
    import audb

    database_name = "projectsmile-salamander-agent-tone"
    db = audb.load(database_name, only_metadata=True)

    df = pd.concat(
        [
            db["agent-tone.train"].get().astype("float"),
            db["agent-tone.dev"].get().astype("float"),
            db["agent-tone.test"].get().astype("float"),
            db["agent-tone.test2"].get().astype("float"),
        ]
    )

    df_sets = db["sets"].get()
    df_core_set = df_sets[df_sets.set == coreset_name]

    print(f"df shapes {df.shape}")
    print(f"Core set shape {df_core_set.shape}")

    def f(x):
        return x in list(df_core_set.index)  # noqa: E731

    select = pd.Series(df.index.get_level_values("file").values).apply(f)
    df = df[(select.values)]
    print(df.shape)

    n_raters = len(df.columns)
    nobs = n_raters - df.isnull().sum(axis=1)
    print(f"Determine total number of raters: {n_raters}")
    print("Distribution of number of ratings per Item:")
    print(nobs.value_counts())
    df_reliability = df[nobs > N_ANSWERS_THRESH]

    return df_reliability


@pytest.fixture(scope="module", autouse=True)
def coreset_df() -> pd.DataFrame:
    """Coreset dataframe for the projectsmile salamander agent tone.

    Args:
        None
    Returns:
        pd.DataFrame

    Initially, two coresets were considered candidates, set142
    and set133. Set 133 proved the right one.

    Unlike the Salamander_Feedback_Agent.ipynb notebook,
    these parts of the data are not considered because these
    overlap with the ones that are only called dev and test:

    - df_unbalanced_test
    - df_unbalanced_dev

    Instead, the data in agent-tone.test2 are added

    The dataset is returned as is and not reshaped to wide.

    """
    dataset_name = "coreset_133"
    dataset_path = os.path.join(
        audpsychometric.datasets.data_directory, f"{dataset_name}.csv"
    )

    if os.path.exists(dataset_path):
        print("reading from disk")
        df = pd.read_csv(dataset_path)
    else:
        print("generating")
        df = generate_coreset_df()
        df.to_csv(dataset_path, index=False)

    return df


def test_audeering_icc(coreset_df):
    """Test the coreset results."""
    # impute Flag
    impute = False
    if impute:
        # f = lambda x: x.fillna(x.mean())
        # f = lambda x: x.fillna(x.mode())
        # coreset_df = coreset_df.apply(f, axis=1)
        coreset_df = coreset_df.apply(lambda x: x.fillna(x.mean()), axis=1)
        # coreset_df = coreset_df.apply(lambda x: x.fillna(x.mode()),axis=1)

    n_nan = coreset_df.isna().sum().sum()
    print(f"currently {n_nan} nan values")
    n_raters_tot = coreset_df.shape[1]

    # convert to LONG format:
    df_long = coreset_df.melt(ignore_index=False)
    df_long["item"] = df_long.index
    df_long.columns = ["rater", "rating", "item"]
    n_raters_tot = coreset_df.shape[1]

    # convert back to wide format:
    data = df_long.pivot_table(index="item", columns="rater", values="rating")

    n_raters_trimmed = data.shape[1]
    print(f"N Raters before/after trimming: {n_raters_tot}/ {n_raters_trimmed}")

    icc_p = pg.intraclass_corr(
        data=df_long,
        targets="item",
        raters="rater",
        ratings="rating",
        nan_policy="omit",
    )

    icc_1 = icc_p.loc[icc_p["Type"] == "ICC1"].iloc[0]

    assert np.isclose(icc_1["ICC"], 0.2180, atol=1e-4)
    assert np.isclose(icc_1["F"], 8.8059, atol=1e-4)

    icc_aa, results = audpsychometric.intra_class_correlation(coreset_df)
    # first value test
    assert np.isclose(icc_aa, 0.2180, atol=1e-4)
    assert len(results) == 3
    # return the right keys
    assert all(
        [
            a == b
            for a, b in zip(
                results.keys(), ["icc_dict", "results_table", "anova_table"]
            )
        ]
    )


def test_audeering_goldstandard_mean(coreset_df):
    r"""Coreset: happy flow for gold standard mean."""
    df = audpsychometric.gold_standard_mean(coreset_df)
    assert isinstance(df, pd.DataFrame)
    assert "gold_standard" in df.columns
    assert "confidence" in df.columns
    assert np.alltrue((df["confidence"] >= 0.0) & (df["confidence"] <= 1.0).values)


def test_audeering_goldstandard_median(coreset_df):
    r"""Coreset: happy flow for gold standard median."""
    df = audpsychometric.gold_standard_median(coreset_df)
    assert isinstance(df, pd.DataFrame)
    assert "gold_standard" in df.columns
    assert "confidence" in df.columns
    assert np.alltrue((df["confidence"] >= 0.0) & (df["confidence"] <= 1.0).values)


def test_audeering_goldstandard_mode(coreset_df):
    r"""Coreset: happy flow for mode."""
    df = audpsychometric.gold_standard_mode(coreset_df)
    assert isinstance(df, pd.DataFrame)
    assert "gold_standard" in df.columns
    assert "confidence" in df.columns
    assert np.alltrue((df["confidence"] >= 0.0) & (df["confidence"] <= 1.0).values)


def test_confidence_values(coreset_df):
    r"""Check that confidences correlate."""
    df_ewe = audpsychometric.evaluator_weighted_estimator(coreset_df)
    df_mode = audpsychometric.gold_standard_mode(coreset_df)

    df_corr = pd.concat([df_mode["confidence"], df_ewe["confidence"]], axis=1)
    corr = audmetric.pearson_cc(df_mode["confidence"], df_ewe["confidence"])
    assert corr > 0.5
    assert df_corr.corr().iloc[0, 1] > 0.5
