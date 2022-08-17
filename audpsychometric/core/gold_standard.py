"""Methods for calculating gold standards over individual raters' judgments"""


import numpy as np
import pandas as pd

import audmetric


def gold_standard_mean(df: pd.DataFrame) -> pd.DataFrame:
    r"""Calculate the gold standard as the mean of raters' votes.

    This functional uses the numerical confidence calculation.

    The returned table
    has an index identical to the input :class:`pd.DataFrame`,
    and additional columns `gold_standard` and `confidence`.

    Args:
        df: DataFrame in wide format, one rater per column

    Returns:
        table containing `gold_standard` and `confidence` columns

    """

    gold_standard = df.mean(axis=1)
    df["gold"] = gold_standard

    confidences = df.apply(_confidence_numerical, axis=1)
    df_result = pd.DataFrame([gold_standard, confidences], columns=df.index).T
    df_result.columns = ["gold_standard", "confidence"]
    return df_result


def gold_standard_median(df):
    r"""Calculate the gold standard as the median of raters' votes.

    The returned table
    has an index identical to the input :class:`pd.DataFrame`,
    and additional columns `gold_standard` and `confidence`.

    Args:
        df: DataFrame in wide format, one rater per column

    Returns:
        table containing `gold_standard` and `confidence` columns

    """

    gold_standard = df.median(axis=1)
    df["gold"] = gold_standard
    confidences = df.apply(_confidence_numerical, axis=1)

    df_result = pd.DataFrame([gold_standard, confidences], columns=df.index).T
    df_result.columns = ["gold_standard", "confidence"]
    return df_result


def gold_standard_mode(df: pd.DataFrame) -> pd.DataFrame:
    r"""Calculate the gold standard as the median of raters' votes.

    The returned table has an index identical to the input df,
    and additional columns `gold_standard` and `confidence`.

    Args:
        df: DataFrame in wide format, one rater per column

    Returns:
        table containing `gold_standard` and `confidence` columns

    """

    gold_standard = np.floor(df.mode(axis=1).mean(axis=1) + 0.5)
    df["gold"] = gold_standard

    # calculate confidence value as the fraction of raters hitting the gold standard
    confidences = df.apply(_confidence_categorical, axis=1)
    df_result = pd.DataFrame([gold_standard, confidences], columns=df.index).T
    df_result.columns = ["gold_standard", "confidence"]
    return df_result


def evaluator_weighted_estimator(df):
    r"""Calculate EWE (evaluator weighted estimator).

    This measure of gold standard calculation is described in
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
        df: DataFrame in wide format, one rater per column

    Returns:
        table containing `gold_standard` and `confidence` columns

    """
    confidences = rater_confidence_pearson(df)
    raters = df.columns.tolist()

    def ewe(row):
        """functional to determine ewe per row"""
        total = sum([row[x] * confidences[x] for x in raters])
        total /= np.sum([confidences[x] for x in raters])
        return total

    df_result = pd.DataFrame(
        data=df.apply(ewe, axis=1), index=df.index, columns=["EWE"]
    )

    df_result["confidence"] = df.apply(_confidence_numerical, axis=1)

    return df_result


def _confidence_categorical(row: pd.Series) -> float:
    r"""Functional to calculate confidence score row-wise - categorical.

    The confidence for categorical data the fraction of raters per item
    with the rating being equal to that of the gold standard

    Args:
        row: one row of the table containing raters' values

    Returns:
        categorical confidence score

    """

    columns = row.index.tolist()

    if "gold" not in columns:
        raise ValueError("Gold column not in DataFrame!")

    raters = [column for column in columns if column != "gold"]
    return np.sum(row[raters] == row["gold"]) / row[raters].count()


def _confidence_numerical(row: pd.Series) -> float:
    """Functional to calculate confidence score row-wise - numerical.

    .. math::
       confidence_{row} = max(0, 1 - std(row) / cutoff_max)

    where
        - std is the standard deviation of the data

    Args:
        row: one row of the table containing raters' values

    Returns:
        numerical confidence score

    """
    raters = row.index.tolist()
    cutoff_max = row[raters].max() / row[raters].mean()
    return max([0., 1 - row[raters].std(ddof=0) / cutoff_max])


def rater_confidence_pearson(df: pd.DataFrame) -> dict:
    """Calculate the rater confidence.

    Calculate the rater confidence
    of a rater as the correlation of a rater
    with the mean score of all other raters.

    This should not be confused with the value
    that relates to a rated stimulus.

    The dictionary returned contains the rater names
    (as in df.columns)
    as keys
    and the corresponding resulting correlation as value.

    Args:
        df: table in wide format

    Returns:
        dict with the rater confidences

    """

    raters = df.columns.tolist()

    confidences = {}
    for rater in raters:
        df_rater = df[rater].dropna().astype(float)
        df_others = df.drop(rater, axis=1).mean(axis=1).dropna()
        indices = df_rater.index.intersection(df_others.index)
        confidences[rater] = audmetric.pearson_cc(df_rater.loc[indices], df_others.loc[indices])
    return confidences
