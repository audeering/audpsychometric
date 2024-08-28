import typing

import numpy as np
import pandas as pd
import pingouin as pg
import sklearn.decomposition
import statsmodels.api as sm
import statsmodels.formula.api


def cronbachs_alpha(
    ratings: typing.Sequence,
    *,
    axis: int = 1,
) -> typing.Tuple[float, typing.Dict]:
    r"""Calculate Cronbach's alpha.

    The Cronbach coefficient quantifying interrater agreement.
    Returns alpha as a float
    and additional information specific to this measure
    collated into a dictionary.

    Cronbach's alpha generalizes Cohen's kappa
    and can handle three or more answers per variable.
    It is suitable for Likert type scale answers.
    A `blogpost on congeneric reliability`_ states
    that Cronbach’s alpha assumes essential tau-equivalence
    and underestimates reliability.
    A tau-equivalent measurement model
    is a special case of a congeneric measurement model
    with all loadings equal :cite:p:`cronbach-alpha-wikipedia`.

    A simplified formula is given in :cite:t:`Hilsdorf`
    that relates the measure to the average reliability:

    .. math::
       \alpha_{st} = \frac{N \times \bar{r}} {1 + (N - 1) \times \bar{r}}

    where

       - :math:`N` is the number of items (labelled chunks)
       - :math:`\bar{r}` is the average correlation between the items

    .. _blogpost on congeneric reliability: http://evaluationdashboard.com/index.php/2012/09/22/congeneric_reliability_r/

    Args:
        ratings: ratings.
            When given as a 1-dimensional array,
            it is treated as a row vector
        axis: axis along which the rater confidence is computed.
            A value of ``1``
            assumes stimuli as rows

    Returns:
         Cronbach's alpha and additional results lumped into dict

    """  # noqa: E501
    ratings = np.atleast_2d(np.array(ratings))
    n_items = ratings.shape[axis]  # K
    total_score = np.sum(ratings, axis=axis)  # X = ∑ Y_i
    variance_sum = np.var(ratings, axis=1 - axis, ddof=1).sum()  # ∑ var(Y_i)
    total_variance = total_score.var(ddof=1)  # var(X)
    alpha = n_items / (n_items - 1) * (1 - variance_sum / total_variance)

    result = {"total_variance": total_variance}
    return alpha, result


def congeneric_reliability(
    ratings: typing.Sequence,
    *,
    axis: int = 1,
) -> typing.Tuple[float, typing.Dict]:
    r"""Congeneric reliability coefficient.

    Extracts the first Principal Component as a measurement model
    and extracts the congeneric reliability coefficient.
    The implementation here is specific
    for a Principal Component Analysis with one factor.
    For implementation details see the wikipedia
    :cite:p:`congeneric-reliability-wikipedia`.

    Args:
        ratings: ratings.
            When given as a 1-dimensional array,
            it is treated as a row vector
        axis: axis along which the rater confidence is computed.
            A value of ``1``
            assumes stimuli as rows

    Returns:
        Congeneric Reliability and additional results lumped into dict

    """
    ratings = np.atleast_2d(np.array(ratings))
    if axis == 0:
        ratings = ratings.T
    pca = sklearn.decomposition.PCA(n_components=1)
    pca.fit(ratings)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)  # λ_i
    loadings_sum = loadings.sum() ** 2  # (∑ λ_i)²
    total_score = np.sum(ratings, axis=1)  # X = ∑ Y_i
    total_variance = total_score.var(ddof=1)  # var(X)

    result = {}
    result["var. explained"] = pca.explained_variance_ratio_
    result["sing. values"] = pca.singular_values_

    reliability = loadings_sum / total_variance
    return reliability, result


def intra_class_correlation(
    ratings: typing.Sequence,
    *,
    axis: int = 1,
    icc_type: str = "ICC_1_1",
    anova_method: str = "pingouin",
) -> typing.Tuple[float, typing.Dict]:
    r"""Intraclass Correlation.

    Intraclass correlation calculates rating reliability by relating
    (i) variability of different ratings of the same subject to
    (ii) the total variation across all ratings and all items.

    The model is based on analysis of variance,
    and ratings must at least be ordinally scaled.

    CCC_ is conceptually and numerically related to the ICC.
    For an implementation see :func:`audmetric.concordance_cc`.

    Args:
        ratings: ratings.
            When given as a 1-dimensional array,
            it is treated as a row vector
        axis: axis along which the rater confidence is computed.
            A value of ``1``
            assumes stimuli as rows
        icc_type: ICC Method, see description below
        anova_method: method for ANOVA calculation,
            can be ``"pingouin"`` or ``"statsmodels"``

    .. _CCC: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient

    Returns:
        icc and additional results lumped into dict

    Notes:
        :cite:ts:`Shrout/Fleiss:1979`
        describe a whole family of forms of coefficients
        that must be selected according
        to following criteria :cite:t:`Koo/Li:2016`:

        **Parametrization**

        :cite:t:`Koo/Li:2016` detail the parametrization:

        *Rater type: 1 versus k raters:*

        "This selection depends
        on how the measurement protocol
        will be conducted in actual application.
        For instance,
        if we plan to use the mean value of 3 raters as an assessment basis,
        the experimental design of the reliability study
        should involve 3 raters,
        and the 'mean of k raters' type should be selected.

        *Rating definition: absolute or consistency*

        "For both 2-way random- and 2-way mixed-effects models,
        there are 2 ICC definitions:
        “absolute agreement” and “consistency.”
        Selection of the ICC definition
        depends on whether we consider absolute agreement
        or consistency between raters to be more important.
        Absolute agreement concerns
        if different raters assign the same score
        to the same subject.
        Conversely,
        consistency definition concerns
        if raters’ scores to the same group of subjects
        are correlated in an additive manner.
        Consider an interrater reliability study of 2 raters as an example.
        In this case,
        consistency definition concerns the degree
        to which one rater’s score (y)
        can be equated to another rater’s score (x)
        plus a systematic error (c)
        (ie, :math:`y = x + c`),
        whereas absolute agreement
        concerns about the extent to which y equals x.

        *Model Choice*

        - One-Way Random-Effects Model
            In this model,
            each subject is rated by a different set of raters
            who were randomly chosen
            from a larger population of possible raters.
            This is used when different sets of raters rate different items
        - Two-Way Random-Effects Model
            If we randomly select our raters
            from a larger population of raters
            with similar characteristics
        - Two-Way Mixed-Effects Model
            We should use the 2-way mixed-effects model
            if the selected raters are the only raters of interest

        .. table::

            +----------------------+------------+--------------------+------------------------+
            | Parametrization      | Rater Type | Rating definition  | Model Choice           |
            +======================+============+====================+========================+
            | ICC(1,1) - icc\_1\_1 | single     | absolute agreement | One-way random effects |
            +----------------------+------------+--------------------+------------------------+
            | ICC(2,1) - icc\_2\_1 | single     | absolute agreement | Two-way random effects |
            +----------------------+------------+--------------------+------------------------+
            | ICC(3,1) - icc\_3\_1 | single     | consistency        | Two-way mixed effects  |
            +----------------------+------------+--------------------+------------------------+
            | ICC(1,k) - icc\_2\_k | k          | absolute agreement | One-way random effects |
            +----------------------+------------+--------------------+------------------------+
            | ICC(2,k) - icc\_2\_k | k          | absolute agreement | Two-way random effects |
            +----------------------+------------+--------------------+------------------------+
            | ICC(3,k) - icc\_3\_k | k          | consistency        | Two-way mixed effects  |
            +----------------------+------------+--------------------+------------------------+

        **Interpretation**

        Interpretation Conventions vary a between several sources,
        and depend also on the use case.
        We list values from the papers of :cite:ts:`Hallgren:2012`
        and :cite:ts:`Koo/Li:2016` respectively:

        .. table::

            +-----------+----------------+
            | Value     | Interpretation |
            +===========+================+
            | < 0.4     | poor           |
            +-----------+----------------+
            | .40 - .59 | fair           |
            +-----------+----------------+
            | .60 - .74 | good           |
            +-----------+----------------+
            | .75 - 1.0 | excellent      |
            +-----------+----------------+

        .. table::

            +-----------+----------------+
            | Value     | Interpretation |
            +===========+================+
            | < 0.5     | poor           |
            +-----------+----------------+
            | .50 - .75 | moderate       |
            +-----------+----------------+
            | .75 - .9  | good           |
            +-----------+----------------+
            | >.9 - 1.0 | excellent      |
            +-----------+----------------+

        **Formulas**:

        Shrout & Fleiss formula (1979, p. 423ff):

        .. math::
          \text{ICC}(1,1) = \frac{(\text{bms} - \text{wms})}{ (\text{bms} + (k - 1) * \text{wms})}

        .. math::
          \text{ICC}(2,1) = \frac{(\text{bms} - \text{ems})}{ (\text{bms} + (k - 1) * \text{ems} + k * (\text{jms} - \text{ems}) / n)}

        .. math::
          \text{ICC}(3,1) = \frac {(\text{bms} - \text{ems})}{ (\text{bms} + (k - 1) * \text{ems})}

        .. math::
          \text{ICC}(1,k) = \frac{(\text{bms} - \text{wms})}{\text{bms}}

        .. math::
          \text{ICC}(2,k) = \frac{(\text{bms} - \text{ems})}{ (\text{bms} + (\text{jms} - \text{ems}) / n)}

        .. math::
          \text{ICC}(3,k) = \frac{(\text{bms} - \text{ems})}{\text{bms}}

        where

        - :math:`\text{bms}` is the between items ("targets")
          mean sqeare of the underlying 2-factor Anova
        - :math:`\text{wms}` is the within items ("target") mean square
        - :math:`\text{jms}` is the between raters ("judges")
          mean square of the underlying 2-factor Anova
        - :math:`\text{ems}` is the error/residual mean square
        - :math:`k` is the number of raters
        - :math:`n` is the number of items

    """  # noqa: E501
    if not isinstance(ratings, pd.DataFrame):
        df = pd.DataFrame(np.atleast_2d(np.array(ratings)))
    else:
        df = ratings

    if axis == 0:
        df = df.T

    def _anova(df_long: pd.DataFrame, anova_method: str = "pingouin") -> pd.DataFrame:
        """Helper to get the anova table.

        Note that pingouin is currently default as statsmodels.
        ols from the statsmodels package is slow under many circumstances
        """
        anova_methods = ["statsmodels", "pingouin"]

        if anova_method not in anova_methods:
            raise ValueError(f"Anova method '{anova_method}' not in {anova_methods}.")

        if anova_method == "pingouin":
            # calculate and remap
            anova_table = pg.anova(
                data_long, dv="rating", between=["item", "rater"], ss_type=2
            )
            anova_table.columns = ["Source", "sum_sq", "df", "mean_sq", "np2"]
            return anova_table
            # aov_pg for wine:
            # Source         SS  DF         MS  np2
            # 0          item  188.21875   7  26.888393  1.0
            # 1         rater    7.34375   3   2.447917  1.0
            # 2  item * rater   47.90625  21   2.281250  1.0
            # 3      Residual    0.00000   0        NaN  NaN
        elif anova_method == "statsmodels":
            formula = "rating ~ C(item) * rater"
            model = statsmodels.formula.api.ols(formula=formula, data=data_long)
            result = model.fit()
            anova_table = sm.stats.anova_lm(result, typ=1)
            anova_table = sm.stats.anova_lm(result, typ=1)
            return anova_table
            # (Pdb) anova_table ols for wine
            #                  df        sum_sq    mean_sq    F  PR(>F)
            # C(item)         7.0  1.882188e+02  26.888393  0.0     NaN
            # rater           3.0  7.343750e+00   2.447917  0.0     NaN
            # C(item):rater  21.0  4.790625e+01   2.281250  0.0     NaN
            # Residual        0.0  9.138719e-27        inf  NaN     NaN
            # data must be (5152, 3)
        elif anova_method not in anova_methods:  # pragma: no cover
            raise ValueError("Undefined Anova method!")

    # data_long = df.reset_index().melt(id_vars=targets, value_name=ratings)
    # data_long = df.melt(ignore_index=False)
    data_long = df.melt(ignore_index=False, var_name="rater", value_name="rating")
    data_long["item"] = data_long.index

    # convert back to wide format to see how many raters dropped:
    data_wide = data_long.pivot_table(index="item", columns="rater", values="rating")

    # delete missing data in listwise manner
    nan_count = data_wide.isna().sum().sum()
    print(f"We have {nan_count} missing data points")
    print("Deleting them!")
    if nan_count > 0:
        data_wide = data_wide.dropna(axis=0, how="any")

    # back to long
    data_long = data_wide.melt(
        ignore_index=False, var_name="rater", value_name="rating"
    )
    data_long["item"] = data_long.index
    # drop missing
    data_long = data_long[data_long.columns].dropna()
    data_long = data_long.reset_index(drop=True)

    anova_table = _anova(data_long, anova_method=anova_method)

    bms = anova_table["mean_sq"].iloc[0]
    wms = (anova_table["sum_sq"].iloc[1] + anova_table["sum_sq"].iloc[2]) / (
        anova_table["df"].iloc[1] + anova_table["df"].iloc[2]
    )
    jms = anova_table["mean_sq"].iloc[1]
    ems = anova_table["mean_sq"].iloc[2]

    k = data_long["rater"].nunique()
    n = data_long["item"].nunique()

    # Calculate ICCs
    # Shrout & Fleiss Formula ICC(1,1), p. 423
    icc_1_1 = (bms - wms) / (bms + (k - 1) * wms)
    # Shrout & Fleiss Formula ICC(2,1), p. 423
    icc_2_1 = (bms - ems) / (bms + (k - 1) * ems + k * (jms - ems) / n)
    # Shrout & Fleiss Formula ICC(3,1), p. 423
    icc_3_1 = (bms - ems) / (bms + (k - 1) * ems)
    # Shrout & Fleiss Formula ICC(1,k), p. 42x
    icc_1_k = (bms - wms) / bms
    # Shrout & Fleiss Formula ICC(2,k), p. 42x
    icc_2_k = (bms - ems) / (bms + (jms - ems) / n)
    # Shrout & Fleiss Formula ICC(3,k), p. 42x
    icc_3_k = (bms - ems) / bms

    # Create output dataframe
    # stats = {
    icc_types = ["ICC_1_1", "ICC_2_1", "ICC_3_1", "ICC_1_k", "ICC_2_k", "ICC_3_k"]
    icc_rater_type = 3 * ["single"] + 3 * ["average"]
    icc_rating_type = ["absolute", "consistency", "consistency"] * 2
    icc_effect_type = ["-", "random", "fixed"] * 2
    icc = [icc_1_1, icc_2_1, icc_3_1, icc_1_k, icc_2_k, icc_3_k]

    results = pd.DataFrame(
        [icc_types, icc_rater_type, icc_rating_type, icc_effect_type, icc]
    ).T

    vars = ["icc_type", "rater type", "rating type", "anova effect type", "icc"]
    results_df = pd.DataFrame(
        [icc_types, icc_rater_type, icc_rating_type, icc_effect_type, icc], index=vars
    ).T

    results_table = results_df.to_dict("records")

    #     "Description": [
    #         "Single raters absolute",
    #         "Single random raters",
    #         "Single fixed raters",
    #         "Average raters absolute",
    #         "Average random raters",
    #         "Average fixed raters",
    #     ],
    #     "ICC": [icc_1_1, icc_2_1, icc_3_1, icc_1_k, icc_2_k, icc_3_k],
    # }

    icc_dict = [x for x in results_table if icc_type == x["icc_type"]]
    icc = icc_dict[0]["icc"]

    results = {
        "icc_dict": icc_dict,
        "results_table": results_table,
        "anova_table": anova_table,
    }
    return icc, results
