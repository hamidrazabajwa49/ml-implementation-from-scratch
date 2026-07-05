"""
hypothesis_tests.py
====================

Classical frequentist hypothesis tests built on top of
:mod:`descriptive_stats` and :mod:`continuous`: one- and two-sample
t-tests (Welch's for unequal variances), paired t-test, z-test,
confidence intervals, chi-squared goodness-of-fit and independence
tests, and one-way ANOVA.

Every test returns a plain ``dict`` with the statistic, p-value, and
supporting details, so results can be inspected, logged, or serialized
without needing a custom result class.

Example
-------
>>> result = ttest_1samp([5.1, 4.9, 5.3, 5.0, 4.8], pop_mean=5.0)
>>> result['conclusion']
'Fail to reject H0'
"""

from __future__ import annotations

import os
import sys
import math
import warnings
from typing import Dict, List, Optional, Sequence, Union



_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
from Probability.descriptive_stats import DescriptiveStats  # type: ignore
from Probability.continuous import (  # type: ignore
    Chi2Distribution,
    FDistribution,
    NormalDistribution,
    TDistribution,
)

Number = Union[int, float]

_ALTERNATIVES = ("two-sided", "greater", "less")


def _validate_alternative(alternative: str) -> None:
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}, got {alternative!r}")


def _validate_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1 exclusive")


def ttest_1samp(
    data: Sequence[Number],
    pop_mean: float,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> Dict:
    """One-sample t-test: is ``mean(data)`` different from ``pop_mean``?

    Parameters
    ----------
    data : Sequence[Number]
        Sample observations; needs at least 2 points.
    pop_mean : float
        Hypothesized population mean under H0.
    alternative : {'two-sided', 'greater', 'less'}, optional
    alpha : float, optional
        Significance level, in ``(0, 1)``.

    Returns
    -------
    dict
        Keys: ``t_statistic``, ``p_value``, ``df``, ``alpha``,
        ``reject_H0``, ``alternative``, ``cohens_d``, ``conclusion``.

    Raises
    ------
    ValueError
        If ``alternative``/``alpha`` are invalid, fewer than 2
        observations are given, or the sample has (effectively) zero
        variance.
    """
    _validate_alternative(alternative)
    _validate_alpha(alpha)

    stats = DescriptiveStats(data)
    n = stats.n
    if n < 2:
        raise ValueError("ttest_1samp requires at least 2 observations")

    sample_mean = stats.mean()
    sample_std = stats.std(ddof=1)

    if sample_std == 0.0:
        raise ValueError("Sample has zero variance; t-statistic is undefined")

    data_scale = max(abs(x) for x in data)
    if data_scale > 0.0 and sample_std / data_scale < 1e-10:
        raise ValueError(
            "Sample standard deviation is negligible relative to the data magnitude.\n"
            "The data may be effectively constant due to floating-point precision limits."
        )

    t_stat = (sample_mean - pop_mean) / (sample_std / math.sqrt(n))
    df = n - 1
    p_value = TDistribution(df).p_value(t_stat, alternative)
    reject = p_value < alpha
    cohens_d = (sample_mean - pop_mean) / sample_std

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "df": df,
        "alpha": alpha,
        "reject_H0": reject,
        "alternative": alternative,
        "cohens_d": cohens_d,
        "conclusion": "Reject H0" if reject else "Fail to reject H0",
    }


def ttest_ind(
    data1: Sequence[Number],
    data2: Sequence[Number],
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> Dict:
    """Welch's two-sample t-test (does not assume equal variances).

    Parameters
    ----------
    data1, data2 : Sequence[Number]
        Independent samples; each needs at least 2 points.
    alternative : {'two-sided', 'greater', 'less'}, optional
    alpha : float, optional

    Returns
    -------
    dict
        Keys: ``t_statistic``, ``p_value``, ``df``, ``alpha``,
        ``reject_H0``, ``alternative``, ``cohens_d``, ``conclusion``.
        ``df`` is the (fractional) Welch-Satterthwaite degrees of freedom.

    Raises
    ------
    ValueError
        If ``alternative``/``alpha`` are invalid, either sample has
        fewer than 2 observations, or both samples have zero variance
        with identical means.
    """
    _validate_alternative(alternative)
    _validate_alpha(alpha)

    stats1 = DescriptiveStats(data1)
    stats2 = DescriptiveStats(data2)
    n1, n2 = stats1.n, stats2.n

    if n1 < 2:
        raise ValueError("data1 requires at least 2 observations")
    if n2 < 2:
        raise ValueError("data2 requires at least 2 observations")

    mean1, mean2 = stats1.mean(), stats2.mean()
    std1, std2 = stats1.std(ddof=1), stats2.std(ddof=1)

    se1 = std1 ** 2 / n1
    se2 = std2 ** 2 / n2
    pooled_se = se1 + se2

    if pooled_se == 0.0:
        raise ValueError(
            "Both samples have zero variance with identical means;\nt-statistic is undefined"
        )

    t_stat = (mean1 - mean2) / math.sqrt(pooled_se)

    num = pooled_se ** 2
    den = (se1 ** 2) / (n1 - 1) + (se2 ** 2) / (n2 - 1)
    df = num / den if den != 0.0 else float("inf")
    if df < 1.0:
        df = 1.0

    p_value = TDistribution(df).p_value(t_stat, alternative)
    reject = p_value < alpha

    # Pooled standard deviation for Cohen's d (assumes equal population
    # variances, unlike the Welch test statistic itself -- this is the
    # conventional effect-size convention even alongside a Welch test).
    pooled_std = math.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0.0 else float("nan")

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "df": df,
        "alpha": alpha,
        "reject_H0": reject,
        "alternative": alternative,
        "cohens_d": cohens_d,
        "conclusion": "Reject H0" if reject else "Fail to reject H0",
    }


def ttest_paired(
    data1: Sequence[Number],
    data2: Sequence[Number],
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> Dict:
    """Paired-sample t-test: one-sample t-test on the within-pair differences.

    Raises
    ------
    ValueError
        If the samples differ in length or are empty (plus everything
        :func:`ttest_1samp` can raise on the differences).
    """
    if len(data1) != len(data2):
        raise ValueError("Paired samples must have the same length")
    if len(data1) == 0:
        raise ValueError("Paired samples must not be empty")
    diff = [x - y for x, y in zip(data1, data2)]
    return ttest_1samp(diff, 0.0, alternative, alpha)


def ztest_1samp(
    data: Sequence[Number],
    pop_mean: float,
    pop_std: float,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> Dict:
    """One-sample z-test, for when the population standard deviation is known.

    Parameters
    ----------
    data : Sequence[Number]
        Sample observations; needs at least 1 point.
    pop_mean : float
        Hypothesized population mean under H0.
    pop_std : float
        *Known* population standard deviation; must be positive.
    alternative : {'two-sided', 'greater', 'less'}, optional
    alpha : float, optional

    Returns
    -------
    dict
        Keys: ``z_statistic``, ``p_value``, ``alpha``, ``reject_H0``,
        ``alternative``, ``conclusion``.

    Raises
    ------
    ValueError
        If ``alternative``/``alpha``/``pop_std`` are invalid, or ``data`` is empty.
    """
    _validate_alternative(alternative)
    _validate_alpha(alpha)
    if pop_std <= 0.0:
        raise ValueError("pop_std must be positive")

    stats = DescriptiveStats(data)
    n = stats.n
    sample_mean = stats.mean()

    z_stat = (sample_mean - pop_mean) / (pop_std / math.sqrt(n))
    std_normal = NormalDistribution(0.0, 1.0)

    if alternative == "two-sided":
        p_value = 2.0 * min(std_normal.cdf(z_stat), std_normal.sf(z_stat))
    elif alternative == "greater":
        p_value = std_normal.sf(z_stat)
    else:  # "less"
        p_value = std_normal.cdf(z_stat)

    reject = p_value < alpha

    return {
        "z_statistic": z_stat,
        "p_value": p_value,
        "alpha": alpha,
        "reject_H0": reject,
        "alternative": alternative,
        "conclusion": "Reject H0" if reject else "Fail to reject H0",
    }


def confidence_interval(data: Sequence[Number], confidence: float = 0.95) -> Dict:
    """A t-based confidence interval for the population mean.

    Raises
    ------
    ValueError
        If ``confidence`` is not in ``(0, 1)`` or fewer than 2 observations are given.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be strictly between 0 and 1")

    stats = DescriptiveStats(data)
    n = stats.n
    if n < 2:
        raise ValueError("Need at least 2 data points for a confidence interval")

    mean = stats.mean()
    std = stats.std(ddof=1)

    alpha = 1.0 - confidence
    t_critical = TDistribution(n - 1).ppf(1.0 - alpha / 2.0)
    margin = t_critical * (std / math.sqrt(n))

    return {
        "mean": mean,
        "std": std,
        "n": n,
        "confidence": confidence,
        "t_critical": t_critical,
        "margin": margin,
        "lower": mean - margin,
        "upper": mean + margin,
    }


def chisquare_gof(
    observed: Sequence[Number],
    expected: Optional[Sequence[Number]] = None,
    alpha: float = 0.05,
) -> Dict:
    """Chi-squared goodness-of-fit test.

    Parameters
    ----------
    observed : Sequence[Number]
        Observed counts; must be non-negative.
    expected : Sequence[Number], optional
        Expected counts; defaults to a uniform distribution over the
        total observed count. Must be positive if given.
    alpha : float, optional

    Raises
    ------
    ValueError
        If ``alpha`` is invalid, ``observed`` is empty, any observed
        count is negative, ``expected`` has a different length, any
        expected count is non-positive, or the observed total is 0
        (when ``expected`` is auto-derived).
    """
    _validate_alpha(alpha)
    if len(observed) == 0:
        raise ValueError("observed must be non-empty")

    for i, o in enumerate(observed):
        if o < 0:
            raise ValueError(f"observed counts must be non-negative. Found {o} at index {i}")

    k = len(observed)

    if expected is None:
        total = sum(observed)
        if total == 0:
            raise ValueError("Sum of observed counts must be > 0")
        expected = [total / k] * k
    else:
        if len(observed) != len(expected):
            raise ValueError("observed and expected must have the same length")
        for i, e in enumerate(expected):
            if e <= 0:
                raise ValueError(f"expected counts must be positive. Found {e} at index {i}")

    chi2 = sum(((o - e) ** 2) / e for o, e in zip(observed, expected))
    df = k - 1

    if df == 0:
        warnings.warn(
            "k=1 provides zero degrees of freedom. The test cannot detect any "
            "departure from expectation. p_value is always 1.0.",
            UserWarning,
            stacklevel=2,
        )
        p_value = 1.0
    else:
        p_value = Chi2Distribution(df).sf(chi2)
    reject = p_value < alpha

    return {
        "chi2_statistic": chi2,
        "p_value": p_value,
        "df": df,
        "alpha": alpha,
        "reject_H0": reject,
        "conclusion": "Reject H0" if reject else "Fail to reject H0",
    }


def chisquare_independence(observed: Sequence[Sequence[Number]], alpha: float = 0.05) -> Dict:
    """Chi-squared test of independence on a contingency table.

    Raises
    ------
    ValueError
        If ``alpha`` is invalid, the table is empty/ragged, any count
        is negative, the grand total is 0, or a row/column totals to 0
        (making an expected count undefined).
    TypeError
        If ``observed`` rows are not themselves sequences.
    """
    _validate_alpha(alpha)
    if not observed:
        raise ValueError("Observed table must be non-empty")

    for i, row in enumerate(observed):
        if not hasattr(row, "__len__"):
            raise TypeError(f"Row {i} must be a sequence (e.g. list) of counts, got {type(row).__name__}")

    n_rows = len(observed)
    n_cols = len(observed[0])

    if n_cols == 0:
        raise ValueError("Rows must not be empty")

    for i, row in enumerate(observed):
        if len(row) != n_cols:
            raise ValueError(f"All rows must have the same length.\nRow {i} has {len(row)} columns, expected {n_cols}")
        for j, val in enumerate(row):
            if val < 0:
                raise ValueError(f"Observed counts must be non-negative.\nFound {val} at ({i}, {j})")

    row_totals = [sum(row) for row in observed]
    col_totals = [sum(observed[i][j] for i in range(n_rows)) for j in range(n_cols)]
    grand_total = sum(row_totals)

    if grand_total == 0:
        raise ValueError("Grand total must be greater than 0")

    expected_table = []
    for i in range(n_rows):
        expected_row = []
        for j in range(n_cols):
            e = (row_totals[i] * col_totals[j]) / grand_total
            if e == 0:
                raise ValueError(f"Expected count at ({i}, {j}) is zero.\nThe table has a row or column that sums to zero")
            expected_row.append(e)
        expected_table.append(expected_row)

    chi2 = sum(
        ((observed[i][j] - expected_table[i][j]) ** 2) / expected_table[i][j]
        for i in range(n_rows)
        for j in range(n_cols)
    )

    df = (n_rows - 1) * (n_cols - 1)
    p_value = Chi2Distribution(df).sf(chi2) if df > 0 else 1.0
    reject = p_value < alpha

    return {
        "chi2_statistic": chi2,
        "p_value": p_value,
        "df": df,
        "alpha": alpha,
        "expected_table": expected_table,
        "reject_H0": reject,
        "conclusion": "Reject H0" if reject else "Fail to reject H0",
    }


def anova_oneway(*groups: Sequence[Number], alpha: float = 0.05) -> Dict:
    """One-way ANOVA F-test across two or more independent groups.

    Parameters
    ----------
    *groups : Sequence[Number]
        Two or more groups, each with at least 2 observations.
    alpha : float, optional

    Returns
    -------
    dict
        Keys: ``F_statistic``, ``p_value``, ``df_between``, ``df_within``,
        ``SSB``, ``SSW``, ``MSB``, ``MSW``, ``group_means``, ``grand_mean``,
        ``alpha``, ``reject_H0``, ``conclusion``.

    Raises
    ------
    ValueError
        If ``alpha`` is invalid, fewer than 2 groups are given, or any
        group has fewer than 2 observations.
    """
    _validate_alpha(alpha)

    k = len(groups)
    if k < 2:
        raise ValueError("Need at least two groups for ANOVA")

    # Reuse DescriptiveStats for validation (numeric, non-empty, no NaN)
    # and mean computation, instead of re-implementing those checks here.
    group_stats = []
    for i, g in enumerate(groups):
        if len(g) == 0:
            raise ValueError(f"Group {i} is empty")
        if len(g) < 2:
            raise ValueError(f"Group {i} has only 1 observation.\nEach group must have at least 2 observations")
        group_stats.append(DescriptiveStats(g))

    group_n = [s.n for s in group_stats]
    group_mean = [s.mean() for s in group_stats]
    N = sum(group_n)
    grand_mean = sum(s.mean() * s.n for s in group_stats) / N

    ssb = sum(ni * (m - grand_mean) ** 2 for ni, m in zip(group_n, group_mean))
    ssw = sum((x - m) ** 2 for g, m in zip(groups, group_mean) for x in g)

    dfb = k - 1
    dfw = N - k
    msb = ssb / dfb
    msw = ssw / dfw

    if msw == 0.0:
        if ssb == 0.0:
            F = float("nan")
            p_value = float("nan")
        else:
            F = float("inf")
            p_value = 0.0
    else:
        F = msb / msw
        p_value = FDistribution(dfb, dfw).sf(F)

    reject = p_value < alpha if not math.isnan(p_value) else False

    return {
        "F_statistic": F,
        "p_value": p_value,
        "df_between": dfb,
        "df_within": dfw,
        "SSB": ssb,
        "SSW": ssw,
        "MSB": msb,
        "MSW": msw,
        "group_means": group_mean,
        "grand_mean": grand_mean,
        "alpha": alpha,
        "reject_H0": reject,
        "conclusion": "Reject H0" if reject else "Fail to reject H0",
    }
