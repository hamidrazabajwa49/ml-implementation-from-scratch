import math
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Probability.descriptive_stats import DescriptiveStats
from Probability.continuous import TDistribution, Chi2Distribution, FDistribution


def ttest_1samp(data: list,pop_mean: float,alternative: str = "two-sided",alpha: float = 0.05,) -> dict:
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1 exclusive")

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
        raise ValueError("Sample standard deviation is negligible relative to the data magnitude.\nThe data may be effectively constant due to floating-point precision limits.")

    t_stat = (sample_mean - pop_mean) / (sample_std / math.sqrt(n))
    df = n - 1
    p_value = TDistribution(df).p_value(t_stat, alternative)
    reject = p_value < alpha

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "df": df,
        "alpha": alpha,
        "reject_H0": reject,
        "alternative": alternative,
        "conclusion": "Reject H0" if reject else "Fail to reject H0",
    }


def ttest_ind(data1: list,data2: list,alternative: str = "two-sided",alpha: float = 0.05,) -> dict:
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1 exclusive")

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
            "Both samples have zero variance with identical means;\nt-statistic is undefined")

    t_stat = (mean1 - mean2) / math.sqrt(pooled_se)

    num = pooled_se ** 2
    den = (se1 ** 2) / (n1 - 1) + (se2 ** 2) / (n2 - 1)
    df = num / den if den != 0.0 else float("inf")
    if df < 1.0:
        df = 1.0

    p_value = TDistribution(df).p_value(t_stat, alternative)
    reject = p_value < alpha

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "df": df,
        "alpha": alpha,
        "reject_H0": reject,
        "alternative": alternative,
        "conclusion": "Reject H0" if reject else "Fail to reject H0",
    }


def ttest_paired(data1: list,data2: list,alternative: str = "two-sided",alpha: float = 0.05,) -> dict:
    if len(data1) != len(data2):
        raise ValueError("Paired samples must have the same length")
    if len(data1) == 0:
        raise ValueError("Paired samples must not be empty")
    diff = [x - y for x, y in zip(data1, data2)]
    return ttest_1samp(diff, 0.0, alternative, alpha)


def confidence_interval(data: list, confidence: float = 0.95) -> dict:
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


def chisquare_gof(observed: list,expected: list = None,alpha: float = 0.05,) -> dict:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1 exclusive")
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
        import warnings
        warnings.warn("k=1 provides zero degrees of freedom. The test cannot detect any "
            "departure from expectation. p_value is always 1.0.",
            UserWarning,
            stacklevel=2,)
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


def chisquare_independence(observed: list, alpha: float = 0.05) -> dict:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1 exclusive")
    if not observed:
        raise ValueError("Observed table must be non-empty")

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

    chi2 = sum(((observed[i][j] - expected_table[i][j]) ** 2) / expected_table[i][j] for i in range(n_rows) for j in range(n_cols))

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


def anova_oneway(*groups: list, alpha: float = 0.05) -> dict:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1 exclusive")

    k = len(groups)
    if k < 2:
        raise ValueError("Need at least two groups for ANOVA")

    for i, g in enumerate(groups):
        if len(g) == 0:
            raise ValueError(f"Group {i} is empty")
        if len(g) < 2:
            raise ValueError(f"Group {i} has only 1 observation.\nEach group must have at least 2 observations")

    group_n = [len(g) for g in groups]
    group_mean = [sum(g) / ni for g, ni in zip(groups, group_n)]
    N = sum(group_n)
    grand_mean = sum(sum(g) for g in groups) / N

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
