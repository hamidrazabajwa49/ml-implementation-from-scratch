import math
from descriptive_stats import DescriptiveStats
from continuous import NormalDistribution

def ttest_1samp(data:list,pop_mean:float,alternative: str="two-sided")->dict:
    stats=DescriptiveStats(data)
    n=stats.n
    sample_mean=stats.mean()
    sample_std=stats.std(ddof=1)

    t_stat=(sample_mean -pop_mean)/(sample_std / math.sqrt(n))
    df=n-1
    normal=NormalDistribution(mu=0.0,sigma=1.0) # using Normal Distribution
    if alternative =="two-sided":
        p_value=2.0*min(normal.cdf(t_stat),1.0-normal.cdf(t_stat))
    elif alternative=="greater":
        p_value=normal.cdf(t_stat)
    elif alternative=="less":
        p_value=normal.cdf(t_stat)
    else:
        raise ValueError("alternative must be 'two-sided'.'greater' or 'less'")

    alpha=0.05
    reject=p_value<alpha

    return{
        "t_statistic":t_stat,
        "p_value":p_value,
        "df":df,
        "alpha":alpha,
        "reject_H0":reject,
        "alternative":alternative,
        "conclusion":"Reject H0" if reject else "Fail to reject H0"
    }


def ttest_ind(data1:list,data2:list,alternative: str="two-sided")->dict:
    
    stats1=DescriptiveStats(data1)
    stats2=DescriptiveStats(data2)
    n1=stats1.n
    n2=stats2.n
    sample_mean1=stats1.mean()
    sample_mean2=stats2.mean()
    sample_std1=stats1.std(ddof=1)
    sample_std2=stats2.std(ddof=1)

    denom=math.sqrt((sample_std1**2/n1)+(sample_std2**2/n2))
    t_stat=(sample_mean1-sample_mean2)/denom

    df=min(n1,n2)-1

    normal=NormalDistribution(mu=0.0,sigma=1.0)

    if (alternative=="two-sided"):
        p_value=2.0 *min(normal.cdf(t_stat),1.0-normal.cdf(t_stat))
    elif (alternative=="greater"):
        p_value=1.0-normal.cdf(t_stat)
    elif (alternative == "less"):
        p_value=normal.cdf(t_stat)
    else:
        raise ValueError("alternative must be 'two-sided','greater', or 'less'")

    alpha = 0.05
    reject = p_value < alpha

    return{
        "t_statistic":t_stat,
        "p_value":p_value,
        "df":df,
        "alpha":alpha,
        "reject_H0":reject,
        "alternative":alternative,
        "conclusion":"Reject H0" if reject else "Fail to reject H0"
    }


def chisquare_gof(observed: list, expected: list = None) -> dict:
    if len(observed) == 0:
        raise ValueError("observed must be non‑empty.")

    if expected is None:
        total = sum(observed)
        k = len(observed)
        if total == 0:
            raise ValueError("Sum of observed counts must be > 0.")
        expected = [total / k] * k
    else:
        if len(observed) != len(expected):
            raise ValueError("observed and expected must have the same length.")
        for i, e in enumerate(expected):
            if e <= 0:
                raise ValueError(f"expected counts must be positive. Found {e} at index {i}.")

    chi2 = 0.0
    k = len(observed)
    for i in range(k):
        o = observed[i]
        e = expected[i]
        chi2 += ((o - e) ** 2) / e

    df = k - 1

    if df == 0:
        p_value = 1.0
    else:
        if chi2 < 0:
            chi2 = 0.0 
        z = math.sqrt(2.0 * chi2) - math.sqrt(2.0 * df - 1.0)
        normal = NormalDistribution(mu=0.0, sigma=1.0)
        # chi‑square test is always right‑tailed
        p_value = 1.0 - normal.cdf(z)

    alpha = 0.05
    reject = p_value < alpha

    return {
        "chi2_statistic": chi2,
        "p_value": p_value,
        "df": df,
        "alpha": alpha,
        "reject_H0": reject,
        "conclusion": "Reject H0" if reject else "Fail to reject H0"
    }

def chisquare_independence(observed: list) -> dict:

    if not observed or len(observed) == 0:
        raise ValueError("Observed table must be non‑empty.")
    n_rows = len(observed)
    n_cols = len(observed[0])
    for i, row in enumerate(observed):
        if len(row) != n_cols:
            raise ValueError(f"All rows must have same length. Row {i} has {len(row)} columns.")

    row_totals = [sum(row) for row in observed]
    col_totals = [0.0] * n_cols
    for j in range(n_cols):
        col_totals[j] = sum(observed[i][j] for i in range(n_rows))
    grand_total = sum(row_totals)
    if grand_total == 0:
        raise ValueError("Grand total must be grater than 0.")

    expected_table = []
    for i in range(n_rows):
        expected_row = []
        for j in range(n_cols):
            e = (row_totals[i] * col_totals[j]) / grand_total
            if e == 0:
                raise ValueError(f"Expected count at ({i},{j}) is zero. Check your data.")
            expected_row.append(e)
        expected_table.append(expected_row)

    chi2 = 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            o = observed[i][j]
            e = expected_table[i][j]
            chi2 += ((o - e) ** 2) / e


    df = (n_rows - 1) * (n_cols - 1)

    if df == 0:
        p_value = 1.0            
    else:
        if chi2 < 0:
            chi2 = 0.0
        z = math.sqrt(2.0 * chi2) - math.sqrt(2.0 * df - 1.0)
        normal = NormalDistribution(mu=0.0, sigma=1.0)
        p_value = 1.0 - normal.cdf(z)    

    alpha = 0.05
    reject = p_value < alpha

    return {
        "chi2_statistic": chi2,
        "p_value": p_value,
        "df": df,
        "alpha": alpha,
        "expected_table": expected_table,
        "reject_H0": reject,
        "conclusion": "Reject H0" if reject else "Fail to reject H0"
    }


