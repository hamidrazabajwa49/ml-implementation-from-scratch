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

