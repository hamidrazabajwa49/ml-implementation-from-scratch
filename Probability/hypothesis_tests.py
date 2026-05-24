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
