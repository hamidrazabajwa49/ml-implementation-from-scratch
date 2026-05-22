import math
import random
from Probability.base import Distribution

class BionomialDistribution(Distribution):

    
    def __init__(self, n: int, p: float):
        if not (0 <= p <= 1):
            raise ValueError("Probability p must be between 0 and 1.")
        self.n = n
        self.p = p

    def mean(self) -> float:
        return self.n * self.p

    def variance(self) -> float:
        return self.n * self.p *(1-self.p)

    def pmf(self,k:int)-> float:
        if not (0 <= k <=self.n):
            return 0.0
        comb=math.comb(self.n,k)
        return comb*(self.p**k)*((1-self.p)**(self.n-k))

    def sample(self,num_samples:int=1)-> list:
        samples=[]
        for _ in range(num_samples):
            successes=sum(1 for _ in range(self.n) if random.random() < self.p)
            samples.append(successes)
        return samples
