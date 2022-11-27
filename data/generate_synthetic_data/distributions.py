from dataclasses import dataclass
from abc import abstractmethod
from typing import List
import numpy as np
from scipy.stats import norm

class ProbabilityDistribution:
    """
    Specification of either a probability mass function or probability density function.

    Attributes
    ----------
    name : str
        name of the type of distribution
    """

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def p_x(self, x: float) -> float:
        """
        Calculates f(x) where f is our probability function
        """
        pass
    
    @abstractmethod
    def c_x(self, x: float) -> float:
        """
        Calculates F(x) where F is the cumulative probability function over our given probability function 
        """
        pass

    @abstractmethod
    def draw(self, n: int):
        """
        Draws `n` random samples from our specified probability distribution
        """
        pass


class NormalDistribution(ProbabilityDistribution):

    def __init__(self, mean=0, stddev=1):
        super().__init__(name='normal')
        self.mean = mean
        self.standard_deviation = stddev
        self.variance = self.standard_deviation**2

    def p_x(self, x: float) -> float:
        norm.pdf(x=x, loc=self.mean, scale=self.standard_deviation)

    def c_x(self, x: float) -> float:
        return norm.cdf(x=x, loc=self.mean, scale=self.standard_deviation)

    def draw(self, n: int):
        return np.random.normal(loc=self.mean, scale=self.standard_deviation, size=n)

    