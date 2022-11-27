from dataclasses import dataclass
from abc import abstractmethod
from typing import List

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
    def draw(self, n: int) -> List[float]:
        """
        Draws `n` random samples from our specified probability distribution
        """
        pass