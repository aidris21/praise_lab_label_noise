from dataclasses import dataclass, field
from distributions import ProbabilityDistribution, NormalDistribution

@dataclass
class ColumnStruct:
    """
    Holds metadata for a column in a synthetically-generated dataset.

    Attributes
    ----------
    name : str
        name of the column
    datatype : str
        data type of the column as a string, see: https://pandas.pydata.org/docs/user_guide/gotchas.html?highlight=type
    distribution : ProbabilityDistribution
        Probability Distribution used to generate data for the column
    """

    name: str = field(init=None)

    datatype: str = field(init='object')

    distribution: ProbabilityDistribution = NormalDistribution
    
