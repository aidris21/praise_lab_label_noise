from typing import List, Dict
from column_struct import ColumnStruct
import pandas as pd

class SyntheticDataFrame:
    """
    Class for generating a synthetic dataset from random samples from specified probability distributions.

    Attributes
    ----------
    schema : List[ColumnStruct]
        A specification of the table schema, including name, datatype, and pmf/pdf
    sample_size : int
        The number of rows that will be generated for the dataset
    """

    def __init__(self, schema: List[ColumnStruct], sample_size: int):
        self.schema = schema
        self.n = sample_size

        self.dataframe = self.generate_dataframe()

    def generate_dataframe(self) -> pd.DataFrame:
        syntheticData = {}
        for struct in self.schema:
            syntheticData[struct.name] = struct.distribution.draw(self.n)

        return pd.DataFrame(data=syntheticData)