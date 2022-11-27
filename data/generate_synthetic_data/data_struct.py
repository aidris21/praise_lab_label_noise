from dataclasses import dataclass

@dataclass
class ColumnStruct:
    """
    Holds metadata for a column in a synthetically-generated dataset.

    Attributes
    ----------
    name : str
        name of the column
    type : str
        data type of the column as a string, see: https://pandas.pydata.org/docs/user_guide/gotchas.html?highlight=type
    distribution : str
        name of the column
    """
    
