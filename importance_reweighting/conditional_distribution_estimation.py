import numpy as np
import pandas as pd
from typing import Dict


def estimate_prior_distribution(labels: pd.Series, colname: str) -> Dict[any, float]:
    return labels.value_counts().div(labels.shape[0]).to_dict()


def estimate_density_ratio(conditional: pd.Series, marginal: pd.Series) -> pd.Series:
    """
    Funtion that estimates the ratio P(X|Y)/P(X) directly, without estimating the individual densities,
    using the KLIEP method. 

    see here: https://github.com/srome/pykliep
    """
    pass