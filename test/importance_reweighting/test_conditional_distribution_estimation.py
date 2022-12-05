from importance_reweighting.conditional_distribution_estimation import estimate_prior_distribution
from data.generate_synthetic_data.synthetic_dataframe import SyntheticDataFrame
from data.generate_synthetic_data.column_struct import ColumnStruct
from data.generate_synthetic_data.distributions import BernoulliDistribution

import unittest
import pandas as pd
import numpy as np


class TestConditionalDistributionEstimation(unittest.TestCase):


    def test_prior_estimation(self):
        colname: str = 'mock_labels'
        mock_data = np.concatenate((np.zeros(50), np.ones(50)))
        mock_labels: pd.Series = pd.Series(data=mock_data, name=colname)
        self.assertDictEqual(
            estimate_prior_distribution(labels=mock_labels, colname=colname), 
            {0: 0.5, 1: 0.5}
        )


if __name__ == '__main__':
    unittest.main()
