import unittest
from constants import PANDAS_DATATYPES
from data.generate_synthetic_data.synthetic_dataframe import SyntheticDataFrame
from data.generate_synthetic_data.distributions import NormalDistribution, BernoulliDistribution

class TestDistributions(unittest.TestCase):


    def test_bernoulli(self):
        p = 0.5
        bernoulli_dist_fair = BernoulliDistribution(p=p)
        self.assertEqual(bernoulli_dist_fair.p_x(0), 1-p)
        self.assertEqual(bernoulli_dist_fair.p_x(1), p)
        self.assertEqual(bernoulli_dist_fair.p_x(2), 0)
        self.assertEqual(bernoulli_dist_fair.c_x(0), 1-p)
        self.assertEqual(bernoulli_dist_fair.c_x(1), 1)

        p = 0
        bernoulli_dist_all_negative = BernoulliDistribution(p=p)
        self.assertEqual(bernoulli_dist_all_negative.p_x(0), 1-p)
        self.assertEqual(bernoulli_dist_all_negative.p_x(1), p)
        self.assertEqual(bernoulli_dist_all_negative.p_x(2), 0)
        self.assertEqual(bernoulli_dist_all_negative.c_x(0), 1-p)
        self.assertEqual(bernoulli_dist_all_negative.c_x(1), 1)
        self.assertListEqual(bernoulli_dist_all_negative.draw(3).tolist(), [0,0,0])



if __name__ == '__main__':
    unittest.main()