import unittest
from data.generate_synthetic_data.synthetic_dataframe import SyntheticDataFrame
from data.generate_synthetic_data.column_struct import ColumnStruct
from data.generate_synthetic_data.distributions import NormalDistribution

class TestSyntheticDataFrame(unittest.TestCase):

    normal_synthetic_df_sample_size = 100
    normal_synthetic_df: SyntheticDataFrame = SyntheticDataFrame(
        schema = [
            ColumnStruct(name='foo', datatype='float64', distribution=NormalDistribution()),
            ColumnStruct(name='bar', datatype='float64', distribution=NormalDistribution(mean=10, stddev=5)),
        ],
        sample_size=normal_synthetic_df_sample_size
    )

    def test_sample_size(self):
        self.assertEqual(self.normal_synthetic_df.dataframe.shape[0], self.normal_synthetic_df_sample_size)

    def test_column_names(self):
        self.assertListEqual(self.normal_synthetic_df.dataframe.columns, ['foo', 'bar'])


if __name__ == '__main__':
    unittest.main()