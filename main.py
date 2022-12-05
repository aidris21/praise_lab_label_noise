from data.generate_synthetic_data.synthetic_dataframe import SyntheticDataFrame
from data.generate_synthetic_data.column_struct import ColumnStruct
from data.generate_synthetic_data.distributions import NormalDistribution

if __name__ == '__main__':
    df: SyntheticDataFrame = SyntheticDataFrame(
        schema = [
            ColumnStruct(name='feature_1', datatype='float64', distribution=NormalDistribution()),
            ColumnStruct(name='feature_2', datatype='float64', distribution=NormalDistribution(mean=10, stddev=5)),
        ],
        sample_size=10
    )
    print(df.dataframe.head())