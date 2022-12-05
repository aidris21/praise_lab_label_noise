from data.generate_synthetic_data.synthetic_dataframe import SyntheticDataFrame
from data.generate_synthetic_data.column_struct import ColumnStruct
from data.generate_synthetic_data.distributions import NormalDistribution, BernoulliDistribution
from data.label_noise.noise import add_noise_to_labels

from pandas import DataFrame
import pandas as pd
import numpy as np

LABEL_COLUMN_NAME: str = 'label'

def insert_label_column(df: DataFrame, label_value: int) -> DataFrame:
    """
    Inserts the label column for a dataframe in place, using a single value
    """
    if (label_value not in [0,1]):
        raise ValueError('Label value must be either 0 or 1.')

    df.insert(loc=len(df.columns.to_list()), column=LABEL_COLUMN_NAME, value=label_value)

if __name__ == '__main__':
    # Generate synthetic dataframe with true labels
    # Each class has distinct underlying distributions
    negative_label_syn_df: DataFrame = SyntheticDataFrame(
        schema = [
            ColumnStruct(name='feature_1', datatype='float64', distribution=NormalDistribution(mean=0, stddev=1)),
            ColumnStruct(name='feature_2', datatype='float64', distribution=NormalDistribution(mean=10, stddev=5)),
            ColumnStruct(name='feature_3', datatype='float64', distribution=BernoulliDistribution(p=0.3)),
        ],
        sample_size=1000
    ).dataframe
    positive_label_syn_df: DataFrame = SyntheticDataFrame(
        schema = [
            ColumnStruct(name='feature_1', datatype='float64', distribution=NormalDistribution(mean=1, stddev=1)),
            ColumnStruct(name='feature_2', datatype='float64', distribution=NormalDistribution(mean=20, stddev=5)),
            ColumnStruct(name='feature_3', datatype='float64', distribution=BernoulliDistribution(p=0.7)),
        ],
        sample_size=1000
    ).dataframe
    insert_label_column(df=negative_label_syn_df, label_value=0)
    insert_label_column(df=positive_label_syn_df, label_value=1)
    syn_df: DataFrame = pd.concat([negative_label_syn_df, positive_label_syn_df], ignore_index=True)

    # Add noise to the labels
    noise_rates = {
        0: 0.2,
        1:0.3
    }
    syn_df_noisy = syn_df.copy(deep=True)
    syn_df_noisy[LABEL_COLUMN_NAME] = add_noise_to_labels(labels=syn_df_noisy[LABEL_COLUMN_NAME], noise_rates=noise_rates)

    