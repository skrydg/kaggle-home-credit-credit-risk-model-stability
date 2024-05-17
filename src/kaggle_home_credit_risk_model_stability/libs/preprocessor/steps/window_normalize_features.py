import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class WindowNormalizeFeaturesStep:
    def __init__(self, window_size, features):
       self.window_size = window_size
       self.features = features

    def process_train_dataset(self, dataframe_generator):
        for dataframe, columns_info in dataframe_generator:
            yield self.process(dataframe, columns_info)
        
    def process_test_dataset(self, dataframe_generator):
        for dataframe, columns_info in dataframe_generator:
            yield self.process(dataframe, columns_info)
    
    def process(self, dataframe, columns_info):
        for feature in self.features:
            tmp_df = dataframe[["date_decision", feature]]
            tmp_df = tmp_df.with_columns(pl.col("date_decision").cast(pl.Date))
            tmp_df = tmp_df.with_columns(pl.col(feature).fill_null(value=tmp_df[feature].mean()))
            tmp_df = tmp_df.sort("date_decision")
            tmp_df = tmp_df.with_columns(
                pl.col(feature).rolling_max(window_size=self.window_size, center=True, by="date_decision").alias(f"{feature}_rolling_max")
            )
            tmp_df = tmp_df.with_columns(
                pl.col(feature).rolling_min(window_size=self.window_size, center=True, by="date_decision").alias(f"{feature}_rolling_min")
            )
            dataframe = dataframe.with_columns(
                  ((tmp_df[feature] - tmp_df[f"{feature}_rolling_min"]) / 
                  (tmp_df[f"{feature}_rolling_max"] - tmp_df[f"{feature}_rolling_min"]).clip(lower_bound=1e-6))
                  .alias(f"window_normalized_{feature}")
            )
            dataframe = dataframe.drop(feature)

        return dataframe, columns_info
