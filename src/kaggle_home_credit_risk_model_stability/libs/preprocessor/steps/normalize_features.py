import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class NormalizeFeaturesStep:
    def __init__(self, features):
       self.features = features

    def process_train_dataset(self, dataframe_generator):
        for dataframe, columns_info in dataframe_generator:
          yield self.process(dataframe, columns_info)
        
    def process_test_dataset(self, dataframe_generator):
        for dataframe, columns_info in dataframe_generator:
          yield self.process(dataframe, columns_info)
    
    def process(self, dataframe, columns_info):
        for feature in self.features:
          dataframe = dataframe.with_columns(
             (pl.col(feature) - pl.col(feature).min()) / max(1e-6, (dataframe[feature].max() - dataframe[feature].min()))
          )
        return dataframe, columns_info
