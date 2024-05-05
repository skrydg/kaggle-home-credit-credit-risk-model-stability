import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class DropAlmostNullFeaturesStep:
    def __init__(self, threashold=0.95):
        self.threashold = threashold
        self.columns = []
        
    def process_train_dataset(self, df_generator):
        df, columns_info = next(df_generator)
        self._fill_columns_to_drop(df, columns_info)
            
        print("Drop {} columns as almost null".format(len(self.columns)))
        print("Columns to drop {}".format(self.columns))
        yield self._process(df, columns_info)
        
    def process_test_dataset(self, df_generator):
        df, columns_info = next(df_generator)
        yield self._process(df, columns_info)
    
    def _fill_columns_to_drop(self, df, columns_info):                    
        for column in df.columns:
            if "SERVICE" in columns_info.get_labels(column):
                continue
                
            isnull = df[column].is_null().mean()

            if isnull > self.threashold:
                self.columns.append(column)
                continue
            
            freq = df[column].n_unique()
            if (freq <= 1):
                self.columns.append(column)
        
    def _process(self, df, columns_info):
        df = df.drop(self.columns)
        return df, columns_info