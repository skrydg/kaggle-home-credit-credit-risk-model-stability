import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class DropSingleValueFeaturesStep:
    def __init__(self):
        self.columns = []
        
    def process_train_dataset(self, dataframe_generator):
        dataframe, columns_info = next(dataframe_generator)
        self._fill_columns_to_drop(dataframe, columns_info)

        yield self._process(dataframe, columns_info)
        
    def process_test_dataset(self, dataframe_generator):
        dataframe, columns_info = next(dataframe_generator)
        yield self._process(dataframe, columns_info)
    
    def _fill_columns_to_drop(self, dataframe, columns_info):                    
        for column in dataframe.columns:
            unique_count = dataframe[column].n_unique()
            if (unique_count == 1):
                self.columns.append(column)

            # if (unique_count == 2) and (dataframe[column].is_null().mean() > 0.):
            #     self.columns.append(column)
            
            # if (unique_count == 2) and (dataframe[column].dtype == pl.Enum) and ((dataframe[column].cast(pl.String) == "__NULL__").mean() > 0.):
            #     self.columns.append(column)
        
    def _process(self, dataframe, columns_info):
        print(f"Drop {len(self.columns)} columns as single value, columns: {self.columns}")
        dataframe = dataframe.drop(self.columns)
        return dataframe, columns_info
