import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class DropVariableEnumFeaturesStep:
    def __init__(self):
        self.columns = []
        
    def process_train_dataset(self, dataframe, columns_info):
        self._fill_columns_to_drop(dataframe, columns_info)
            
        print("Drop {} columns as variable enum value".format(len(self.columns)))
        return self._process(dataframe, columns_info)
        
    def process_test_dataset(self, dataframe, columns_info):
        return self._process(dataframe, columns_info)
    
    def _fill_columns_to_drop(self, dataframe, columns_info):                    
        for column in dataframe.columns:
            if (dataframe[column].dtype == pl.Enum) and (dataframe[column].n_unique() > 30):
                self.columns.append(column)
        
    def _process(self, dataframe, columns_info):
        dataframe = dataframe.drop(self.columns)
        return dataframe, columns_info
