import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class DropVariableEnumFeaturesStep:
    def __init__(self):
        self.columns = []
        
    def process_train_dataset(self, dataset, columns_info):
        for name, table in dataset.get_tables():
            self._fill_columns_to_drop(table, columns_info)
            
        print("Drop {} columns as variable enum".format(len(self.columns)))
        return self._process(dataset, columns_info)
        
    def process_test_dataset(self, dataset, columns_info):
        return self._process(dataset, columns_info)
    
    def _fill_columns_to_drop(self, table, columns_info):                    
        for column in table.columns:
            if table[column].dtype == pl.Enum:
                freq = table[column].n_unique()

                if (freq > 200):
                    self.columns.append(column)
        
    def _process(self, dataset, columns_info):
        assert(type(dataset) is Dataset)
        for name, table in dataset.get_tables():
            for column in self.columns:
                table = table.drop(column)
            dataset.set(name, table)
        return dataset, columns_info
