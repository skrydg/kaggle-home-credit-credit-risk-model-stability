import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class DropAlmostNullFeaturesStep:
    def __init__(self):
        self.columns = []
        
    def process_train_dataset(self, dataset, columns_info):
        size = dataset.get_base().shape[0]
        for name, table in dataset.get_tables():
            self._fill_columns_to_drop(table, size, columns_info)
            
        print("Drop {} columns as almost null".format(len(self.columns)))
        print("Columns to drop {}".format(self.columns))
        return self._process(dataset, columns_info)
        
    def process_test_dataset(self, dataset, columns_info):
        return self._process(dataset, columns_info)
    
    def _fill_columns_to_drop(self, table, base_size, columns_info):                    
        for column in table.columns:
            if "SERVICE" in columns_info.get_labels(column):
                continue
                
            if table[column].shape[0] == 0:
                self.columns.append(column)
            else:
                isnull = (base_size - table[column].is_not_null().sum()) / base_size

                if isnull > 0.99:
                    self.columns.append(column)
                
                freq = table[column].n_unique()
                if (freq <= 1):
                    self.columns.append(column)
        
    def _process(self, dataset, columns_info):
        assert(type(dataset) is Dataset)
        for name, table in dataset.get_tables():
            for column in self.columns:
                table = table.drop(column)
            dataset.set(name, table)
        return dataset, columns_info