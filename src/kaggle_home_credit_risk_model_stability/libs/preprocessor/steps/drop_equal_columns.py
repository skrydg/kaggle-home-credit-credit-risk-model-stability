import numpy as np
import polars as pl
import hashlib

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class DropEqualColumnsStep:
    def __init__(self):
        self.columns = []
        
    def process_train_dataset(self, dataset, columns_info):
        for name, table in dataset.get_tables():
            self._fill_columns_to_drop(table, columns_info)
            
        print("Drop {} columns as duplicates".format(len(self.columns)))
        return self._process(dataset, columns_info)
        
    def process_test_dataset(self, dataset, columns_info):
        return self._process(dataset, columns_info)
    
    def _fill_columns_to_drop(self, table, columns_info):  
        if table.shape[0] == 0:
            return
        
        unique_columns = dict()        
        raw_columns = sorted(list(set(columns_info.get_columns_with_label("RAW")) & set(table.columns)))
        other_columns = sorted([column for column in table.columns if column not in set(raw_columns)])
        for column in raw_columns + other_columns: # Try raw columns at first
            hash_result = hashlib.sha256(table[column].hash().to_numpy())
            hash_result.update(str(table[column].dtype).encode('utf-8'))
            hash_result = hash_result.hexdigest()
            if hash_result in unique_columns:
                self.columns.append(column)
                column1 = column
                column2 = unique_columns[hash_result]
                assert(table[column1].equals(table[column2].alias(column1).cast(table[column1].dtype)))
            else:
                unique_columns[hash_result] = column
        
    def _process(self, dataset, columns_info):
        assert(type(dataset) is Dataset)
        for name, table in dataset.get_tables():
            table = table.drop(self.columns)
            dataset.set(name, table)
        return dataset, columns_info