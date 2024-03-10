import numpy as np
import polars as pl

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
        unique_columns = set()
        hashed_table = table.select(pl.all().hash()).sum()
        
        raw_columns = list(set(columns_info.get_columns_with_label("RAW")) & set(hashed_table.columns))
        other_columns = [column for column in hashed_table.columns if column not in set(raw_columns)]
        for column in raw_columns + other_columns: # Try raw columns at first
            hash = hashed_table[column][0]
            if hash in unique_columns:
                self.columns.append(column)
            else:
                unique_columns.add(hash)
        
    def _process(self, dataset, columns_info):
        assert(type(dataset) is Dataset)
        for name, table in dataset.get_tables():
            for column in self.columns:
                table = table.drop(column)
            dataset.set(name, table)
        return dataset, columns_info