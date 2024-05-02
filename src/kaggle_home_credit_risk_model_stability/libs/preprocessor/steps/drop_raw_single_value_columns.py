import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class DropRawSingleValueColumnsStep:
    def __init__(self):
        self.columns_to_drop = []
        
    def process_train_dataset(self, dataset_generator):
        dataset, columns_info = next(dataset_generator)
        self._fill_columns_to_drop(dataset, columns_info)
        yield self._process(dataset, columns_info)

        for dataset, columns_info in dataset_generator:
          yield self._process(dataset, columns_info)
        
    def process_test_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
          return self._process(dataset, columns_info)
    
    def _fill_columns_to_drop(self, dataset, columns_info):
        raw_tables_info = columns_info.get_raw_tables_info()
        for name, table in dataset.get_depth_tables([0, 1, 2]):
            for column in table.columns:
                if (raw_tables_info[name].get_min_value(column) == raw_tables_info[name].get_max_value(column)):
                    self.columns_to_drop.append(column)
 
    def _process(self, dataset, columns_info):
        print(f"Drop {len(self.columns_to_drop)} raw single value columns, columns: {self.columns_to_drop}")
        for name, table in dataset.get_depth_tables([0, 1, 2]):
            columns_to_drop = list(set(table.columns) & set(self.columns_to_drop))
            table = table.drop(columns_to_drop)
            dataset.set(name, table)
        return dataset, columns_info
