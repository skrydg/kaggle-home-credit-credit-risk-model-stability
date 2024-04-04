import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class ProcessCategoricalStep:    
    def __init__(self):
        self.column_to_type = {}
        
    def process_train_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            self._fill_types(dataset, columns_info)
            yield self._process(dataset, columns_info)
            
    def process_test_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self._process(dataset, columns_info)
        
    def _fill_types(self, dataset, columns_info):
        for name, table in dataset.get_tables():
          self._fill_table_types(name, table, columns_info)

    def _fill_table_types(self, name, table, columns_info):
        for column in table.columns:
            if "CATEGORICAL" in columns_info.get_labels(column):
                unique_values = sorted(columns_info.get_raw_tables_info()[name].get_unique_values(column) + ["__UNKNOWN__", "__NULL__", "__OTHER__"])
                self.column_to_type[column] = pl.Enum(unique_values)
    
    def _process(self, dataset, columns_info):
        for name, table in dataset.get_tables():
            dataset.set(name, self._process_table(table, columns_info))
        return dataset, columns_info

    def _process_table(self, table, columns_info):
        for column in table.columns:
            if "CATEGORICAL" in columns_info.get_labels(column):
                column_type = self.column_to_type[column]
                table = table.with_columns(table[column].set(~table[column].is_in(column_type.categories), "__UNKNOWN__"))
                table = table.with_columns(table[column].fill_null("__NULL__").cast(column_type))
        return table
