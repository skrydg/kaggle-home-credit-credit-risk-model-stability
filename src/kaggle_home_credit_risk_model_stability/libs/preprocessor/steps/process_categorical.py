import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class ProcessCategoricalStep:    
    def __init__(self):
        self.column_to_type = {}
        
    def process_train_dataset(self, dataset):
        assert(type(dataset) is Dataset)
        self._fill_types(dataset)
        return self._process(dataset)
        
    def process_test_dataset(self, dataset):
        assert(type(dataset) is Dataset)
        return self._process(dataset)
    
    def _fill_types(self, dataset):
        for name, table in dataset.get_tables():
          self._fill_table_types(table)

    def _fill_table_types(self, table):
        for column in table.columns:
            if table[column].dtype == pl.String:
                unique_values = list(table[column].filter(~table[column].is_null()).unique())
                self.column_to_type[column] = pl.Enum(unique_values + ["__UNKNOWN__"])
    
    def _process(self, dataset):
        for name, table in dataset.get_tables():
            dataset.set(name, self._process_table(table))
        return dataset

    def _process_table(self, table):
        for column in table.columns:
            if table[column].dtype == pl.String:
                column_type = self.column_to_type[column]
                table = table.with_columns(table[column].set(~table[column].is_in(column_type.categories), "__UNKNOWN__"))
                table = table.with_columns(table[column].fill_null("__UNKNOWN__").cast(column_type))
        return table
