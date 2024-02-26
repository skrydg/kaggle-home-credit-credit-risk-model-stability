import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class SetTypesStep:
    def __init__(self):
        self.column_to_type = {}
        
    def process_train_dataset(self, train_dataset):
        for name, table in train_dataset.get_tables():
            for column in table.columns:
                if column in ("WEEK_NUM", "case_id", "MONTH", "num_group1", "num_group2", "target"):
                    self.column_to_type[column] = pl.Int64
                elif (column[-1] == "D" or column == "date_decision"):
                    self.column_to_type[column] = pl.Date
                elif (column[-1] in ['M']) or (table[column].dtype == pl.String):
                    self.column_to_type[column] = pl.String
                else:
                    self.column_to_type[column] = pl.Float32
        return self.process(train_dataset)
    
    def process_test_dataset(self, test_dataset):
        return self.process(test_dataset)
    
    def process(self, dataset):
        assert(type(dataset) is Dataset)
        for name, table in dataset.get_tables():
            dataset.set(name, self.process_table(table))

        return dataset
    
    def process_table(self, table):
        for column in table.columns:
            assert column in self.column_to_type, "Unknown column: {}".format(column)
            table = table.with_columns(table[column].cast(self.column_to_type[column]))
            if (column[-1] == "D"):
               table = table.with_columns(table[column].cast(pl.Int32))
        return table
