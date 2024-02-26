import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class SetTypesStep:
    def __init__(self):
        self.column_to_type = {}
        
    def process_train_dataset(self, train_dataset):
        for df in [train_dataset.base] + train_dataset.depth_0 + train_dataset.depth_1 + train_dataset.depth_2:
            for column in df.columns:
                if column in ("WEEK_NUM", "case_id", "MONTH", "num_group1", "num_group2", "target"):
                    self.column_to_type[column] = pl.Int64
                elif (column[-1] == "D" or column == "date_decision"):
                    self.column_to_type[column] = pl.Date
                elif (column[-1] in ['M']) or (df[column].dtype == pl.String):
                    self.column_to_type[column] = pl.String
                else:
                    self.column_to_type[column] = pl.Float32
        return self.process(train_dataset)
    
    def process_test_dataset(self, test_dataset):
        return self.process(test_dataset)
    
    def process(self, dataset):
        assert(type(dataset) is Dataset)
        dataset.base = self.process_tables([dataset.base])[0]
        dataset.depth_0 = self.process_tables(dataset.depth_0)
        dataset.depth_1 = self.process_tables(dataset.depth_1)
        dataset.depth_2 = self.process_tables(dataset.depth_2)
        return dataset
    
    def process_tables(self, dfs):
        for i in range(len(dfs)):
            for column in dfs[i].columns:
                assert column in self.column_to_type, "Unknown column: {}".format(column)
                dfs[i] = dfs[i].with_columns(dfs[i][column].cast(self.column_to_type[column]))
                if (column[-1] == "D"):
                    dfs[i] = dfs[i].with_columns(dfs[i][column].cast(pl.Int32))
        return dfs
