import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class DropColumnsStep:
    def __init__(self):
        self.columns = []
        
    def process_train_dataset(self, dataset):
        for name, table in dataset.get_tables():
            self._fill_columns_to_drop(table)
            

        self.columns.append("date_decision")
        self.columns.append("MONTH")
                
        print("Columns to drop: {}".format(len(self.columns)))
        return self._process(dataset)
        
    def process_test_dataset(self, dataset):
        return self._process(dataset)
    
    def _fill_columns_to_drop(self, table):
        for column in table.columns:
            if column == "case_id":
                continue
            if table[column].shape[0] == 0:
                self.columns.append(column)
            else:
                isnull = table[column].is_null().mean()

                if isnull > 0.95:
                    self.columns.append(column)

                freq = table[column].n_unique()
                if (freq <= 1):
                    self.columns.append(column)
        for column in table.columns:
            if table[column].dtype == pl.Enum:
                freq = table[column].n_unique()

                if (freq <= 1) or (freq > 200):
                    self.columns.append(column)
                    
        unique_columns = set()
        hashed_table = table.all().hash().sum()
        for column in hashed_table.columns:
            hash = hashed_table[column][0]
            if hash in unique_columns:
                self.columns.append(column)
            else:
                unique_columns.add(hash)
        
    def _process(self, dataset):
        assert(type(dataset) is Dataset)
        for name, table in dataset.get_tables():
            for column in self.columns:
                table = table.drop(column)
            dataset.set(name, table)
        return dataset