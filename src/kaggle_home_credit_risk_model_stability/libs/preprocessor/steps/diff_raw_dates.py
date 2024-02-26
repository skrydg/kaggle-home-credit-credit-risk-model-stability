import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class DiffRawDatesStep:        
    def process_train_dataset(self, train_dataset):
        return self.process(train_dataset)
        
    def process_test_dataset(self, test_dataset):
        return self.process(test_dataset)
    
    def process(self, dataset):
        assert(type(dataset) is Dataset)
        for name, table in dataset.get_tables():
            dataset.set(name, self._process_table(table))
        return dataset

    def _process_table(self, table):
        count_new_columns = 0
        dates_columns = [column for column in table.columns if column[-1] == 'D']
        for i in range(len(dates_columns)):
            for j in range(i + 1, len(dates_columns)):
              column_name = f"{dates_columns[i]}_{dates_columns[j]}_diff"
              table = table.with_columns((table[dates_columns[i]] - table[dates_columns[j]]).alias(column_name))
              count_new_columns = count_new_columns + 1

        print("Create {} new columns as dates diff", count_new_columns)
        return table