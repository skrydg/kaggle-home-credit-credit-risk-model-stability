import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class PairwiseDateDiffStep:   
    def process_train_dataset(self, train_dataset):
        return self.process(train_dataset)
        
    def process_test_dataset(self, test_dataset):
        return self.process(test_dataset)
    
    def process(self, dataset):
        self.count_new_columns = 0
        assert(type(dataset) is Dataset)
        dataset.base = self._process_table(dataset.base)

        for i in range(len(dataset.depth_0)):
            dataset.depth_0[i] = self._process_table(dataset.depth_0[i])

        for i in range(len(dataset.depth_1)):
            dataset.depth_1[i] = self._process_table(dataset.depth_1[i])

        for i in range(len(dataset.depth_2)):
            dataset.depth_2[i] = self._process_table(dataset.depth_2[i])

        print("Create {} new columns as pairwise dates diff".format(self.count_new_columns))
        return dataset

    def _process_table(self, table):
        dates_columns = [column for column in table.columns if column[-1] == 'D']
        for i in range(len(dates_columns)):
            for j in range(i + 1, len(dates_columns)):
                column_name = f"{dates_columns[i]}_{dates_columns[j]}_diff"
                table = table.with_columns((table[dates_columns[i]] - table[dates_columns[j]]).dt.total_days().alias(column_name))
                self.count_new_columns = self.count_new_columns + 1

        return table