import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class PairwiseDateDiffStep:   
    def process_train_dataset(self, dataset_generator):
        for train_dataset, columns_info in dataset_generator:
            yield self.process(train_dataset, columns_info)
        
    def process_test_dataset(self, dataset_generator):
        for test_dataset, columns_info in dataset_generator:
            yield self.process(test_dataset, columns_info)
    
    def process(self, dataset, columns_info):
        self.count_new_columns = 0
        for name, table in dataset.get_tables():
            dataset.set(name, self._process_table(table, columns_info))

        print("Create {} new columns as pairwise dates diff".format(self.count_new_columns))
        return dataset, columns_info

    def _process_table(self, table, columns_info):
        dates_columns = [column for column in table.columns if "DATE" in columns_info.get_labels(column)]
        for i in range(len(dates_columns)):
            for j in range(i + 1, len(dates_columns)):
                column_name = f"{dates_columns[i]}_{dates_columns[j]}_diff"
                table = table.with_columns((table[dates_columns[i]] - table[dates_columns[j]]).alias(column_name))
                self.count_new_columns = self.count_new_columns + 1
                columns_info.add_labels(column_name, {"DATE_DIFF"})

        return table