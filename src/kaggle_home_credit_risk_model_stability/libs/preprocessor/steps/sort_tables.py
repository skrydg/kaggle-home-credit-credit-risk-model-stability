import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class SortRawTablesStep:
    def process_train_dataset(self, dataframe_generator):
        for dataset, columns_info in dataframe_generator:
            dataset = self.sort_dataset(dataset)
            yield dataset, columns_info

    def process_test_dataset(self, dataframe_generator):
        for dataset, columns_info in dataframe_generator:
            dataset = self.sort_dataset(dataset)
            yield dataset, columns_info

    def sort_dataset(self, dataset):
        for name, table in dataset.get_tables():
            columns_to_sort = ["case_id", "num_group1", "num_group2"]
            columns_to_sort = list(set(columns_to_sort) & set(table.columns))
            dataset.set(name, table.sort(by=columns_to_sort))
        return dataset