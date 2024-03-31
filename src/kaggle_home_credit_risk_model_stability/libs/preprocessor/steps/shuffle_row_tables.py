import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class ShuffleRowTablesStep:
    def __init__(self, shuffle_train_dataset = False, shuffle_test_dataset=False):
        self.shuffle_train_dataset = shuffle_train_dataset
        self.shuffle_test_dataset = shuffle_test_dataset
        
    def process_train_dataset(self, dataframe_generator):
        for dataset, columns_info in dataframe_generator:
            if self.shuffle_train_dataset:
              dataset = self.shuffle_dataset(dataset)
            yield dataset, columns_info

    def process_test_dataset(self, dataframe_generator):
        for dataset, columns_info in dataframe_generator:
            if self.shuffle_test_dataset:
              dataset = self.shuffle_dataset(dataset)
            yield dataset, columns_info

    def shuffle_dataset(self, dataset):
        for name, table in dataset.get_tables():
            dataset.set(name, table.sample(fraction=1, with_replacement=False, shuffle=True, seed=42))
        return dataset