import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class DropTablesStep:
    def __init__(self, tables):
        self.tables = tables

    def process_train_dataset(self, dataframe_generator):
        for dataset, columns_info in dataframe_generator:
            yield self.process(dataset, columns_info)

    def process_test_dataset(self, dataframe_generator):
        for dataset, columns_info in dataframe_generator:
            yield self.process(dataset, columns_info)

    def process(self, dataset, columns_info):
        for table in self.tables:
            dataset.delete(table)
        return dataset, columns_info