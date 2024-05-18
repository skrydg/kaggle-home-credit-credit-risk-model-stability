import numpy as np
import polars as pl
import gc

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class JoinTablesStep:        
    def process_train_dataset(self, train_dataset_generator):
        for train_dataset, columns_info in train_dataset_generator:
            yield self.process(train_dataset, columns_info)
        
    def process_test_dataset(self, test_dataset_generator):
        for train_dataset, columns_info in test_dataset_generator:
            yield self.process(train_dataset, columns_info)
    
    def process(self, dataset, columns_info):
        result = dataset.get_base()
        for name, table in dataset.get_depth_tables([0, 1, 2]):
            result = result.join(table, how="left", on="case_id", suffix=f"_{name}")
            dataset.set(name, None) # Release memory
            gc.collect()
        del dataset
        gc.collect()
        return result, columns_info