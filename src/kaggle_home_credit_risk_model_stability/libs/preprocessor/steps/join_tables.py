import numpy as np
import polars as pl
import gc

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

    
class JoinTablesStep:        
    def process_train_dataset(self, train_dataset):
        return self.process(train_dataset)
        
    def process_test_dataset(self, test_dataset):
        return self.process(test_dataset)
    
    def process(self, dataset):
        result = dataset.get_base()
        for name, table in dataset.get_depth_tables([0, 1, 2]):
            result = result.join(table, how="left", on="case_id", suffix=f"_{name}")
            dataset.set(name, None) # To clean memory
            gc.collect()
        return result