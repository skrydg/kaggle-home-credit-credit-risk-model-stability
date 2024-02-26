import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

    
class JoinTablesStep:        
    def process_train_dataset(self, train_dataset):
        return self.process(train_dataset)
        
    def process_test_dataset(self, test_dataset):
        return self.process(test_dataset)
    
    def process(self, dataset):
        result = dataset.base
        for i, df in enumerate(dataset.depth_0 + dataset.depth_1 + dataset.depth_2):
            result = result.join(df, how="left", on="case_id", suffix=f"_{i}")
        return result