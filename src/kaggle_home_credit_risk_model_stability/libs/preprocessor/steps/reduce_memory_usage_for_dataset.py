import numpy as np
import polars as pl
import gc
from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset
from .reduce_memory_usage_for_dataframe import ReduceMemoryUsageForDataFrameStep

class ReduceMemoryUsageForDatasetStep:     
    def process_train_dataset(self, dataset_generator):  
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)
        
    def process_test_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)
    
    def process(self, dataset, columns_info):
        for table_name, table in dataset.get_depth_tables([0, 1, 2]):
            table, columns_info = ReduceMemoryUsageForDataFrameStep().process(table, columns_info)
            dataset.set(table_name, table)
            gc.collect()

        return dataset, columns_info
