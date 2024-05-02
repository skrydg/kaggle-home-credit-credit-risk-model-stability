import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset


class GenerateCumFeaturesStep:  
    def __init__(self, tables):
        self.tables = tables

    def process_train_dataset(self, dataset_generator):
        for train_dataset, columns_info in dataset_generator:
            yield self.process(train_dataset, columns_info)
        
    def process_test_dataset(self, dataset_generator):
        for train_dataset, columns_info in dataset_generator:
            yield self.process(train_dataset, columns_info)
    
    def process(self, dataset, columns_info):
        self.count_new_columns = 0
        for name in self.tables:
            table = dataset.get_table(name)
            dataset.set(name, self._process_table(table)) 

        print("Create {} new cumulative columns".format(self.count_new_columns))
        return dataset, columns_info

    def _is_numeric_type(self, type):
        return str(type).startswith("Int") or str(type).startswith("Float")
    
    def _process_table(self, table):
        columns = table.columns
        for column_name in columns:
            if column_name in ["case_id", "num_group1", "num_group2"]:
                continue
            if not self._is_numeric_type(table[column_name].dtype):
                continue
                
            table = table.with_columns([
                pl.col(column_name).diff().over("case_id").alias(f"{column_name}_cum_diff"),
                pl.col(column_name).rolling_sum(window_size=10000, min_periods=1).over("case_id").alias(f"{column_name}_cum_sum"),
                pl.col(column_name).rolling_mean(window_size=10000, min_periods=1).over("case_id").alias(f"{column_name}_cum_mean"),
                pl.col(column_name).rolling_max(window_size=10000, min_periods=1).over("case_id").alias(f"{column_name}_cum_max"),
                pl.col(column_name).rolling_min(window_size=10000, min_periods=1).over("case_id").alias(f"{column_name}_cum_min"),
                pl.col(column_name).rolling_median(window_size=10000, min_periods=1).over("case_id").alias(f"{column_name}_cum_median"),
                pl.col(column_name).is_null().rolling_mean(window_size=10000, min_periods=1).over("case_id").alias(f"{column_name}_cum_is_null_mean")
            ])
            self.count_new_columns = self.count_new_columns + 7
  
        return table