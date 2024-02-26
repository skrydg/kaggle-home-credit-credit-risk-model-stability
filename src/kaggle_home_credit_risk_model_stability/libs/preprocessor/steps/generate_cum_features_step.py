import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset


class GenerateCumFeaturesStep:        
    def process_train_dataset(self, train_dataset):
        return self.process(train_dataset)
        
    def process_test_dataset(self, test_dataset):
        return self.process(test_dataset)
    
    def process(self, dataset):
        self.count_new_columns = 0
        assert(type(dataset) is Dataset)

        for i in range(len(dataset.depth_1)):
            dataset.depth_1[i] = self._process_table(dataset.depth_1[i])

        for i in range(len(dataset.depth_2)):
            dataset.depth_2[i] = self._process_table(dataset.depth_2[i])

        print("Create {} new cumulative columns".format(self.count_new_columns))
        return dataset

    def _is_numeric_type(self, type):
        return str(type).startswith("Int") or str(type).startswith("Float")
    
    def _process_table(self, table):
        columns = table.columns
        for column_name in columns:
            if column_name in ["case_id", "num_group1", "num_group2"]:
                continue
            if not self._is_numeric_type(table[column_name].dtype):
                continue
                
            new_column_name = f"{column_name}_cum_diff"
            table = table.with_columns(
                pl.col(column_name).diff().over("case_id").alias(new_column_name)
            )
            self.count_new_columns = self.count_new_columns + 1
            
            new_column_name = f"{column_name}_cum_sum"
            table = table.with_columns(
                pl.col(column_name).rolling_sum(window_size=10000, min_periods=1).over("case_id").alias(new_column_name)
            )
            self.count_new_columns = self.count_new_columns + 1
            
            
            new_column_name = f"{column_name}_cum_mean"
            table = table.with_columns(
                pl.col(column_name).rolling_mean(window_size=10000, min_periods=1).over("case_id").alias(new_column_name)
            )
            self.count_new_columns = self.count_new_columns + 1
            
            new_column_name = f"{column_name}_cum_max"
            table = table.with_columns(
                pl.col(column_name).rolling_max(window_size=10000, min_periods=1).over("case_id").alias(new_column_name)
            )
            self.count_new_columns = self.count_new_columns + 1
            
            new_column_name = f"{column_name}_cum_min"
            table = table.with_columns(
                pl.col(column_name).rolling_min(window_size=10000, min_periods=1).over("case_id").alias(new_column_name)
            )
            self.count_new_columns = self.count_new_columns + 1
            
            new_column_name = f"{column_name}_cum_median"
            table = table.with_columns(
                pl.col(column_name).rolling_median(window_size=10000, min_periods=1).over("case_id").alias(new_column_name)
            )
            self.count_new_columns = self.count_new_columns + 1
            
            new_column_name = f"{column_name}_cum_is_null_mean"
            table = table.with_columns(
                pl.col(column_name).is_null().rolling_mean(window_size=10000, min_periods=1).over("case_id").alias(new_column_name)
            )
            self.count_new_columns = self.count_new_columns + 1
            
        return table