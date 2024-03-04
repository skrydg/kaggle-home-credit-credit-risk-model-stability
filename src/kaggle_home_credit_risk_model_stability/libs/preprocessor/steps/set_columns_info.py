import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class SetColumnsInfoStep:        
    def process_train_dataset(self, train_dataset, columns_info):
        for name, table in train_dataset.get_tables():
            for column in table.columns:
                if column in ("WEEK_NUM", "case_id", "MONTH", "num_group1", "num_group2", "target"):
                    columns_info.add_label(column, "SERVICE")
                    continue

                if (column[-1] == "D" or column in ["date_decision"]):
                    columns_info.add_label(column, "DATE")
                elif (column[-1] in ['M']) or (table[column].dtype == pl.String):
                    columns_info.add_label(column, "CATEGORICAL")
                    
                if column in ["mainoccupationinc_384A"]:
                    columns_info.add_label(column, "MONEY")
                    
        return train_dataset, columns_info
        
    def process_test_dataset(self, test_dataset, columns_info):
        return test_dataset, columns_info