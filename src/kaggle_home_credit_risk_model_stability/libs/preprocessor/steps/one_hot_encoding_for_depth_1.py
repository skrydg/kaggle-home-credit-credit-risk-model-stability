import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class OneHotEncodingForDepth1Step:
    def __init__(self):
        self.features = []
        
    def process_train_dataset(self, dataset):
        depth_1 = dataset.get_depth_tables(1)
        for name, table in depth_1:
            for column in table.columns:
                if (table[column].dtype == pl.Enum) and (table[column].n_unique() < 10) and (table[column].n_unique() > 1):
                    self.features.append(column)
                    
        return self.process(dataset)
        
    def process_test_dataset(self, dataset):
        return self.process(dataset)
    
    def process(self, dataset):
        assert(type(dataset) is Dataset)
        count_new_columns = 0
        depth_1 = dataset.get_depth_tables(1)
        
        table_names = [name for name, table in dataset.get_depth_tables(1)]
        new_tables = {}
        for name in table_names:
            table = dataset.get_table(name)
            columns_to_transform = list(set(self.features) & set(list(table.columns)))
            if len(columns_to_transform) == 0:
                continue
            print(columns_to_transform)
            one_hot_encoding_table = table[["case_id"] + columns_to_transform].to_dummies(columns_to_transform).group_by("case_id").sum()
            
            dataset.set(f"{name}_one_hot_encoding_0", one_hot_encoding_table)
            table = table.drop(columns_to_transform)
            dataset.set(name, table)
            count_new_columns = count_new_columns + one_hot_encoding_table.shape[1]

        print(f"Create {count_new_columns} new columns as one hot encoding")
        return dataset