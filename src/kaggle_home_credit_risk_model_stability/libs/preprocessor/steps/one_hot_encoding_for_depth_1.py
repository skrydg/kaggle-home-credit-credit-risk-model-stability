import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class OneHotEncodingForDepth1Step:
    def __init__(self):
        self.features = []
        
    def process_train_dataset(self, dataset, columns_info):
        depth_1 = dataset.get_depth_tables(1)
        for name, table in depth_1:
            for column in table.columns:
                if ("CATEGORICAL" in columns_info.get_labels(column)) and (1 < table[column].n_unique() < 15):
                    self.features.append(column)
                    
        return self.process(dataset, columns_info)
        
    def process_test_dataset(self, dataset, columns_info):
        return self.process(dataset, columns_info)
    
    def process(self, dataset, columns_info):
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
            one_hot_encoding_table = table[["case_id"] + columns_to_transform].to_dummies(columns_to_transform).group_by("case_id").sum()
            
            dataset.set(f"{name}_one_hot_encoding_0", one_hot_encoding_table)
            table = table.drop(columns_to_transform)
            dataset.set(name, table)

            new_columns = list(one_hot_encoding_table.columns)
            new_columns.remove("case_id")
            count_new_columns = count_new_columns + len(new_columns)

            for column in new_columns:
                columns_info.add_label(column, "ONE_HOT_ENCODING")

        print(f"Create {count_new_columns} new columns as one hot encoding")
        return dataset, columns_info