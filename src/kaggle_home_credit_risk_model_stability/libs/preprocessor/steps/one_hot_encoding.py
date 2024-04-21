import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class OneHotEncodingStep:
    def __init__(self):
        self.feature_to_values = {}
        
    def process_train_dataset(self, dataset_generator):
        dataset, columns_info = next(dataset_generator)
        raw_tables_info = columns_info.get_raw_tables_info()

        depth_tables = dataset.get_depth_tables([1, 2])
        for _, table in depth_tables:
            for column in table.columns:
                table_name = columns_info.get_table_name(column)
                if ("CATEGORICAL" in columns_info.get_labels(column)) and (len(raw_tables_info[table_name].get_unique_values(column)) > 1):
                    value_count = raw_tables_info[table_name].get_value_counts(column)
                    top_10_categories = value_count.sort(["count", column])[-10:]
                    top_10_count = top_10_categories["count"].sum()
                    if (top_10_count / value_count["count"].sum() > 0.9):
                        self.feature_to_values[column] = list(top_10_categories[column]) + ["__UNKNOWN__", "__NULL__", "__OTHER__"]
                        
        yield self.process(dataset, columns_info)
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)
        
    def process_test_dataset(self, dataset_generator):
      for dataset, columns_info in dataset_generator:
          yield self.process(dataset, columns_info)
    
    def process(self, dataset, columns_info):
        count_new_columns = 0
        
        table_names = [name for name, table in dataset.get_depth_tables(1)]
        for name in table_names:
            table = dataset.get_table(name)
            columns_to_transform = sorted(list(set(self.feature_to_values.keys()) & set(list(table.columns))))
            if len(columns_to_transform) == 0:
                continue

            table_to_transform = table[["case_id"] + columns_to_transform]

            for column in columns_to_transform:
                mask = table_to_transform[column].is_in(self.feature_to_values[column])
                table_to_transform = table_to_transform.with_columns(table_to_transform[column].cast(pl.String).set(~mask, "__OTHER__"))
            
            one_hot_encoding_table = table_to_transform.to_dummies(columns_to_transform)
            for column in columns_to_transform:
                for value in self.feature_to_values[column]:
                    new_column_name = f"{column}_{value}"
                    if new_column_name not in one_hot_encoding_table.columns:
                        one_hot_encoding_table = one_hot_encoding_table.with_columns(pl.lit(0).alias(new_column_name))

            one_hot_encoding_table = one_hot_encoding_table.group_by("case_id").sum().sort("case_id")

            dataset.set(f"{name}_one_hot_encoding_0", one_hot_encoding_table)

            new_columns = list(one_hot_encoding_table.columns)
            new_columns.remove("case_id")
            count_new_columns = count_new_columns + len(new_columns)

            for column in new_columns:
                columns_info.add_label(column, "ONE_HOT_ENCODING")

        print(f"Create {count_new_columns} new columns as one hot encoding")
        return dataset, columns_info