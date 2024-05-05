import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class OneHotEncodingStep:
    def __init__(self, columns = []):
        self.columns = columns
        self.column_to_values = {}
        
    def process_train_dataset(self, dataset_generator):
        dataset, columns_info = next(dataset_generator)
        self.set_values(dataset, columns_info)
        yield self.process(dataset, columns_info)

        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)
        
    def process_test_dataset(self, dataset_generator):
      for dataset, columns_info in dataset_generator:
          yield self.process(dataset, columns_info)
    
    def process(self, dataset, columns_info):
        count_new_columns = 0
        
        for name, table in dataset.get_depth_tables([1, 2]):
            table = dataset.get_table(name)
            columns_to_transform = list(set(self.columns) & set(table.columns))
            if len(columns_to_transform) == 0:
                continue

            table_to_transform = table[["case_id"] + columns_to_transform]

            for column in columns_to_transform:
                mask = table_to_transform[column].is_in(self.column_to_values[column])
                table_to_transform = table_to_transform.with_columns(table_to_transform[column].cast(pl.String).set(~mask, "__OTHER__"))
            
            one_hot_encoding_table = table_to_transform.to_dummies(columns_to_transform)
            for column in columns_to_transform:
                for value in self.column_to_values[column]:
                    new_column_name = f"{column}_{value}"
                    if new_column_name not in one_hot_encoding_table.columns:
                        one_hot_encoding_table = one_hot_encoding_table.with_columns(pl.lit(0).alias(new_column_name))

            one_hot_encoding_table = one_hot_encoding_table.group_by("case_id").agg([
                pl.all().sum().cast(pl.Int16).name.suffix("_sum"),
                pl.all().mean().cast(pl.Float32).name.suffix("_mean")
            ]).sort("case_id")

            dataset.set(f"{name}_one_hot_encoding_0", one_hot_encoding_table)

            new_columns = list(one_hot_encoding_table.columns)
            new_columns.remove("case_id")
            count_new_columns = count_new_columns + len(new_columns)

            for column in new_columns:
                columns_info.add_label(column, "ONE_HOT_ENCODING")

        print(f"Create {count_new_columns} new columns as one hot encoding")
        return dataset, columns_info

    def set_values(self, dataset, columns_info):
        raw_tables_info = columns_info.get_raw_tables_info()
        for table_name, table in dataset.get_depth_tables([1, 2]):
            columns_to_transform = list(set(self.columns) & set(table.columns))
            for column in columns_to_transform:
                values = raw_tables_info[table_name].get_unique_values(column)
                values = sorted(np.unique(values + ["__OTHER__", "__NULL__", "__UNKNOWN__"]))
                self.column_to_values[column] = values