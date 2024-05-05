import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class OneHotEncodingStep:
    def __init__(self):
        self.column_to_values = {
            "relationshiptoclient_415T": ['OTHER', '__OTHER__', 'GRAND_PARENT', 'CHILD', 'NEIGHBOR', 'PARENT', 'SIBLING', 'COLLEAGUE', 'SPOUSE', 'OTHER_RELATIVE', 'FRIEND'],
            "familystate_726L": ['DIVORCED', '__OTHER__', 'SINGLE', 'MARRIED', 'WIDOWED', 'LIVING_WITH_PARTNER'],
            "collaterals_typeofguarante_359M_close_credit_bureau_a_2": ['0e63c0f0', '168ad9f3', '2fd21cf1', '3cbe86ba', '46ab00a7', '5224034a', '7b62420e', '9276e4bb', '940efad7', '__NULL__', '__OTHER__', 'c7a5ad39'],
            "status_219L": ['K', 'N', 'Q', 'P', 'L', 'H', '__OTHER__', 'R', 'A', 'S', 'D', 'T']
        }
        
    def process_train_dataset(self, dataset_generator):
        dataset, columns_info = next(dataset_generator)
        yield self.process(dataset, columns_info)

        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)
        
    def process_test_dataset(self, dataset_generator):
      for dataset, columns_info in dataset_generator:
          yield self.process(dataset, columns_info)
    
    def process(self, dataset, columns_info):
        count_new_columns = 0
        
        table_names = [name for name, table in dataset.get_depth_tables([1, 2])]
        for name in table_names:
            table = dataset.get_table(name)
            columns_to_transform = list(set(self.column_to_values.keys()) & set(table.columns))
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