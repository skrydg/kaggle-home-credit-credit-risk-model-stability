import numpy as np
import polars as pl
import gc

class CreateMoneyFeatureFractionStep:
    def __init__(self, base_column):
        self.base_column = base_column

    def process_train_dataset(self, df_generator):
        for df, columns_info in df_generator:
            yield self.process(df, columns_info)
        
    def process_test_dataset(self, df_generator):
        for df, columns_info in df_generator:
            yield self.process(df, columns_info)
    
    def process(self, df, columns_info):
        self.count_new_columns = 0
        for column in df.columns:
            if (column != self.base_column) and ("MONEY" in columns_info.get_labels(column)):
                new_column_name = f"{column}/{self.base_column}_fraction"

                new_column = (df[column] / df[self.base_column]).cast(pl.Float32)
                inf_mask = new_column.is_infinite() | new_column.is_nan()
                new_column = new_column.set(inf_mask, 1e10)

                df = df.with_columns(new_column.alias(new_column_name))
                columns_info.add_label(new_column_name, "MONEY_FRACTION")
                self.count_new_columns = self.count_new_columns + 1

        print(f"Create {self.count_new_columns} new columns as money feature fraction with base '{self.base_column}'")
        gc.collect()
        return df, columns_info
