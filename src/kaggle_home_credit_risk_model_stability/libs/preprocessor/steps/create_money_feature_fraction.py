import numpy as np
import polars as pl


class CreateMoneyFeatureFractionStep:
    def __init__(self, base_column):
        self.base_column = base_column

    def process_train_dataset(self, df, columns_info):
        return self.process(df, columns_info)
        
    def process_test_dataset(self, df, columns_info):
        return self.process(df, columns_info)
    
    def process(self, df, columns_info):
        for column in df.columns:
            if (column != self.base_column) and ("MONEY" in columns_info.get_labels(column)):
                new_column = f"{column}/{self.base_column}_fraction"
                df = df.with_columns((df[column] / df[self.base_column]).alias(new_column))
                columns_info.add_label(new_column, "MONEY_FRACTION")
        return df, columns_info
