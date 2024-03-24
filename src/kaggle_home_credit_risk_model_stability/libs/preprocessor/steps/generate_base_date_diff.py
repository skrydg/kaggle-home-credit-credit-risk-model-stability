import numpy as np
import polars as pl


class GenerateBaseDateDiffStep:
    def __init__(self, base_column):
        self.base_column = base_column

    def process_train_dataset(self, df_generator):
        for df, columns_info in df_generator:
            yield self.process(df, columns_info)
        
    def process_test_dataset(self, df_generator):
        for df, columns_info in df_generator:
            yield self.process(df, columns_info)
    
    def process(self, df, columns_info):
        dates_columns = [
            column 
            for column in df.columns 
            if ("DATE" in columns_info.get_labels(column)) and (column != self.base_column)
        ]
        for column in dates_columns:
            new_column_name = f"{column}_{self.base_column}_diff"
            df = df.with_columns((pl.col(column) - pl.col(self.base_column)).alias(new_column_name))
            columns_info.add_labels(new_column_name, {"DATE_DIFF"})
        print("Create {} new date diff columns, with base_column={}".format(len(dates_columns), self.base_column))
        return df, columns_info