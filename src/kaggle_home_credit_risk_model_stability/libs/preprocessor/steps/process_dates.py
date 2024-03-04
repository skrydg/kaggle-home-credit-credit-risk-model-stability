import numpy as np
import polars as pl


class ProcessDatesStep:        
    def process_train_dataset(self, df, columns_info):
        return self.process(df, columns_info)
        
    def process_test_dataset(self, df, columns_info):
        return self.process(df, columns_info)
    
    def process(self, df, columns_info):
        for column in df.columns:
            if (df[column].dtype == pl.Date) and (column != "date_decision"):
                df = df.with_columns(pl.col(column) - pl.col("date_decision"))
                df = df.with_columns(pl.col(column).dt.total_days())
        return df, columns_info