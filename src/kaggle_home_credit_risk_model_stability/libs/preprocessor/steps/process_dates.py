import numpy as np
import polars as pl


class ProcessDatesStep:        
    def process_train_dataset(self, df):
        return self.process(df)
        
    def process_test_dataset(self, df):
        return self.process(df)
    
    def process(self, df):
        for column in df.columns:
            if (df[column].dtype == pl.Date) and (column != "date_decision"):
                df = df.with_columns(pl.col(column) - pl.col("date_decision"))
                df = df.with_columns(pl.col(column).dt.total_days())
        return df