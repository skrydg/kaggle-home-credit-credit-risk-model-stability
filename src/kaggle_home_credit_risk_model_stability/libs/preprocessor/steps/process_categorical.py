import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class ProcessCategoricalStep:    
    def __init__(self):
        self.column_to_type = {}
        
    def process_train_dataset(self, df):
        for column in df.columns:
            if df[column].dtype == pl.String:
                unique_values = list(df[column].filter(~df[column].is_null()).unique())
                self.column_to_type[column] = pl.Enum(unique_values + ["__UNKNOWN__"])
            
        return self.process(df)
        
    def process_test_dataset(self, df):
        return self.process(df)
    
    def process(self, df):
        for column in df.columns:
            if df[column].dtype == pl.String:
                column_type = self.column_to_type[column]
                df = df.with_columns(df[column].set(~df[column].is_in(column_type.categories), "__UNKNOWN__"))
                df = df.with_columns(df[column].fill_null("__UNKNOWN__").cast(column_type))
        return df