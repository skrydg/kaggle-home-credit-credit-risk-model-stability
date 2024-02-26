import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class DropColumnsStep:
    def __init__(self):
        self.columns = []
        
    def process_train_dataset(self, df):
        self.
        for column in df.columns:
            isnull = df[column].is_null().mean()
            if isnull > 0.95:
                self.columns.append(column)

        for column in df.columns:
            if df[column].dtype == pl.Enum:
                freq = df[column].n_unique()

                if (freq == 1) or (freq > 200):
                    self.columns.append(column)

        self.columns.append("date_decision")
        self.columns.append("MONTH")
                
        print("Columns to drop: {}".format(self.columns))            
        return self.process(df)
        
    def process_test_dataset(self, df):
        return self.process(df)
    
    def _process(self, df):
        for column in self.columns:
            df = df.drop(column)
        return df