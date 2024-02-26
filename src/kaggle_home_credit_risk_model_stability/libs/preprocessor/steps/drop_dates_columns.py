import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset


class DropDatesColumnsStep:            
    def process_train_dataset(self, df):  
        return self.process(df)
        
    def process_test_dataset(self, df):
        return self.process(df)
    
    def process(self, df):
        for column in df.columns:
            if (column[-1] == 'D'):
                df = df.drop(column)
        return df