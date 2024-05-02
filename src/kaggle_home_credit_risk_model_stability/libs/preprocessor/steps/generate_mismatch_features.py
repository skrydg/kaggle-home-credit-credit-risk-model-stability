import numpy as np
import polars as pl

from collections import defaultdict
from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset


class GenerateMismatchFeaturesStep:
    def __init__(self):
        self.features = []
    
    def process_train_dataset(self, df_generator):
        df, columns_info = next(df_generator)
        
        raw_columns = sorted(list(set(columns_info.get_columns_with_label("RAW")) & set(df.columns)))
        equal_rate = defaultdict(lambda: defaultdict())
        diverse_columns = []
        for column in raw_columns:
            sum_2 = df[column].value_counts().sort(["count", column])[-2:]["count"].sum()
            if (sum_2 <= df.shape[0] * 0.99):
                diverse_columns.append(column)
                
        diverse_columns = sorted(diverse_columns)
        for column1 in diverse_columns:
            comparable_columns = [
                column for column in diverse_columns
                if (df[column1].dtype == df[column].dtype) and (column1 < column)
            ]
            for column2 in comparable_columns:
                if ((df[column1] == df[column2]).is_null().mean() <= 0.9):
                  equal_rate[column1][column2] = (df[column1] == df[column2]).mean()


        for column1 in equal_rate.keys():
            for column2 in equal_rate[column1]:
                if (equal_rate[column1][column2] is None):
                    continue

                if 0.9 <= equal_rate[column1][column2] < 1:
                    self.features.append((column1, column2))

        yield self.process(df, columns_info)
        
    def process_test_dataset(self, df_generator):
        df, columns_info = next(df_generator)
        yield self.process(df, columns_info)
    
    def process(self, df, columns_info):
        for feature1, feature2 in self.features:
            column_name = f"{feature1}_{feature2}_mismatch"
            df = df.with_columns((df[feature1] == df[feature2]).alias(column_name).cast(pl.Boolean))
            columns_info.add_label(column_name, "MISMATCH")

        print(f"Create {len(self.features)} new columns as feature mismatch")
        return df, columns_info