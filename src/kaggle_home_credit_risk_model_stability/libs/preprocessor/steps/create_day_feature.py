import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class CreateDayFeaturesStep:
    def process_train_dataset(self, dataset, columns_info):
        return self.process(dataset, columns_info)
        
    def process_test_dataset(self, dataset, columns_info):
        return self.process(dataset, columns_info)
    
    def process(self, dataset, columns_info):
        base = dataset.get_base_table()
        base.with_features(
            month_decision = pl.col("date_decision").dt.month(),
            weekday_decision = pl.col("date_decision").dt.weekday(),
        )
        dataset.set("base", base)
        return dataset, columns_info
