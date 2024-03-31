import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class CreateDayFeatureStep:
    def process_train_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
          yield self.process(dataset, columns_info)
        
    def process_test_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
          yield self.process(dataset, columns_info)
    
    def process(self, dataset, columns_info):
        base = dataset.get_base()
        base = base.with_columns(
            month_decision = pl.col("date_decision").cast(pl.Date).dt.month(),
            weekday_decision = pl.col("date_decision").cast(pl.Date).dt.weekday(),
        )
        dataset.set("base", base)
        return dataset, columns_info
