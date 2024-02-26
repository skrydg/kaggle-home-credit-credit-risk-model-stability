import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class Aggregator:
    num_aggregators = [pl.max, pl.min, pl.first, pl.last, pl.mean]
    enum_aggregators = [pl.first, pl.last, pl.n_unique]
    
    @staticmethod
    def num_expr(df):
        columns = [column for column in df.columns if type(column) is not pl.Enum]
        expr_all = []
        for method in Aggregator.num_aggregators:
            expr = [method(column).alias(f"{method.__name__}_{column}") for column in columns]
            expr_all += expr

        return expr_all

    @staticmethod
    def enum_expr(df):
        columns = [column for column in df.columns if type(column) is pl.Enum]
        
        expr_all = []
        for method in Aggregator.enum_aggregators:
            expr = [method(column).alias(f"{method.__name__}_{column}") for column in columns]  
            expr_all += expr
          
        return expr_all


    @staticmethod
    def get_exprs(df):
        return Aggregator.num_expr(df) + Aggregator.enum_expr(df)
    

class AggregateDepthTableStep:        
    def process_train_dataset(self, train_dataset):
        return self.process(train_dataset)
        
    def process_test_dataset(self, test_dataset):
        return self.process(test_dataset)
    
    def process(self, dataset):
        assert(type(dataset) is Dataset)
        
        for name, table in dataset.get_depth_tables([1, 2]):
            dataset.set(name, table.group_by("case_id").agg(Aggregator.get_exprs(table)))

        return dataset
