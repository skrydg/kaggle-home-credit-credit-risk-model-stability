import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class Aggregator:
    num_aggregators = [pl.max, pl.min, pl.first, pl.last, pl.mean]
    str_aggregators = [pl.max, pl.min, pl.first, pl.last] # n_unique
    group_aggregators = [pl.max, pl.min, pl.first, pl.last]
    
    @staticmethod
    def num_expr(df):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]
        expr_all = []
        for method in Aggregator.num_aggregators:
            expr = [method(col).alias(f"{method.__name__}_{col}") for col in cols]
            expr_all += expr

        return expr_all

    @staticmethod
    def date_expr(df):
        cols = [col for col in df.columns if col[-1] in ("D",)]
        expr_all = []
        for method in Aggregator.num_aggregators:
            expr = [method(col).alias(f"{method.__name__}_{col}") for col in cols]  
            expr_all += expr

        return expr_all

    @staticmethod
    def str_expr(df):
        cols = [col for col in df.columns if col[-1] in ("M",)]
        
        expr_all = []
        for method in Aggregator.str_aggregators:
            expr = [method(col).alias(f"{method.__name__}_{col}") for col in cols]  
            expr_all += expr
            
        expr_mode = [
            pl.col(col)
            .drop_nulls()
            .mode()
            .first()
            .alias(f"mode_{col}")
            for col in cols
        ]

        return expr_all + expr_mode

    @staticmethod
    def other_expr(df):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        
        expr_all = []
        for method in Aggregator.str_aggregators:
            expr = [method(col).alias(f"{method.__name__}_{col}") for col in cols]  
            expr_all += expr

        return expr_all
    
    @staticmethod
    def count_expr(df):
        cols = [col for col in df.columns if "num_group" in col]

        expr_all = []
        for method in Aggregator.group_aggregators:
            expr = [method(col).alias(f"{method.__name__}_{col}") for col in cols]  
            expr_all += expr
            
#         if len(cols) > 0:
#             method = pl.count
#             expr = [method(col).alias(f"{method.__name__}_{col}") for col in [cols[0]]]
#             expr_all += expr

        return expr_all

    @staticmethod
    def get_exprs(df):
        exprs = Aggregator.num_expr(df) + \
                Aggregator.date_expr(df) + \
                Aggregator.str_expr(df) + \
                Aggregator.other_expr(df) + \
                Aggregator.count_expr(df)

        return exprs
    

class AggregateDepthTableStep:        
    def process_train_dataset(self, train_dataset):
        return self.process(train_dataset)
        
    def process_test_dataset(self, test_dataset):
        return self.process(test_dataset)
    
    def process(self, dataset):
        assert(type(dataset) is Dataset)
        for i in range(len(dataset.depth_1)):
            dataset.depth_1[i] = dataset.depth_1[i].group_by("case_id").agg(Aggregator.get_exprs(dataset.depth_1[i]))
        for i in range(len(dataset.depth_2)):
            dataset.depth_2[i] = dataset.depth_2[i].group_by("case_id").agg(Aggregator.get_exprs(dataset.depth_2[i]))
        return dataset
