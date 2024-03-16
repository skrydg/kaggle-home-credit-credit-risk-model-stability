import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class Aggregator:
    num_aggregators = [pl.max, pl.min, pl.first, pl.last, pl.mean, pl.std, pl.var, pl.count]
    enum_to_num_aggregators = [pl.n_unique, pl.count]

    @staticmethod
    def num_expr(df, columns_info):
        columns = [
            column for column in df.columns 
            if (df[column].dtype != pl.Enum) and ("SERVICE" not in columns_info.get_labels(column))
        ]
        expr_all = []
        for method in Aggregator.num_aggregators:
            expr = [method(column).alias(f"{method.__name__}_{column}") for column in columns]
            expr_all += expr
            
            for column in columns:
                labels = columns_info.get_labels(column)
                if "RAW" in labels:
                    labels.remove("RAW")
                columns_info.add_labels(f"{method.__name__}_{column}", labels)

        return expr_all

    @staticmethod
    def enum_expr(df, columns_info):
        columns = [column for column in df.columns if df[column].dtype == pl.Enum]
        
        expr_all = []
        for column in columns:
            filtered_column = pl.col(column).filter(pl.col(column).cast(pl.String) != "__UNKNOWN__")
            expr_all.extend([
                filtered_column.max().alias(f"max_{column}"),
                filtered_column.min().alias(f"min_{column}"),
                filtered_column.first().alias(f"first_{column}"),
                filtered_column.last().alias(f"last_{column}"),
                filtered_column.mode().max().alias(f"mode_{column}")
            ])

            labels = columns_info.get_labels(column)
            if "RAW" in labels:
                labels.remove("RAW")
            columns_info.add_labels(f"max_{column}", labels)
            columns_info.add_labels(f"min_{column}", labels)
            columns_info.add_labels(f"first_{column}", labels)
            columns_info.add_labels(f"last_{column}", labels)
            columns_info.add_labels(f"mode_{column}", labels)
        
        for method in Aggregator.enum_to_num_aggregators:
            expr = [method(column).alias(f"{method.__name__}_{column}") for column in columns]
            expr_all += expr

        return expr_all


    @staticmethod
    def get_exprs(df, columns_info):
        return Aggregator.num_expr(df, columns_info) + Aggregator.enum_expr(df, columns_info)
    

class AggregateDepthTableStep:        
    def process_train_dataset(self, train_dataset, columns_info):
        return self.process(train_dataset, columns_info)
        
    def process_test_dataset(self, test_dataset, columns_info):
        return self.process(test_dataset, columns_info)
    
    def process(self, dataset, columns_info):
        assert(type(dataset) is Dataset)
        
        for name, table in dataset.get_depth_tables([1, 2]):
            dataset.set(name, table.group_by("case_id").agg(Aggregator.get_exprs(table, columns_info)).sort("case_id"))

        return dataset, columns_info
