import numpy as np
import polars as pl
import gc

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class Aggregator:
    num_aggregators = [pl.max, pl.min, pl.mean]
    enum_to_num_aggregators = [pl.n_unique, pl.count]

    @staticmethod
    def num_expr(table_name, df, columns_info):
        columns = [
            column for column in df.columns 
            if (df[column].dtype != pl.Enum) and ("SERVICE" not in columns_info.get_labels(column))
        ]
        expr_all = []
        for method in Aggregator.num_aggregators:
            for column in columns:
                new_column = f"{method.__name__}_{column}"
                expr_all.append(method(column).alias(new_column))

                labels = columns_info.get_labels(column)
                if "RAW" in labels:
                    labels.remove("RAW")
                columns_info.set_ancestor(new_column, column)
                columns_info.add_labels(new_column, labels)

        # for column in columns:
        #     new_column = f"std_{column}"
        #     expr_all.append(pl.std(column).alias(new_column))
        #     columns_info.set_ancestor(new_column, column)

        return expr_all

    @staticmethod
    def enum_expr(table_name, df, columns_info):
        columns = [column for column in df.columns if df[column].dtype == pl.Enum]
        
        expr_all = []
        for column in columns:
            filtered_column = pl.col(column).filter(~pl.col(column).cast(pl.String).is_in(["__UNKNOWN__", "__NULL__", "__OTHER__"]))
            expr_all.extend([
                filtered_column.cast(pl.String).max().alias(f"max_{column}").cast(df[column].dtype),
                filtered_column.cast(pl.String).min().alias(f"min_{column}").cast(df[column].dtype),
                filtered_column.cast(pl.String).mode().max().alias(f"mode_{column}").cast(df[column].dtype)
            ])

            labels = columns_info.get_labels(column)
            if "RAW" in labels:
                labels.remove("RAW")

            columns_info.add_labels(f"max_{column}", labels)
            columns_info.add_labels(f"min_{column}", labels)
            columns_info.add_labels(f"mode_{column}", labels)
            
            columns_info.set_ancestor(f"max_{column}", column)
            columns_info.set_ancestor(f"min_{column}", column)
            columns_info.set_ancestor(f"mode_{column}", column)
            
        
        for method in Aggregator.enum_to_num_aggregators:
            for column in columns:
                new_column = f"{method.__name__}_{column}"
                expr_all.append(method(column).alias(new_column))
                columns_info.set_ancestor(new_column, column)

        return expr_all

    @staticmethod
    def num_group_expr(table_name, df, columns_info):
        columns = [column for column in df.columns if column in ["num_group1", "num_group2"]]
        expr = []
        for column in columns:
            expr.extend([
                pl.col(column).max().alias(f"max_{column}_{table_name}"),
                pl.col(column).min().alias(f"min_{column}_{table_name}"),
                pl.col(column).count().alias(f"count_{column}_{table_name}")
            ])
            columns_info.set_ancestor(f"max_{column}_{table_name}", column)
            columns_info.set_ancestor(f"min_{column}_{table_name}", column)
            columns_info.set_ancestor(f"count_{column}_{table_name}", column)
        return expr

    @staticmethod
    def get_exprs(table_name, df, columns_info):
        return Aggregator.num_expr(table_name, df, columns_info) + \
            Aggregator.enum_expr(table_name, df, columns_info) + \
            Aggregator.num_group_expr(table_name, df, columns_info)
    

class AggregateDepthTableStep:
    def process_train_dataset(self, train_dataset_generator):
        for train_dataset, columns_info in train_dataset_generator:
            yield self.process(train_dataset, columns_info)
        
    def process_test_dataset(self, test_dataset_generator):
        for test_dataset, columns_info in test_dataset_generator:
            yield self.process(test_dataset, columns_info)
    
    def process(self, dataset, columns_info):
        count_columns = 0
        for name, table in dataset.get_depth_tables([1, 2]):
            expr = Aggregator.get_exprs(name, table, columns_info)
            dataset.set(name, table.group_by("case_id").agg(expr).sort("case_id"))
            count_columns += len(expr)
            gc.collect()
        print(f"Generate {count_columns} columns as aggregates")

        return dataset, columns_info
