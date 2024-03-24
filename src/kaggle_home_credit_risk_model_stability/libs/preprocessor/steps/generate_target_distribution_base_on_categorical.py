import numpy as np
import polars as pl
import gc


class GenerateTargetDistributionBasedOnCategoricalStep:
    def __init__(self, min_observation_count = 1000):
        self.feature_to_target_distribution_dataframe = {}
        self.min_observation_count = min_observation_count
        
    def process_train_dataset(self, dataframe_generator):
        dataframe, columns_info = next(dataframe_generator)
        for column in dataframe.columns:
            if dataframe[column].dtype != pl.Enum:
                continue
                
            value_counts = dataframe[column].value_counts()
            frequent_value_counts = value_counts.filter(pl.col("count") > self.min_observation_count)
            non_frequent_value_counts = value_counts.filter(pl.col("count") <= self.min_observation_count)
            
            frequent_values = list(frequent_value_counts[column].to_numpy())
            non_frequent_values = list(non_frequent_value_counts[column].to_numpy())
            
            column_with_target = dataframe[[column, "target"]]
            column_with_target = column_with_target.with_columns(column_with_target[column].cast(pl.String()))

            column_with_target = column_with_target.with_columns(
                column_with_target[column].set(
                    column_with_target[column].is_in(non_frequent_values),
                    "__other__"
                )
            )
            
            target_distribution = column_with_target.group_by(column).agg([
                pl.col("target").mean().alias(f"{column}_target_distribution"), 
                pl.col("target").count().alias("count")])


            target_distribution = target_distribution.with_columns(
                (pl.col("count") / pl.col("count").sum()).alias(f"{column}_value_persent")
            )
            target_distribution = target_distribution.drop("count")
            
            if (non_frequent_value_counts["count"].sum() <= self.min_observation_count):
                other_target_distribution = dataframe["target"].mean()
                other_value_persent = 0.
            else:
                other_target_distribution = target_distribution.filter(pl.col(column) == "__other__")[f"{column}_target_distribution"][0]
                other_value_persent = target_distribution.filter(pl.col(column) == "__other__")[f"{column}_value_persent"][0]
            
            current_df = pl.DataFrame({
                column: frequent_values + non_frequent_values,
            })
            
            current_df = current_df.join(target_distribution, on=column, how="left")
            current_df = current_df.with_columns(
                current_df[f"{column}_target_distribution"].fill_null(other_target_distribution),
                current_df[f"{column}_value_persent"].fill_null(other_value_persent),
                current_df[column].cast(dataframe[column].dtype)
            )
#            print(current_df)
            self.feature_to_target_distribution_dataframe[column] = current_df
                
        yield self.process(dataframe, columns_info)
        
    def process_test_dataset(self, dataframe_generator):
        dataframe, columns_info = next(dataframe_generator)
        yield self.process(dataframe, columns_info)
    
    def process(self, dataframe, columns_info):
        count_new_features = 0
        for column, df in self.feature_to_target_distribution_dataframe.items():
            dataframe = dataframe.join(df, on=column, how="left")
            new_columns = df.columns
            new_columns.remove(column)
            for new_column in new_columns:
                columns_info.add_labels(new_column, {"TARGET_BASE"})
                count_new_features = count_new_features + 1
                
        print(f"Create {count_new_features} as target distribution by categorical feature")
        return dataframe, columns_info
