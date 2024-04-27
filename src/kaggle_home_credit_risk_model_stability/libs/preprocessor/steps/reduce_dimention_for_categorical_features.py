import polars as pl
import numpy as np

from collections import defaultdict

class ReduceDimentionForCategoricalFeaturesStep:
    def __init__(self):
        self.non_significant_treashold = 0.001
        self.feature_to_values = {}

    def process_train_dataset(self, dataframe_generator):
        dataframe, columns_info = next(dataframe_generator)
        self.set_values(dataframe)
        yield self.process(dataframe, columns_info)

    def process_test_dataset(self, dataframe_generator):
        for dataframe, columns_info in dataframe_generator:
            yield self.process(dataframe, columns_info)

    def process(self, table, columns_info):
        for feature in table.columns:
            if (table[feature].dtype == pl.Enum):
                table = table.with_columns(
                    table[feature]
                        .cast(pl.String)
                        .set(~table[feature].is_in(self.feature_to_values[feature]), "__OTHER__")
                        .cast(pl.Enum(self.feature_to_values[feature]))
                    )
        return table, columns_info

    def set_values(self, dataframe):
        for feature in dataframe.columns:
            if (dataframe[feature].dtype == pl.Enum):
                value_counts = dataframe[feature].value_counts()
                total_count = value_counts.sum()["count"][0]
                threashold = self.non_significant_treashold * total_count
                values = value_counts.filter(pl.col("count") > threashold)[feature].unique().to_numpy().tolist()
                values = sorted(np.unique(values + ["__OTHER__", "__NULL__", "__UNKNOWN__"]))
                self.feature_to_values[feature] = values
