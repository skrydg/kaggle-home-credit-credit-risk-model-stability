import polars as pl
import numpy as np
from collections import defaultdict

class ReduceDimentionForCategoricalFeaturesStep:
    def __init__(self):
        self.non_significant_treashold = 0.001
        self.feature_to_values = {}

    def process_train_dataset(self, dataframe_generator):
        dataset, columns_info = next(dataframe_generator)
        self.set_values(columns_info)
        yield self.process(dataset, columns_info)

        for dataset, columns_info in dataframe_generator:
            yield self.process(dataset, columns_info)

    def process_test_dataset(self, dataframe_generator):
        for dataset, columns_info in dataframe_generator:
            yield self.process(dataset, columns_info)

    def process(self, dataset, columns_info):
        raw_tables_info = columns_info.get_raw_tables_info()
        for table_name, table in dataset.get_tables():
              for feature in table.columns:
                  if "CATEGORICAL" in columns_info.get_labels(feature):
                      table = table.with_columns(
                          table[feature]
                              .cast(pl.String)
                              .set(~table[feature].is_in(self.feature_to_values[feature]), "__OTHER__")
                              .cast(pl.Enum(self.feature_to_values[feature]))
                      )
              dataset.set_table(table_name, table)
        return dataset, columns_info

    def set_values(self, columns_info):
        raw_tables_info = columns_info.get_raw_tables_info()
        for table_name in raw_tables_info:
            for feature in raw_tables_info.get_columns():
                if "CATEGORICAL" in columns_info.get_labels(feature):
                    value_counts = raw_tables_info[table_name].get_value_counts(feature)
                    total_count = value_counts.sum()["count"][0]
                    threashold = self.non_significant_treashold * total_count
                    values = value_counts.filter(pl.col("count") > threashold)[feature].unique().to_numpy().tolist()
                    self.feature_to_values[feature] = values + ["__OTHER__"]
