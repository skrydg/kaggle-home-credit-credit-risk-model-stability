import polars as pl
import numpy as np
from collections import defaultdict

class DropCompositeFeaturesStep:
    def __init__(self):
        self.composite_features = []

    def process_train_dataset(self, dataframe_generator):
        dataset, columns_info = next(dataframe_generator)
        self.set_composite_features(columns_info.get_raw_tables_info())
        yield self.process(dataset, columns_info)

        for dataset, columns_info in dataframe_generator:
            yield self.process(dataset, columns_info)

    def process_test_dataset(self, dataframe_generator):
        for dataset, columns_info in dataframe_generator:
            yield self.process(dataset, columns_info)

    def process(self, dataset, columns_info):
        for table_name, table in dataset.get_tables():
            feature_to_drop = list(set(self.composite_features) & set(table.columns))
            table = table.drop(feature_to_drop)
            dataset.set(table_name, table)
        return dataset, columns_info

    def set_composite_features(self, raw_tables_info):
        self.composite_features = []
        for table_name, feature in self.get_composite_feature(raw_tables_info):
            self.composite_features.append(feature)
        print(f"Drop composite features {self.composite_features}")

    def get_composite_feature(self, raw_tables_info):
        for table_name, raw_table_info in raw_tables_info.items():
            for column in raw_table_info.get_columns():
                if raw_table_info.get_dtype(column) == pl.String:
                    uv = raw_table_info.get_unique_values(column)
                    is_composite_value = np.array([("_" in v) and (v[0] == "P") for v in uv])
                    if np.mean(is_composite_value) > 0.5:
                        yield table_name, column
