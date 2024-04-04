import polars as pl
import numpy as np
from collections import defaultdict

class SplitCompositeFeaturesStep:
    def __init__(self):
        self.table_to_composite_features = defaultdict(lambda: list())

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
        for table_name, composite_features in self.table_to_composite_features.items():
            table = dataset.get_table(table_name)
            for feature in composite_features:
                for part in range(3):
                    new_feature_name = f"{feature}_part_{part}"
                    table = table.with_columns(
                        table[feature].cast(pl.String).str.split(by="_").list.get(part).alias(new_feature_name)
                    )
                    unique_values = np.unique(table[new_feature_name].unique().to_numpy().tolist() + ["__UNKNOWN__", "__NULL__", "__OTHER__"])
                    table = table.with_columns(
                        pl.col(new_feature_name).cast(pl.String).fill_null("__NULL__").cast(pl.Enum(unique_values))
                    )

            dataset.set_table(table_name, table)
        return dataset, columns_info

    def set_composite_features(self, raw_tables_info):
        for table_name, feature in self.get_composite_feature(raw_tables_info):
            self.table_to_composite_features[table_name].append(feature)

    def get_composite_feature(self, raw_tables_info):
        for table_name, raw_table_info in raw_tables_info.items():
            for column in raw_table_info.get_columns():
                if raw_table_info.get_dtype(column) == pl.String:
                    uv = raw_table_info.get_unique_values(column)
                    is_composite_value = np.array([("_" in v) and (v[0] == "P") for v in uv])
                    if np.mean(is_composite_value) > 0.5:
                        yield table_name, column
