import polars as pl
import numpy as np
import gc

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
class PsiFeatureSelector:
    def __init__(self, threashold = 0.25, batch_size = 50):
        self.threashold = threashold
        self.batch_size = batch_size

    def fill_nulls(self, dataframe):
        dataframe = dataframe.fill_null(strategy="min")
        dataframe = dataframe.with_columns(*[
            pl.col(column).to_physical()
            for column in dataframe.columns if dataframe[column].dtype == pl.Enum
        ])
        return dataframe

    def calc_psi(self, dataframe, week_to_split):
        from feature_engine.selection import DropHighPSIFeatures

        pandas_dataframe = dataframe.to_pandas()
        drop_features = DropHighPSIFeatures(
            cut_off = week_to_split,
            split_col = "WEEK_NUM",
            bins=10,
            missing_values="ignore"
        )
        drop_features.fit(pandas_dataframe)
        return drop_features.psi_values_

    def select_for_batch(self, train_dataframe, test_dataframe, features):
        features = features + ["WEEK_NUM"]
        features = list(set(sorted(features)))
        train_dataframe = train_dataframe[features]
        test_dataframe = test_dataframe[features]
        max_train_week = train_dataframe["WEEK_NUM"].max()
        min_test_week = test_dataframe["WEEK_NUM"].min()
        assert max_train_week < min_test_week
        dataframe = pl.concat([train_dataframe, test_dataframe], how="vertical_relaxed")
        dataframe = self.fill_nulls(dataframe)

        psi = self.calc_psi(dataframe, max_train_week)
        low_psi_features = [feature for feature in psi.keys() if psi[feature] < self.threashold]
        high_psi_features = [feature for feature in psi.keys() if psi[feature] >= self.threashold]
        return low_psi_features

    def select(self, train_dataframe_path, test_dataframe_path, features):
        selected_features = []
        for feature_chunk in chunker(features, self.batch_size):
            train_dataframe = pl.read_parquet(train_dataframe_path, columns=feature_chunk + ["WEEK_NUM"])
            test_dataframe = pl.read_parquet(test_dataframe_path, columns=feature_chunk + ["WEEK_NUM"])
            
            selected_features.extend(self.select_for_batch(train_dataframe, test_dataframe, feature_chunk))
            del train_dataframe
            del test_dataframe
            gc.collect()
        return selected_features