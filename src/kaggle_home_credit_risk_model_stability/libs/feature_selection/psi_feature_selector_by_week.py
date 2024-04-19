import polars as pl
import gc
import os
import time
import numpy as np

from multiprocessing import Pool

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

class PsiFeatureSelectorByWeek:
    def __init__(self):
        self.treashold = 0.25

    def select(self, dataframe, features):
        count_weeks = dataframe["WEEK_NUM"].max() + 1

        dataframe = dataframe.fill_null(strategy="min")
        dataframe = dataframe.with_columns(*[
            pl.col(column).to_physical()
            for column in dataframe.columns if dataframe[column].dtype == pl.Enum
        ])

        n_cpu = os.cpu_count()
        batch_size = min(100, len(features) // n_cpu)

        cell_start = time.time()
        with Pool(n_cpu) as p:
            res = p.map(self.calc_psi_by_week, [dataframe[["WEEK_NUM"] + feature_batch] for feature_batch in chunker(features, batch_size)])

        psi_by_week = {}
        for psi_dict in res:
            psi_by_week.update(psi_dict)

        psi_by_week_df = pl.DataFrame({"WEEK_NUM": range(0, count_weeks)})
        psi_by_week_df = psi_by_week_df.with_columns(**{key: np.array(value) for key, value in psi_by_week.items()})


        low_psi_by_week_features = []
        high_psi_by_week_features = []
        for column in psi_by_week_df.drop("WEEK_NUM").max().columns:
            if psi_by_week_df.drop("WEEK_NUM").quantile(0.9)[column][0] < self.treashold:
                low_psi_by_week_features.append(column)
            else:
                high_psi_by_week_features.append(column)
        print(f"len(low_psi_by_week_features): {len(low_psi_by_week_features)}, len(high_psi_by_week_features): {len(high_psi_by_week_features)}")

        cell_finish = time.time()
        print(f"Finish, time: {cell_finish - cell_start}", flush=True)
        return low_psi_by_week_features, psi_by_week_df
    
    def calc_psi_by_week(self, dataframe):
        from feature_engine.selection import DropHighPSIFeatures

        count_weeks = dataframe["WEEK_NUM"].max() + 1
        psi_by_week = {feature: [] for feature in dataframe.columns if feature != "WEEK_NUM"}
        start = time.time()

        pandas_dataframe = dataframe.to_pandas()
        for week_num in range(0, count_weeks):
            weeks_in_train = list(range(0, count_weeks))
            weeks_in_train.remove(week_num)
            drop_features = DropHighPSIFeatures(
                cut_off = weeks_in_train,
                split_col = "WEEK_NUM",
                bins=10,
                missing_values="ignore"
            )
            drop_features.fit(pandas_dataframe)
            # assert(len(drop_features.psi_values_) == len(dataframe.columns) - 1)
            for feature in psi_by_week.keys():
                psi_for_feature = drop_features.psi_values_[feature] if feature in drop_features.psi_values_ else 0
                psi_by_week[feature].append(psi_for_feature)
            gc.collect()
        

        finish = time.time()
        print(f"Finish batch, time: {finish - start}", flush=True)
        gc.collect()
        return psi_by_week