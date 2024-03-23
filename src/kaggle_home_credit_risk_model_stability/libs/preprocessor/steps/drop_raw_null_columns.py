import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class DropRawNullColumns:
    def __init__(self, feature_threashold = 0.05, week_threashold = 0.5):
        self.feature_threashold = feature_threashold
        self.week_threashold = week_threashold
        self.columns_to_drop = []
        
    def process_train_dataset(self, dataset, columns_info):
        self._fill_columns_to_drop(dataset, columns_info)
            
        print("Drop {} columns as null".format(len(self.columns_to_drop)))
        return self._process(dataset, columns_info)
        
    def process_test_dataset(self, dataset, columns_info):
        return self._process(dataset, columns_info)
    
    def _fill_columns_to_drop(self, dataset, columns_info):
        base = dataset.get_base()
        count_weeks = base["WEEK_NUM"].n_unique();
        total_cases_by_week = base["WEEK_NUM"].value_counts().sort("count").rename({"count": "total_cases"})
        total_cases_by_week

        feature_to_count_week_num = {}
        for name, table in dataset.get_depth_tables([0, 1, 2]):
            base_joined_table = table.join(dataset.get_base(), how="left", on="case_id")
            for column in table.columns:
                if "SERVICE" in columns_info.get_labels(column):
                    continue
                null_mask = table[column].is_not_null()
                vc = base_joined_table[["case_id", "WEEK_NUM", column]].filter(null_mask).unique(subset=["case_id"])["WEEK_NUM"].value_counts().sort("count")
                vc = vc.join(total_cases_by_week, how="left", on="WEEK_NUM")
                vc = vc.with_columns((pl.col("count") / pl.col("total_cases")).alias("persent_not_null_cases")).sort("persent_not_null_cases")
                
                assert((vc.shape[0] == 0) or (vc["persent_not_null_cases"].max() <= 1.))
                hight_null_count_mask = vc["persent_not_null_cases"] > self.feature_threashold
                vc = vc.filter(hight_null_count_mask)
                count_week_num = vc["WEEK_NUM"].n_unique()
                feature_to_count_week_num[column] = count_week_num
        self.columns_to_drop = [column for column in feature_to_count_week_num.keys() if feature_to_count_week_num[column] < count_weeks * self.week_threashold]
        
    def _process(self, dataset, columns_info):
        for name, table in dataset.get_depth_tables([0, 1, 2]):
            columns_to_drop = list(set(table.columns) & set(self.columns_to_drop))
            table = table.drop(columns_to_drop)
            dataset.set(name, table)
        return dataset, columns_info
