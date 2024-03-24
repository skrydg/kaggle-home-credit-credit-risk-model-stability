import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

class DropAlmostNullFeaturesWithRespectToTargetStep:
    def __init__(self, threashold = 3.):
        self.columns = []
        self.threashold = threashold
        
    def process_train_dataset(self, dataset, columns_info):
        for name, table in dataset.get_tables():
            self._fill_columns_to_drop(table, dataset.get_base(), columns_info)
            
        print("Drop {} columns as almost null".format(len(self.columns)))
        print(f"Columns to drop: {self.columns}")
        return self._process(dataset, columns_info)
        
    def process_test_dataset(self, dataset, columns_info):
        return self._process(dataset, columns_info)
    
    def _fill_columns_to_drop(self, table, base, columns_info):
        table_with_target = base.join(table, how="left", on="case_id")
        
        for column in table.columns:
            if "SERVICE" in columns_info.get_labels(column):
                continue
                
            if table_with_target[column].shape[0] == 0:
                self.columns.append(column)
            elif (table_with_target[column].n_unique() <= 1):
                self.columns.append(column)
            else:
                isnull = table_with_target[column].is_null().mean()
                if isnull < 0.95:
                    continue
                if isnull > 0.9999:
                    self.columns.append(column)
                    continue
                    
                null_target = table_with_target.filter(pl.col(column).is_null())["target"].mean()
                not_null_target = max(1e-7, table_with_target.filter(pl.col(column).is_not_null())["target"].mean())

                if ((1 / self.threashold) < (null_target / not_null_target) < (self.threashold)):
                    self.columns.append(column)

        
    def _process(self, dataset, columns_info):
        for name, table in dataset.get_tables():
            table = table.drop(self.columns)
            dataset.set(name, table)
        return dataset, columns_info