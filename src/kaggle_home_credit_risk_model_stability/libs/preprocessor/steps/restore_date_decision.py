import numpy as np
import polars as pl

from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset
from kaggle_home_credit_risk_model_stability.libs.date_decision_restorer.date_decision_restorer_by_overdue_amount import DateDecisionRestorerByOverdueAmount

class RestoreDateDecisionStep:
    def __init__(self, env):
       self.env = env

    def process_train_dataset(self, dataset_generator):
        restorer = DateDecisionRestorerByOverdueAmount(self.env, is_test=False)
        date_decision_table = restorer.restore()

        for dataset, columns_info in dataset_generator:
          yield self.process(dataset, columns_info, date_decision_table)
        
    def process_test_dataset(self, dataset_generator):
        restorer = DateDecisionRestorerByOverdueAmount(self.env, is_test=True)
        date_decision_table = restorer.restore()

        for dataset, columns_info in dataset_generator:
          yield self.process(dataset, columns_info, date_decision_table)
    
    def process(self, dataset, columns_info, date_decision_table):
        base = dataset.get_base()

        date_decision_table = date_decision_table.with_columns(pl.col("date_decision").cast(pl.Date).alias("real_date_decision"))
        base = base.join(date_decision_table, how="left", on="case_id")
        base = base.with_columns((pl.col("real_date_decision") - pl.col("date_decision").cast(pl.Date)).dt.total_days().fill_null(0).alias("date_decision_diff"))
        
        for table_name, table in dataset.get_tables():
           table = self.process_table(table, columns_info, base[["case_id", "date_decision_diff"]])
           dataset.set(table_name, table)
        return dataset, columns_info
    
    def process_table(self, table, columns_info, date_decision_diff_table):
        table = table.join(date_decision_diff_table, how="left", on="case_id")
        for column in table.columns:
            if ("DATE" in columns_info.get_labels(column)) or (column == "date_decision"):
                table = table.with_columns((table[column] + table["date_decision_diff"]))
        table = table.drop("date_decision_diff")
        return table
