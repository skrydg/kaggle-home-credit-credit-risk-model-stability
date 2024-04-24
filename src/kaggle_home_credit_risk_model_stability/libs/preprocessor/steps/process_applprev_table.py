import numpy as np
import polars as pl


class ProcessApplprevTableStep:
    def __init__(self):
        self.table_name = "applprev_1"

    def process_train_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)
    
    def process_test_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)

    def process(self, dataset, columns_info):
        table = dataset.get_table(self.table_name)
        table = table.drop("actualdpd_943P")
        table = table.drop("credacc_maxhisbal_375A")

        table_col = table.filter(pl.col("credtype_587L") == "COL")
        table_cal = table.filter(pl.col("credtype_587L") == "CAL")
        table_rel = table.filter(pl.col("credtype_587L") == "REL")
        table_null = table.filter(pl.col("credtype_587L").is_null())

        dataset.set(f"applprev_col_1", table_col)
        dataset.set(f"applprev_cal_1", table_cal)
        dataset.set(f"applprev_rel_1", table_rel)
        dataset.set(f"applprev_null_1", table_null)
        dataset.delete(self.table_name)
        return dataset, columns_info
