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
        
        table_col = table.filter(pl.col("credtype_587L") == "COL")
        table_cal = table.filter(pl.col("credtype_587L") == "CAL")
        table_rel = table.filter(pl.col("credtype_587L") == "REL")
        table_null = table.filter(pl.col("credtype_587L").is_null())

        dataset.set_table(f"{self.table_name}_col", table_col)
        dataset.set_table(f"{self.table_name}_cal", table_cal)
        dataset.set_table(f"{self.table_name}_rel", table_rel)
        dataset.set_table(f"{self.table_name}_null", table_null)
        dataset.delete_table(self.table_name)
        return dataset, columns_info
