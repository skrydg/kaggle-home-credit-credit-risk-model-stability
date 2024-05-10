import numpy as np
import polars as pl
import gc

class TruncateTableByDurationStep:
    def __init__(self, table_name, duration, date_column):
        self.table_name = table_name
        self.duration = duration
        self.date_column = date_column

    def process_train_dataset(self, dataset_generator):
        for dataset, columm_info in dataset_generator:
            yield self.process(dataset, columm_info)
    
    def process_test_dataset(self, dataset_generator):
        for dataset, columm_info in dataset_generator:
            yield self.process(dataset, columm_info)

    def process(self, dataset, columns_info):
        base = dataset.get_base()
        table = dataset.get_table(self.table_name)

        joined_table = table[["case_id", self.date_column]].join(base[["case_id", "date_decision"]], how="left", on="case_id")
        joined_table = joined_table.with_columns((joined_table["date_decision"] - joined_table[self.date_column]).alias("duration"))
        mask = (joined_table["duration"] <= self.duration)

        new_table = table.filter(mask)
        new_table_name = f"truncated_{self.duration}_{self.table_name}"
        dataset.set(new_table_name, new_table)
        gc.collect()
        return dataset, columns_info


