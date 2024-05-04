import numpy as np
import polars as pl


class SplitTableByCreditDurationStep:
    def __init__(self, table_name, intervals):
        self.table_name = table_name
        self.intervals = intervals

        self.credit_duration_column = "credit_duration"

    def process_train_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)
    
    def process_test_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)

    def process(self, dataset, columns_info):
        table = dataset.get_table(self.table_name)
        for interval_name, interval in self.intervals.items():
            mask = (interval[0] <= table[self.credit_duration_column]) & (table[self.credit_duration_column] < interval[1])
            new_table_name = f"{interval_name}_{self.table_name}"
            dataset.set(new_table_name, table.filter(mask))

        dataset.delete(self.table_name)
        return dataset, columns_info

