import numpy as np
import polars as pl


class ProcessCreditBureaua1TableStep:
    def __init__(self):
        self.table_name = "credit_bureau_a_1"
    
    def process_train_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)
    
    def process_test_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)

    def process(self, dataset, columns_info):
        table = dataset.get_table(self.table_name)
        mask = table["dateofcredstart_181D"].is_not_null() & table["dateofcredend_353D"].is_not_null()
    
        dataset.set(self.table_name, table.filter(mask))
        return dataset, columns_info
