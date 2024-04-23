import numpy as np
import polars as pl


class ProcessTaxRegestryA1TableStep:
    def __init__(self):
        self.table_name = "tax_registry_a_1"

    def process_train_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)
    
    def process_test_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)

    def process(self, dataset, columns_info):
        table = dataset.get_table(self.table_name)
        dataset.set(self.table_name, table.drop("name_4527232M"))
        return dataset, columns_info
