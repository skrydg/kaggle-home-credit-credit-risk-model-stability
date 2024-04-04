import numpy as np
import polars as pl


class ProcessPersonTableStep:
    def __init__(self):
        self.table_name = "person_1"
    
    def process_train_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)
    
    def process_test_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)

    def process(self, dataset, columns_info):
        table = dataset.get_table(self.table_name)
    
        customer_info_table = table.filter((pl.col("num_group1") == 0))
        dataset.set("customer_info_table_0", customer_info_table)
        
        guarantors = table.filter((pl.col("num_group1") != 0))
        dataset.set("guarantors_1", guarantors)

        dataset.delete("person_1")
        return dataset, columns_info
