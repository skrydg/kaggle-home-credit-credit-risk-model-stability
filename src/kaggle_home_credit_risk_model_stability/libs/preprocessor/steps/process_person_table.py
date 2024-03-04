import numpy as np
import polars as pl


class ProcessPersonTableStep:
    def process_train_dataset(self, dataset):
        return self.process(dataset)
        
    def process_test_dataset(self, dataset):
        return self.process(dataset)
    
    def process(self, dataset):
        table = dataset.get_table("person_1")
    
        customer_info_table = table.filter((pl.col("num_group1") == 0))        
        dataset.set("customer_info_table_0", customer_info_table)
        
        guarantors = table.filter((pl.col("num_group1") != 0))
        dataset.set("guarantors_1", guarantors)
        
        dataset.delete("person_1")
        return dataset
