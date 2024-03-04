import numpy as np
import polars as pl


class ProcessPersonTable:        
    def process_train_dataset(self, dataset):
        return self.process(dataset)
        
    def process_test_dataset(self, dataset):
        return self.process(dataset)
    
    def process(self, dataset):
        table = dataset.get_table("person_1")
    
        customer_info_table = table.filter((pl.col("num_group1") == 0))
        
        guarantors = table.filter((pl.col("num_group1") != 0))
        guarantors_relation = guarantors[["case_id", "relationshiptoclient_642T", "relationshiptoclient_415T"]].to_dummies(["relationshiptoclient_642T", "relationshiptoclient_415T"]).group_by("case_id").sum()
        
        customer_info_table = customer_info_table.join(guarantors_relation, on="case_id")
        dataset.set("customer_info_table_0", customer_info_table)
        
        dataset.set("guarantors_1", guarantors)
        dataset.delete("person_1")
        return dataset
