import numpy as np
import polars as pl


class RenameFinancialInstitutionForCreditBureauA1TableStep:
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

        for column in ["financialinstitution_382M", "financialinstitution_591M"]:
          dtype = table[column].dtype
          table = table.with_columns(table[column].cast(pl.String).replace({
              "7e4feb1b": "66b2baaa",
              "cc2c2610": "55b002a9",
              "9325d851": "0d39f5db",
              "b619fa46": "P204_66_73",
              "71de340a": "cb830fec",
              "952e9882": "P40_25_35"
          }).cast(dtype))

        dataset.set(self.table_name, table)
        return dataset, columns_info
    