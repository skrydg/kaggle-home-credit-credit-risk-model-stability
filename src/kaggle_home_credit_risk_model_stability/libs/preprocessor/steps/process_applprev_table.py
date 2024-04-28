import numpy as np
import polars as pl


class ProcessApplprevTableStep:
    def __init__(self):
        self.table_name = "applprev_1"
        self.service_columns = ["case_id"]
        self.credit_types = {
            "col": {
                "filter": pl.col("credtype_587L") == "COL"
            },
            "cal": {
                "filter": pl.col("credtype_587L") == "CAL"
            },
            "rel": {
                "filter": pl.col("credtype_587L") == "REL"
            },
            "null": {
                "filter": pl.col("credtype_587L").is_null()
            }
        }
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

        for credit_type in self.credit_types:
            filter = self.credit_types[credit_type]["filter"]
            credit_type_table = table.filter(filter)
            table_name = f"{credit_type}_{self.table_name}"

            columns = credit_type_table.columns
            columns = [column for column in columns if column not in self.service_columns]

            for column in columns:
                new_column_name = f"{column}_{table_name}"
                labels = columns_info.get_labels(column)
                columns_info.add_labels(new_column_name, labels)
            
            credit_type_table = credit_type_table.rename({column: f"{column}_{table_name}" for column in columns})
            dataset.set(table_name, credit_type_table)

        dataset.set(self.table_name, table)
        return dataset, columns_info
