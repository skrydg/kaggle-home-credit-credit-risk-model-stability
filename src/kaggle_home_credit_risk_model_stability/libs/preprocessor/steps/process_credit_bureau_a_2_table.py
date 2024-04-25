import numpy as np
import polars as pl


class ProcessCreditBureaua2TableStep:
    def __init__(self):
        self.table_name = "credit_bureau_a_2"
        self.config = {
            "active": {
                "filter_column": "collater_valueofguarantee_1124L",
                "columns": ["collater_typofvalofguarant_298M", "collater_valueofguarantee_1124L", "collaterals_typeofguarante_669M", "pmts_dpd_1073P", "pmts_month_158T", "pmts_overdue_1140A", "subjectroles_name_838M"]
            },
            "close": {
                "filter_column": "collater_valueofguarantee_876L",
                "columns": ["collater_typofvalofguarant_407M", "collater_valueofguarantee_876L", "collaterals_typeofguarante_359M", "pmts_dpd_303P", "pmts_month_706T", "pmts_overdue_1152A", "subjectroles_name_541M"]
            },
            "terminated": {
                "filter_column": "pmts_overdue_1152A",
                "columns": ["pmts_dpd_303P", "pmts_month_706T", "pmts_overdue_1152A", "subjectroles_name_541M"]
            },
            "existed": {
                "filter_column": "pmts_overdue_1140A",
                "columns": ["pmts_dpd_1073P", "pmts_month_158T", "pmts_overdue_1140A", "subjectroles_name_838M"]
            }
        }
        
        self.service_columns = ["case_id", "num_group1", "num_group2"]

    def process_train_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)
    
    def process_test_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)

    def process(self, dataset, columns_info):
        table = dataset.get_table(self.table_name)

        for contract_type in self.config:
            columns = self.config[contract_type]["columns"]
            filter_column = self.config[contract_type]["filter_column"]
            mask = table[filter_column].is_not_null()
            assert(filter_column in columns)
            table_name = f"{contract_type}_{self.table_name}"

            contract_table = table.filter(mask)[columns + self.service_columns]
            for column in columns:
                new_column_name = f"{column}_{table_name}"
                labels = columns_info.get_labels(column)
                columns_info.add_labels(new_column_name, labels)

            contract_table = contract_table.rename({column: f"{column}_{table_name}" for column in columns})
            dataset.set(table_name, contract_table)

        dataset.delete(self.table_name)
        return dataset, columns_info
