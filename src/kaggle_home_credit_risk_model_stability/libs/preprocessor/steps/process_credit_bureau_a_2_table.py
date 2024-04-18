import numpy as np
import polars as pl


class ProcessCreditBureaua2TableStep:
    def __init__(self):
        self.table_name = "credit_bureau_a_2"
        self.active_contracts_columns = ["collater_typofvalofguarant_298M", "collater_valueofguarantee_1124L", "collaterals_typeofguarante_669M", "pmts_dpd_1073P", "pmts_month_158T", "pmts_overdue_1140A", "pmts_year_1139T", "subjectroles_name_838M"]
        self.close_contracts_columns = ["collater_typofvalofguarant_407M", "collater_valueofguarantee_876L", "collaterals_typeofguarante_359M", "pmts_dpd_303P", "pmts_month_706T", "pmts_overdue_1152A", "pmts_year_507T", "subjectroles_name_541M"]
        self.terminated_contracts_columns = ["pmts_dpd_303P", "pmts_month_706T", "pmts_overdue_1152A", "pmts_year_507T", "subjectroles_name_541M"]
        self.existed_contracts_columns = ["pmts_dpd_1073P", "pmts_month_158T", "pmts_overdue_1140A", "pmts_year_1139T", "subjectroles_name_838M"]
        self.service_columns = ["case_id", "num_group1", "num_group2"]

    def process_train_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)
    
    def process_test_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)

    def process(self, dataset, columns_info):
        table = dataset.get_table(self.table_name)

        # active contracts
        column = "collater_valueofguarantee_1124L"
        mask = table[column].is_not_null()
        assert(column in self.active_contracts_columns)
        dataset.set("credit_bureau_a_active_contracts_2", table.filter(mask)[self.active_contracts_columns + self.service_columns])

        # close contracts
        column = "collater_valueofguarantee_876L"
        mask = table[column].is_not_null()
        assert(column in self.close_contracts_columns)
        dataset.set("credit_bureau_a_close_contracts_2", table.filter(mask)[self.close_contracts_columns + self.service_columns])

        # existed contracts
        column = "pmts_overdue_1140A"
        mask = table[column].is_not_null()
        assert(column in self.existed_contracts_columns)
        dataset.set("credit_bureau_a_existed_contracts_2", table.filter(mask)[self.existed_contracts_columns + self.service_columns])

        # terminated contracts
        column = "pmts_overdue_1152A"
        mask = table[column].is_not_null()
        assert(column in self.terminated_contracts_columns)
        dataset.set("credit_bureau_a_terminated_contracts_2", table.filter(mask)[self.terminated_contracts_columns + self.service_columns])

        dataset.delete(self.table_name)
        return dataset, columns_info
