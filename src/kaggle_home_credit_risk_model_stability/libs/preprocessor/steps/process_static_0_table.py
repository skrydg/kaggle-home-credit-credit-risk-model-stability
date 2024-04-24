import numpy as np
import polars as pl


class ProcessStatic0TableStep:
    def __init__(self):
        self.table_name = "static_0"

    def process_train_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)
    
    def process_test_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)

    def get_credit_duration(self, table):
        inittransactionamount_650A = table["inittransactionamount_650A"].fill_null(0)
        annuitynextmonth_57A = table["annuitynextmonth_57A"].fill_null(0)
        annuity_780A = table["annuity_780A"]
        credamount_770A = table["credamount_770A"]
        interest_rate = table["interestrate_311L"].fill_null(0)
        duration = ((credamount_770A * (1 + interest_rate) - inittransactionamount_650A - annuitynextmonth_57A) / annuity_780A + 1) * 30
        mean_duration = duration.mean()
        inf_mask = duration.is_infinite() | duration.is_nan()
        duration = duration.set(inf_mask, mean_duration)
        return duration
    
    def process(self, dataset, columns_info):
        table = dataset.get_table(self.table_name)

        table = table.with_columns(self.get_credit_duration(table).alias("credit_duration"))

        dataset.set(self.table_name, table)
        return dataset, columns_info
