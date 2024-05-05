import numpy as np
import polars as pl


class SplitTableByCategoricalFeatureStep:
    def __init__(self, table_name, column, column_values):
        self.table_name = table_name
        self.column = column
        self.column_values = column_values

    def process_train_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)
    
    def process_test_dataset(self, dataset_generator):
        for dataset, columns_info in dataset_generator:
            yield self.process(dataset, columns_info)

    def process(self, dataset, columns_info):
        table = dataset.get_table(self.table_name)
        for column_value in self.column_values:
            mask = (table[self.column].is_in(column_value))

            table_prefix = column_value[0].replace(" ", "_")
            new_table_name = f"{table_prefix}_{self.table_name}"
            new_table = table.filter(mask)

            columns = [c for c in new_table.columns if "SERVICE" not in columns_info.get_labels(c)]
            for column in columns:
                new_column_name = f"{column}_{new_table_name}"
                labels = columns_info.get_labels(column)
                columns_info.add_labels(new_column_name, labels)

            new_table = new_table.rename({column: f"{column}_{new_table_name}" for column in columns})

            dataset.set(new_table_name, new_table)
        dataset.delete(self.table_name)
        return dataset, columns_info

