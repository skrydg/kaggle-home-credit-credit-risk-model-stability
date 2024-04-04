import gc
import polars as pl
import numpy as np

class RawChunkedTableInfo:
    def __init__(self, table_path):
        self.table_path = table_path
        table = pl.read_parquet(table_path, low_memory=True)
        self.columns = table.columns
        self.dtypes = table[self.columns].dtypes
        self.column_to_unique_values = {
            column: table[column].filter(table[column].is_not_null()).unique() for column in self.columns 
            if (table[column].dtype == pl.String) or (column == "case_id")
        }
        self.column_to_min = {}
        for column in self.columns:
            if (table[column].shape[0] == 0) or (table[column].is_null().mean() == 1.):
                self.column_to_min[column] = 0
            else:
                self.column_to_min[column] = table[column].min()

        self.column_to_max = {}
        for column in self.columns:
            if (table[column].shape[0] == 0) or (table[column].is_null().mean() == 1.):
                self.column_to_max[column] = 0
            else:
                self.column_to_max[column] = table[column].max()
        
        self.value_counts = {
            column: table[column].fill_null("__NULL__").value_counts() for column in self.columns
            if (table[column].dtype == pl.String)
        }

        del table
        gc.collect()

    def get_columns(self):
        return self.columns

    def get_dtype(self, column):
        return self.dtypes[self.columns.index(column)]
        
    def get_min_value(self, column):
        return self.column_to_min[column]
    
    def get_max_value(self, column):
        return self.column_to_max[column]

    def get_unique_values(self, column):
        return self.column_to_unique_values[column].to_numpy().tolist()
    
    def get_value_counts(self, column):
        return self.value_counts[column]
    
class RawTableInfo:
    def __init__(self, table_name, table_paths):
        self.table_name = table_name
        self.table_paths = table_paths
        self.chunked_table_info = {}
        for table_path in table_paths:
            self.chunked_table_info[table_path] = RawChunkedTableInfo(table_path)

    def get_table_name(self):
        return self.table_name
        
    def get_chunk_info(self, path):
        return self.chunked_table_info[path]
        
    def get_columns(self):
        return list(self.chunked_table_info.values())[0].get_columns()

    def get_dtype(self, column):
        return list(self.chunked_table_info.values())[0].get_dtype(column)
        
    def get_min_value(self, column):
        return min([chunk.get_min_value(column) for chunk in self.chunked_table_info.values()])
    
    def get_max_value(self, column):
        return max([chunk.get_max_value(column) for chunk in self.chunked_table_info.values()])

    def get_unique_values(self, column):
        return np.unique(np.concatenate([chunk.get_unique_values(column) for chunk in self.chunked_table_info.values()])).tolist()
    
    def get_value_counts(self, column):
        table = pl.concat([chunk.get_value_counts(column) for chunk in self.chunked_table_info.values()], how="vertical_relaxed")
        return table.group_by(column).sum()
