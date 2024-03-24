import numpy as np
import polars as pl
import gc

class FillNullsInCategoricalFeaturesStep:        
    def process_train_dataset(self, dataframe_generator):
        for dataframe, columns_info in dataframe_generator:
            yield self.process(dataframe, columns_info)
        
    def process_test_dataset(self, dataframe_generator):
        for dataframe, columns_info in dataframe_generator:
            yield self.process(dataframe, columns_info)
    
    def process(self, dataframe, columns_info):
        for column in dataframe.columns:
            if "CATEGORICAL" in columns_info.get_labels(column):
                column_type = dataframe[column].dtype
                dataframe = dataframe.with_columns(dataframe[column].cast(pl.String).fill_null("__UNKNOWN__").cast(column_type))
        return dataframe, columns_info
