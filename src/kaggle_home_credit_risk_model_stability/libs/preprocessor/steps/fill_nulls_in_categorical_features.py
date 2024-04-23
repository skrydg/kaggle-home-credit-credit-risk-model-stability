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
            if dataframe[column].dtype == pl.Enum:
                column_type = dataframe[column].dtype
                dataframe = dataframe.with_columns(dataframe[column].cast(pl.String).fill_null("__NULL__").cast(column_type))
        return dataframe, columns_info
