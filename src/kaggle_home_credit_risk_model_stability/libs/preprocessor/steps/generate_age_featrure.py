import numpy as np
import polars as pl

class GenerateAgeFeatureStep:
    def process_train_dataset(self, dataframe_generator):
        for dataframe, columns_info in dataframe_generator:
            yield self._process(dataframe, columns_info)
        
    def process_test_dataset(self, dataframe_generator):
        for dataframe, columns_info in dataframe_generator:
            yield self._process(dataframe, columns_info)
            
    def _process(self, dataframe, columns_info):
        dataframe = dataframe.with_columns(
            ((dataframe["date_decision"] - dataframe["birth_259D"]) // 365).alias("age")
        )
        dataframe = dataframe.with_columns(
            (dataframe["age"] // 10).alias("age_bucket").cast(pl.String).cast(pl.Enum([str(i) for i in range(0, 10)]))
        )
        return dataframe, columns_info
