import numpy as np
import polars as pl
import gc

    
class FillNullStep:        
    def process_train_dataset(self, dataframe, columns_info):
        return self.process(dataframe, columns_info)
        
    def process_test_dataset(self, dataframe, columns_info):
        return self.process(dataframe, columns_info)
    
    def process(self, dataframe, columns_info):
        for column in dataframe.columns:
          if dataframe[column].dtype == pl.Categorical:
              dataframe = dataframe.with_columns(dataframe[column].cast(pl.String).fill_null("__UNKNOWN__").cast(pl.Categorical))
        return dataframe, columns_info
