import numpy as np
import polars as pl
import gc
from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset

    
class ReduceMemoryUsageForDataFrameStep:     
    def process_train_dataset(self, df_generator):  
        df, columns_info = next(df_generator)
        yield self.process(df, columns_info)
        
    def process_test_dataset(self, df_generator):
        df, columns_info = next(df_generator)
        yield self.process(df, columns_info)
    
    def process(self, df, columns_info):
        column_to_type = {}
        for column in df.columns:
            column_type = df[column].dtype

            if (column_type != pl.Enum) and \
              (column_type != pl.String) and \
              (column_type != pl.Categorical) and \
              ("SERVICE" not in columns_info.get_labels(column)):
                c_min = df[column].min()
                c_max = df[column].max()
                if (c_min is None) or (c_max is None):
                    column_to_type[column] = column_type
                    continue
                    
                if str(column_type)[:3] == 'Int':
                    if c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        column_to_type[column] = pl.Int16
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        column_to_type[column] = pl.Int32
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        column_to_type[column] = pl.Int64
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        column_to_type[column] = pl.Float32
                    else:
                        column_to_type[column] = pl.Float64
            else:
                column_to_type[column] = column_type
        for column in df.columns:
            column_type = column_to_type[column]
            df = df.with_columns(df[column].cast(column_type))
        gc.collect()
        return df, columns_info