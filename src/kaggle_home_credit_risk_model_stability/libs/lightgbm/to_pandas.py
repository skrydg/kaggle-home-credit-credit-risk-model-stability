import polars as pl
import numpy as np

def to_pandas(dataframe):
    column_dtypes = {}
    for column in dataframe.columns:
        if dataframe[column].dtype in [pl.Float32, pl.Float64]:
            column_dtypes[column] = np.float64
        elif dataframe[column].dtype in [pl.Int8]:
            column_dtypes[column] = "Int8"
        elif dataframe[column].dtype in [pl.Int16]:
            column_dtypes[column] = "Int16"
        elif dataframe[column].dtype in [pl.Int32]:
            column_dtypes[column] = "Int32"
        elif dataframe[column].dtype in [pl.Int64]:
            column_dtypes[column] = "Int64"
        elif dataframe[column].dtype in [pl.UInt8]:
            column_dtypes[column] = "UInt8"
        elif dataframe[column].dtype in [pl.UInt16]:
            column_dtypes[column] = "UInt16"
        elif dataframe[column].dtype in [pl.UInt32]:
            column_dtypes[column] = "UInt32"
        elif dataframe[column].dtype in [pl.UInt64]:
            column_dtypes[column] = "UInt64"
        elif dataframe[column].dtype in [pl.Boolean]:
            column_dtypes[column] = bool
        elif dataframe[column].dtype in [pl.Enum, pl.Categorical]:
            column_dtypes[column] = "category"
        else:
            raise Exception(f"Unknown column dtype: {dataframe[column].dtype}, column: {column}")

    pandas_dataframe = dataframe.to_pandas()
    for column in dataframe.columns:
        pandas_dataframe[column] = pandas_dataframe[column].astype(column_dtypes[column])
    return pandas_dataframe