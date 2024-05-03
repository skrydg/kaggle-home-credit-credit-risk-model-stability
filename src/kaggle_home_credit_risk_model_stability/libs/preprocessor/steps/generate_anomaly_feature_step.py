import numpy as np
import polars as pl
import gc

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

class GenerateAnomalyFeatureStep:
    def __init__(self, quantile=0.95, threashold=1.5):
        self.quantile = quantile
        self.column_batch_size = 100
        self.threashold = threashold
        self.columns = []

    def process_train_dataset(self, df_generator):
        df, columns_info = next(df_generator)
        self.set_columns(df, columns_info)
        yield self.process(df, columns_info)
        
    def process_test_dataset(self, df_generator):
        df, columns_info = next(df_generator)
        yield self.process(df, columns_info)
    
    def set_columns(self, dataframe, columns_info):
        numerical_features = [c for c in dataframe.columns if dataframe[c].dtype != pl.Enum]
        numerical_features = [c for c in numerical_features if "SERVICE" not in columns_info.get_labels(c)]
        numerical_features = [c for c in numerical_features if "DATE" not in columns_info.get_labels(c)]
        
        for column_batch in chunker(numerical_features, self.column_batch_size):
            self.set_columns_for_batch(dataframe[column_batch + ["target"]], columns_info)

        print(f"Find split in {len(self.columns)} columns for anomaly feature")

    def set_columns_for_batch(self, dataframe, columns_info):
        target = dataframe["target"]
        
        columns = dataframe.columns
        columns.remove("target")

        dataframe = dataframe.select(pl.col(columns).map_batches(lambda x: x.fill_null(value=x.median())))
        quantile_95_dataframe = dataframe.quantile(self.quantile)

        mask_dataframe = dataframe.select(pl.all().map_batches(lambda x: x < quantile_95_dataframe[x.name]))
        mean_mask_dataframe = mask_dataframe.select(pl.all().mean())
        columns_mask = mean_mask_dataframe.to_numpy()[0] > (self.quantile - 0.5)
        columns = np.array(mean_mask_dataframe.columns)[columns_mask].tolist()

        anomaly_dataframe = mask_dataframe.select(pl.col(columns).map_batches(
            lambda x: target.filter(~x).mean() / target.filter(x).mean() if target.filter(x).mean() is not None else None
        ))

        columns_mask = anomaly_dataframe.to_numpy()[0] > self.threashold
        columns = np.array(anomaly_dataframe.columns)[columns_mask].tolist()
        
        self.columns.extend(columns)
        gc.collect()

    def process(self, dataframe, columns_info):
        filled_dataframe = dataframe.select(pl.col(self.columns).map_batches(lambda x: x.fill_null(value=x.median())))
        quantile_95_dataframe = filled_dataframe.quantile(self.quantile)
        mask_dataframe = filled_dataframe.select(pl.all().map_batches(lambda x: x >= quantile_95_dataframe[x.name]))
        anomaly_feature = np.sum(mask_dataframe.to_numpy(), axis=1)

        dataframe = dataframe.with_columns(pl.Series(anomaly_feature).alias("anomaly_feature"))

        return dataframe, columns_info
