import numpy as np
import polars as pl
import gc

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

class GenerateAnomalyFeatureStep:
    def __init__(self, quantile=0.95, threashold=1.5, use_w=True):
        self.quantile = quantile
        self.column_batch_size = 100
        self.threashold = threashold
        self.use_w = use_w
        self.positive_columns = []
        self.positive_columns_w = []
        self.negative_columns = []
        self.negative_columns_w = []

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

        print(f"Find split in {len(self.positive_columns)} columns for positive anomaly feature")
        print(f"Find split in {len(self.negative_columns)} columns for negative anomaly feature")

    def set_columns_for_batch(self, dataframe, columns_info):
        target = dataframe["target"]
        
        columns = dataframe.columns
        columns.remove("target")

        dataframe = dataframe.select(pl.col(columns).map_batches(lambda x: x.fill_null(value=x.median())))
        quantile_dataframe = dataframe.quantile(self.quantile)

        mask_dataframe = dataframe.select(pl.all().map_batches(lambda x: x < quantile_dataframe[x.name]))
        mean_mask_dataframe = mask_dataframe.select(pl.all().mean())
        columns_mask = mean_mask_dataframe.to_numpy()[0] > (self.quantile - 0.5)
        columns = np.array(mean_mask_dataframe.columns)[columns_mask].tolist()

        anomaly_dataframe = mask_dataframe.select(pl.col(columns).map_batches(
            lambda x: target.filter(~x).mean() / max(target.filter(x).mean(), 1e-6) if (target.filter(x).mean() is not None) else None
        ))
        anomaly_dataframe_np = anomaly_dataframe.to_numpy()[0]

        
        negative_columns_mask = anomaly_dataframe_np > self.threashold
        negative_columns = np.array(anomaly_dataframe.columns)[negative_columns_mask].tolist()
        self.negative_columns.extend(negative_columns)
        self.negative_columns_w.extend(anomaly_dataframe_np[negative_columns_mask].tolist())
        
        positive_columns_mask = anomaly_dataframe_np < 1. / self.threashold
        positive_columns = np.array(anomaly_dataframe.columns)[positive_columns_mask].tolist()
        self.positive_columns.extend(positive_columns)
        self.positive_columns_w.extend(anomaly_dataframe_np[positive_columns_mask].tolist())

        gc.collect()

    def process(self, dataframe, columns_info):
        dataframe, columns_info = self.process_impl(dataframe, columns_info, "positive", self.positive_columns, self.positive_columns_w)
        dataframe, columns_info = self.process_impl(dataframe, columns_info, "negative", self.negative_columns, self.negative_columns_w)
        return dataframe, columns_info

    def process_impl(self, dataframe, columns_info, name, columns, w):
        if (len(columns) == 0):
            return dataframe, columns_info
        w = np.array(w)
        
        filled_dataframe = dataframe.select(pl.col(columns).map_batches(lambda x: x.fill_null(value=x.median())))
        quantile_dataframe = filled_dataframe.quantile(self.quantile)
        mask_dataframe = filled_dataframe.select(pl.all().map_batches(lambda x: x >= quantile_dataframe[x.name]))

        if self.use_w:
            new_feature_name = f"weight_{name}_anomaly_feature_{self.quantile}_{self.threashold}"
            anomaly_feature = np.sum(mask_dataframe.to_numpy() * w[np.newaxis, :], axis=1)
        else:
            new_feature_name = f"{name}_anomaly_feature_{self.quantile}_{self.threashold}"
            anomaly_feature = np.sum(mask_dataframe.to_numpy(), axis=1)

        dataframe = dataframe.with_columns(pl.Series(anomaly_feature).alias(new_feature_name))
        columns_info.add_labels(new_feature_name, {"ANOMALY_FEATURE"})
        return dataframe, columns_info
