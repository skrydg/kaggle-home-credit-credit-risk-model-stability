import lightgbm as lgb
import shutil
import os
import polars as pl

from pathlib import Path
from glob import glob

from kaggle_home_credit_risk_model_stability.libs.lightgbm.to_pandas import to_pandas

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

class LightGbmDatasetSerializer:
    def __init__(self, directory, dataset_params):
        self.directory = Path(directory)
        self.dataset_params = dataset_params
        
    def serialize(self, X, Y):
        chunk_size = 1000
        shutil.rmtree(self.directory, ignore_errors=True)
        os.makedirs(self.directory, exist_ok=True)
        columns = X.columns
        for chunk_index, columns_chunk in enumerate(chunker(columns, chunk_size)):
            self.serialize_impl(
                self.directory / f"data_{chunk_index}.bin",
                X[columns_chunk],
                Y,
            )
    
    def deserialize(self):
        size = len(glob(str(self.directory / "data_*.bin")))
        datasets = []
        for i in range(size):
            path = self.directory / f"data_{i}.bin"
            current_dataset = lgb.Dataset(
                path, 
                params=self.dataset_params
            )
            current_dataset.construct()
            datasets.append(current_dataset)

        dataset = datasets[0]
        for i in range(1, size):
            dataset.add_features_from(datasets[i])
        
        return dataset
    
    def serialize_impl(self, file, X, Y):
        categorical_features = [feature for feature in X.columns if X[feature].dtype == pl.Enum]

        physical_X = X.with_columns(*[
            pl.col(column).to_physical()
            for column in categorical_features
        ])

        data = lgb.Dataset(
            physical_X.to_numpy(),
            Y.to_numpy(),
            params=self.dataset_params,
            categorical_feature=categorical_features,
            feature_name=physical_X.columns,
            free_raw_data=False
        )
        data.save_binary(file)


    def clear(self):
        shutil.rmtree(self.directory, ignore_errors=True)