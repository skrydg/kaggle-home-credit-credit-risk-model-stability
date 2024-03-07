import lightgbm as lgb
import shutil
import os
import polars as pl

from pathlib import Path
from glob import glob

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

class LightGbmDatasetSerializer:
    def __init__(self, directory, dataset_params):
        self.directory = Path(directory)
        self.dataset_params = dataset_params

        self.chunk_size = 100
        
    def serialize(self, X, Y):
        shutil.rmtree(self.directory, ignore_errors=True)
        os.makedirs(self.directory, exist_ok=True)
        columns = X.columns
        for chunk_index, columns_chunk in enumerate(chunker(columns, self.chunk_size)):
            self.serialize_impl(
                self.directory / f"data_{chunk_index}.bin",
                X[columns_chunk],
                Y
            )
    
    def deserialize(self):
        size = len(glob(str(self.directory / "data_*.bin")))
        datasets = []
        for i in range(size):
            path = self.directory / f"data_{i}.bin"
            current_dataset = lgb.Dataset(
                path, 
                params=self.dataset_params, 
                free_raw_data=True
            ).construct()
            datasets.append(current_dataset)

        for i in range(1, size):
            datasets[0].add_features_from(datasets[i])

        return datasets[0]
    
    def serialize_impl(self, file, X, Y):
        data = lgb.Dataset(
            X.to_pandas(), 
            Y.to_pandas(),
            params=self.dataset_params,
            free_raw_data=True
        )
        data.save_binary(file)