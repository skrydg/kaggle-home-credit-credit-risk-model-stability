import polars as pl

from .dataset import Dataset
from pathlib import Path
from glob import glob


class DataLoader:
    def __init__(self, data_path = Path("/kaggle/input/home-credit-credit-risk-model-stability")):
        self.data_path = data_path
        self.train_dir = self.data_path / "parquet_files/train/"
        self.test_dir = self.data_path /  "parquet_files/test/"
        
    def load_train_dataset(self) -> Dataset:
        return Dataset(self._get_train_data())
        
    def load_test_dataset(self) -> Dataset:
        return Dataset(self._get_test_data())
    
    def _get_train_data(self):
        base = {
            "base": self._read_file(self.train_dir / "train_base.parquet")
        }
        depth_0 = {
            "static_cb_0": self._read_files(self.train_dir / "train_static_cb_*.parquet"),
            "static_0": self._read_files(self.train_dir / "train_static_0_*.parquet")
        }  
        depth_1 = {
            "applprev_1": self._read_files(self.train_dir / "train_applprev_1_*.parquet"),
            "tax_registry_a_1": self._read_file(self.train_dir / "train_tax_registry_a_1.parquet"),
            "tax_registry_b_1": self._read_file(self.train_dir / "train_tax_registry_b_1.parquet"),
            "tax_registry_c_1": self._read_file(self.train_dir / "train_tax_registry_c_1.parquet"),
            "credit_bureau_b_1": self._read_file(self.train_dir / "train_credit_bureau_b_1.parquet"),
            "other_1": self._read_file(self.train_dir / "train_other_1.parquet"),
            "person_1": self._read_file(self.train_dir / "train_person_1.parquet"),
            "deposit_1": self._read_file(self.train_dir / "train_deposit_1.parquet"),
            "debitcard_1": self._read_file(self.train_dir / "train_debitcard_1.parquet")
        }
        depth_2 = {
            "credit_bureau_b_2": self._read_file(self.train_dir / "train_credit_bureau_b_2.parquet"),
        }
        return {**base, **depth_0, **depth_1, **depth_2}
    
    def _get_test_data(self):
        base = {
            "base": self._read_file(self.test_dir / "test_base.parquet")
        }
        depth_0 = {
            "static_cb_0": self._read_files(self.test_dir / "test_static_cb_*.parquet"),
            "static_0": self._read_files(self.test_dir / "test_static_0_*.parquet")
        }
        depth_1 = {
            "applprev_1": self._read_files(self.test_dir / "test_applprev_1_*.parquet"),
            "tax_registry_a_1": self._read_files(self.test_dir / "test_tax_registry_a_1.parquet"),
            "tax_registry_b_1": self._read_file(self.test_dir / "test_tax_registry_b_1.parquet"),
            "tax_registry_c_1": self._read_file(self.test_dir / "test_tax_registry_c_1.parquet"),
            "credit_bureau_b_1": self._read_file(self.test_dir / "test_credit_bureau_b_1.parquet"),
            "other_1": self._read_file(self.test_dir / "test_other_1.parquet"),
            "person_1": self._read_file(self.test_dir / "test_person_1.parquet"),
            "deposit_1": self._read_file(self.test_dir / "test_deposit_1.parquet"),
            "debitcard_1": self._read_file(self.test_dir / "test_debitcard_1.parquet")
        }
        depth_2 = {
            "credit_bureau_b_2": self._read_file(self.test_dir / "test_credit_bureau_b_2.parquet"),
        }
        return {**base, **depth_0, **depth_1, **depth_2}
    
    def _read_file(self, path):
        return pl.read_parquet(path)
    

    def _read_files(self, regex_path):
        chunks = []
        for path in glob(str(regex_path)):
            chunks.append(self._read_file(path))

        return pl.concat(chunks, how="vertical_relaxed")