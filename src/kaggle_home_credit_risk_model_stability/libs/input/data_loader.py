import polars as pl

from .dataset import Dataset
from pathlib import Path
from glob import glob

from enum import Enum

class Mode(Enum):
    Predict = 0
    Train = 1

class DataLoader:
    def __init__(
            self, 
            data_path = Path("/kaggle/input/home-credit-credit-risk-model-stability"), 
            mode = Mode.Predict,
            train_persent_size = 0.5):
        self.data_path = data_path
        self.train_dir = self.data_path / "parquet_files/train/"
        self.test_dir = self.data_path /  "parquet_files/test/"
        self.mode = mode
        self.train_persent_size = train_persent_size
        
    def load_train_dataset(self) -> Dataset:
      train_data = self._get_train_data()
      if self.mode == Mode.Train:
        case_id_set = self._get_train_case_id_set()
        return Dataset(train_data).filter(lambda df: df.filter(df["case_id"].is_in(case_id_set)))
      else:
        return Dataset(train_data)
        
    def load_test_dataset(self) -> Dataset:
        if self.mode == Mode.Train:
            case_id_set = self._get_test_case_id_set()
            return Dataset(self._get_train_data()).filter(lambda df: df.filter(df["case_id"].is_in(case_id_set)))
        else:
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
    
    def _get_train_case_id_set(self):
        case_id_info = pl.read_parquet(self.train_dir / "train_base.parquet", columns=["case_id", "WEEK_NUM"])
        
        min_week_id = case_id_info["WEEK_NUM"].min()
        max_week_id = case_id_info["WEEK_NUM"].max()
        week_id_threashold = min_week_id + int((max_week_id - min_week_id) * self.train_persent_size)
        case_id_info = case_id_info.filter(case_id_info["WEEK_NUM"] <= week_id_threashold)
        return case_id_info["case_id"]
    
    def _get_test_case_id_set(self):
        case_id_info = pl.read_parquet(self.train_dir / "train_base.parquet", columns=["case_id", "WEEK_NUM"])
        
        min_week_id = case_id_info["WEEK_NUM"].min()
        max_week_id = case_id_info["WEEK_NUM"].max()
        week_id_threashold = min_week_id + int((max_week_id - min_week_id) * self.train_persent_size)
        case_id_info = case_id_info.filter(case_id_info["WEEK_NUM"] > week_id_threashold)
        return case_id_info["case_id"]