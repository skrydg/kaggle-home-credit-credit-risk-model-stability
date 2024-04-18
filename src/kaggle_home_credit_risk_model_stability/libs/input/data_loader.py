import polars as pl
import gc

from pathlib import Path
from glob import glob

from enum import Enum

from kaggle_home_credit_risk_model_stability.libs.env import Env
from kaggle_home_credit_risk_model_stability.libs.input.dataset import Dataset
from kaggle_home_credit_risk_model_stability.libs.input.raw_table_info import RawTableInfo

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


class DataLoader:
    def __init__(
            self, 
            env: Env, 
            tables = ["base", "static_cb_0", "static_0", 
                      "applprev_1", "tax_registry_a_1", "tax_registry_b_1", "tax_registry_c_1",
                      "credit_bureau_a_1", "credit_bureau_b_1", "other_1", "person_1", "deposit_1", 
                      "debitcard_1", "credit_bureau_a_2", "credit_bureau_b_2"]
    ):
        self.tables = tables
        self.data_path = env.input_directory
        self.train_dir = self.data_path / "home-credit-credit-risk-model-stability" / "parquet_files/train/"
        self.test_dir = self.data_path /  "home-credit-credit-risk-model-stability" / "parquet_files/test/"
      
    def load_train_base_table(self):
        return pl.read_parquet(self.train_dir / "train_base.parquet")
    
    def load_test_base_table(self):
        return pl.read_parquet(self.test_dir / "test_base.parquet")
    
    def load_train_dataset(self, chunk_size=300000, count_rows=None):
        train_data_paths = self._get_train_data_paths()
        raw_tables_info = self.get_raw_tables_info(train_data_paths)

        unique_case_ids = sorted(raw_tables_info["base"].get_unique_values("case_id"))
        count_rows = count_rows or len(unique_case_ids)
        unique_case_ids = unique_case_ids[:count_rows]

        for case_id_chunk in chunker(unique_case_ids, chunk_size):
            l_case_id = min(case_id_chunk)
            r_case_id = max(case_id_chunk)
            yield self.create_dataset(train_data_paths, raw_tables_info, (l_case_id, r_case_id + 1)), raw_tables_info
        
    def load_test_dataset(self, chunk_size=300000, count_rows=None):
        test_data_paths = self._get_test_data_paths()
        raw_tables_info = self.get_raw_tables_info(test_data_paths)
        
        unique_case_ids = sorted(raw_tables_info["base"].get_unique_values("case_id"))
        count_rows = count_rows or len(unique_case_ids)
        unique_case_ids = unique_case_ids[:count_rows]
        
        for case_id_chunk in chunker(unique_case_ids, chunk_size):
            l_case_id = min(case_id_chunk)
            r_case_id = max(case_id_chunk)
            yield self.create_dataset(test_data_paths, raw_tables_info, (l_case_id, r_case_id + 1)), raw_tables_info
    
    def create_dataset(self, data_paths, raw_tables_info, case_id_seqment) -> Dataset:
        tables = {}
        for table_name in self.tables:
            table_paths = data_paths[table_name]
            table = self._read_files(table_paths, raw_tables_info[table_name], case_id_seqment)
            tables[table_name] = table
            gc.collect()
        return Dataset(tables)

    def get_raw_tables_info(self, data_paths):
        raw_tables_info = {}
        for table_name in self.tables:
            table_paths = data_paths[table_name]
            raw_table_info = RawTableInfo(
                table_name,
                table_paths
            )
            raw_tables_info[table_name] = raw_table_info

        return raw_tables_info
        
    def _get_train_data_paths(self):
        base = {
            "base": glob(str(self.train_dir / "train_base.parquet"))
        }
        depth_0 = {
            "static_cb_0": glob(str(self.train_dir / "train_static_cb_*.parquet")),
            "static_0": glob(str(self.train_dir / "train_static_0_*.parquet"))
        }  
        depth_1 = {
            "applprev_1": glob(str(self.train_dir / "train_applprev_1_*.parquet")),
            "tax_registry_a_1": glob(str(self.train_dir / "train_tax_registry_a_1.parquet")),
            "tax_registry_b_1": glob(str(self.train_dir / "train_tax_registry_b_1.parquet")),
            "tax_registry_c_1": glob(str(self.train_dir / "train_tax_registry_c_1.parquet")),
            "credit_bureau_a_1": glob(str(self.train_dir / "train_credit_bureau_a_1_*.parquet")),
            "credit_bureau_b_1": glob(str(self.train_dir / "train_credit_bureau_b_1.parquet")),
            "other_1": glob(str(self.train_dir / "train_other_1.parquet")),
            "person_1": glob(str(self.train_dir / "train_person_1.parquet")),
            "deposit_1": glob(str(self.train_dir / "train_deposit_1.parquet")),
            "debitcard_1": glob(str(self.train_dir / "train_debitcard_1.parquet"))
        }
        depth_2 = {
            "credit_bureau_a_2": glob(str(self.train_dir / "train_credit_bureau_a_2_*.parquet")),
            "credit_bureau_b_2": glob(str(self.train_dir / "train_credit_bureau_b_2.parquet")),
            "applprev_2":  glob(str(self.train_dir / "train_applprev_2.parquet")),
            "person_2":  glob(str(self.train_dir / "train_person_2.parquet")),
        }
        return {**base, **depth_0, **depth_1, **depth_2}
    
    def _get_test_data_paths(self):
        base = {
            "base": glob(str(self.test_dir / "test_base.parquet"))
        }
        depth_0 = {
            "static_cb_0": glob(str(self.test_dir / "test_static_cb_*.parquet")),
            "static_0": glob(str(self.test_dir / "test_static_0_*.parquet"))
        }
        depth_1 = {
            "applprev_1": glob(str(self.test_dir / "test_applprev_1_*.parquet")),
            "tax_registry_a_1": glob(str(self.test_dir / "test_tax_registry_a_1.parquet")),
            "tax_registry_b_1": glob(str(self.test_dir / "test_tax_registry_b_1.parquet")),
            "tax_registry_c_1": glob(str(self.test_dir / "test_tax_registry_c_1.parquet")),
            "credit_bureau_a_1": glob(str(self.test_dir / "test_credit_bureau_a_1_*.parquet")),
            "credit_bureau_b_1": glob(str(self.test_dir / "test_credit_bureau_b_1.parquet")),
            "other_1": glob(str(self.test_dir / "test_other_1.parquet")),
            "person_1": glob(str(self.test_dir / "test_person_1.parquet")),
            "deposit_1": glob(str(self.test_dir / "test_deposit_1.parquet")),
            "debitcard_1": glob(str(self.test_dir / "test_debitcard_1.parquet"))
        }
        depth_2 = {
            "credit_bureau_a_2": glob(str(self.test_dir / "test_credit_bureau_a_2_*.parquet")),
            "credit_bureau_b_2": glob(str(self.test_dir / "test_credit_bureau_b_2.parquet")),
            "applprev_2":  glob(str(self.test_dir / "test_applprev_2.parquet")),
            "person_2":  glob(str(self.test_dir / "test_person_2.parquet")),
        }
        return {**base, **depth_0, **depth_1, **depth_2}
    
    def _read_file(self, path):
        return pl.read_parquet(path)
    
    def _read_files(self, paths, raw_table_info, case_id_seqment):
        chunks = []
        for path in paths:
            min_table_case_id = raw_table_info.get_chunk_info(path).get_min_value("case_id")
            max_table_case_id = raw_table_info.get_chunk_info(path).get_max_value("case_id")

            min_case_id, max_case_id = case_id_seqment
            
            if (max_case_id <= min_table_case_id) or (max_table_case_id <= min_case_id):
                pass
            else:
                chunks.append(self._read_file(path).filter((min_case_id <= pl.col("case_id")) & (pl.col("case_id") < max_case_id)))
        if len(chunks) > 0:
            return pl.concat(chunks, how="vertical_relaxed")
        else:
            return pl.DataFrame(schema = {column: raw_table_info.get_dtype(column) for column in raw_table_info.get_columns()})