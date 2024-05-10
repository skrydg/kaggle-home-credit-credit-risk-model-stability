import polars as pl

from kaggle_home_credit_risk_model_stability.libs.env import Env
from kaggle_home_credit_risk_model_stability.libs.input.data_loader import DataLoader

class TableLoader:
    def __init__(self, env: Env):
        self.env = env
        self.data_loader = DataLoader(env)

    def load(self, table_name, filter=True, is_test=False, columns=None):
        if is_test:
            table_paths = self.data_loader._get_test_data_paths()[table_name]
        else:
            table_paths = self.data_loader._get_train_data_paths()[table_name]

        if columns is None:
            chunks = [pl.read_parquet(path).filter(filter) for path in table_paths]
        else:
            chunks = [pl.read_parquet(path, columns=columns).filter(filter) for path in table_paths]

        return pl.concat(chunks, how="vertical_relaxed")
