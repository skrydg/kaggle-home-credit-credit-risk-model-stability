import polars as pl

from kaggle_home_credit_risk_model_stability.libs.env import Env
from kaggle_home_credit_risk_model_stability.libs.input.data_loader import DataLoader

class TableLoader:
    def __init__(self, env: Env):
        self.env = env
        self.data_loader = DataLoader(env)

    def load(self, table_name, filter=True):
        table_paths = self.data_loader._get_train_data_paths()[table_name]

        chunks = [self.data_loader._read_file(path).filter(filter) for path in table_paths]
        return pl.concat(chunks, how="vertical_relaxed")
