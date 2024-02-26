import polars as pl

from pathlib import Path

class FreatureDescription:
  def __init__(self, data_path = Path("/kaggle/input/home-credit-credit-risk-model-stability")):
    self.fd = pl.read_csv(data_path / "feature_definitions.csv")

  def get(self, feature_name):
    return self.fd.filter(self.fd["Variable"] == feature_name)["Description"][0]
